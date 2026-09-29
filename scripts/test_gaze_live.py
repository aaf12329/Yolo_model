# -*- coding: utf-8 -*-
"""眼球转动（注视方向）模型实时测试 / Live gaze-direction model test.

测的是模型 B：models/gaze_yolo26n.pt（上看 look_up / 平视 look_center / 下看 look_down）。

协议 / Protocol（照 compare_yolo_vs_mediapipe.py 的"协议法"）:
    屏幕上出现一个目标点（顶部 / 中央 / 底部）→ 按要求把视线移过去 → 每帧记录模型输出
    → 与"目标位置"这个地面真值对比，算准确率与混淆矩阵。
    每阶段首尾各留 BUFFER 秒过渡帧不计分（视线移动需要时间）。

    默认：3 个方向 × 2 轮 × 每段 6 秒 ≈ 45 秒（含过渡），随时按 Q 中止。

门控 / Gating:
    闭眼时注视方向没有物理意义（模型 B 的训练样本均为睁眼），所以先跑模型 A：
    眼睛不是 open_eye 的帧记为 skipped，**不计入准确率**，单独统计数量。

运行 / Run:
    python scripts/test_gaze_live.py                       # 默认 2 轮
    python scripts/test_gaze_live.py --rounds 3 --look-sec 8
    python scripts/test_gaze_live.py --max-frames 30 --no-window   # 自检（不弹窗，只写文件）

输出 / Output（docs/gaze_test/）:
    frames.csv                  逐帧：阶段标签、是否计分、眼睛、注视预测、置信度、耗时
    report.md                   准确率表 + 混淆矩阵 + 判读
    gaze_confusion_matrix.png   混淆矩阵图

被谁调用 / Called by: 手动 / manual（有摄像头时跑）
内部调用 / Calls: mediapipe.tasks(VIDEO 模式), ultralytics.YOLO ×2, cv2, matplotlib(可选)
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
GAZE_MODEL = ROOT / "models" / "gaze_yolo26s.pt"   # 主控 3 类版（95.62%）；5 类实验版是 gaze5_yolo26s.pt
EYE_MODEL = ROOT / "models" / "eye_yolo26n.pt"
OUT_DIR = ROOT / "docs" / "gaze_test"

# 与训练数据同款眼周关键点与扩边 / same landmarks & margins as training data（见 predict.py）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EXPAND_X, EXPAND_Y = 1.6, 2.2

BUFFER = 0.7          # 每阶段首尾不计分的过渡时长（视线移动比眨眼慢）
LOOK_SEC = 6.0        # 每个方向的注视时长
PHASE_ORDER = ["look_up", "look_center", "look_down"]

PHASE_TEXT = {        # 屏幕提示（含目标点该画在哪：y 比例）
    "look_up": ("抬眼向上看 / LOOK UP", 0.12),
    "look_center": ("平视中央 / LOOK CENTER", 0.5),
    "look_down": ("垂眼向下看 / LOOK DOWN", 0.88),
}

# 注视方向 → 指令语义（与 predict.py / live_detect.py 一致）
GAZE_ACTION = {"look_up": "FORWARD", "look_center": "STOP", "look_down": "BACKWARD"}
CHANGE_STABLE_FRAMES = 2   # 新方向连续出现这么多帧才播报（防单帧抖动刷屏）


class DirectionAnnouncer:
    """方向变更播报器：只在"稳定确认的新方向"出现时产出一行，不逐帧刷屏。

    调用关系：被 main() 每帧调用一次（update），返回 None 或一行待打印的文字。
    三条规则（每条都能单独测）：
      1. 无效帧（无人脸 / 闭眼）不参与判断，也不打断已播报的方向；
      2. 新方向必须连续出现 CHANGE_STABLE_FRAMES 帧才作数（防单帧抖动）；
      3. 与上次播报相同则不重复播报（状态没变就不打印）。
    """

    def __init__(self, stable_frames: int = CHANGE_STABLE_FRAMES) -> None:
        self._stable_frames = stable_frames
        self.current: str | None = None     # 已经播报出去的方向
        self._pending: str | None = None    # 正在攒帧数的新方向
        self._count = 0

    def update(self, gaze_pred, conf: float, valid: bool) -> str | None:
        if not valid or gaze_pred is None:
            return None                     # 无效帧：不判断，也不清空已播报状态
        if gaze_pred == self.current:
            self._pending, self._count = None, 0
            return None
        if gaze_pred == self._pending:
            self._count += 1
        else:
            self._pending, self._count = gaze_pred, 1
        if self._count < self._stable_frames:
            return None
        previous, self.current = self.current, gaze_pred
        self._pending, self._count = None, 0
        action = GAZE_ACTION.get(gaze_pred, "STOP(fail-safe)")
        return (f"[方向变更] {previous or '--'} → {gaze_pred}  "
                f"conf={conf:.2f}  动作={action}")


def eye_crops(frame, lms):
    """裁左右眼（与 predict.py / compare_yolo_vs_mediapipe.py 同款扩边）。"""
    h, w = frame.shape[:2]
    crops = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in idx]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_w = max((max(xs) - min(xs)) / 2 * EXPAND_X, 12)
        half_h = max((max(ys) - min(ys)) / 2 * EXPAND_Y, 12)
        x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            crops.append(frame[y0:y1, x0:x1])
    return crops


def top_pred(model, crop, device, conf_thres):
    """单个裁剪 → 最可信类别 + 置信度（与 predict.py 同款）。"""
    r = model.predict(crop, imgsz=128, device=device, verbose=False, conf=conf_thres)[0]
    if not len(r.boxes):
        return None, 0.0
    top = int(r.boxes.conf.argmax())
    return r.names[int(r.boxes.cls[top])], float(r.boxes.conf[top])


def combine(per_eye):
    """双眼取一致；不一致取置信度高者（与 predict.py 同款）。"""
    per_eye = [(c, f) for c, f in per_eye if c is not None]
    if not per_eye:
        return None, 0.0
    if len(per_eye) == 2 and per_eye[0][0] == per_eye[1][0]:
        return per_eye[0][0], min(per_eye[0][1], per_eye[1][1])
    return max(per_eye, key=lambda t: t[1])


def draw_stage(frame, phase, remain, pred, conf, eye_state, fps):
    """屏幕提示：目标点 + 大字提示 + 倒计时 + 模型实时输出。"""
    h, w = frame.shape[:2]
    text, y_ratio = PHASE_TEXT[phase]
    tx, ty = int(w / 2), int(h * y_ratio)          # 注视目标点
    cv2.circle(frame, (tx, ty), 22, (60, 220, 60), -1)
    cv2.circle(frame, (tx, ty), 26, (255, 255, 255), 2)
    cv2.putText(frame, text, (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 60), 3)
    cv2.putText(frame, f"{remain:4.1f}s", (w - 170, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    lines = [
        f"GAZE: {pred or '--'} {conf:.2f}",
        f"EYE:  {eye_state or '--'}",
        f"FPS:  {fps:.1f}",
    ]
    cv2.rectangle(frame, (0, 0), (290, 26 * len(lines) + 10), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255) if i else (60, 220, 60), 2)
    return frame


def summarize(samples):
    """samples = [(label, pred)]，只含"计分且检出"的帧 → 总体/分类准确率 + 3×3 混淆矩阵。"""
    labels = PHASE_ORDER
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for label, pred in samples:
        if label in matrix and pred in matrix[label]:
            matrix[label][pred] += 1
    per_class = {}
    for t in labels:
        total = sum(matrix[t].values())
        per_class[t] = matrix[t][t] / total if total else 0.0
    correct = sum(matrix[t][t] for t in labels)
    return {
        "n": len(samples),
        "acc": correct / max(1, len(samples)),
        "per_class": per_class,
        "matrix": matrix,
    }


def write_files(rows, samples, skipped, args, elapsed):
    """写 frames.csv + report.md + 混淆矩阵图（照 compare_yolo_vs_mediapipe.py 的产出风格）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase_label", "counted", "eye_pred", "eye_conf", "gaze_pred", "gaze_conf", "ms"])
        w.writerows(rows)

    m = summarize(samples)
    lines = [
        "# 注视方向模型（YOLO26）实时测试 / Live gaze-direction benchmark",
        "",
        f"日期 / date: {time.strftime('%Y-%m-%d %H:%M')}　"
        f"协议: {args.rounds} 轮 × ({' + '.join(PHASE_ORDER)})，每段 {args.look_sec:.0f}s，"
        f"首尾 {BUFFER}s 过渡不计分；总耗时 {elapsed:.0f}s",
        "地面真值 = 屏幕目标点位置（协议法）。闭眼帧经模型 A 门控后不计分。",
        "",
        f"- 计分帧 / counted: **{m['n']}**　"
        f"未计分 / skipped: 无人脸 {skipped['no_face']}，闭眼 {skipped['eyes_closed']}"
        + ("（门控已关闭 / gate off）" if args.no_gate else ""),
        f"- **总准确率 / overall accuracy: {m['acc']*100:.1f}%**",
        "",
        "| 目标 / target | 准确率 / accuracy |",
        "|---|---|",
    ]
    for t in PHASE_ORDER:
        lines.append(f"| {t} | {m['per_class'][t]*100:.1f}% |")
    lines += ["", "## 混淆矩阵 / confusion matrix（行=目标，列=预测）", "",
              "| target \\ pred | " + " | ".join(PHASE_ORDER) + " |",
              "|---|" + "---|" * len(PHASE_ORDER)]
    for t in PHASE_ORDER:
        lines.append(f"| {t} | " + " | ".join(str(m["matrix"][t][p]) for p in PHASE_ORDER) + " |")
    lines += [
        "",
        "## 判读 / notes",
        "- 若 up/down 互相串（上↔下），说明模型过度依赖「眼球在眼眶中的位置」，个人化微调时要补数据",
        "  / up↔down confusion means the model over-relies on eye position — personalise with your own data",
        "- 单帧耗时含 MediaPipe 定位 + 双眼 2 模型；纯 CPU 约 60~80ms 属正常",
        f"- 逐帧数据 / per-frame data: frames.csv（{len(rows)} 行）",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        mat = np.array([[m["matrix"][t][p] for p in PHASE_ORDER] for t in PHASE_ORDER], dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(mat, cmap="Blues")
        for i in range(len(PHASE_ORDER)):
            for j in range(len(PHASE_ORDER)):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() * 0.6 else "black", fontsize=13)
        ax.set_xticks(range(len(PHASE_ORDER)), PHASE_ORDER, rotation=20)
        ax.set_yticks(range(len(PHASE_ORDER)), PHASE_ORDER)
        ax.set_xlabel("预测 / predicted")
        ax.set_ylabel("目标 / target")
        ax.set_title(f"注视方向混淆矩阵 / gaze confusion（acc={m['acc']*100:.1f}%）")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "gaze_confusion_matrix.png", dpi=150)
    except Exception as e:  # 画图失败不影响报告
        print("chart skipped:", e)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="注视方向模型实时测试 / live gaze-direction test")
    ap.add_argument("--rounds", type=int, default=2, help="轮数 / rounds")
    ap.add_argument("--look-sec", type=float, default=LOOK_SEC, help="每段注视时长（秒）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="处理这么多帧后自动退出（0=按协议跑完；自检用）")
    ap.add_argument("--no-window", action="store_true", help="不弹窗口（自检用）")
    ap.add_argument("--no-gate", action="store_true",
                    help="关闭模型A闭眼门控：闭眼帧也计分（诊断模型A误判时用，2026-09-29 实测"
                         "93%% 帧被门控误杀，根因=眼镜反光+低头姿态的域差）")
    ap.add_argument("--dump-crops", type=int, default=0,
                    help="每 N 帧把眼部裁剪图存到 docs/gaze_test/crops/<阶段>/（诊断模型输入质量用）")
    args = ap.parse_args()
    if args.look_sec <= 2 * BUFFER:
        raise SystemExit(
            f"--look-sec 太小（现在 {args.look_sec}s）：每段首尾各有 {BUFFER}s 过渡不计分，"
            f"必须 > {2 * BUFFER:.1f}s 才可能计分，建议 ≥ 4s。")

    import torch
    from ultralytics import YOLO
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    device = 0 if torch.cuda.is_available() else "cpu"
    print("加载模型 / loading models...")
    gaze_model = YOLO(str(GAZE_MODEL))
    eye_model = YOLO(str(EYE_MODEL))
    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            num_faces=1, running_mode=mp_vision.RunningMode.VIDEO))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"摄像头打开失败 / cannot open camera {args.camera}")

    phases = [p for _ in range(args.rounds) for p in PHASE_ORDER]
    rows, samples = [], []
    skipped = {"no_face": 0, "eyes_closed": 0}
    announcer = DirectionAnnouncer()      # 方向变更只打印一次（边沿触发，见类的文档）
    ts_ms, frame_count = 0, 0
    t_start = time.perf_counter()
    phase_idx, phase_t0 = 0, time.perf_counter()
    print("测试开始：跟着屏幕上的绿点看（上/中/下），Q 中止。")

    try:
        while phase_idx < len(phases):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("读帧失败 / frame read failed")
            # ⚠️ frame 保持未镜像：模型吃原始帧（镜像会把左/右语义反转）；
            # 镜像仅用于显示（下面 disp），绿点在水平中线不受影响。
            now = time.perf_counter()
            if now - phase_t0 >= args.look_sec:
                phase_idx += 1
                phase_t0 = now
                if phase_idx >= len(phases):
                    break
            phase = phases[phase_idx]

            ts_ms += 33
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts_ms)

            t0 = time.perf_counter()
            eye_state, eye_conf, gaze_pred, gaze_conf = None, 0.0, None, 0.0
            if res.face_landmarks:
                crops = eye_crops(frame, res.face_landmarks[0])
                if crops:
                    eye_preds = [top_pred(eye_model, c, device, args.conf) for c in crops]
                    gaze_preds = [top_pred(gaze_model, c, device, args.conf) for c in crops]
                    eye_state, eye_conf = combine(eye_preds)
                    gaze_pred, gaze_conf = combine(gaze_preds)
            ms = (time.perf_counter() - t0) * 1000

            in_buffer = (now - phase_t0 < BUFFER) or (args.look_sec - (now - phase_t0) < BUFFER)
            counted = "skip" if in_buffer else "count"
            if not res.face_landmarks:
                skipped["no_face"] += 1
                counted = "skip"
            elif eye_state != "open_eye" and not args.no_gate:
                skipped["eyes_closed"] += 1
                counted = "skip"          # 闭眼帧：注视方向无意义，不计分（--no-gate 关闭此门控）
            if res.face_landmarks and args.dump_crops and frame_count % args.dump_crops == 0:
                crop_dir = OUT_DIR / "crops" / phase
                crop_dir.mkdir(parents=True, exist_ok=True)
                for side, c in zip(("L", "R"), eye_crops(frame, res.face_landmarks[0])):
                    cv2.imwrite(str(crop_dir / f"f{frame_count:05d}_{side}.jpg"), c)
            rows.append([phase, counted, eye_state, round(eye_conf, 3), gaze_pred,
                         round(gaze_conf, 3), round(ms, 1)])
            if counted == "count" and gaze_pred is not None:
                samples.append((phase, gaze_pred))

            frame_count += 1
            # 方向变更播报：只在稳定确认的新方向出现时打印（闭眼/无人脸帧不算）
            line = announcer.update(gaze_pred, gaze_conf, valid=(eye_state == "open_eye"))
            if line:
                print(f"{line}   @ {now - t_start:.1f}s")
            remain = max(0.0, args.look_sec - (now - phase_t0))
            fps = frame_count / max(now - t_start, 0.001)
            if not args.no_window:
                disp = cv2.flip(frame, 1)   # 显示用镜像（体感），绿点在中线不受影响
                cv2.imshow("gaze test | Q abort", draw_stage(disp, phase, remain, gaze_pred,
                                                             gaze_conf, eye_state, fps))
            if args.max_frames and frame_count >= args.max_frames:
                print(f"已处理 {frame_count} 帧，自动退出（--max-frames）")
                break
            if not args.no_window and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("用户中止 / aborted by user")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    report = write_files(rows, samples, skipped, args, elapsed)
    print("\n" + report)
    print("\n报告已保存 / report saved:", OUT_DIR / "report.md")


if __name__ == "__main__":
    main()
