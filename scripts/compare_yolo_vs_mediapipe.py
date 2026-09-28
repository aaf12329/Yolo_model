# -*- coding: utf-8 -*-
"""睁/闭眼判定正面对比：YOLO26(本项目) vs MediaPipe EAR(EyeWheelchairProject 现状)
/ Head-to-head benchmark: YOLO26 (this repo) vs MediaPipe EAR (current pipeline).

协议 / Protocol:
    1) 校准 3s：保持睁眼，取 EAR 基线（与 blink_preview.py 完全同款：中间 80% 截尾均值）
    2) N 轮：睁眼 OPEN_SEC 秒 → 闭眼 CLOSE_SEC 秒，阶段即标签（前后留 BUFFER 过渡缓冲）
    3) 每帧同时跑两种方法：
       - MediaPipe: FaceLandmarker → 双眼 EAR（左[33,160,158,133,153,144]右[362,385,387,263,373,380]）
                    → ear < 基线*0.78 判闭（blink_preview.py 的 CLOSE_RATIO）
       - YOLO26:    同一组关键点裁出眼眶 → runs/eye_yolo26n/weights/best.pt → 睁/闭 + 置信度
    4) 输出：双方准确率/混淆矩阵/逐方法耗时 + docs/comparison/ 报告、曲线、逐帧 CSV

运行 / Run:
    python scripts/compare_yolo_vs_mediapipe.py            # 按屏幕提示做 35 秒即可
    python scripts/compare_yolo_vs_mediapipe.py --rounds 3 # 更长测试

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: mediapipe.tasks, ultralytics.YOLO, cv2
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACE_MODEL = ROOT / "models" / "face_landmarker.task"  # 复制自 EyeWheelchairProject（同款模型文件）
OUT_DIR = ROOT / "docs" / "comparison"

# 与 EyeWheelchairProject/src/interaction/blink_preview.py 保持一致 / same as upstream
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
CLOSE_RATIO = 0.78
CALIBRATION_SECONDS = 3.0
BUFFER = 0.5          # 每阶段开头/结尾不计分（换动作的过渡帧 / transition frames）
OPEN_SEC = 8.0
CLOSE_SEC = 8.0


# ---------------- MediaPipe: EAR（移植自 blink_preview.py）----------------
def _distance(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def eye_aspect_ratio(lms, idx) -> float:
    p0, p1, p2, p3, p4, p5 = [lms[i] for i in idx]
    horizontal = _distance(p0, p3)
    if horizontal < 1e-6:
        return 0.0
    return (_distance(p1, p5) + _distance(p2, p4)) / (2.0 * horizontal)


def average_ear(lms) -> float:
    return (eye_aspect_ratio(lms, LEFT_EYE) + eye_aspect_ratio(lms, RIGHT_EYE)) / 2.0


def eye_crop(frame, lms, idx, expand_x=1.6, expand_y=2.2):
    """按关键点外接框裁眼眶（YOLO 输入）/ crop eye region from landmarks."""
    h, w = frame.shape[:2]
    pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in idx]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max((max(xs) - min(xs)) / 2 * expand_x, 12)
    half_h = max((max(ys) - min(ys)) / 2 * expand_y, 12)
    x0, x1 = max(0, int(cx - half_w)), min(w, int(cx + half_w))
    y0, y1 = max(0, int(cy - half_h)), min(h, int(cy + half_h))
    return frame[y0:y1, x0:x1]


def banner(frame, text, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), color, -1)
    cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from ultralytics import YOLO

    yolo = YOLO(str(ROOT / "runs" / "eye_yolo26n" / "weights" / "best.pt"))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    phases = [("open", OPEN_SEC), ("closed", CLOSE_SEC)] * args.rounds
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"摄像头打开失败 / cannot open camera {args.camera}")

    # ---- 阶段0：校准（MediaPipe 专属优势：个人基线）/ calibration ----
    baseline, calib_vals, t_start = None, [], time.perf_counter()
    ts_ms = 0
    while baseline is None:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("读帧失败 / frame read failed")
        ts_ms += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, ts_ms)
        remaining = CALIBRATION_SECONDS - (time.perf_counter() - t_start)
        if res.face_landmarks:
            calib_vals.append(average_ear(res.face_landmarks[0]))
        banner(frame, f"校准: 保持睁眼 {max(0, remaining):.1f}s", (160, 160, 160))
        cv2.imshow("YOLO26 vs MediaPipe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit("中止 / aborted")
        elapsed = time.perf_counter() - t_start
        if elapsed >= CALIBRATION_SECONDS and len(calib_vals) >= 10:
            vals = sorted(calib_vals)
            trim = max(1, len(vals) // 10)
            baseline = sum(vals[trim:-trim] or vals) / len(vals[trim:-trim] or vals)
    mp_close_thr = baseline * CLOSE_RATIO

    # ---- 主实验：逐帧双方法 / main benchmark ----
    rows, samples, t0 = [], [], time.perf_counter()
    phases_left = list(phases)
    phase_name, phase_len = phases_left.pop(0)
    phase_t0 = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now - phase_t0 >= phase_len:
            if not phases_left:
                break
            phase_name, phase_len = phases_left.pop(0)
            phase_t0 = time.perf_counter()
            continue
        ok, frame = cap.read()
        if not ok:
            continue
        ts_ms += 33

        # MediaPipe 分支
        t_mp0 = time.perf_counter()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, ts_ms)
        ear = average_ear(res.face_landmarks[0]) if res.face_landmarks else None
        mp_pred = None if ear is None else ("closed" if ear < mp_close_thr else "open")
        t_mp = (time.perf_counter() - t_mp0) * 1000

        # YOLO26 分支（增量 = 裁眼 + 推理；总耗时 = MediaPipe + 增量）
        t_y0 = time.perf_counter()
        if res.face_landmarks:
            lms = res.face_landmarks[0]
            crops = [eye_crop(frame, lms, LEFT_EYE), eye_crop(frame, lms, RIGHT_EYE)]
            crops = [c for c in crops if c.size and min(c.shape[:2]) >= 16]
            preds = yolo.predict(crops, imgsz=128, device=0, verbose=False, conf=args.conf)
            eye_states = [p.names[int(p.boxes.cls[int(p.boxes.conf.argmax())])] for p in preds if len(p.boxes)]
            yolo_pred = ("closed" if eye_states and all(s == "closed" for s in eye_states)
                         else "open" if eye_states else None)
            yolo_conf = float(np.mean([float(p.boxes.conf.max()) for p in preds if len(p.boxes)])) if eye_states else 0.0
        else:
            yolo_pred, yolo_conf = None, 0.0
        t_y = (time.perf_counter() - t_y0) * 1000

        label = phase_name
        in_buffer = (now - phase_t0 < BUFFER) or (phase_len - (now - phase_t0) < BUFFER)
        rows.append([label, "skip" if in_buffer else "count", ear, mp_pred, yolo_pred,
                     yolo_conf, round(t_mp, 2), round(t_y, 2)])
        if not in_buffer:
            samples.append((label, mp_pred, yolo_pred))

        color = {"open": (80, 200, 120), "closed": (60, 120, 230)}[phase_name]
        banner(frame, f"{'睁眼 OPEN' if phase_name == 'open' else '闭眼 CLOSED'} "
                      f"{phase_len - (now - phase_t0):.1f}s", color)
        cv2.putText(frame, f"MediaPipe: {mp_pred}  ear={ear:.3f}" if ear else
                    "MediaPipe: no face", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"YOLO26: {yolo_pred}  conf={yolo_conf:.2f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)
        cv2.imshow("YOLO26 vs MediaPipe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---- 统计 / metrics ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase_label", "counted", "ear", "mp_pred", "yolo_pred", "yolo_conf",
                    "mp_ms", "yolo_increment_ms"])
        w.writerows(rows)

    def acc(pred_idx):
        counted = [(l, p[pred_idx]) for l, _, p in samples if p[pred_idx] is not None]
        miss = len(samples) - len(counted)
        correct = sum(1 for l, p in counted if l == p)
        open_ok = [l == p for l, p in counted if l == "open"]
        closed_ok = [l == p for l, p in counted if l == "closed"]
        return {
            "n": len(samples), "miss": miss, "acc": correct / max(1, len(counted)),
            "open_acc": sum(open_ok) / max(1, len(open_ok)),
            "closed_acc": sum(closed_ok) / max(1, len(closed_ok)),
        }

    mp_m, yolo_m = acc(1), acc(2)
    mp_ms = np.mean([r[6] for r in rows])
    yolo_inc = np.mean([r[7] for r in rows])
    n_total = len(rows)

    lines = [
        "# YOLO26 vs MediaPipe(EAR) 睁/闭眼判定对比 / Benchmark",
        "",
        f"日期 / date: {time.strftime('%Y-%m-%d %H:%M')}　协议: 校准3s + "
        f"{args.rounds}轮 x (睁眼{OPEN_SEC:.0f}s + 闭眼{CLOSE_SEC:.0f}s)，共 {n_total} 帧",
        "地面真值 = 阶段标签（协议法）；阶段首尾 0.5s 过渡帧不计分。",
        "",
        "| 方法 / method | 总准确率 / acc | 睁眼准确率 | 闭眼准确率 | 漏检帧 | 单帧耗时 |",
        "|---|---|---|---|---|---|",
        f"| MediaPipe EAR（现状） | {mp_m['acc']*100:.1f}% | {mp_m['open_acc']*100:.1f}% | "
        f"{mp_m['closed_acc']*100:.1f}% | {mp_m['miss']} | {mp_ms:.1f} ms |",
        f"| YOLO26（本项目） | {yolo_m['acc']*100:.1f}% | {yolo_m['open_acc']*100:.1f}% | "
        f"{yolo_m['closed_acc']*100:.1f}% | {yolo_m['miss']} | 增量 {yolo_inc:.1f} ms "
        f"(含MediaPipe共 {mp_ms + yolo_inc:.1f} ms) |",
        "",
        "## 判读 / notes",
        "- 睁眼段的自然眨眼会被双方同时误判为闭眼（协议标签不含瞬时眨眼），双方同罪，对比仍公平",
        "  / natural blinks during open phase penalize both methods equally",
        "- MediaPipe 依赖个人 EAR 基线校准（本实验已给足 3s 校准）；YOLO26 无需校准，跨人即用",
        f"- 逐帧数据 / per-frame data: frames.csv（{n_total} 行）",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ["MediaPipe EAR\n(现状)", "YOLO26\n(本项目)"]
        accs = [mp_m["acc"] * 100, yolo_m["acc"] * 100]
        bars = ax.bar(methods, accs, color=["#2e7d32", "#ef6c00"], width=0.5)
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=13)
        ax.set_ylim(0, 105)
        ax.set_ylabel("准确率 / accuracy %")
        ax.set_title("睁/闭眼判定对比 / open-closed eye benchmark (同一批帧)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "accuracy_chart.png", dpi=150)
    except Exception as e:  # 画图失败不影响报告
        print("chart skipped:", e)

    print("\n".join(lines))
    print("\n报告已保存 / report saved:", OUT_DIR / "report.md")


if __name__ == "__main__":
    main()
