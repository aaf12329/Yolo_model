# -*- coding: utf-8 -*-
"""眼球状态实时查看器 / Live eye-state viewer（无校准、无协议，直接输出）。

模型仍用 YOLO：模型 A 判睁/闭眼（models/eye_yolo26n.pt），
模型 B 判注视方向（默认 models/gaze5_yolo26s.pt，5 类含左右；3 类主控版用 --model 切换）。

**没有绿点、没有阶段协议、没有校准**——启动后直接持续输出当前眼球状态：
    屏幕面板：当前：看左 / 直视 / ……（中文大字 + 动作 + 置信度 + FPS）
    终端：只在状态稳定变化时打印一行（[方向变更] 直视 → 看左 | 动作=TURN_LEFT）

按 Q 退出；按 S 存一帧截图（runs/live_snapshots/，存未镜像原始帧）。

诊断开关（2026-09-29 首次实测复盘时加入，保留）：
    --no-gate     闭眼帧也更新注视输出（默认闭眼时注视不更新——方向无物理意义）
    --dump-crops N  每 N 帧把眼部裁剪图存 docs/gaze_test/crops/（诊断模型输入质量）

文件分三段（结构参考 EyeWheelchairProject/src/interaction/gaze_direction_preview.py）：
  1) 判定逻辑（DirectionAnnouncer + 双眼合并，纯逻辑可单测）
  2) 摄像头与画面（开摄像头、建识别器、中文面板）
  3) 主流程 main()：读帧 → 判 → 显 → 播

⚠️ 两条铁律（AGENTS.md / README 已知限制）：
  1. 模型输入必须是未镜像帧（镜像会把看左/看右反转）；镜像只用于显示层。
  2. 实验室指标 ≠ 部署域指标：本工具即"部署域实测"工具（域差诊断见
     docs/gaze_test/domain_gap_analysis.md）。

被谁调用 / Called by: 手动 / manual（有摄像头时跑）
内部调用 / Calls: mediapipe.tasks(VIDEO 模式), ultralytics.YOLO ×2, cv2, PIL
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np  # noqa: F401  （draw_text 的 cv2↔PIL 转换需要）
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
EYE_MODEL = ROOT / "models" / "eye_yolo26n.pt"
GAZE_MODEL_DEFAULT = ROOT / "models" / "gaze5_yolo26s.pt"   # 5 类（含左右）
SNAPSHOT_DIR = ROOT / "runs" / "live_snapshots"
CROP_DIR = ROOT / "docs" / "gaze_test" / "crops"

# 与训练数据同款眼周关键点与扩边 / same landmarks & margins as training data
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EXPAND_X, EXPAND_Y = 1.6, 2.2

# 类别的中文与动作映射（3 类模型只有前三个键，一样能用）
GAZE_ZH = {"look_up": "上看", "look_center": "直视", "look_down": "下看",
           "look_left": "看左", "look_right": "看右"}
GAZE_ACTION = {"look_up": "FORWARD", "look_center": "STOP", "look_down": "BACKWARD",
               "look_left": "TURN_LEFT", "look_right": "TURN_RIGHT"}
EYE_ZH = {"open_eye": "睁眼", "closed_eye": "闭眼"}

CHANGE_STABLE_FRAMES = 2   # 新方向连续出现这么多帧才播报（防单帧抖动）
CAMERA_INDEX = 0
CAMERA_W = 1280
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)


# ============================ 1) 判定逻辑（纯逻辑） ============================


class DirectionAnnouncer:
    """方向变更播报器：只在"稳定确认的新方向"出现时产出一行，不逐帧刷屏。

    三条规则（每条都能单独测）：
      1. 无效帧（无人脸/闭眼）不参与判断，也不打断已播报的方向；
      2. 新方向必须连续出现 stable_frames 帧才作数（防单帧抖动）；
      3. 与上次播报相同则不重复播报（状态没变就不打印）。
    """

    def __init__(self, stable_frames: int = CHANGE_STABLE_FRAMES) -> None:
        self._stable_frames = stable_frames
        self.current: str | None = None
        self._pending: str | None = None
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
        return (f"[方向变更] {previous or '--'} → {GAZE_ZH.get(gaze_pred, gaze_pred)}"
                f"({gaze_pred})  conf={conf:.2f}  动作={action}")


def eye_crops(frame, lms):
    """裁左右眼（与 predict.py / 训练数据同款扩边：x1.6 / y2.2）。"""
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


# ============================ 2) 摄像头与画面 ============================


def open_camera(index: int):
    """打开摄像头：先 DirectShow（Windows 成功率高），失败回退默认后端。"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头 {index} 无法打开：关闭占用它的软件，或换 --camera 1")
    return cap


def create_landmarker():
    """建 MediaPipe 人脸识别器（VIDEO 模式，478 点，含虹膜）。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


_font_cache: dict[tuple[Path, int], "ImageFont.FreeTypeFont"] = {}


def chinese_font(size: int):
    """找第一个可用中文字体并缓存（OpenCV 自带字体画不了中文，用 PIL 代替）。"""
    for path in FONT_CANDIDATES:
        if not path.exists():
            continue
        cached = _font_cache.get((path, size))
        if cached is None:
            cached = ImageFont.truetype(str(path), size)
            _font_cache[(path, size)] = cached
        return cached
    raise FileNotFoundError("找不到可用中文字体：" + "、".join(str(p) for p in FONT_CANDIDATES))


def draw_text(frame, text, xy, size, color):
    """在画面上画一行中文；不改传入 frame，返回画好的新图。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(xy, text, font=chinese_font(size), fill=color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_panel(disp, eye_state, eye_conf, gaze_dir, gaze_conf, fps):
    """左上角面板：当前眼球状态（大字）+ 动作 + 眼睛状态 + FPS。"""
    gaze_zh = GAZE_ZH.get(gaze_dir, "无人脸" if gaze_dir is None and eye_state is None
                          else gaze_dir or "--")
    action = GAZE_ACTION.get(gaze_dir, "STOP(fail-safe)")
    eye_zh = EYE_ZH.get(eye_state, "无人脸" if eye_state is None else "--")
    lines = [
        (f"当前：{gaze_zh}", 34, (60, 220, 60)),
        (f"动作: {action}   置信度: {gaze_conf:.2f}", 20, (255, 255, 255)),
        (f"眼睛: {eye_zh} ({eye_conf:.2f})   FPS: {fps:.1f}", 18, (190, 190, 190)),
    ]
    cv2.rectangle(disp, (0, 0), (400, 128), (0, 0, 0), -1)
    y = 12
    for text, size, color in lines:
        disp = draw_text(disp, text, (12, y), size, color)
        y += size + 12
    return disp


# ============================ 3) 主流程 ============================


def main() -> None:
    ap = argparse.ArgumentParser(description="眼球状态实时查看器（无校准无协议，YOLO 直出）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", default=str(GAZE_MODEL_DEFAULT),
                    help="注视模型：默认 5 类（含左右）；3 类主控版 models/gaze_yolo26s.pt")
    ap.add_argument("--device", default=None, help="推理设备 / device：默认自动")
    ap.add_argument("--stable-frames", type=int, default=CHANGE_STABLE_FRAMES,
                    help="新方向连续多少帧才播报变化（防抖）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="处理这么多帧后自动退出（0=一直跑，自检用）")
    ap.add_argument("--no-window", action="store_true",
                    help="不弹窗口只打印（自检用 / headless smoke test）")
    ap.add_argument("--no-gate", action="store_true",
                    help="闭眼帧也更新注视输出（默认闭眼时注视不更新——方向无物理意义）")
    ap.add_argument("--dump-crops", type=int, default=0,
                    help="每 N 帧把眼部裁剪图存 docs/gaze_test/crops/（诊断模型输入质量）")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python  # noqa: F401
    from mediapipe.tasks.python import vision as mp_vision
    device = args.device or (0 if torch.cuda.is_available() else "cpu")

    print("加载模型 / loading models...")
    eye_model = YOLO(str(EYE_MODEL))
    gaze_model = YOLO(str(args.model))
    print(f"注视模型类别 / gaze classes: {gaze_model.names}")
    landmarker = create_landmarker()
    announcer = DirectionAnnouncer(args.stable_frames)
    print(f"设备 / device: {device}")

    cap = open_camera(args.camera)
    print("眼球状态实时查看器已启动（无校准无协议）：Q 退出，S 存截图。")
    print(f"终端只在状态稳定变化时打印（连续 {args.stable_frames} 帧确认）。")

    ts_ms, frame_count, crop_count = 0, 0, 0
    t0 = time.perf_counter()
    eye_state, eye_conf, gaze_dir, gaze_conf = None, 0.0, None, 0.0
    try:
        while True:
            # ---- 看：取一帧（模型吃未镜像帧；镜像只用于显示）----
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败 / camera read failed")
            now = time.perf_counter()
            ts_ms += 33
            h, w = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(mp_img, ts_ms)
            lms = res.face_landmarks[0] if res.face_landmarks else None

            # ---- 判：裁眼 → 模型A（睁/闭）+ 模型B（方向）双眼合并 ----
            if lms is not None:
                crops = eye_crops(frame, lms)
                if crops:
                    eye_preds = [top_pred(eye_model, c, device, args.conf) for c in crops]
                    gaze_preds = [top_pred(gaze_model, c, device, args.conf) for c in crops]
                    eye_state, eye_conf = combine(eye_preds)
                    gaze_dir, gaze_conf = combine(gaze_preds)
                if args.dump_crops and frame_count % args.dump_crops == 0:
                    CROP_DIR.mkdir(parents=True, exist_ok=True)
                    for side, c in zip(("L", "R"), eye_crops(frame, lms)):
                        cv2.imwrite(str(CROP_DIR / f"f{frame_count:05d}_{side}.jpg"), c)
                        crop_count += 1

            frame_count += 1
            fps = frame_count / max(now - t0, 0.001)

            # ---- 显：镜像显示（体感）+ 中文面板（当前眼球状态）----
            if not args.no_window:
                disp = cv2.flip(frame, 1)
                disp = draw_panel(disp, eye_state, eye_conf, gaze_dir, gaze_conf, fps)
                cv2.imshow("eye state | Q quit | S snapshot", disp)

            # ---- 播：终端只在状态稳定变化时打印一行 ----
            valid = (eye_state == "open_eye") or args.no_gate
            line = announcer.update(gaze_dir, gaze_conf, valid=valid and gaze_dir is not None)
            if line:
                print(f"{line}   @ {now - t0:.1f}s")

            if args.max_frames and frame_count >= args.max_frames:
                print(f"已处理 {frame_count} 帧，自动退出（--max-frames）")
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                snap = SNAPSHOT_DIR / f"live_{int(time.time())}.jpg"
                cv2.imwrite(str(snap), frame)   # 存未镜像原始帧
                print(f"截图已存 / snapshot: {snap}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if crop_count:
            print(f"诊断裁剪已存 / diag crops: {CROP_DIR}（{crop_count} 张）")


if __name__ == "__main__":
    main()
