# -*- coding: utf-8 -*-
"""开摄像头实时检测 / Live camera detection with both models.

把 predict.py 的离线管线搬进摄像头循环——本机有摄像头时"克隆即用"的实时版：

    python scripts/live_detect.py                  # 摄像头 0，实时检测
    python scripts/live_detect.py --camera 1       # 换摄像头
    python scripts/live_detect.py --conf 0.4       # 放宽置信度
    python scripts/live_detect.py --max-frames 8 --no-window   # 无窗口冒烟（自检用）

流程 / pipeline（与 predict.py 完全同款，只是输入从图片换成摄像头帧）:
    摄像头帧（水平镜像，符合体感）→ MediaPipe 478 点定位 → 裁左右眼（训练同款扩边）
    → 模型A（models/eye_yolo26n.pt）判 睁/闭
    → 模型B（models/gaze_yolo26n.pt）判 上看/中看/下看 → 映射 前进/停/后退 提示
    窗口叠加：双眼框线 + 左上结论面板（眼睛/注视/动作/FPS）+ 按 S 存截图、Q 退出。

按键 / Keys:  Q 退出 · S 存截图（runs/live_snapshots/）
被谁调用 / Called by: 手动 / manual
内部调用 / Calls: mediapipe.tasks(IMAGE 模式，每帧重定位), ultralytics.YOLO, cv2
移植说明 / Notes: eye_boxes / top_pred / combine 三个函数原样取自 predict.py，勿分叉修改。
"""
import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "runs" / "live_snapshots"

# 与训练数据同款眼周关键点 / same landmarks as training data（见 predict.py）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 注视方向 → 指令语义（同 predict.py 的 action_hint / control semantics）
GAZE_ACTION = {"look_up": "FORWARD", "look_center": "STOP", "look_down": "BACKWARD"}


def eye_boxes(lms, w, h):
    """由 478 点算左右眼裁剪框（与 predict.py 同款扩边：x1.6 / y2.2）。"""
    boxes = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in idx]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half_w = max((max(xs) - min(xs)) / 2 * 1.6, 12)
        half_h = max((max(ys) - min(ys)) / 2 * 2.2, 12)
        x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
        if x1 - x0 >= 24 and y1 - y0 >= 24:
            boxes.append((x0, y0, x1, y1))
    return boxes


def top_pred(model, crop, device, conf_thres):
    """单眼裁剪 → 模型最可信类别（与 predict.py 同款）。"""
    r = model.predict(crop, imgsz=128, device=device, verbose=False,
                      conf=conf_thres)[0]
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


def open_camera(index: int):
    """打开摄像头：先 DirectShow（Windows 成功率高），失败回退默认后端。"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头 {index} 无法打开：关闭占用它的软件，或换 --camera 1")
    return cap


def draw_overlay(frame, boxes, eye_state, eye_cf, gaze_dir, gaze_cf, fps):
    """结论画上画面：双眼框 + 左上面板（纯 cv2，英文，避免中文字体依赖）。"""
    color = (0, 200, 120) if eye_state == "open_eye" else (0, 80, 230)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    action = GAZE_ACTION.get(gaze_dir, "STOP(fail-safe)")
    lines = [
        f"EYE: {eye_state or '--'} {eye_cf:.2f}",
        f"GAZE: {gaze_dir or '--'} {gaze_cf:.2f}",
        f"ACTION: {action}",
        f"FPS: {fps:.1f}",
    ]
    cv2.rectangle(frame, (0, 0), (300, 26 * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color if i == 0 else (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description="双模型实时眼动检测 / live dual-model eye detection")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号 / camera index")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值 / confidence threshold")
    ap.add_argument("--device", default=None,
                    help="推理设备 / device：默认自动（有 GPU 用 GPU，否则 CPU）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="处理这么多帧后自动退出（0=一直跑，自检用 / auto-stop for smoke tests）")
    ap.add_argument("--no-window", action="store_true",
                    help="不弹窗口只打印结论（自检用 / headless smoke test）")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = args.device or (0 if torch.cuda.is_available() else "cpu")

    print("加载模型 / loading models...")
    eye_model = YOLO(str(ROOT / "models" / "eye_yolo26n.pt"))
    gaze_model = YOLO(str(ROOT / "models" / "gaze_yolo26n.pt"))
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
            num_faces=1))
    print(f"设备 / device: {device}")

    cap = open_camera(args.camera)
    print("实时检测已启动：Q 退出，S 存截图。CPU 上约 2~5 FPS 属正常。")
    frame_count = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败 / camera read failed")
            frame = cv2.flip(frame, 1)          # 水平镜像，符合照镜子体感
            h, w = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = face_lm.detect(mp_img)        # IMAGE 模式：每帧独立定位，鲁棒优先

            eye_state, eye_cf = None, 0.0
            gaze_dir, gaze_cf = None, 0.0
            boxes = []
            if res.face_landmarks:
                boxes = eye_boxes(res.face_landmarks[0], w, h)
                per_eye = []
                for box in boxes:
                    crop = frame[box[1]:box[3], box[0]:box[2]]
                    eye_cls, e_cf = top_pred(eye_model, crop, device, args.conf)
                    g_cls, g_cf = top_pred(gaze_model, crop, device, args.conf)
                    per_eye.append((eye_cls, e_cf))
                    per_eye.append((g_cls, g_cf))
                # 双眼各自合并：eye 与 gaze 分开结算（与 predict.py 的 final 字段一致）
                eye_state, eye_cf = combine([(v[0], v[1]) for v in per_eye[0::2]])
                gaze_dir, gaze_cf = combine([(v[0], v[1]) for v in per_eye[1::2]])

            frame_count += 1
            fps = frame_count / max(time.perf_counter() - t0, 0.001)
            if not args.no_window:
                draw_overlay(frame, boxes, eye_state, eye_cf, gaze_dir, gaze_cf, fps)
                cv2.imshow("live detect | Q quit | S snapshot", frame)

            action = GAZE_ACTION.get(gaze_dir, "STOP(fail-safe)" if gaze_dir else "--")
            print(f"[{frame_count:04d}] eye={eye_state or '--'}({eye_cf:.2f})  "
                  f"gaze={gaze_dir or '--'}({gaze_cf:.2f})  ->  {action}")

            if args.max_frames and frame_count >= args.max_frames:
                print(f"已处理 {frame_count} 帧，自动退出（--max-frames）")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                path = SNAPSHOT_DIR / f"live_{int(time.time())}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"截图已保存 / snapshot: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
