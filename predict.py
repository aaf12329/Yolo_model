# -*- coding: utf-8 -*-
"""克隆即用的预测入口 / Out-of-the-box prediction entry.

git clone 本仓库 → pip install -r requirements.txt → 直接运行：

    python predict.py --image path/to/face.jpg        # 单张含人脸的完整画面
    python predict.py --image some_dir --out my_out   # 整个目录批量
    python predict.py --image eye_crop.jpg --raw      # 图本身已是眼部特写（跳过人脸检测）

每张图输出：终端摘要 + 标注图（画框写结论）保存到 --out 目录。

流程 / pipeline:
    完整画面 → MediaPipe 478 点定位 → 裁左右眼（与训练数据同款扩边）
    → 模型A（models/eye_yolo26n.pt）判 睁/闭
    → 模型B（models/gaze_yolo26n.pt）判 上看/中看/下看
    （模型输入的定位由 MediaPipe 负责；两模型吃同一份裁剪）

被谁调用 / Called by: 手动 / manual（clone 后第一件事就是它）
内部调用 / Calls: ultralytics, mediapipe, cv2
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent

# 与训练数据同款眼周关键点与扩边 / same landmarks & margins as training data
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_boxes(lms, w, h):
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
    r = model.predict(crop, imgsz=128, device=device, verbose=False,
                      conf=conf_thres)[0]
    if not len(r.boxes):
        return None, 0.0
    top = int(r.boxes.conf.argmax())
    return r.names[int(r.boxes.cls[top])], float(r.boxes.conf[top])


def combine(per_eye):
    """双眼取一致；不一致取置信度高者 / prefer agreement, else higher confidence."""
    per_eye = [(c, f) for c, f in per_eye if c is not None]
    if not per_eye:
        return None, 0.0
    if len(per_eye) == 2 and per_eye[0][0] == per_eye[1][0]:
        return per_eye[0][0], min(per_eye[0][1], per_eye[1][1])
    return max(per_eye, key=lambda t: t[1])


def main():
    ap = argparse.ArgumentParser(description="眼动双模型预测 / dual-model eye prediction")
    ap.add_argument("--image", required=True, help="图片文件或目录 / image file or directory")
    ap.add_argument("--out", default="runs/predict", help="标注图输出目录 / annotated output dir")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值 / confidence threshold")
    ap.add_argument("--raw", action="store_true",
                    help="输入图已是眼部特写，跳过人脸检测 / input is already an eye crop")
    args = ap.parse_args()

    src = Path(args.image)
    files = sorted([p for p in ([src] if src.is_file() else src.glob("*"))
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    if not files:
        sys.exit(f"未找到图片 / no images under {src}")

    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"设备 / device: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    eye_model = YOLO(str(ROOT / "models" / "eye_yolo26n.pt"))
    gaze_model = YOLO(str(ROOT / "models" / "gaze_yolo26n.pt"))

    face_lm = None
    if not args.raw:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
            num_faces=1)
        face_lm = mp_vision.FaceLandmarker.create_from_options(opts)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for img_path in files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[跳过 / skip] 读不了图 / unreadable: {img_path.name}")
            continue
        h, w = frame.shape[:2]

        if args.raw:
            eyes = {"L": frame, "R": None}
            boxes = {}
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = face_lm.detect(mp_img)
            if not res.face_landmarks:
                print(f"[{img_path.name}] 未检出人脸 / no face detected")
                summary.append({"image": img_path.name, "error": "no_face"})
                continue
            boxes = dict(zip(("L", "R"),
                             eye_boxes(res.face_landmarks[0], w, h)))
            eyes = {s: frame[b[1]:b[3], b[0]:b[2]] for s, b in boxes.items()}

        result = {"image": img_path.name, "eyes": {}}
        for side, crop in eyes.items():
            if crop is None or crop.size == 0:
                continue
            eye_cls, eye_conf = top_pred(eye_model, crop, device, args.conf)
            gaze_cls, gaze_conf = top_pred(gaze_model, crop, device, args.conf)
            result["eyes"][side] = {"eye": eye_cls, "eye_conf": round(eye_conf, 3),
                                    "gaze": gaze_cls, "gaze_conf": round(gaze_conf, 3)}
            if not args.raw and side in boxes:
                x0, y0, x1, y1 = boxes[side]
                color = (0, 200, 120) if eye_cls == "open_eye" else (0, 80, 230)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{eye_cls} {eye_conf:.2f}", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"{gaze_cls} {gaze_conf:.2f}", (x0, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 60), 2)

        eye_state, eye_cf = combine([(v.get("eye"), v.get("eye_conf"))
                                     for v in result["eyes"].values()])
        gaze_dir, gaze_cf = combine([(v.get("gaze"), v.get("gaze_conf"))
                                     for v in result["eyes"].values()])
        result["final"] = {"eye": eye_state, "eye_conf": round(eye_cf, 3),
                           "gaze": gaze_dir, "gaze_conf": round(gaze_cf, 3)}
        # 指令语义 / control semantics（映射可按上层状态机调整）
        result["action_hint"] = {"look_up": "FORWARD", "look_center": "STOP",
                                 "look_down": "BACKWARD"}.get(gaze_dir, "STOP(fail-safe)")

        dst = out_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(dst), frame)
        summary.append(result)
        print(f"[{img_path.name}] 眼睛: {eye_state}({eye_cf:.2f})  "
              f"注视: {gaze_dir}({gaze_cf:.2f})  →  {result['action_hint']}  |  {dst.name}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 / done: {len(summary)} 张 → {out_dir}（含 summary.json）")


if __name__ == "__main__":
    main()
