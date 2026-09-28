# -*- coding: utf-8 -*-
"""Columbia Gaze → 眼部裁剪 YOLO 数据集（模型 B 初版数据）
/ Convert Columbia Gaze to eye-crop YOLO dataset for gaze-direction Model B.

数据 / Dataset:
    Columbia Gaze Data Set（56 人 × 5 头姿 × 7 水平 × 3 垂直 = 5,880 张，5184×3456）
    下载 / download: https://cave.cs.columbia.edu/repository/ColumbiaGazeDataSet
    许可 / license: 仅非商用，引用 Smith et al., UIST 2013

文件名标注（已抽样目视验证符号）/ filename labels (sign visually verified):
    0001_2m_-15P_-10V_-10H.jpg
    [0]受试者 [1]距离 [3]头部姿态P [4]垂直注视V [5]水平注视H
    V: -10 = 往上看(look_up) | 0 = 平视(look_center) | +10 = 往下看(look_down)
    注意：正号在文件名里省略，"往下"是正值，与直觉相反！

流水线 / pipeline:
    1) 全图等比缩到宽 1280（18MP 原图 CPU 上直接跑 MediaPipe 太慢）
    2) MediaPipe FaceLandmarker 检测 478 点 → 裁左/右眼（含眉毛上下文，与
       compare_yolo_vs_mediapipe.py 同款关键点与扩边）
    3) 每张源图产出 2 个眼部裁剪（左右眼共享同一垂直注视标签）
    4) 整图单框 YOLO 标签；按受试者划分 train/val（每第 10 人进 val）
    5) 写 configs/columbia_gaze.yaml（3 类）

产出 / output: datasets/columbia_gaze_yolo/ + configs/columbia_gaze.yaml
    裁剪统一缩放到 128px 长边（与模型 A 的 MRL 83px 同量级）

运行 / Run:
    python scripts/prepare_columbia_gaze_dataset.py

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: cv2, mediapipe, pathlib（MediaPipe 在 CPU 上跑，约 15-20 分钟）
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "datasets" / "columbia_gaze" / "Columbia Gaze Data Set"
OUT_DIR = ROOT / "datasets" / "columbia_gaze_yolo"
CONFIG_PATH = ROOT / "configs" / "columbia_gaze.yaml"
FACE_MODEL = ROOT / "models" / "face_landmarker.task"

PROCESSED_W = 1280     # 送 MediaPipe 前的缩放宽 / downscale before detection
CROP_LONG = 128        # 裁剪图长边像素 / output crop long side
VAL_EVERY = 10         # 每第 10 个受试者进 val / every 10th subject -> val

# 与 EyeWheelchairProject/src/interaction/blink_preview.py 一致的眼周关键点
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def v_to_class(v_token: str):
    """'-10V'→look_up(0) '0V'→look_center(1) '10V'→look_down(2) / None=不认识."""
    v = v_token.replace("V", "")
    if v == "-10":
        return 0
    if v == "0":
        return 1
    if v == "10":
        return 2
    return None


def parse_name(path: Path):
    """0001_2m_-15P_-10V_-10H.jpg → (subject, class) / parse label from filename.
    共 5 段：[0]受试者 [1]距离 [2]头姿P [3]垂直V [4]水平H，V 在索引 3。"""
    tokens = path.stem.split("_")
    if len(tokens) != 5:
        return None
    return tokens[0], v_to_class(tokens[3])


def eye_boxes(lms, w, h):
    """左右眼外接框（归一化关键点→像素框，含扩边）/ eye bboxes from landmarks."""
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


def main():
    if not SRC_DIR.exists():
        sys.exit(f"未找到数据目录 / missing: {SRC_DIR}（先解压 zip）")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    src_files = sorted(SRC_DIR.glob("*/*.jpg"))
    if not src_files:
        sys.exit(f"未找到图片 / no jpg under {SRC_DIR}")
    subjects = sorted({p.name.split("_")[0] for p in src_files})
    val_subjects = {s for i, s in enumerate(subjects) if i % VAL_EVERY == VAL_EVERY - 1}
    print(f"受试者 / subjects: {len(subjects)} | val: {sorted(val_subjects)}")

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        num_faces=1)
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    stats = {"train": {0: 0, 1: 0, 2: 0}, "val": {0: 0, 1: 0, 2: 0}}
    no_face, done = 0, 0
    class_names = {0: "look_up", 1: "look_center", 2: "look_down"}

    for img_path in src_files:
        parsed = parse_name(img_path)
        if parsed is None:
            continue
        subject, cls = parsed
        split = "val" if subject in val_subjects else "train"

        img = cv2.imread(str(img_path))
        if img is None:
            no_face += 1
            continue
        scale = PROCESSED_W / img.shape[1]
        small = cv2.resize(img, (PROCESSED_W, int(img.shape[0] * scale)))
        h, w = small.shape[:2]

        res = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            no_face += 1
            continue

        for eye_i, (x0, y0, x1, y1) in enumerate(eye_boxes(res.face_landmarks[0], w, h)):
            crop = small[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            long_side = max(ch, cw)
            if long_side > CROP_LONG:
                r = CROP_LONG / long_side
                crop = cv2.resize(crop, (int(cw * r), int(ch * r)))
            stem = f"{subject}_{img_path.stem.split('_', 1)[1]}_{'L' if eye_i == 0 else 'R'}"
            out_name = f"{stem}.jpg"
            cv2.imwrite(str(OUT_DIR / "images" / split / out_name), crop)
            (OUT_DIR / "labels" / split / (stem + ".txt")).write_text(f"{cls} 0.5 0.5 1.0 1.0\n")
            stats[split][cls] += 1

        done += 1
        if done % 500 == 0:
            print(f"进度 / progress: {done} 张源图 ...", flush=True)

    total = sum(stats[s][c] for s in stats for c in stats[s])
    print(f"\n完成 / done: {total} 个眼部裁剪（来自 {done} 张源图；未检出人脸 {no_face} 张）")
    for split in ("train", "val"):
        print(f"  {split}: " + " ".join(f"{class_names[c]}={stats[split][c]}" for c in (0, 1, 2)))

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "# Columbia Gaze — 眼部裁剪 YOLO 格式（自动生成 / auto-generated）\n"
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 3\n"
        "names:\n"
        "  0: look_up\n"
        "  1: look_center\n"
        "  2: look_down\n",
        encoding="utf-8",
    )
    print("数据配置 / data config:", CONFIG_PATH)


if __name__ == "__main__":
    main()
