# -*- coding: utf-8 -*-
"""手机视频 → 眼部裁剪 YOLO 数据集（模型 B 个人化流水线）
/ Phone videos → eye-crop YOLO dataset (Model B personalization pipeline).

输入约定 / input convention（见 docs/gaze_data_collection_guide.md）:
    phone_videos/<人名>/<文件名>.mp4，文件名必须含方向字：
        上 → look_up(0)   中 → look_center(1)   下 → look_down(2)
    每段视频只拍一个方向；标签来自文件名，无需人工画框。

流水线 / pipeline:
    1) 每段视频按 --fps 抽帧（默认 5fps）
    2) MediaPipe FaceLandmarker 逐帧检测 → 裁左/右眼（与模型 B 底座数据同款扩边）
    3) 存 datasets/phone_gaze_yolo/images/{train,val}/ + 整图框标签
       （按"人"划分 train/val，与 Columbia 底座同规则）
    4) 质量检查（QC）：用虹膜关键点几何估计垂直注视，与文件名标签比对，
       汇总每段视频的一致率——一致率过低说明拍摄者头动了或方向拍错

用法 / usage:
    python scripts/auto_label_phone_videos.py                    # 处理 phone_videos/
    python scripts/auto_label_phone_videos.py --fps 5 --val-every 4
    python scripts/auto_label_phone_videos.py --src <其他目录>    # 冒烟测试用

输出 / output: datasets/phone_gaze_yolo/ + configs/phone_gaze.yaml + 控制台 QC 报告
被谁调用 / Called by: 手动（视频收齐后）/ manual
内部调用 / Calls: cv2, mediapipe
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "phone_videos"
OUT_DIR = ROOT / "datasets" / "phone_gaze_yolo"
CONFIG_PATH = ROOT / "configs" / "phone_gaze.yaml"
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
PROCESSED_W = 1280
CROP_LONG = 128

# 文件名方向字 → 类别 / direction char in filename -> class
CHAR_MAP = {"上": 0, "中": 1, "下": 2}
CLASS_NAMES = {0: "look_up", 1: "look_center", 2: "look_down"}

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def class_of(filename: str):
    """文件名方向字 → 类别 / direction char in filename -> class id."""
    for ch, cls in CHAR_MAP.items():
        if ch in filename:
            return cls
    return None


def eye_boxes(lms, w, h):
    """左右眼外接框 + 眼眶中心y跨度 / eye bboxes (same margins as bootstrap data)."""
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


def iris_vertical_metric(lms):
    """虹膜中心y 相对内外眼角连线y（按眼宽归一化，双眼取均值）。
    数值越大=虹膜越高=往上看；实验（Columbia 0002, 0P）：上≈-0.09 > 中≈-0.14 > 下≈-0.17，
    三区间零重叠 / canthus-relative iris height; higher = looking up; empirically
    well-separated on labeled data. 竖直注视时眼睑会跟随眼珠，用眼睑做参考系无区分度
    （实测上/中/下均≈0.37），故必须用眼角连线。"""
    vals = []
    for a_i, b_i, ic in ((33, 133, 468), (362, 263, 473)):
        a, b, iris = lms[a_i], lms[b_i], lms[ic]
        width = abs(a.x - b.x)
        if width > 1e-6:
            vals.append((iris.y - (a.y + b.y) / 2) / width)
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--fps", type=float, default=5.0, help="抽帧率 / frames per second")
    ap.add_argument("--val-every", type=int, default=4,
                    help="每第 N 个人进 val / every Nth person -> val")
    ap.add_argument("--camera-w", type=int, default=1280)
    args = ap.parse_args()

    src = Path(args.src)
    # Windows 下 glob 不区分大小写，*.mp4 与 *.MP4 会重复匹配 → 按 normcase 去重
    seen = set()
    videos = []
    for pat in ("*/*.mp4", "*/*.mov", "*/*.MOV", "*/*.MP4", "*/*.avi", "*/*.AVI"):
        for v in sorted(src.glob(pat)):
            key = os.path.normcase(str(v))
            if key not in seen:
                seen.add(key)
                videos.append(v)
    if not videos:
        raise SystemExit(f"未找到视频 / no videos under {src}\\<人名>\\*.mp4")

    # 人的划分 / person-wise split
    persons = sorted({v.parent.name for v in videos})
    val_persons = {p for i, p in enumerate(persons) if i % args.val_every == args.val_every - 1}
    print(f"人 / persons: {len(persons)} | val: {sorted(val_persons)}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        num_faces=1)
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: {"crops": 0, "frames": 0, "no_face": 0})
    qc = defaultdict(list)  # (person, clip) -> [geo_ratio,...]

    for video in videos:
        person = video.parent.name
        cls = class_of(video.stem)
        if cls is None:
            print(f"[跳过 / skip] 文件名无方向字（上/中/下）: {video.name}")
            continue
        split = "val" if person in val_persons else "train"
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / args.fps)))
        frame_i, saved = 0, 0

        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                scale = args.camera_w / frame.shape[1]
                if scale < 1:
                    frame = cv2.resize(frame, (args.camera_w, int(frame.shape[0] * scale)))
                h, w = frame.shape[:2]
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = landmarker.detect(mp_img)
                stats[(person, video.stem)]["frames"] += 1
                if not res.face_landmarks:
                    stats[(person, video.stem)]["no_face"] += 1
                    frame_i += 1
                    continue
                lms = res.face_landmarks[0]
                for eye_i, (x0, y0, x1, y1) in enumerate(eye_boxes(lms, w, h)):
                    side = "L" if eye_i == 0 else "R"
                    crop = frame[y0:y1, x0:x1]
                    ch, cw = crop.shape[:2]
                    long_side = max(ch, cw)
                    if long_side > CROP_LONG:
                        r = CROP_LONG / long_side
                        crop = cv2.resize(crop, (int(cw * r), int(ch * r)))
                    stem = f"{person}_{video.stem}_{frame_i:06d}_{side}"
                    (OUT_DIR / "images" / split / (stem + ".jpg")).write_bytes(
                        cv2.imencode(".jpg", crop)[1].tobytes())
                    (OUT_DIR / "labels" / split / (stem + ".txt")).write_text(f"{cls} 0.5 0.5 1.0 1.0\n")
                    stats[(person, video.stem)]["crops"] += 1
                    # QC：眼角参考系虹膜高度 / canthus-relative iris height
                    ir = iris_vertical_metric(lms)
                    if ir is not None:
                        qc[(person, video.stem)].append((cls, ir))
                    saved += 1
            frame_i += 1
        cap.release()
        s = stats[(person, video.stem)]
        print(f"[{person}/{video.name}] 抽帧 {s['frames']} | 裁剪 {s['crops']} | 无脸帧 {s['no_face']}")

    # ---- QC 汇总：同人跨段排序校验 / per-person ordering check ----
    # 数值越大=虹膜越高=越往上看；同人先上后下应严格递减。眼型差异被同人对消。
    print("\n==== QC：虹膜几何（同人排序校验，期望 上 > 中 > 下）====")
    by_person = defaultdict(dict)
    for (person, clip), pairs in sorted(qc.items()):
        label = pairs[0][0]
        mean_r = sum(r for _, r in pairs) / len(pairs)
        by_person[person][label] = mean_r
        print(f"  {person}/{clip}: label={CLASS_NAMES[label]:12s} 虹膜高={mean_r:+.4f}")
    for person, zones in sorted(by_person.items()):
        if len(zones) >= 2:
            order_ok = all(
                zones[a] > zones[b]
                for a, b in ((0, 1), (1, 2), (0, 2))
                if a in zones and b in zones
            )
            print(f"  [{person}] 排序校验: {'OK ✅' if order_ok else '⚠️ 不满足 上>中>下，请人工复查该人素材'}")

    # ---- 写配置 / write config ----
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "# 自采手机视频 — 眼部裁剪 YOLO 格式（自动生成 / auto-generated）\n"
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
    total = sum(s["crops"] for s in stats.values())
    print(f"\n完成 / done: {total} 个裁剪 → {OUT_DIR}")
    print("下一步 / next: python scripts/train_eye.py --data configs/phone_gaze.yaml "
          "--name gaze_personal --model runs/gaze_yolo26n/weights/best.pt --epochs 30")


if __name__ == "__main__":
    main()
