# -*- coding: utf-8 -*-
"""Columbia Gaze 5 类重标注 v2（复用裁剪 + 翻转增强 + 修均衡）
/ 5-class re-label v2: reuse crops + flip-swap augmentation + fixed balancing.

v1 的两个教训 / lessons from v1:
    1) 过采样只复制了标签没复制图片 → center 类实际只有 1/4 数据（40% 准确率）。
       本版 center 复制图片+标签（含翻转副本，是真变化不是重复）。
    2) fliplr=0 后垂直类失去翻转增强。本版离线做"翻转+标签互换"：
       左看图翻转后外观=右看，标签必须 3↔4 互换才正确；上/中/下翻转标签不变。
       这样所有 5 类都恢复了翻转增强，且左右语义不被污染
       （不能开 ultralytics 的 fliplr：它翻转图片但保留标签，左右会互相污染）。

5 类标注规则（V/H 符号均已目视验证 / both signs visually verified）:
    V ∈ {0,±10}（-10=上看）  H ∈ {0,±5,±10,±15}（-15=看左）
    |V| >= |H| → 垂直类（平局算垂直，前进/后退优先）  |H| > |V| → 水平类

产出 / output: datasets/columbia_gaze5_yolo/ + configs/columbia_gaze5.yaml
运行 / Run:  python scripts/prepare_columbia_gaze5_dataset.py
被谁调用 / Called by: 手动 / manual
内部调用 / Calls: cv2, pathlib（重标注+翻转写盘，约 1 分钟）
"""
import cv2
import numpy as np
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "columbia_gaze_yolo"   # 已有 3 类版（图片可复用）
OUT = ROOT / "datasets" / "columbia_gaze5_yolo"
CONFIG = ROOT / "configs" / "columbia_gaze5.yaml"

NAMES = {0: "look_up", 1: "look_center", 2: "look_down",
         3: "look_left", 4: "look_right"}
FLIP_SWAP = {3: 4, 4: 3}  # 翻转后左右互换 / horizontal flip swaps left/right


def classify(v_deg: int, h_deg: int) -> int:
    """主轴优先 5 类 / dominant-axis 5-class. 返回类别 id。"""
    av, ah = abs(v_deg), abs(h_deg)
    if av == 0 and ah == 0:
        return 1
    if av >= ah:
        return 0 if v_deg < 0 else 2          # 垂直优先（含平局）
    return 3 if h_deg < 0 else 4              # 水平


def parse_vh(stem: str):
    """0001_2m_-15P_-10V_-10H_L → (V 度数, H 度数)。共 6 段：[3]=V [4]=H。"""
    parts = stem.split("_")
    if len(parts) != 6:
        return None
    try:
        return int(parts[3].replace("V", "")), int(parts[4].replace("H", ""))
    except ValueError:
        return None


def emit(split: str, stem: str, img, cls: int, counts: Counter):
    dst = OUT / "images" / split / (stem + ".jpg")
    if not dst.exists():
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            return
        dst.write_bytes(buf.tobytes())
    (OUT / "labels" / split / (stem + ".txt")).write_text(f"{cls} 0.5 0.5 1.0 1.0\n")
    counts[(split, cls)] += 1


def main():
    if not SRC.exists():
        raise SystemExit(f"未找到 3 类版数据 / missing: {SRC}")
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for split in ("train", "val"):
        for img_path in sorted((SRC / "images" / split).glob("*.jpg")):
            vh = parse_vh(img_path.stem)
            if vh is None:
                continue
            cls = classify(*vh)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            flipped = cv2.flip(img, 1)
            flip_cls = FLIP_SWAP.get(cls, cls)

            emit(split, img_path.stem, img, cls, counts)
            emit(split, img_path.stem + "_f", flipped, flip_cls, counts)
            # center 原始组合只有一种（V0H0），再补一份原始副本缓解 1:4 不均衡
            if cls == 1:
                emit(split, img_path.stem + "_c", img, cls, counts)

    print("5 类分布（含翻转增强）/ class distribution:")
    for split in ("train", "val"):
        row = " ".join(f"{NAMES[c]}={counts[(split, c)]}" for c in range(5))
        print(f"  {split}: {row}")

    CONFIG.write_text(
        "# Columbia Gaze 5 类 v2（主轴优先+翻转互换增强，自动生成 / auto-generated）\n"
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 5\n"
        "names:\n"
        "  0: look_up\n  1: look_center\n  2: look_down\n"
        "  3: look_left\n  4: look_right\n",
        encoding="utf-8",
    )
    print("数据配置 / data config:", CONFIG)


if __name__ == "__main__":
    main()
