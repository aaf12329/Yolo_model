# -*- coding: utf-8 -*-
"""MRL Eye Dataset → YOLO 检测格式转换 / Convert MRL Eye Dataset to YOLO format.

数据集 / Dataset:
    mrlEyes_2018_01 — 84,898 张眼部特写灰度图（约 64~128px）
    下载 / Download: http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip (~326 MB)

文件名标注规则 / Filename label convention（已抽样目视验证 / visually verified）:
    s0001_00001_0_0_0_0_0_01.png
    [0]受试者 [1]序号 [2]性别 [3]眼镜 [4]眼睛状态 [5]反光 [6]光照 [7]传感器
    token[4]: 0 = 睁眼 open_eye (class 0), 1 = 闭眼 closed_eye (class 1)

YOLO 标签: 整图单目标框（特写图本身即眼部区域，定位交给
EyeWheelchairProject 里的 MediaPipe，本模型负责在眼部裁剪图上判定睁/闭）。
/ YOLO label: one full-image box per close-up crop. Eye localization is
/ MediaPipe's job in EyeWheelchairProject; this model judges open/closed
/ on eye crops.

划分策略 / Split: 按受试者划分（每第 10 个受试者进 val），避免同一人
同时出现在训练和验证集造成泄漏 / subject-wise split to avoid identity leakage.

运行 / Run:
    python scripts/prepare_mrl_dataset.py

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: zipfile, pathlib, os
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "datasets" / "mrlEyes_2018_01.zip"
EXTRACT_DIR = ROOT / "datasets" / "mrlEyes_2018_01"
OUT_DIR = ROOT / "datasets" / "mrl_eye_yolo"
CONFIG_PATH = ROOT / "configs" / "mrl_eye.yaml"

VAL_EVERY = 10  # 每 10 个受试者取 1 个进验证集 / every 10th subject -> val


def extract():
    """解压数据集（已存在则跳过）/ Extract dataset (skip if present)."""
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.glob("s*/")):
        print(f"已解压，跳过 / already extracted: {EXTRACT_DIR}")
        return
    if not ZIP_PATH.exists():
        sys.exit(f"缺少数据集压缩包 / missing zip: {ZIP_PATH}")
    print("解压中 / extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR.parent)
    print("解压完成 / done")


def link_or_copy(src: Path, dst: Path):
    """优先硬链接省磁盘，失败则复制 / hardlink first, fall back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def eye_state_class(stem: str):
    """从文件名解析类别 / parse class from filename. 0=open, 1=closed."""
    tokens = stem.split("_")
    if len(tokens) != 8:
        return None
    return 0 if tokens[4] == "0" else 1


def main():
    extract()

    img_files = sorted(EXTRACT_DIR.glob("s*/*.png"))
    if not img_files:
        sys.exit(f"未找到图片 / no images found in {EXTRACT_DIR}")

    subjects = sorted({p.parent.name for p in img_files})
    val_subjects = {s for i, s in enumerate(subjects) if i % VAL_EVERY == VAL_EVERY - 1}
    print(f"受试者 / subjects: {len(subjects)} | val: {len(val_subjects)} -> {sorted(val_subjects)}")

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}}
    skipped = 0
    for img in img_files:
        cls = eye_state_class(img.stem)
        if cls is None:
            skipped += 1
            continue
        split = "val" if img.parent.name in val_subjects else "train"
        stats[split][cls] += 1

        img_dst = OUT_DIR / "images" / split / img.name
        lbl_dst = OUT_DIR / "labels" / split / (img.stem + ".txt")
        if not img_dst.exists():
            link_or_copy(img, img_dst)
        # 整图单框: 类别 + 归一化 cx cy w h / full-image single box
        lbl_dst.write_text(f"{cls} 0.5 0.5 1.0 1.0\n")

    total = sum(stats[s][c] for s in stats for c in stats[s])
    print(f"转换完成 / converted: {total} 张（跳过 / skipped: {skipped}）")
    for split in ("train", "val"):
        print(
            f"  {split}: open_eye={stats[split][0]} closed_eye={stats[split][1]}"
        )

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        f"# MRL Eye Dataset — YOLO 格式 / YOLO format（自动生成 / auto-generated）\n"
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: open_eye\n"
        f"  1: closed_eye\n",
        encoding="utf-8",
    )
    print(f"数据配置 / data config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
