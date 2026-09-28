# -*- coding: utf-8 -*-
"""眼部数据集训练启动器 / Eye dataset training launcher.

用 YOLO26n 在 MRL 眼部数据集（2 类：open_eye / closed_eye）上做迁移学习。
/ Transfer-learn YOLO26n on the MRL eye dataset (2 classes: open_eye / closed_eye).

运行 / Run:
    python scripts/train_eye.py              # 正式训练 / full training
    python scripts/train_eye.py --epochs 3   # 短跑 / short run

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: ultralytics.YOLO
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=128, help="MRL 是小尺寸特写图 / close-ups are tiny")
    ap.add_argument("--batch", type=int, default=0, help="0 = 自动 / auto")
    ap.add_argument("--model", default="yolo26n.pt")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "configs" / "mrl_eye.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch if args.batch > 0 else -1,
        device=0,
        project=str(ROOT / "runs"),
        name="eye_yolo26n",
        exist_ok=True,
        patience=10,
        plots=True,
    )
    print("训练完成，结果目录 / done, results dir:", ROOT / "runs" / "eye_yolo26n")


if __name__ == "__main__":
    main()
