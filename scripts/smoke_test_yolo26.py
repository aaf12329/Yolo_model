# -*- coding: utf-8 -*-
"""YOLO26 GPU 冒烟测试 / YOLO26 GPU smoke test.

做两件事 / Does two things:
1. 下载 yolo26n.pt 预训练权重，GPU 推理一张样图 / download weights, GPU inference
2. 在 COCO8 上训练 1 epoch，验证训练循环 + GPU 可用 / 1-epoch train on COCO8

运行 / Run:
    python scripts/smoke_test_yolo26.py

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: ultralytics.YOLO
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    model = YOLO("yolo26n.pt")  # 自动下载 / auto-download (~6 MB)
    # 克隆即用：样图缺失时自动合成一张（推理只验证管线，不关心内容）
    sample = ROOT / "datasets" / "label_check" / "tok4_0_s0001_00001_0_0_0_0_0_01.png"
    if not sample.exists():
        import numpy as np
        sample.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample), np.random.randint(0, 255, (83, 83), dtype=np.uint8))
    results = model.predict(str(sample), imgsz=640, device=0, verbose=False)
    r = results[0]
    print("推理 OK / inference OK | device:", r.speed, "| detections:", len(r.boxes))

    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=64,
        batch=8,
        device=0,
        project=str(ROOT / "runs" / "smoke"),
        name="yolo26n_coco8",
        exist_ok=True,
        verbose=False,
    )
    print("训练循环 OK / training loop OK（COCO8, 1 epoch, GPU）")


if __name__ == "__main__":
    main()
