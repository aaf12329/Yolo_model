# -*- coding: utf-8 -*-
"""GPU 环境验证 / GPU environment verification.

运行 / Run:
    python scripts/verify_gpu.py

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: torch, ultralytics
"""
import torch


def main():
    print("torch:", torch.__version__)
    print("cuda available / CUDA 可用:", torch.cuda.is_available())
    print("cuda version / CUDA 版本:", torch.version.cuda)
    if torch.cuda.is_available():
        print("device / 设备:", torch.cuda.get_device_name(0))
        print("capability / 算力:", torch.cuda.get_device_capability(0))
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print("GPU matmul OK / GPU 矩阵乘法正常:", round(y, 2))
    try:
        import ultralytics

        print("ultralytics:", ultralytics.__version__)
    except ImportError:
        print("ultralytics NOT installed / 未安装 ultralytics")


if __name__ == "__main__":
    main()
