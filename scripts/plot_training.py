# -*- coding: utf-8 -*-
"""训练损失/指标曲线绘制 / Plot training loss & metric curves.

读取 ultralytics 训练产生的 results.csv，用 matplotlib 保存：
/ Reads results.csv written by ultralytics training and saves via matplotlib:
    runs/eye_yolo26n/curves_loss.png     损失曲线 / loss curves
    runs/eye_yolo26n/curves_metrics.png  精度指标 / metric curves

运行 / Run:
    python scripts/plot_training.py [results.csv 所在目录 / dir containing results.csv]

被谁调用 / Called by: 手动或训练结束后自动 / manual or after training
内部调用 / Calls: pandas, matplotlib
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent

LOSS_COLS = [
    ("train/box_loss", "train box / 训练框损失"),
    ("train/cls_loss", "train cls / 训练分类损失"),
    ("train/dfl_loss", "train dfl / 训练分布焦点损失"),
    ("val/box_loss", "val box / 验证框损失"),
    ("val/cls_loss", "val cls / 验证分类损失"),
    ("val/dfl_loss", "val dfl / 验证分布焦点损失"),
]
METRIC_COLS = [
    ("metrics/precision(B)", "precision / 精确率"),
    ("metrics/recall(B)", "recall / 召回率"),
    ("metrics/mAP50(B)", "mAP@0.5"),
    ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
]


def plot(run_dir: Path):
    csv = run_dir / "results.csv"
    if not csv.exists():
        sys.exit(f"未找到 / not found: {csv}")
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"] + 1

    # ---- 损失曲线 / loss curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in LOSS_COLS:
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, linewidth=1.6)
    ax.set_xlabel("epoch / 轮次")
    ax.set_ylabel("loss / 损失")
    ax.set_title(f"YOLO26n 眼部训练损失曲线 / eye training loss ({run_dir.name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out1 = run_dir / "curves_loss.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # ---- 指标曲线 / metric curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in METRIC_COLS:
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, linewidth=1.6)
    ax.set_xlabel("epoch / 轮次")
    ax.set_ylabel("metric / 指标")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"YOLO26n 眼部验证指标曲线 / eye validation metrics ({run_dir.name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out2 = run_dir / "curves_metrics.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    print("已保存 / saved:", out1)
    print("已保存 / saved:", out2)
    last = df.iloc[-1]
    print(
        f"最新一轮 / latest epoch {int(last['epoch']) + 1}: "
        f"mAP50={last.get('metrics/mAP50(B)', float('nan')):.4f} "
        f"mAP50-95={last.get('metrics/mAP50-95(B)', float('nan')):.4f} "
        f"precision={last.get('metrics/precision(B)', float('nan')):.4f} "
        f"recall={last.get('metrics/recall(B)', float('nan')):.4f}"
    )


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "eye_yolo26n"
    plot(run_dir)
