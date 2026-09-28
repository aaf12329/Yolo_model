# -*- coding: utf-8 -*-
"""检测模型准确率评估（配置驱动，MRL / Columbia Gaze 通用）/ Config-driven accuracy eval.

在指定数据集的验证集上：
    1) YOLO 权重逐张预测 vs 标签文件（labels/val/<stem>.txt 首个 token）
       → 总准确率 / 各类准确率 / 混淆矩阵 / 官方 mAP / 单张耗时
    2) MediaPipe FaceLandmarker 在验证图上的检出率探测（衡量 EAR 法适用性）

用法 / usage:
    python scripts/eval_eye_accuracy.py                                   # 模型A（MRL 默认）
    python scripts/eval_eye_accuracy.py --config configs/columbia_gaze.yaml \
        --weights runs/gaze_yolo26n/weights/best.pt                       # 模型B（3 类注视方向）

输出 / outputs（docs/comparison/，文件名随 config 命名）:
    <config>_eval_report.md / <config>_confusion_matrix.png / <config>_accuracy_chart.png

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: ultralytics.YOLO, mediapipe, matplotlib, pyyaml
"""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "comparison"
MP_PROBE_N = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mrl_eye.yaml")
    ap.add_argument("--weights", default="runs/eye_yolo26n/weights/best.pt")
    ap.add_argument("--probe-n", type=int, default=MP_PROBE_N)
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg_dir = Path(cfg["path"])
    names = {int(k): v for k, v in cfg["names"].items()}
    stem = cfg_path.stem  # mrl_eye / columbia_gaze

    val_img_dir = cfg_dir / cfg["val"]                              # .../images/val
    val_lbl_dir = cfg_dir / cfg["val"].replace("images", "labels")  # .../labels/val
    imgs = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    print(f"验证集 / val images: {len(imgs)} | 类别 / classes: {names}")

    from ultralytics import YOLO
    yolo = YOLO(str(ROOT / args.weights))

    # ---- 1) YOLO 批量预测 / batch prediction ----
    labels, preds = [], []
    t_infer = 0.0
    B = 128
    for i in range(0, len(imgs), B):
        chunk = imgs[i : i + B]
        t0 = time.perf_counter()
        results = yolo.predict([str(p) for p in chunk], imgsz=128, device=0,
                               verbose=False, conf=0.5)
        t_infer += time.perf_counter() - t0
        for p, path in zip(results, chunk):
            lbl_file = val_lbl_dir / (path.stem + ".txt")
            labels.append(int(lbl_file.read_text().split()[0]) if lbl_file.exists() else -1)
            preds.append(int(p.boxes.cls[int(p.boxes.conf.argmax())]) if len(p.boxes) else -1)

    valid = [(l, p) for l, p in zip(labels, preds) if l >= 0 and p >= 0]
    no_det = sum(1 for p in preds if p < 0)
    n_cls = len(names)
    correct = sum(1 for l, p in valid if l == p)
    acc = correct / max(1, len(valid))
    per_cls_acc = {
        c: (sum(1 for l, p in valid if l == c and p == c)
            / max(1, sum(1 for l, p in valid if l == c)))
        for c in range(n_cls)
    }
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for l, p in valid:
        conf[l][p] += 1
    yolo_ms = t_infer / len(imgs) * 1000

    # ---- 2) 官方 val 指标 / official ultralytics metrics ----
    m = yolo.val(data=str(cfg_path), imgsz=128, device=0, verbose=False)
    map50, map5095 = float(m.box.map50), float(m.box.map)

    # ---- 3) MediaPipe 适用性探测 / applicability probe ----
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
        num_faces=1)
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    rng = random.Random(0)
    probe = rng.sample(imgs, min(args.probe_n, len(imgs)))
    detected, t_mp = 0, 0.0
    for p in probe:
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        res = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        t_mp += time.perf_counter() - t0
        if res.face_landmarks:
            detected += 1
    mp_rate, mp_ms = detected / len(probe), t_mp / len(probe) * 1000

    # ---- 4) 报告 / report ----
    cls_rows = "\n".join(
        f"| {names[c]} | {per_cls_acc[c]*100:.2f}% | {int(conf[c].sum())} |"
        for c in range(n_cls))
    conf_txt = "，".join(
        f"{names[i]}→{names[j]} {conf[i][j]}"
        for i in range(n_cls) for j in range(n_cls) if conf[i][j])
    lines = [
        f"# 准确率评估 / accuracy eval — {', '.join(str(v) for v in names.values())}",
        "",
        f"日期 / date: {time.strftime('%Y-%m-%d %H:%M')}　配置 / config: {args.config}",
        f"权重 / weights: {args.weights}　验证集 / val: {len(imgs)} 张",
        "",
        f"- 判定准确率 / accuracy: **{acc*100:.2f}%**（{correct}/{len(valid)}，无检测帧 {no_det}）",
        f"- 官方指标 / official: mAP50={map50:.4f}　mAP50-95={map5095:.4f}",
        f"- 单张耗时 / latency: {yolo_ms:.2f} ms/张（RTX 5060 Ti, imgsz=128）",
        "",
        "| 类别 / class | 准确率 / acc | 样本数 / n |",
        "|---|---|---|",
        cls_rows,
        "",
        f"- 混淆矩阵 / confusion（行=真值, 列=预测）: {conf_txt}",
        "",
        "## MediaPipe 适用性 / applicability probe",
        "",
        f"- {len(probe)} 张验证图上检出人脸 / face detected: {detected}（{mp_rate*100:.1f}%），"
        f"单张 {mp_ms:.2f} ms",
        "- EAR 法需要整脸 478 关键点；眼部裁剪/近距离场景结构性不可用，YOLO 是可行选项"
        " / EAR needs full-face landmarks; structurally unusable on eye crops & close range",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # ---- 5) 画图 / charts ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(min(9, 4 + n_cls), 5.5))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n_cls), [f"预测 {names[c]}" for c in range(n_cls)],
                  rotation=15 if n_cls > 2 else 0, fontsize=9)
    ax.set_yticks(range(n_cls), [f"真值 {names[c]}" for c in range(n_cls)], fontsize=9)
    for (i, j), v in np.ndenumerate(conf):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                color="white" if v > conf.max() / 2 else "black")
    ax.set_title(f"混淆矩阵 / confusion matrix（准确率 {acc*100:.2f}%）", fontsize=12)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    cat_names = [f"YOLO26\n{names[c]}" for c in range(n_cls)] + ["MediaPipe\n检出率"]
    vals = [per_cls_acc[c] * 100 for c in range(n_cls)] + [mp_rate * 100]
    colors = ["#ef6c00"] * n_cls + ["#2e7d32"]
    bars = ax.bar(cat_names, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%")
    ax.set_title(f"YOLO26 vs MediaPipe — {stem} 验证集（跨受试者）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}_accuracy_chart.png", dpi=150)
    plt.close(fig)
    print("图已保存 / charts saved:", OUT_DIR / f"{stem}_confusion_matrix.png",
          OUT_DIR / f"{stem}_accuracy_chart.png")


if __name__ == "__main__":
    main()
