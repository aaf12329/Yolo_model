# YOLO26n 注视方向模型 B 训练记录 / Gaze Model B training record

日期 / Date: 2026-09-28　环境 / Env: conda `yolo`（py3.10, torch 2.11.0+cu128, RTX 5060 Ti）

## 结果 / Results

| 项 / Item | 值 / Value |
|---|---|
| 模型 / Model | YOLO26n（COCO 预训练迁移 / transfer from COCO），3 类 |
| 数据 / Data | Columbia Gaze 眼部裁剪 11,760 张（5,880 源图 × 左右眼，MediaPipe 裁剪） |
| 类别 / Classes | `look_up`（-10V）/ `look_center`（0V）/ `look_down`（10V）——符号已目视验证 |
| 划分 / Split | 按受试者：56 人取 s0010/20/30/40/50 共 5 人进 val（1,050 张），其余 10,710 张训练 |
| 参数 / Params | imgsz=128, batch=auto, epochs 上限 50, patience=10 |
| 训练时长 / Time | 27 epochs / **2.6 分钟**（早停 / early stopping） |
| 最佳轮次 / Best epoch | 第 18 轮 |
| mAP@0.5 = mAP@0.5:0.95 | **0.9754** |
| 总判定准确率 / accuracy | **93.62%**（983/1,050，跨受试者） |
| 各类准确率 / per class | look_center **96.86%** / look_down **96.00%** / look_up **88.00%** |
| 单张耗时 / latency | 16.66 ms（验证管线） / 0.5 ms（纯推理） |

## 混淆矩阵要点 / Confusion notes

```
              预测up  预测center  预测down
真值 up         308       42          0
真值 center       3      339          8
真值 down         0       14        336
```

- 主要混淆是 **look_up → look_center（42/350，12%）**：+10° 的向上注视幅度本身小，
  且数据里部分人被眼睑/眼镜部分遮挡，"微抬头"与"平视"在裁剪图上本来就接近。
  对轮椅控制的影响可控（up→center 的错误是"减速为停"，不是反向误动作），
  且方向模型输出还会经过上层状态机的持续注视确认（0.7s 稳定才成候选）。
- down 几乎不误判为 up（0 例）——**前进/后退两个最关键方向不会互相窜**。

## 产物 / Artifacts

- 权重 / weights: `runs/gaze_yolo26n/weights/best.pt`（不入库 / not committed）
- 曲线 / curves: `runs/gaze_yolo26n/curves_loss.png`、`curves_metrics.png`
- 评估 / eval: `docs/comparison/columbia_gaze_eval_report.md`、
  `columbia_gaze_confusion_matrix.png`、`columbia_gaze_accuracy_chart.png`

## 下一步 / Next

- 个人化微调：用户自采手机视频（≥3 人）→ MediaPipe 自动打标 → 在 best.pt 上继续训练
- 预计能显著修复 look_up 的偏低（自采数据会覆盖真实使用角度/光照/本人眼型）

## 2026-09-29 容量对比：yolo26n → yolo26s 升级 / capacity comparison

| | yolo26n | **yolo26s（采用）** |
|---|---|---|
| 总准确率 | 93.62% | **95.62%** |
| look_up | 88.00% | **92.00%** |
| look_center | 96.86% | 96.29% |
| look_down | 96.00% | **98.57%** |
| mAP50-95 | 0.9776 | 0.9828 |
| up→center 误判 | 42/350 | **28/350** |
| 单张耗时 | 0.5ms | 2.43ms（远低于 33ms 实时预算） |

同数据同划分（11,760 裁剪，5 人 val），s 版早停@27、最佳@18、训练 3.4 分钟。
**结论**：容量确实 helped——up 误判减少三分之一，采纳 s 为 Model B 交付权重
（`models/gaze_yolo26s.pt`）；up 剩余 8% 误差仍需个人化微调解决（方向不变）。


## 2026-09-29 5类边界修复实验：|H|>=10 才算左/右

**背景**：v1 训练集上 center 只有 74.6%（见上文）——模型在自己见过的图上都判不对，
排除过拟合与数据量，定位为**标签边界自相矛盾**：V0H±5（水平偏 5°）被标为左/右，
但外观与直视几乎无差。

**改动**：|H|<=5° 并回 center/垂直类；左/右类只保留 |H|>=10°（外观可区分）。

| 指标 | v1（±5°算左右） | **v2（左右≥10°）** |
|---|---|---|
| center 训练集 | 74.6% | **82.1%** |
| center 验证集 | 42.1% | **76.5%** |
| look_left | 87.4% | 86.3% |
| look_right | 80.2% | **88.7%** |
| look_up | 84.1% | 81.3% |
| look_down | 83.2% | 81.2% |
| 总体 | 80.9% | **82.5%** |

**结论**：假设成立——center 弱的主因是类边界画错，修正后 center +34.4pp、
总体 +1.6pp，left/right 基本无损。v2 标注已采纳为 5 类交付版
（`models/gaze5_yolo26s.pt`）。
**遗留**：center 训练集 82.1% 仍未到 90% 目标——剩余困难是 center 在注视空间
中是"一个点"的结构性限制 + 对角样本歧义，根治靠个人化数据（center 段多录），
必要时改两阶段判定。
