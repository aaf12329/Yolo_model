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
