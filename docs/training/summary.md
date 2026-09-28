# YOLO26n 睁/闭眼模型训练记录 / Training record

日期 / Date: 2026-09-28　环境 / Env: conda `yolo`（py3.10, torch 2.11.0+cu128, RTX 5060 Ti）

## 结果 / Results

| 项 / Item | 值 / Value |
|---|---|
| 模型 / Model | YOLO26n（COCO 预训练迁移 / transfer from COCO） |
| 数据 / Data | MRL mrlEyes_2018_01，84,898 张眼部特写，2 类（open_eye / closed_eye） |
| 划分 / Split | 按受试者：37 人训练，s0010/s0020/s0030 验证 / subject-wise split |
| 参数 / Params | imgsz=128, batch=auto, epochs 上限 50, patience=10, device=GPU |
| 训练时长 / Time | 22 epochs / 9.2 分钟（早停 / early stopping） |
| 最佳轮次 / Best epoch | 第 13 轮（1-based） |
| mAP@0.5 | **0.9718** |
| mAP@0.5:0.95 | **0.9401** |
| precision / 精确率 | 0.9506 |
| recall / 召回率 | 0.9253 |

样图冒烟 / sample check: 6/6 预测正确，置信度 0.88~0.98。
验证集与训练集**无同一受试者**，故上述指标可视为跨人泛化估计。
/ Val subjects never appear in train, so metrics estimate cross-user generalization.

## 产物 / Artifacts

- 权重 / weights: `runs/eye_yolo26n/weights/best.pt`（未入库 / not committed）
- 曲线 / curves: `docs/training/curves_loss.png`、`docs/training/curves_metrics.png`
- 原始数据 / raw csv: `docs/training/results.csv`

## 曲线解读 / Curve notes

- 验证 cls_loss 在第 5 轮附近有尖峰后回落，属 LR warmup 结束后的正常波动
  / val cls_loss spikes around epoch 5 then settles — normal post-warmup noise
- 早停在 22 轮触发：后 10 轮 mAP50-95 无提升，best.pt 取自第 13 轮
  / early stop at 22: no mAP50-95 gain in last 10 epochs, best.pt from epoch 13

## 与轮椅控制的衔接 / Integration notes

- 推理输入：MediaPipe 裁剪出的单眼特写图（webcam 帧），
  输出 open_eye / closed_eye + 置信度，交给上层眨眼状态机（闭眼 0.04~0.8s 计为有效眨眼）
- 上下文：EyeWheelchairProject `src/interaction/blink_preview.py` 已有同款判定逻辑（EAR 阈值法），
  本模型可作为深度学习替代/冗余通道
