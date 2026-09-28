# 准确率评估 / accuracy eval — look_up, look_center, look_down

日期 / date: 2026-09-28 22:16　配置 / config: configs/columbia_gaze.yaml
权重 / weights: runs/gaze_yolo26n/weights/best.pt　验证集 / val: 1050 张

- 判定准确率 / accuracy: **93.62%**（983/1050，无检测帧 0）
- 官方指标 / official: mAP50=0.9776　mAP50-95=0.9776
- 单张耗时 / latency: 16.66 ms/张（RTX 5060 Ti, imgsz=128）

| 类别 / class | 准确率 / acc | 样本数 / n |
|---|---|---|
| look_up | 88.00% | 350 |
| look_center | 96.86% | 350 |
| look_down | 96.00% | 350 |

- 混淆矩阵 / confusion（行=真值, 列=预测）: look_up→look_up 308，look_up→look_center 42，look_center→look_up 3，look_center→look_center 339，look_center→look_down 8，look_down→look_center 14，look_down→look_down 336

## MediaPipe 适用性 / applicability probe

- 300 张验证图上检出人脸 / face detected: 0（0.0%），单张 1.82 ms
- EAR 法需要整脸 478 关键点；眼部裁剪/近距离场景结构性不可用，YOLO 是可行选项 / EAR needs full-face landmarks; structurally unusable on eye crops & close range