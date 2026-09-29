# 准确率评估 / accuracy eval — look_up, look_center, look_down

日期 / date: 2026-09-29 10:21　配置 / config: configs/columbia_gaze.yaml
权重 / weights: runs/gaze_yolo26s/weights/best.pt　验证集 / val: 1050 张

- 判定准确率 / accuracy: **95.62%**（1004/1050，无检测帧 0）
- 官方指标 / official: mAP50=0.9837　mAP50-95=0.9828
- 单张耗时 / latency: 2.43 ms/张（RTX 5060 Ti, imgsz=128）

| 类别 / class | 准确率 / acc | 样本数 / n |
|---|---|---|
| look_up | 92.00% | 350 |
| look_center | 96.29% | 350 |
| look_down | 98.57% | 350 |

- 混淆矩阵 / confusion（行=真值, 列=预测）: look_up→look_up 322，look_up→look_center 28，look_center→look_up 2，look_center→look_center 337，look_center→look_down 11，look_down→look_center 5，look_down→look_down 345

## MediaPipe 适用性 / applicability probe

- 300 张验证图上检出人脸 / face detected: 0（0.0%），单张 1.91 ms
- EAR 法需要整脸 478 关键点；眼部裁剪/近距离场景结构性不可用，YOLO 是可行选项 / EAR needs full-face landmarks; structurally unusable on eye crops & close range