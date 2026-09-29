# 准确率评估 / accuracy eval — look_up, look_center, look_down, look_left, look_right

日期 / date: 2026-09-29 10:51　配置 / config: configs/columbia_gaze5.yaml
权重 / weights: runs/gaze5_yolo26s/weights/best.pt　验证集 / val: 2150 张

- 判定准确率 / accuracy: **80.90%**（1724/2131，无检测帧 19）
- 官方指标 / official: mAP50=0.8393　mAP50-95=0.8109
- 单张耗时 / latency: 8.41 ms/张（RTX 5060 Ti, imgsz=128）

| 类别 / class | 准确率 / acc | 样本数 / n |
|---|---|---|
| look_up | 84.10% | 497 |
| look_center | 42.07% | 145 |
| look_down | 83.23% | 495 |
| look_left | 87.37% | 499 |
| look_right | 80.20% | 495 |

- 混淆矩阵 / confusion（行=真值, 列=预测）: look_up→look_up 418，look_up→look_center 1，look_up→look_left 49，look_up→look_right 29，look_center→look_up 1，look_center→look_center 61，look_center→look_down 6，look_center→look_left 42，look_center→look_right 35，look_down→look_center 1，look_down→look_down 412，look_down→look_left 54，look_down→look_right 28，look_left→look_up 25，look_left→look_center 20，look_left→look_down 15，look_left→look_left 436，look_left→look_right 3，look_right→look_up 35，look_right→look_center 22，look_right→look_down 32，look_right→look_left 9，look_right→look_right 397

## MediaPipe 适用性 / applicability probe

- 300 张验证图上检出人脸 / face detected: 0（0.0%），单张 1.67 ms
- EAR 法需要整脸 478 关键点；眼部裁剪/近距离场景结构性不可用，YOLO 是可行选项 / EAR needs full-face landmarks; structurally unusable on eye crops & close range