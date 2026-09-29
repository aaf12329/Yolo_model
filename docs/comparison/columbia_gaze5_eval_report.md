# 准确率评估 / accuracy eval — look_up, look_center, look_down, look_left, look_right

日期 / date: 2026-09-29 18:13　配置 / config: configs/columbia_gaze5.yaml
权重 / weights: runs/gaze5_yolo26s/weights/best.pt　验证集 / val: 2250 张

- 判定准确率 / accuracy: **82.54%**（1849/2240，无检测帧 10）
- 官方指标 / official: mAP50=0.8887　mAP50-95=0.8517
- 单张耗时 / latency: 1.65 ms/张（RTX 5060 Ti, imgsz=128）

| 类别 / class | 准确率 / acc | 样本数 / n |
|---|---|---|
| look_up | 81.33% | 498 |
| look_center | 76.52% | 443 |
| look_down | 81.20% | 500 |
| look_left | 86.25% | 400 |
| look_right | 88.72% | 399 |

- 混淆矩阵 / confusion（行=真值, 列=预测）: look_up→look_up 405，look_up→look_center 5，look_up→look_left 47，look_up→look_right 41，look_center→look_up 3，look_center→look_center 339，look_center→look_down 8，look_center→look_left 43，look_center→look_right 50，look_down→look_down 406，look_down→look_left 46，look_down→look_right 48，look_left→look_up 18，look_left→look_center 21，look_left→look_down 14，look_left→look_left 345，look_left→look_right 2，look_right→look_up 19，look_right→look_center 14，look_right→look_down 11，look_right→look_left 1，look_right→look_right 354

## MediaPipe 适用性 / applicability probe

- 300 张验证图上检出人脸 / face detected: 1（0.3%），单张 1.87 ms
- EAR 法需要整脸 478 关键点；眼部裁剪/近距离场景结构性不可用，YOLO 是可行选项 / EAR needs full-face landmarks; structurally unusable on eye crops & close range