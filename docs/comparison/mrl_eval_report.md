# 眨眼判定准确率评估 / open-closed eye accuracy eval

日期 / date: 2026-09-28 20:22　数据 / data: MRL 验证集 2487 张（受试者 s0010/s0020/s0030，与训练集零重叠）
标签来源 / labels: 官方文件名（第5字段），已抽样目视核验 / visually verified

## YOLO26（本项目 best.pt）

- 判定准确率 / accuracy: **94.33%**（2344/2485，无检测帧 2）
- 睁眼准确率 / open: 89.80%　闭眼准确率 / closed: 96.05%
- 官方指标 / official: mAP50=0.9719　mAP50-95=0.9399
- 混淆矩阵 / confusion (行=真值, 列=预测): 闭眼→闭眼 1728，闭眼→睁眼 71，睁眼→闭眼 70，睁眼→睁眼 616
- 单张耗时 / latency: 1.43 ms/张（RTX 5060 Ti, imgsz=128, batch=128）

## MediaPipe FaceLandmarker 适用性 / applicability probe

- 在 300 张眼部特写上检出人脸 / face detected: 0 张 （0.0%），单张 1.49 ms
- 结论 / conclusion: EAR 法需要整脸 478 关键点；在 MRL 这类眼部特写上结构上不可用（检不出脸）
- 即在"近距离摄像头/眼部裁剪"场景下，MediaPipe 无鼻可施，YOLO26 是可行选项

## 标注噪声说明 / label noise
- MRL 文件名标签在文献中约有少量噪声（估 90~97% 一致率），因此此处准确率是下限估计 / filename labels carry minor noise; accuracy is a lower bound