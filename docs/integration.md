# EyeWheelchairProject 接入指南 / Integration Guide

> 给 EyeWheelchairProject（眼控轮椅）侧的开发者：怎么把本仓库训练好的两个
> YOLO26 模型接进现有"视线选方向 + 眨眼确认"管线。一页纸说清文件、接口、阈值、安全。

---

## 1. 模型清单 / Models

| 模型 | 权重（本项目仓库内） | 类别 | 用途 | 跨受试者准确率 |
|---|---|---|---|---|
| A：睁/闭眼 | `yolo_model/runs/eye_yolo26n/weights/best.pt`（ONNX 同目录 `best.onnx`，9.2MB） | `open_eye` / `closed_eye` | 眨眼确认 + 闭眼过久安全停止 | 94.33% |
| B：注视方向 | `yolo_model/runs/gaze_yolo26n/weights/best.pt`（`best.onnx` 同理） | `look_up` / `look_center` / `look_down` | 上看=前进 / 中看=停 / 下看=后退（映射可调） | 93.62% |

两个权重都基于 YOLO26n（237 万参数），GPU 单张 <2ms，CPU 也可实时。

## 2. 调用契约 / Calling contract

输入**不是**整帧画面，而是**MediaPipe 裁出的眼部图**（本机无摄像头场景下已验证的分工）：

```python
# EyeWheelchairProject 侧（已有 MediaPipe 478 点管线）每帧：
# 1) FaceLandmarker 出 478 点（运行模式 VIDEO，num_faces=1，与现有脚本一致）
# 2) 用眼周 6 点裁左右眼，扩边系数与本项目训练数据一致：
#    左眼索引 [33,160,158,133,153,144]  右眼 [362,385,387,263,373,380]
#    水平扩 1.6 倍、垂直扩 2.2 倍（含眉毛上下文），参考 yolo_model/scripts/compare_yolo_vs_mediapipe.py 的 eye_boxes()
# 3) 裁剪图送对应模型
from ultralytics import YOLO
eye_model  = YOLO(r"...\yolo_model\runs\eye_yolo26n\weights\best.pt")   # 模型A
gaze_model = YOLO(r"...\yolo_model\runs\gaze_yolo26n\weights\best.pt")  # 模型B

r = eye_model.predict(eye_crop, imgsz=128, verbose=False)[0]
cls = r.names[int(r.boxes.cls[int(r.boxes.conf.argmax())])]   # 'open_eye' / 'closed_eye'
conf = float(r.boxes.conf.max())                              # 建议 ≥0.5 才采信
```

- 两只眼分别判，**取一致结果**；不一致或无检测帧 → 沿用上一帧状态（防抖）
- 模型 A 和 B 吃同一个眼部裁剪，一帧两次推理即可（GPU 上合计 <4ms）

## 3. 与现有状态机的对接点 / State machine hooks

现有阈值**全部保留**（`blink_preview.py` 的判定参数已实测可靠）：

| 逻辑 | 现有实现 | 模型接入后 |
|---|---|---|
| 有效眨眼 | EAR < 基线×0.78，闭眼 0.04~0.8s，0.3s 不应期 | EAR 判定换成模型 A 的 `closed_eye` 连续帧段，时长/不应期逻辑不变 |
| 方向候选 | 视线几何 + 稳定停留 0.7s | 模型 B 的 `look_*` 连续 0.7s 才成候选 |
| 映射 | 左/中/右 | `look_up`=前进，`look_center`=停，`look_down`=后退 |

**重要**：模型 B 当前版本的主要误差是 up→center（约 12%，见
`docs/training/summary_gaze.md`），语义上是"该前进时变成停"——**宁停勿错走**，
符合安全设计；个人化微调后（自采视频 ≥3 人）会显著改善。

## 4. 无独显机器 / CPU-only deployment

- 用 ONNX：`pip install onnxruntime`，`YOLO("best.onnx")` 接口完全一致
  （ultralytics 自动识别），或直接用 onnxruntime API
- 实测参考：纯推理 0.5ms/张（GPU）；CPU 预计 10~30ms/帧，30fps 管线仍可实时

## 5. 安全边界（不可妥协）⚠️

1. 模型输出只是**意图信号**；`serial_link.py` 默认模拟模式必须保持，
   台架验证清单全过 + 物理急停按钮加装之前**禁止任何实机测试**
2. 低置信度（<0.5）、无脸帧、左右眼判定冲突 → 一律按"停"处理（fail-safe）
3. 模型 B 的 up/down 互窜率为 0（已验证），但状态机仍应保留"方向切换需经过
   中间态"的约束（up→down 必须路过 center 停止态）
