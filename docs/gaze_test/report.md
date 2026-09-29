# 注视方向模型（YOLO26）实时测试 / Live gaze-direction benchmark

日期 / date: 2026-09-29 17:31　协议: 2 轮 × (look_up + look_center + look_down)，每段 6s，首尾 0.7s 过渡不计分；总耗时 37s
地面真值 = 屏幕目标点位置（协议法）。闭眼帧经模型 A 门控后不计分。

- 计分帧 / counted: **5**　未计分 / skipped: 无人脸 0，闭眼 408
- **总准确率 / overall accuracy: 20.0%**

| 目标 / target | 准确率 / accuracy |
|---|---|
| look_up | 0.0% |
| look_center | 100.0% |
| look_down | 0.0% |

## 混淆矩阵 / confusion matrix（行=目标，列=预测）

| target \ pred | look_up | look_center | look_down |
|---|---|---|---|
| look_up | 0 | 0 | 0 |
| look_center | 0 | 1 | 0 |
| look_down | 3 | 1 | 0 |

## 判读 / notes
- 若 up/down 互相串（上↔下），说明模型过度依赖「眼球在眼眶中的位置」，个人化微调时要补数据
  / up↔down confusion means the model over-relies on eye position — personalise with your own data
- 单帧耗时含 MediaPipe 定位 + 双眼 2 模型；纯 CPU 约 60~80ms 属正常
- 逐帧数据 / per-frame data: frames.csv（413 行）