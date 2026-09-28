# yolo_model — 眼动追踪模型（YOLO26）

服务于 [EyeWheelchairProject](../EyeWheelchairProject) 的目标检测模型子项目：
用 YOLO26 识别**睁眼 / 闭眼**，配合已有的 MediaPipe 视线+眨眼管线，
实现"眼球活动操控轮椅前进 / 后退"。

Serves the [EyeWheelchairProject](../EyeWheelchairProject): a YOLO26 model that
classifies **open eye / closed eye** on webcam eye crops, complementing the
existing MediaPipe gaze+blink pipeline for eye-controlled wheelchair operation.

---

## 当前进度 / Progress

| 阶段 / Stage | 内容 / Content | 状态 / Status |
|---|---|---|
| 1 | conda 环境 `yolo`（Python 3.10）+ PyTorch 2.11.0 cu128 + 项目骨架 | ✅ |
| 2 | 下载 MRL Eye Dataset（84,898 张）并抽样目视验证标签 | ✅ |
| 3 | YOLO26 GPU 冒烟测试（推理 + 1 epoch 训练） | ✅ |
| 4 | 数据集转 YOLO 格式（按受试者划分 train/val） | ✅ |
| 5 | YOLO26n 睁/闭眼训练（50 epochs, imgsz=128, GPU）+ matplotlib 损失曲线 | 🔄 训练中 |
| 6 | 注视方向模型（上看/中看/下看 → 前进/停/后退） | ⏸ 等待录制素材 |

> 阶段 6 说明 / Stage 6 note: 公开数据集没有"注视方向+检测框"标注，需用
> `scripts/collect_gaze_video.py` 录制本人素材（按屏幕提示往上看/看中间/往下看），
> 再用 MediaPipe 自动打标训练。录制方法见下方"注视方向数据采集"。
> No public dataset offers gaze-direction detection labels, so stage 6 needs
> self-recorded footage (auto-labeled with MediaPipe afterwards).

## 环境信息 / Environment

| 项 / Item | 值 / Value |
|---|---|
| conda env | `yolo`（Python 3.10），`C:\Users\Guards\.conda\envs\yolo` |
| PyTorch | 2.11.0+cu128（RTX 5060 Ti, sm_120, CUDA 12.8） |
| ultralytics | 8.4.164（YOLO26 从 v8.4.0 起支持） |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |

## 快速开始 / Quick Start

```bash
conda activate yolo
python scripts/verify_gpu.py            # 验证 GPU / verify GPU
python scripts/prepare_mrl_dataset.py   # 数据集转换 / dataset conversion
python scripts/train_eye.py             # 训练 / train
python scripts/plot_training.py         # 损失曲线 / loss curves
```

全部指令详见 [COMMANDS.md](COMMANDS.md)。

## 数据集 / Dataset

**MRL Eye Dataset**（mrlEyes_2018_01，Media Research Lab, VSB-TU Ostrava）

- 84,898 张眼部特写灰度图（83×83 左右的小图）
- 类别 / classes：`open_eye`（睁眼，41,946 张）/ `closed_eye`（闭眼，42,952 张）
- 标注来源 / labels from：文件名第 5 个字段（已抽样目视验证 / visually verified）
- 下载 / download: <http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip>（~326 MB）
- 划分 / split：按受试者划分（每第 10 人进 val），避免同一人跨集泄漏 / subject-wise split

```
datasets/mrl_eye_yolo/
├─ images/{train,val}/*.png      硬链接到解压目录 / hardlinks
├─ labels/{train,val}/*.txt      整图单框: "cls 0.5 0.5 1.0 1.0" / full-image box
configs/mrl_eye.yaml             YOLO 数据配置 / YOLO data config
```

## 注视方向数据采集 / Gaze data capture（阶段 6 素材）

```bash
conda activate yolo
python scripts/collect_gaze_video.py            # 3 zones x 2 rounds x 20s ≈ 2 分钟
python scripts/collect_gaze_video.py --seconds 15 --rounds 3
```

操作 / how to:

1. 坐在平时使用轮椅的**同一位置、同一距离**，正常戴眼镜（如果你平时戴）/ sit as in real use
2. 面对摄像头，按 **空格** 开始，按提示依次"往上看 → 看中间 → 往下看"，每段 20 秒带倒计时
3. **关键：头保持不动，只动眼球**——否则模型学到的会是头部姿态而不是注视方向
   / keep the head still, move eyes only, otherwise the model learns head pose
4. 自然眨眼即可；录满 2 轮（约 2 分钟）后按 q 结束
5. 建议在不同光照下再录 1~2 次（白天/晚上），增强鲁棒性 / record extra sessions in different lighting

输出 / output（不进 git，位于 EyeWheelchairProject/data/raw_videos/）:

- `gaze_capture_日期时间.mp4` — 原始视频 / raw footage
- `gaze_capture_日期时间.json` — 每段 {zone, t_start, t_end}，供自动打标 / segments for auto-labeling

## 与 EyeWheelchairProject 的关系 / Integration

该项目的 Python 端已用 **MediaPipe 人脸 478 点**实现"视线选方向 + 眨眼确认"。
本模型的分工 / division of labor：

- **MediaPipe**：定位眼部（眼框裁剪）/ locate eyes, provide crops
- **YOLO26 模型 A（本项目，训练中）**：眼部裁剪图上判定 睁眼/闭眼 → 眨眼确认 + 闭眼过久安全停止
- **YOLO26 模型 B（阶段 6）**：注视方向 上看/中看/下看 → 轮椅 前进/停/后退
- 上层状态机把判定结果映射为轮椅指令
  （详见 EyeWheelchairProject 的 `src/hardware/serial_link.py` 与安全说明）

> ⚠️ 安全边界同 EyeWheelchairProject：任何硬件控制前必须通过台架验证并加装物理急停。
> Same safety boundary: no hardware control before bench validation + physical e-stop.

## 目录结构 / Structure

```
yolo_model/
├─ README.md                     本文件（每阶段更新 / updated per stage）
├─ COMMANDS.md                   全部指令整理 / command reference
├─ requirements.txt              依赖清单 / dependencies
├─ configs/mrl_eye.yaml          YOLO 数据配置 / data config
├─ scripts/
│  ├─ verify_gpu.py              GPU 验证 / GPU check
│  ├─ prepare_mrl_dataset.py     数据集转换 / dataset conversion
│  ├─ smoke_test_yolo26.py       YOLO26 冒烟测试 / smoke test
│  ├─ train_eye.py               训练入口 / training entry
│  └─ plot_training.py           matplotlib 损失/指标曲线 / loss & metric curves
├─ datasets/                     数据（gitignore，不入库 / not committed）
└─ runs/                         训练输出（gitignore；曲线图会另行提交 / curves committed）
```

## Git 提交规范 / Commit convention

每个阶段一次提交，注释中英双语 / one commit per stage, bilingual message:

```
阶段X：中文说明 / Stage X: English description
```
