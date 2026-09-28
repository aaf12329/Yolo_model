# yolo_model — 眼动追踪模型（YOLO26）· Eye Tracking Models for Eye-Controlled Wheelchair

> 中文版在上，英文版在下方 / Chinese version above, English version below.

---

# 中文版

## 一、项目简介

本项目是 [EyeWheelchairProject](../EyeWheelchairProject)（眼控轮椅原型）的**视觉模型子项目**，
用 YOLO26 训练两个轻量检测模型，把"眼球活动"变成轮椅可以执行的信号：

- **模型 A（已完成）**：睁眼 / 闭眼判定 → 支撑"眨眼确认"与"闭眼过久安全停止"
- **模型 B（规划中）**：注视方向判定（上看 / 中看 / 下看）→ 映射轮椅 **前进 / 停 / 后退**

上位机的交互状态机、串口输出层、Arduino 固件全部在 EyeWheelchairProject 仓库；
本仓库只负责"训练模型 + 验证效果 + 交付权重"。

> ⚠️ **安全边界**（同 EyeWheelchairProject）：模型输出只是"意图"信号。
> 任何硬件控制前必须通过台架验证清单并加装物理急停按钮，不进行任何载人测试。

## 二、当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | conda 环境 `yolo`（Python 3.10）+ PyTorch 2.11.0 cu128 + 项目骨架 | ✅ |
| 2 | 下载 MRL Eye Dataset（84,898 张）并抽样目视验证标签 | ✅ |
| 3 | YOLO26 GPU 冒烟测试（推理 + COCO8 训练循环） | ✅ |
| 4 | 数据集转 YOLO 格式（按受试者划分 train/val） | ✅ |
| 5 | 模型 A 训练（早停于第 22 轮，最佳第 13 轮，GPU 共 9.2 分钟） | ✅ |
| 6 | 眨眼判定准确率评估 + MediaPipe 对比实验 | ✅ |
| 7 | 模型 B 数据：Columbia Gaze 获取（5,880 张）+ 眼部裁剪转换（11,760 裁剪） | ✅ |
| 8 | 模型 B 初版训练（早停于第 27 轮，最佳第 18 轮，GPU 2.6 分钟） | ✅ |
| 9 | 模型 B 个人化微调（等手机采集素材 → 自动打标 → 精调） | ⏸ 等素材 |

**阶段 8 结果 / Stage 8 results**（详见 [docs/training/summary_gaze.md](docs/training/summary_gaze.md)）：
总判定准确率 **93.62%**（look_center 96.86% / look_down 96.00% / look_up 88.00%），
mAP50-95=**0.9754**；权重 `runs/gaze_yolo26n/weights/best.pt`。
图：`docs/comparison/columbia_gaze_confusion_matrix.png`、`columbia_gaze_accuracy_chart.png`。
关键结论：down 几乎不会误判为 up（0 例）——**前进/后退不会互相窜**；
主要误差是 up→center（+10° 上看幅度小），对控制语义无害且将由个人化微调修复。

## 三、环境信息

| 项 | 值 |
|---|---|
| conda env | `yolo`，Python 3.10，位于 `C:\Users\Guards\.conda\envs\yolo` |
| PyTorch | 2.11.0+cu128（RTX 50 系必须 cu128 及以上；官方 cu128 源安装） |
| ultralytics | 8.4.164（YOLO26 从 v8.4.0 起支持） |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB（Blackwell，sm_120，CUDA 12.8） |
| MediaPipe | 仅用于对比实验与后续自动打标（CPU 运行） |
| 其他 | opencv-python、matplotlib、pandas、numpy |

> 本机**没有摄像头**，所有实验基于数据集离线完成；实时 demo 脚本保留，
> 到有摄像头的机器上即可运行。

## 四、项目架构

```
yolo_model/
├─ README.md                        本文件：架构、进度、指标、规范（每阶段更新）
├─ COMMANDS.md                      全部指令速查（按阶段整理，含英文注释）
├─ requirements.txt                 依赖清单
├─ .gitignore                       数据集/权重/视频不入库
│
├─ configs/
│  └─ mrl_eye.yaml                  模型A的 YOLO 数据配置（路径、2 类名）
│
├─ models/
│  └─ face_landmarker.task          MediaPipe 人脸 478 点模型（复制自 EyeWheelchairProject，
│                                   供对比实验与自动打标复用，实验自包含）
│
├─ scripts/                         全部可执行脚本（每个文件头部有调用关系图注释）
│  ├─ verify_gpu.py                 ① 环境验证：torch/cuda/ultralytics 自检
│  ├─ prepare_mrl_dataset.py        ② 数据工程：解压 MRL → 解析文件名标签 →
│  │                                   按受试者划分 train/val → 硬链接建 YOLO 目录
│  │                                   → 生成 configs/mrl_eye.yaml
│  ├─ smoke_test_yolo26.py          ③ 冒烟测试：GPU 推理 + COCO8 训练 1 epoch
│  ├─ train_eye.py                  ④ 模型A训练入口（epochs/imgsz/batch 可调）
│  ├─ plot_training.py              ⑤ 训练可视化：读 results.csv 画损失/指标曲线
│  ├─ eval_eye_accuracy.py          ⑥ 准确率评估：验证集逐张判定 vs 文件名标签，
│  │                                   混淆矩阵 + MediaPipe 适用性探测 + 图表
│  ├─ compare_yolo_vs_mediapipe.py  ⑦ 实时对比（需摄像头，本机无，保留备用）
│  ├─ collect_gaze_video.py         ⑧ 模型B素材自采工具（电脑摄像头+屏幕提示，
│  │                                   输出到 gaze_captures/，备用）
│  └─ auto_label_phone_videos.py    ⑨ 手机视频自动打标（抽帧+裁眼+QC虹膜排序校验，
│                                      已用合成视频端到端验证；输出 datasets/phone_gaze_yolo/）
│
├─ phone_videos/                    模型B手机素材归档区（一人一文件夹，文件名带上/中/下）
│  └─ 转发给拍摄的人.txt             可直接整段复制发微信群的大白话拍摄说明
│
├─ data/                           数据来源与下载专用文件夹（详见其 README）
│  ├─ README.md                    两个数据集的来源/许可/符号约定/下载指令
│  └─ download_datasets.py         一键下载器（断点续传+C盘检查+体积校验+可选解压）
│
├─ docs/
│  ├─ yolo_tutorial.md              YOLO 入门教程：原理/结构/训练，全部用本项目实例讲解
│  ├─ dataset_sources.md            数据集查找渠道记录（网站/直链/许可/磁盘策略）
│  ├─ ai_dev_resources.md           AI 开发网站与工具大全（文档/算力/标注/部署/学习资源）
│  ├─ gaze_data_collection_guide.md 模型B采集指南（手机多人版，含转发文案与归档规则）
│  ├─ integration.md                EyeWheelchairProject 接入指南（接口/阈值/安全）
│  ├─ training/                     模型A训练档案：summary.md、损失/指标曲线、results.csv
│  └─ comparison/                   对比实验档案：评估报告、混淆矩阵图、对比柱状图
│
├─ datasets/                        （不入库）MRL 原始 zip、解压目录、YOLO 格式数据集
│
├─ gaze_captures/                   （不入库）脚本自采的方向素材视频
│
└─ runs/                            （不入库）训练输出；模型A权重：
                                    runs/eye_yolo26n/weights/best.pt
```

## 五、模型 A：睁眼/闭眼判定（已交付）

### 5.1 数据

- **来源**：MRL Eye Dataset（mrlEyes_2018_01，捷克 VSB-TU Ostrava Media Research Lab），
  官方下载 <http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip>（~326MB）
- **规模**：84,898 张眼部特写灰度图（约 83×83），含眼镜/光照/人种多样性
- **标签**：文件名第 5 字段（`0`=睁眼，`1`=闭眼），已抽样目视核验
- **划分**：**按受试者**划分——37 人中 s0010/s0020/s0030 全人进验证集（2,487 张），
  其余进训练集（82,411 张），杜绝"同一人既当教材又当考题"的指标虚高
  - train：睁眼 41,259 / 闭眼 41,152（均衡）
  - val：睁眼 687 / 闭眼 1,800
- **YOLO 标签**：每图一个整幅单框（特写图本身即眼部）；定位交给 MediaPipe，
  本模型专注判定状态

### 5.2 训练

- `yolo26n.pt`（COCO 预训练）迁移学习；imgsz=128，batch 自动，最多 50 轮，patience=10
- 早停：第 22 轮停止，**best.pt 取自第 13 轮**；GPU 实际用时 9.2 分钟

### 5.3 指标

| 指标 | 值 | 说明 |
|---|---|---|
| mAP@0.5 | 0.9718 | ultralytics 官方 val |
| mAP@0.5:0.95 | 0.9401 | 同上 |
| precision / recall | 0.9506 / 0.9253 | 同上 |
| **判定准确率** | **94.33%** | 逐张 vs 文件名标签（2,485/2,487） |
| 闭眼准确率 | 96.05% | 1,728/1,800 |
| 睁眼准确率 | 89.80% | 616/687（受 MRL 标签噪声影响，为下限） |
| 单张耗时 | 1.43 ms | RTX 5060 Ti，imgsz=128，batch=128 |

### 5.4 与 MediaPipe 的对比（为什么引入 YOLO26）

| 方法 | 结果 |
|---|---|
| YOLO26（本项目） | 跨受试者判定准确率 94.33%，单张 1.43ms |
| MediaPipe FaceLandmarker | 300 张眼部特写 **0 张检出人脸（0%）** |

- EAR（眼纵横比）法必须看到**整张脸**才能取 478 关键点；轮椅场景人离摄像头近、
  画面主体是眼部，MediaPipe 结构性失效，YOLO26 不受此限制。
- MediaPipe 保留"定位"价值：出 478 点 → 裁眼眶 → 交 YOLO26 判定；近距离找不到脸时
  YOLO26 仍是唯一可用判定器。
- 曲线与图：`docs/training/curves_loss.png`、`curves_metrics.png`、
  `docs/comparison/confusion_matrix.png`、`accuracy_chart.png`

### 5.5 产物

- 权重：`runs/eye_yolo26n/weights/best.pt`（~6MB，不入库，训练可复现）
- 输入约定：MediaPipe 眼眶裁剪图（含眉毛上下文）；输出：`open_eye` / `closed_eye` + 置信度

## 六、模型 B：注视方向判定（数据已就位）

- **目标**：三类 `look_up` / `look_center` / `look_down` → 上层映射 前进 / 停 / 后退
- **底座数据（已获取）**：Columbia Gaze Data Set——5,880 张、56 人、5 头姿 × 3 垂直
  注视（0/±10°），21 人戴眼镜。**符号已目视验证：`-10V`=上看、`0V`=平视、
  `10V`=下看**（正号表示视线下方，反直觉，写脚本时务必注意）。
  来源/许可/下载方式见 [data/README.md](data/README.md)
- **转换流水线**：18MP 原图缩放 → MediaPipe 裁左/右眼（每源图 2 个裁剪）→
  3 类整图框 YOLO 数据（`scripts/prepare_columbia_gaze_dataset.py`，
  预计约 1.1 万个裁剪，~20 分钟 CPU）
- **个人化微调（等素材）**：每人手机拍 3 段视频（上看/中看/下看各 30s~1min，
  **头不动只动眼睛**），微信"文件"方式发送，归档 `phone_videos/一人一文件夹/`
  （协议与归档规则见 [docs/gaze_data_collection_guide.md](docs/gaze_data_collection_guide.md)，
  启动门槛 ≥3 人，建议 5~8 人）
- **流水线**：视频 + 文件名方向 → MediaPipe 自动打标 → YOLO 格式 → 在底座权重上
  微调 → 曲线+报告 → 对接状态机

## 七、快速开始

```bash
conda activate yolo
cd C:\Users\Guards\Desktop\yolo_model

python data/download_datasets.py --dataset all --extract  # ⓪ 下载数据（断点续传+校验）
python scripts/verify_gpu.py             # ① 验证 GPU 环境
python scripts/prepare_mrl_dataset.py    # ② 数据集转换（已执行过，可重复）
python scripts/train_eye.py              # ③ 训练模型A（复现）
python scripts/plot_training.py          # ④ 画损失曲线
python scripts/eval_eye_accuracy.py      # ⑤ 准确率评估+对比图
# python scripts/collect_gaze_video.py   # ⑥（有摄像头时）自采方向素材
```

全部指令逐条解释见 [COMMANDS.md](COMMANDS.md)。

## 八、与 EyeWheelchairProject 的集成

- **现管线**：`gaze_blink_confirm_demo.py` 用 MediaPipe 478 点做"视线选方向 + 眨眼确认"，
  眨眼判定基于 EAR 阈值 + 个人基线校准
- **本仓库的接入点**：以 `best.pt` 替换/冗余其 EAR 判定——MediaPipe 出眼眶裁剪图，
  YOLO26 出 `open_eye/closed_eye` + 置信度，上层眨眼状态机逻辑不变
  （闭眼 0.04~0.8s 计有效眨眼、0.3s 不应期等阈值全部保留）
- **安全**：`serial_link.py` 默认模拟模式、低电锁、心跳包等机制不变；
  模型错误判定不应直接变成轮椅动作，状态机层的确认逻辑继续生效

## 九、Git 提交规范 / 提交分工

**分工规则 / Responsibilities**（2026-09-28 约定）：

- 日常 `git add` + `git commit` 由 ZCode（助手）负责执行
  / day-to-day `git add` + `git commit` is ZCode's (the assistant's) job
- **`git push` 只有仓库所有者本人执行**；ZCode 需要 push 时**必须先征得同意**，
  除非所有者明确说过可以推
  / **`git push` is reserved for the repo owner**; ZCode must ask for permission
  before pushing, unless the owner has explicitly said otherwise

每阶段一次提交，注释**中英双语**（中文在前，英文在后）：

```
阶段X：中文说明 / Stage X: English description
```

## 十、已知限制

- 本机无摄像头：实时脚本未实测，接口与离线链路已验证
- MRL 文件名标签含少量噪声（文献报告一致率约 90~97%），故准确率为下限
- MRL 以非亚裔面孔为主；模型 B 自采数据将补足本群体特征
- `datasets/`、`runs/`、视频均不入库；权重与数据集可由脚本完整复现

---

# English Version

## 1. Overview

This repo is the **vision-model sub-project** of [EyeWheelchairProject](../EyeWheelchairProject)
(an eye-controlled wheelchair prototype). It trains two lightweight YOLO26 detectors that turn
eye activity into signals a wheelchair can act on:

- **Model A (done)**: open-eye / closed-eye judgment → powers "blink = confirm" and
  "eyes closed too long = safety stop"
- **Model B (v1 trained)**: gaze-direction judgment (look up / center / down) → mapped to
  **forward / stop / backward**

The interaction state machines, serial link, and Arduino firmware live in the
EyeWheelchairProject repo; this repo only trains, validates, and ships models.

> ⚠️ **Safety boundary** (same as EyeWheelchairProject): model output is an "intent" signal only.
> No hardware control before bench-validation checklists pass and a physical e-stop is installed.
> No on-human testing.

## 2. Progress

| Stage | Content | Status |
|---|---|---|
| 1 | conda env `yolo` (Python 3.10) + PyTorch 2.11.0 cu128 + skeleton | ✅ |
| 2 | Download MRL Eye Dataset (84,898 imgs), visually verify labels | ✅ |
| 3 | YOLO26 GPU smoke test (inference + COCO8 training loop) | ✅ |
| 4 | Convert dataset to YOLO format (subject-wise train/val) | ✅ |
| 5 | Train Model A (early stop @22, best @13, 9.2 min on GPU) | ✅ |
| 6 | Accuracy benchmark vs MediaPipe | ✅ |
| 7 | Model B data: Columbia Gaze acquired (5,880 imgs) + eye-crop conversion (11,760 crops) | ✅ |
| 8 | Model B v1 training (early stop @27, best @18, 2.6 min on GPU) | ✅ |
| 9 | Model B personalization (awaiting phone footage → auto-label → fine-tune) | ⏸ waiting |

**Stage 8 results** (see [docs/training/summary_gaze.md](docs/training/summary_gaze.md)):
overall accuracy **93.62%** (look_center 96.86% / look_down 96.00% / look_up 88.00%),
mAP50-95=**0.9754**; weights `runs/gaze_yolo26n/weights/best.pt`.
Key finding: down is never misread as up (0 cases) — **forward/backward never swap**;
main error is up→center (small +10° amplitude), harmless for control semantics.

## 3. Environment

| Item | Value |
|---|---|
| conda env | `yolo`, Python 3.10, at `C:\Users\Guards\.conda\envs\yolo` |
| PyTorch | 2.11.0+cu128 (RTX 50-series requires cu128+; installed from official cu128 index) |
| ultralytics | 8.4.164 (YOLO26 shipped in v8.4.0) |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell, sm_120, CUDA 12.8) |
| MediaPipe | for the comparison probe and upcoming auto-labeling (CPU) |
| Others | opencv-python, matplotlib, pandas, numpy |

> This machine has **no camera**; all experiments are dataset-based and offline.
> Live-demo scripts are kept and ready for any camera-equipped machine.

## 4. Repository Layout

```
yolo_model/
├─ README.md                        this file: architecture, progress, metrics (updated per stage)
├─ COMMANDS.md                      command reference organized by stage
├─ requirements.txt                 dependencies
├─ .gitignore                       datasets / weights / videos stay out of git
│
├─ configs/
│  └─ mrl_eye.yaml                  Model-A YOLO data config (paths, 2 class names)
│
├─ models/
│  └─ face_landmarker.task          MediaPipe 478-pt face model (copied from
│                                   EyeWheelchairProject; reused for probing & auto-labeling)
│
├─ scripts/                         every script has a call-graph header comment
│  ├─ verify_gpu.py                 ① environment check: torch/cuda/ultralytics
│  ├─ prepare_mrl_dataset.py        ② data eng: unzip MRL → parse filename labels →
│  │                                   subject-wise split → hardlinked YOLO tree → data yaml
│  ├─ smoke_test_yolo26.py          ③ smoke test: GPU inference + 1-epoch COCO8 training
│  ├─ train_eye.py                  ④ Model-A training entry (epochs/imgsz/batch flags)
│  ├─ plot_training.py              ⑤ visualize: loss/metric curves from results.csv
│  ├─ eval_eye_accuracy.py          ⑥ accuracy eval: per-image judgment vs filename labels,
│  │                                   confusion matrix + MediaPipe applicability probe + charts
│  ├─ compare_yolo_vs_mediapipe.py  ⑦ live head-to-head (needs camera; kept for later)
│  ├─ collect_gaze_video.py         ⑧ Model-B self-capture tool (webcam + on-screen prompts)
│  └─ auto_label_phone_videos.py    ⑨ phone-video auto-labeling (frame sampling + eye crops
│                                      + iris-ordering QC; e2e verified on synthetic videos)
│
├─ phone_videos/                    Model-B phone-footage archive (one folder per person;
│  └─ 转发给拍摄的人.txt             copy-paste WeChat instructions for contributors
│
├─ data/                           dataset sources & downloads (see its README)
│  ├─ README.md                    sources / licenses / sign conventions / commands
│  └─ download_datasets.py         one-shot downloader (resume + disk check + verify)
│
├─ docs/
│  ├─ yolo_tutorial.md              YOLO primer: principles/architecture/training, all
│  │                                   illustrated with this repo's real artifacts
│  ├─ dataset_sources.md            where to find datasets (sites, direct links, licenses, disk policy)
│  ├─ ai_dev_resources.md           AI-dev sites & tools catalog (docs/compute/labeling/deploy/learning)
│  ├─ gaze_data_collection_guide.md Model-B collection guide (multi-person phone edition)
│  ├─ integration.md                EyeWheelchairProject integration guide (contract/thresholds/safety)
│  ├─ training/                     Model-A record: summary.md, loss/metric curves, results.csv
│  └─ comparison/                   benchmark record: report, confusion matrix, accuracy bars
│
├─ datasets/                        (not committed) MRL zip, extracted tree, YOLO-format dataset
├─ gaze_captures/                   (not committed) self-captured gaze footage
└─ runs/                            (not committed) training output; Model-A weights:
                                    runs/eye_yolo26n/weights/best.pt
```

## 5. Model A: Open/Closed Eye (delivered)

### 5.1 Data

- **Source**: MRL Eye Dataset (mrlEyes_2018_01, Media Research Lab, VSB-TU Ostrava),
  <http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip> (~326 MB)
- **Scale**: 84,898 grayscale eye close-ups (~83×83) with glasses/lighting/ethnicity diversity
- **Labels**: 5th filename field (`0`=open, `1`=closed), spot-checked visually
- **Split**: **subject-wise** — 37 subjects; s0010/s0020/s0030 (2,487 imgs) form the val set,
  the rest (82,411 imgs) train. No identity overlap between train and val.
  - train: 41,259 open / 41,152 closed (balanced)
  - val: 687 open / 1,800 closed
- **YOLO labels**: one full-image box per close-up; localization is MediaPipe's job,
  this model judges the state.

### 5.2 Training

- Transfer learning from `yolo26n.pt` (COCO); imgsz=128, auto batch, max 50 epochs, patience=10
- Early stop at epoch 22; **best.pt is from epoch 13**; 9.2 minutes on GPU

### 5.3 Metrics

| Metric | Value | Note |
|---|---|---|
| mAP@0.5 | 0.9718 | official ultralytics val |
| mAP@0.5:0.95 | 0.9401 | official |
| precision / recall | 0.9506 / 0.9253 | official |
| **Judgment accuracy** | **94.33%** | per-image vs filename labels (2,485/2,487) |
| Closed-eye accuracy | 96.05% | 1,728/1,800 |
| Open-eye accuracy | 89.80% | 616/687 (limited by MRL label noise; a lower bound) |
| Latency | 1.43 ms/img | RTX 5060 Ti, imgsz=128, batch=128 |

### 5.4 Why YOLO26 instead of MediaPipe EAR

| Method | Result |
|---|---|
| YOLO26 (this repo) | 94.33% cross-subject accuracy, 1.43 ms/img |
| MediaPipe FaceLandmarker | face detected in **0 of 300** eye close-ups (0%) |

- The EAR method requires **478 full-face landmarks**; in the wheelchair scenario the user
  sits close to the camera and the frame is dominated by the eye region, where MediaPipe
  fails structurally. YOLO26 has no such constraint.
- MediaPipe keeps its "localization" value: 478 points → eye crops → YOLO26 judges;
  when the face is too close to detect, YOLO26 remains the only working judge.
- Curves & charts: `docs/training/curves_loss.png`, `curves_metrics.png`,
  `docs/comparison/confusion_matrix.png`, `accuracy_chart.png`

### 5.5 Artifacts

- Weights: `runs/eye_yolo26n/weights/best.pt` (~6 MB, not committed; training is reproducible)
- Input contract: MediaPipe eye crops (with brow context); output: `open_eye` / `closed_eye` + confidence

## 6. Model B: Gaze Direction (data acquired)

- **Goal**: 3 classes `look_up` / `look_center` / `look_down` → forward / stop / backward
- **Bootstrap data (acquired)**: Columbia Gaze Data Set — 5,880 images, 56 subjects,
  5 head poses × 3 vertical gaze (0/±10°), 21 wear glasses. **Sign convention visually
  verified: `-10V` = up, `0V` = center, `10V` = down** (positive means below horizon —
  counterintuitive; watch out in scripts). Sources/licenses/downloads: [data/README.md](data/README.md)
- **Conversion pipeline**: downscale 18MP originals → MediaPipe crops left/right eye
  (2 crops per source) → 3-class full-image-box YOLO data
  (`scripts/prepare_columbia_gaze_dataset.py`, ~11k crops expected, ~20 min on CPU)
- **Personalization (waiting)**: each contributor records 3 phone clips (up/center/down,
  30 s–1 min each, **head still, eyes only**), sent via WeChat "file" mode, archived under
  `phone_videos/<person>/` (protocol: [docs/gaze_data_collection_guide.md](docs/gaze_data_collection_guide.md);
  kick-off threshold ≥3 contributors, 5–8 recommended)
- **Pipeline**: footage + filename direction → MediaPipe auto-labeling → YOLO format →
  fine-tune on bootstrap weights → curves + report → state machine

## 7. Quick Start

```bash
conda activate yolo
cd C:\Users\Guards\Desktop\yolo_model

python data/download_datasets.py --dataset all --extract  # ⓪ download datasets
python scripts/verify_gpu.py             # ① verify GPU environment
python scripts/prepare_mrl_dataset.py    # ② dataset conversion (idempotent)
python scripts/train_eye.py              # ③ train Model A (reproduce)
python scripts/plot_training.py          # ④ plot loss curves
python scripts/eval_eye_accuracy.py      # ⑤ accuracy eval + comparison charts
# python scripts/collect_gaze_video.py   # ⑥ (camera required) capture gaze footage
```

Every command explained line-by-line in [COMMANDS.md](COMMANDS.md).

## 8. Integration with EyeWheelchairProject

- **Current pipeline**: `gaze_blink_confirm_demo.py` uses MediaPipe 478-pt landmarks for
  "gaze picks direction + blink confirms"; blink judgment is EAR-threshold with per-user calibration
- **Integration point**: `best.pt` replaces / redundandizes the EAR judgment — MediaPipe
  provides eye crops, YOLO26 returns `open_eye/closed_eye` + confidence; the blink state
  machine is untouched (0.04–0.8 s valid blink, 0.3 s refractory period, etc.)
- **Safety**: `serial_link.py` stays in simulation-by-default mode with heartbeat and
  low-battery lock; a wrong model verdict never maps directly to a wheelchair action —
  the state machine's confirmation logic still gates it

## 9. Git Commit Convention & Responsibilities

**Responsibilities** (agreed 2026-09-28):

- Day-to-day `git add` + `git commit` is ZCode's (the assistant's) job
- **`git push` is reserved for the repo owner**; ZCode must ask for permission
  before pushing, unless the owner has explicitly said otherwise

One commit per stage, **bilingual messages** (Chinese first, then English):

```
阶段X：中文说明 / Stage X: English description
```

## 10. Known Limitations

- No camera on this machine: live scripts untested in situ; offline interfaces fully verified
- MRL filename labels carry minor noise (~90–97% agreement reported); accuracy is a lower bound
- MRL is mostly non-Asian faces; Model B's self-collected data will cover our user population
- `datasets/`, `runs/`, and videos are not committed; weights and datasets are fully
  reproducible from the scripts
