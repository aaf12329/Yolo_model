# 常用指令整理 / Command Reference

> 本文件记录项目搭建与训练过程中用到的所有指令，按阶段整理，方便复现。
> This file records every command used to build & train the project, organized by stage.

## 0. 环境 / Environment

- GPU: NVIDIA GeForce RTX 5060 Ti（Blackwell, sm_120, 16GB）
- conda 25.11.1（安装于 `D:\Anaconda`）
- 环境位置 / env location: `C:\Users\Guards\.conda\envs\yolo`

## 1. 创建 conda 环境 / Create conda environment

```bash
# 创建名为 yolo 的环境，Python 3.10 / create env "yolo" with Python 3.10
conda create -n yolo python=3.10 -y

# 激活 / activate
conda activate yolo
```

## 2. 安装 PyTorch（CUDA 12.8）/ Install PyTorch

> RTX 50 系是 sm_120 架构，必须用 cu128 及以上的 PyTorch 构建版本，
> 用官方 cu128 源安装（约 3GB，较慢）。
> RTX 50-series needs a cu128+ build; install from the official cu128 index (~3 GB).

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

验证 / verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出类似 / expect like: 2.11.0+cu128 True NVIDIA GeForce RTX 5060 Ti
```

> **无独显的机器 / CPU-only machine**（2026-09-29 追加）：直接从 PyPI 装 CPU 构建，
> 版本号相同、API 一致，只是推理/训练慢；不要再加 cu128 的 --index-url。
> / On a machine without a discrete GPU, install the CPU build from PyPI — same
> version, same API, just slower; drop the cu128 --index-url.

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0
# 期望输出类似 / expect like: 2.11.0+cpu False
# 实测（D:\Anaconda\envs\yolo，含全部 requirements）：环境共 1.3GB
```

## 3. 安装 YOLO26（ultralytics）/ Install YOLO26

```bash
# YOLO26 从 ultralytics v8.4.0 开始支持 / YOLO26 shipped in ultralytics v8.4.0
python -m pip install "ultralytics>=8.4.0" opencv-python pyyaml matplotlib pandas
```

验证 / verify:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python scripts/verify_gpu.py
```

## 4. 下载数据集 / Download dataset

> 数据集查找渠道与磁盘策略（解压 >10GB 放 `D:\yolo_datasets\`）统一记录在
> [docs/dataset_sources.md](docs/dataset_sources.md)。
> All dataset sources & the disk policy are catalogued in dataset_sources.md.

MRL Eye Dataset（mrlEyes_2018_01，84,898 张眼部特写，睁眼/闭眼按文件名标注）：

```bash
# ~326 MB，放到 datasets/ 目录（已 gitignore）/ ~326 MB into datasets/ (gitignored)
curl -L -o datasets/mrlEyes_2018_01.zip http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip
```

官方页面 / official page: https://mrl.cs.vsb.cz/eyedataset.html

## 5. 数据集转换 / Dataset conversion

```bash
# 解压 + 生成 YOLO 格式（硬链接省磁盘）+ 写 configs/mrl_eye.yaml
# extract + build YOLO format (hardlinks) + write configs/mrl_eye.yaml
python scripts/prepare_mrl_dataset.py
```

## 6. YOLO26 冒烟测试 / Smoke test

```bash
# GPU 推理 + COCO8 上 1 epoch 训练验证 / GPU inference + 1-epoch COCO8 training
python scripts/smoke_test_yolo26.py
```

## 6b. 实时测试（有摄像头时）/ Live testing

```bash
# 实时检测：5类注视+睁闭眼，状态变化时终端播报一行（连续2帧确认防抖）
python scripts/live_detect.py                  # Q 退出 / S 存截图
# 眼球状态实时查看器（无校准无协议）：屏幕显示当前状态，终端播报变化
python scripts/test_gaze_live.py
# 历史版本（绿点协议实测）在 git 历史中：commit 00e86da 及之前
```
⚠️ 两脚本均以未镜像帧送模型（镜像会左右反转）；镜像只用于显示。

## 7. 训练 / Training

```bash
# 完整训练（50 epochs，自动 batch）/ full training (50 epochs, auto batch)
python scripts/train_eye.py

# 短跑测试 / short run
python scripts/train_eye.py --epochs 3
```

## 8. 绘制损失曲线 / Plot loss curves

```bash
# 读取 runs/eye_yolo26n/results.csv，保存 curves_loss.png / curves_metrics.png
python scripts/plot_training.py

# 指定其他结果目录 / other run dir
python scripts/plot_training.py runs/smoke/yolo26n_coco8
```

## 8b. 模型 B 流水线（Columbia 3类/5类）/ Model B pipeline

```bash
# 3类：解压 Columbia → MediaPipe 裁眼 → YOLO 格式（约20分钟）
python scripts/prepare_columbia_gaze_dataset.py

# 5类（上/中/下/左/右）：复用已有裁剪重标注（主轴优先+翻转互换增强，秒级）
python scripts/prepare_columbia_gaze5_dataset.py

# 训练（5类必须 --fliplr 0，否则左右标签被翻转增强污染）
python scripts/train_eye.py --data configs/columbia_gaze.yaml --name gaze_yolo26s --model yolo26s.pt --imgsz 128
python scripts/train_eye.py --data configs/columbia_gaze5.yaml --name gaze5_yolo26s --model yolo26s.pt --imgsz 128 --fliplr 0

# 评估（配置驱动，报告/图表按 config 命名）
python scripts/eval_eye_accuracy.py --config configs/columbia_gaze5.yaml --weights runs/gaze5_yolo26s/weights/best.pt
```

⚠️ 含左右方向的模型：①训练必须 --fliplr 0 ②推理输入必须未镜像帧（镜像会
左右反转）——live_detect/test_gaze_live 已按此实现。

```bash
# 读取 runs/eye_yolo26n/results.csv，保存 curves_loss.png / curves_metrics.png
python scripts/plot_training.py

# 指定其他结果目录 / other run dir
python scripts/plot_training.py runs/smoke/yolo26n_coco8
```

## 9. 准确率评估 / Accuracy evaluation

```bash
# MRL 验证集上：YOLO26 准确率/混淆矩阵 + MediaPipe 适用性探测，自动出报告和图
# accuracy + confusion + MediaPipe probe on the MRL val split, report & charts auto-generated
python scripts/eval_eye_accuracy.py
```

输出 / outputs: `docs/comparison/mrl_eval_report.md`、`confusion_matrix.png`、`accuracy_chart.png`

## 10. 录制注视方向素材 / Capture gaze-direction footage

**详细指南 / detailed guide: [docs/gaze_data_collection_guide.md](docs/gaze_data_collection_guide.md)**

```bash
# 默认 3 个方向 x 2 轮 x 20 秒 / 3 zones x 2 rounds x 20s
python scripts/collect_gaze_video.py
python scripts/collect_gaze_video.py --seconds 15 --rounds 3 --camera 1
```

操作要点：空格开始、q 结束；**头不动只动眼睛**；正常眨眼；尽量在真实使用光照下录。
Key points: SPACE to start, q to quit; keep head still (eyes only); natural blinks; record in real lighting.

## 11. 摄像头实时检测 / Live webcam detection

```bash
python scripts/live_detect.py                             # 摄像头 0，MediaPipe 定位 + 模型A/B 判定
python scripts/live_detect.py --camera 1 --conf 0.4       # 换摄像头 / 放宽阈值
```

窗口叠加眼框 + 结论面板（EYE/GAZE/ACTION/FPS）；**Q 退出、S 截图**（存 `runs/live_snapshots/`）。
当前接入 5 类模型 `models/gaze5_yolo26s.pt`（含左右，动作映射含 TURN_LEFT/RIGHT）。
Overlay shows eye boxes + verdict panel; **Q quit, S snapshot** (into `runs/live_snapshots/`).
Currently wired to the 5-class model `models/gaze5_yolo26s.pt` (left/right included).

## 12. 眼球状态实时查看 / Live eye-state viewer

```bash
python scripts/test_gaze_live.py                    # 持续输出当前眼球状态，Q 中止
python scripts/test_gaze_live.py --model models/gaze_yolo26s.pt   # 换 3 类主控版
python scripts/test_gaze_live.py --dump-crops 30    # 每30帧存眼部裁剪图（诊断用）
```

**无校准、无协议**：启动即持续输出当前眼球状态——屏幕面板显示"当前：直视/看左/…"
（中文大字 + 动作 + 置信度 + FPS），**终端只在状态稳定变化时打印一行**
（新方向需连续 2 帧确认）。历史版本（绿点协议实测）在 git 历史中（commit 00e86da）。
*No calibration, no protocol — just a continuous stream of the current eye state;
terminal prints only on stable state changes. The old green-dot benchmark lives in git history.*
Closed-eye frames are gated by Model A and not scored; the terminal prints one line
per **stable** direction change (2-frame confirmation).

## 13. 训练时监控 GPU / Monitor GPU during training

```bash
nvidia-smi -l 2        # Linux/Git Bash 每 2 秒刷新 / refresh every 2s
nvidia-smi             # 单次查看 / one shot
```

## 14. Git 提交规范 / Git commit convention

每个阶段一次提交，注释中英双语 / one commit per stage, bilingual messages:

```bash
git add -A
git commit -m "阶段X：中文说明 / Stage X: English description"
```
