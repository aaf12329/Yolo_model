# 磁盘变更日志 / Disk Change Log

> 约定（2026-09-28）：**大规模磁盘操作先报备、后执行、再记录**。
> Convention (2026-09-28): large disk operations are announced in the conversation
> first, then executed, then recorded here.
>
> 需要报备+记录的操作 / What gets logged:
> - 单次写入 ≥ 1GB（下载/解压/复制/训练输出）/ any single write ≥ 1 GB
> - 项目目录之外的任何写入或移动（如 D 盘）/ anything written or moved outside this repo (e.g. D:)
> - ≥ 1GB 的删除 / any deletion ≥ 1 GB
> - conda 环境等系统级安装 / system-level installs like conda environments
>
> 已获授权的常规位置 / pre-approved locations:
> - `C:\Users\Guards\.conda\envs\yolo`（conda 环境）
> - 本仓库内 `datasets/`、`runs/`、`gaze_captures/`（<10GB 的数据与产物）
> - `D:\yolo_datasets\`（解压后 >10GB 的数据集及其压缩包，2026-09-28 约定新建）

---

## 2026-09-28

| 时间 | 盘 | 路径 | 大小 | 操作 | 用途 | 状态 |
|---|---|---|---|---|---|---|
| 19:07 | C | `C:\Users\Guards\.conda\envs\yolo` | **5.0GB** | 新建 | conda 环境：Python 3.10 + PyTorch 2.11 cu128 + ultralytics 8.4.164 + mediapipe 1.0.1 | ✅ |
| 19:21 | C | `yolo_model\datasets\mrlEyes_2018_01.zip` | 326MB | 下载 | 模型 A 数据（MRL Eye Dataset 官方直链） | ✅ |
| 19:47 | C | `yolo_model\datasets\mrlEyes_2018_01\` | 496MB | 解压 | 同上，84,898 张眼部特写 | ✅ |
| 19:52 | C | `yolo_model\datasets\mrl_eye_yolo\` | 166MB* | 生成 | YOLO 格式数据（图片为硬链接指向解压目录，实际增量≈标签文本） | ✅ |
| 20:10 | C | `yolo_model\runs\` | 29MB | 训练输出 | 模型 A 权重与曲线（best.pt 约 6MB） | ✅ |
| 20:57 | C | `yolo_model\datasets\columbia_gaze\columbia_gaze_data_set.zip` | **2.2GB** | 下载中 | 模型 B 数据（Columbia Gaze，官方直链） | 🔄 约 21:10 已到 547MB |
| 21:03 | D | `D:\yolo_datasets\` | 0（空目录） | 新建 | 预留：解压后 >10GB 的数据集统一放这里（当日约定） | ✅ |

\* du 把硬链接目标也计了一次；本目录真实增量是 labels 文本 + 少量非链接文件。

## 待执行 / Planned

| 盘 | 路径 | 预计大小 | 操作 | 条件 |
|---|---|---|---|---|
| 待定（D 优先） | `<D或C>:\…\columbia_gaze\` 解压目录 | **待定**（看 zip 内清单，可能 2.4GB~20GB+） | 解压 | 解压后 >10GB → `D:\yolo_datasets\columbia_gaze\`；否则留 C 盘 `datasets/`。执行前在对话中报备 |
| 待定 | 上述 zip | 2.2GB | 解压完成后移动到解压目录同盘存放 | 与解压同批报备 |

---
*由 ZCode 维护；每条变更执行前在对话中报备，执行后登记。/ Maintained by ZCode.*
