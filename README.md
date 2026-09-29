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
| 8 | 模型 B 训练（n 版 93.62% → 容量对比后升级 s 版 **95.62%**） | ✅ |
| 9 | 模型 B 个人化微调（等手机采集素材 → 自动打标 → 精调） | ⏸ 等素材 |
| 10 | 有摄像头机器：实时工具落地（`live_detect.py` + `test_gaze_live.py`）+ CPU 环境 + 双机同步 | 🔶 通路已验证（CPU 15FPS、真实人脸截图），**真人实测待补** |

**阶段 8 结果 / Stage 8 results**（详见 [docs/training/summary_gaze.md](docs/training/summary_gaze.md)）：
总判定准确率 **93.62%**（look_center 96.86% / look_down 96.00% / look_up 88.00%），
mAP50-95=**0.9828**；权重已入库 `models/gaze_yolo26s.pt`（n→s 容量对比后升级，训练原始输出在 runs/）。
图：`docs/comparison/columbia_gaze_confusion_matrix.png`、`columbia_gaze_accuracy_chart.png`。
关键结论：down 几乎不会误判为 up（0 例）——**前进/后退不会互相窜**；
主要误差是 up→center（+10° 上看幅度小），对控制语义无害且将由个人化微调修复。

## 三、环境信息

| 项 | 值 |
|---|---|
| conda env | `yolo`，Python 3.10。训练机：`C:\Users\Guards\.conda\envs\yolo`；本机（无独显）：`D:\Anaconda\envs\yolo` |
| PyTorch | 2.11.0。训练机 +cu128（RTX 50 系要求）；无独显机器装 **CPU 构建**（同一版本号，API 一致、只是慢） |
| ultralytics | 8.4.164（YOLO26 从 v8.4.0 起支持） |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB（Blackwell，sm_120，CUDA 12.8） |
| MediaPipe | 仅用于对比实验与后续自动打标（CPU 运行） |
| 其他 | opencv-python、matplotlib、pandas、numpy |

> 训练机没有摄像头，实验全部基于数据集离线完成；有摄像头的机器上可直接跑实时脚本
> （见"十一、本机实时工具"）。两处依赖锁定同一套版本，代码在两边通用。

## 四、项目架构

```
yolo_model/
├─ README.md                        本文件：架构、进度、指标、规范（每阶段更新）
├─ COMMANDS.md                      全部指令速查（按阶段整理，含英文注释）
├─ requirements.txt                 依赖清单
├─ .gitignore                       数据集/视频不入库；交付权重在 models/ 随 git 分发
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
│  ├─ auto_label_phone_videos.py    ⑨ 手机视频自动打标（抽帧+裁眼+QC虹膜排序校验，
│  │                                   已用合成视频端到端验证；输出 datasets/phone_gaze_yolo/）
│  ├─ live_detect.py                ⑩ 摄像头实时检测：MediaPipe 定位 + 双模型判定，
│  │                                   叠加眼框与结论面板（Q 退出 / S 截图）
│  └─ test_gaze_live.py             ⑪ 注视方向实测：屏幕绿点目标协议 → 准确率 + 混淆矩
│                                      阵报告（闭眼帧由模型A门控、不计分）
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
│  ├─ results_explained.md          结果解读手册：每个模型/每张图是什么、原理
│  ├─ training_journey.md           训练全程复盘：数据集→训练→评估，问题→根因→解决→预防
│  ├─ training/                     模型A训练档案：summary.md、损失/指标曲线、results.csv
│  └─ comparison/                   对比实验档案：评估报告、混淆矩阵图、对比柱状图
│
├─ datasets/                        （不入库）MRL 原始 zip、解压目录、YOLO 格式数据集
│
├─ gaze_captures/                   （不入库）脚本自采的方向素材视频
│
└─ runs/                            （不入库）训练输出与曲线；交付权重已复制到 models/
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

- 权重：`models/eye_yolo26n.pt`（~5MB，随 git 分发；训练原始输出在 runs/，可复现）
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
- **5 类扩展版**（`models/gaze5_yolo26s.pt`，80.90%）：加入 look_left/look_right
  （主轴优先标注+翻转互换增强），center 因"注视空间一个点"结构性偏弱（42%），
  逐图解读见 [docs/results_explained.md](docs/results_explained.md)

## 七、快速开始

```bash
conda activate yolo
cd <repo path on this machine>   # e.g. C:\Users\AAF12\Desktop\Yolo_model

python data/download_datasets.py --dataset all --extract  # ⓪ 下载数据（断点续传+校验）
python scripts/verify_gpu.py             # ① 验证 GPU 环境
python scripts/prepare_mrl_dataset.py    # ② 数据集转换（已执行过，可重复）
python scripts/train_eye.py              # ③ 训练模型A（复现）
python scripts/plot_training.py          # ④ 画损失曲线
python scripts/eval_eye_accuracy.py      # ⑤ 准确率评估+对比图
python predict.py --image 某张人脸照片.jpg  # ⑥ 一行命令跑预测（权重已入库）
# python scripts/collect_gaze_video.py   # ⑦（有摄像头时）自采方向素材
python scripts/live_detect.py            # ⑩ 摄像头实时检测（Q 退出 / S 截图）
python scripts/test_gaze_live.py         # ⑪ 注视方向实测（跟随屏幕绿点看上/中/下）
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

**提交前检查清单（2026-09-29 追加约定，每次 commit 前逐项过）**：

1. 本次改动涉及的**功能/脚本/数据**是否已在 README 的架构树与进度表里反映；
2. 是否引入了**新的文件类型或目录** → `.gitignore` 需要增补或加白名单例外；
3. 新增了**指令或脚本** → `COMMANDS.md` 是否同步；
4. 触发了**磁盘策略条款**（≥1GB 写入/删除、跨盘、环境安装）→ `log.md` 是否登记；
5. 有了**新的实测结果或图表** → `docs/`（summary/results_explained 等）是否归档

## 十、已知限制

- 无摄像头的机器上无法实测实时脚本（离线接口与链路已验证）；在有摄像头的机器上，
  `live_detect.py` 已跑通无窗口自检并用真实人脸截图验证，`test_gaze_live.py` 的统计与
  协议通路已验证，**真人实测准确率待补**
- MRL 文件名标签含少量噪声（文献报告一致率约 90~97%），故准确率为下限
- 模型 A 在"睁眼但视线朝下"的画面上偶判为 closed_eye（MRL 闭眼类混有垂目样本的域差异），
  个人化微调可修复；predict.py 实测快照时已观察到并如实记录
- MRL 以非亚裔面孔为主；模型 B 自采数据将补足本群体特征
- `datasets/`、`runs/`、视频均不入库；权重与数据集可由脚本完整复现

## 十一、本机实时工具（有摄像头时）

| 脚本 | 用途 | 输出 |
|---|---|---|
| `scripts/live_detect.py` | 摄像头实时检测：MediaPipe 定位 478 点 → 裁双眼 → 模型A 判睁/闭 + 模型B 判上/中/下 → 画面叠加眼框、结论面板、FPS | 窗口；按 `S` 存到 `runs/live_snapshots/` |
| `scripts/test_gaze_live.py` | **注视方向实测**：屏幕出现绿点（上/中/下），按提示看 → 逐帧记录模型输出 → 准确率 + 混淆矩阵 | `docs/gaze_test/`：`frames.csv`、`report.md`、`gaze_confusion_matrix.png` |

```bash
conda activate yolo
python scripts/live_detect.py                             # Q 退出，S 截图
python scripts/live_detect.py --camera 1 --conf 0.4       # 换摄像头 / 放宽阈值
python scripts/test_gaze_live.py                          # 3 方向 × 2 轮 ≈ 45 秒，Q 中止
python scripts/test_gaze_live.py --rounds 3 --look-sec 8  # 更长的测试
```

**实测（本机 CPU，无独显）**：

- 全管线（MediaPipe 定位 + 双眼 2 模型）**66 ms/帧 ≈ 15 FPS**，CPU 上做验证够用
- `live_detect.py`：无窗口自检 8 帧跑通；用真实人脸截图验证——检出人脸、2 个眼框、双模型出结论
- `test_gaze_live.py`：统计逻辑经合成预测校验（准确率/混淆矩阵计数正确），完整协议通路跑通；
  **真人实测准确率待补**（坐到摄像头前跑一次即自动生成报告）

**门控设计**：闭眼帧的"注视方向"没有物理意义，两个脚本里模型 B 的输出都由模型 A 门控——
`test_gaze_live.py` 把闭眼帧不计分并单独统计；`live_detect.py` 面板照常显示，但闭眼时不可采信。

**终端播报（边沿触发）**：`test_gaze_live.py` **只在方向发生变化时打印一行**，不逐帧刷屏：

```
[方向变更] look_center → look_up  conf=0.80  动作=FORWARD   @ 12.3s
```

- 新方向需**连续 2 帧**确认才播报（单帧抖动忽略）；同一方向不重复打印；闭眼/无人脸帧不参与判断
- 播报内容含映射动作（`look_up→FORWARD` / `look_center→STOP` / `look_down→BACKWARD`），可直接观察控制语义
- 注意：**3 类主控版**（`gaze_yolo26s.pt`，test_gaze_live.py 用的）类别只有上/中/下；
  **5 类实验版**（`gaze5_yolo26s.pt`，live_detect.py 已接入）才含左右

**代码约定**：`eye_boxes/eye_crop`、`top_pred`、`combine` 三组函数与 `predict.py` **同款**，
两处需同步修改（各脚本头部注释已标注）。

## 十二、模型改进路线（2026-09-29 首次真人实测后制定）

**实测结论**：`test_gaze_live.py` 管线本身跑通——418 帧、约 11 FPS、全程检出人脸、双模型都
在出结果。但 **93% 的帧被模型 A 门控判为"闭眼"跳过，模型 B 没有测成**；过门控的 20 帧里
19 帧预测 look_up。另有一个输入质量事实：本摄像头裁出的眼部特写仅约 **77×28 像素**且模糊
（对比 MRL 83×83 / Columbia 高清裁剪，域差很大）。

改进按下面顺序做，每步都带验收方式：

1. **门控诊断（下一步，零代码改动）**：
   `python scripts/test_gaze_live.py --no-gate --dump-crops 30`
   - `--no-gate`：闭眼帧也计分，拿到模型 B 的全量预测（报告会标注"门控已关闭"）
   - `--dump-crops 30`：每 30 帧存一对眼部裁剪到 `docs/gaze_test/crops/`（gitignore，隐私）
   - 判读：若 frames.csv 里 eye_pred 大面积 closed、而裁剪图里眼睛明显睁开 →
     模型 A 跨人域差实锤，走第 3 步；若裁剪图里眼睛确实闭着或小得看不清 → 先做第 2 步
2. **输入质量与几何**：摄像头抬到与眼睛平齐；必要时提高摄像头分辨率——
   目标是把眼部裁剪从 ~77×28 提到 ≥120×60；检查对焦与光照
3. **个人化微调（阶段 9，治本）**：用**测试者本人**的素材精调模型 A 与模型 B——
   - 采集：`scripts/collect_gaze_video.py`（本机已可用）或手机拍摄（`phone_videos/`，
     指南见 docs/gaze_data_collection_guide.md）
   - 打标：`scripts/auto_label_phone_videos.py`（虹膜排序 QC）
   - 训练：`scripts/train_eye.py` 流程；**翻转增强必须标签互换**（5 类版教训：
     左看图翻转后标签必须换成右看，否则左右互相污染）
   - 数据量：每人每方向 ≥200 张裁剪起步，睁眼/闭眼类保持均衡
4. **门控策略演进**：a) 模型 A 用本人数据精调后继续当门控；b) 改用 EAR 保守门控
   （MediaPipe 现成 EAR，低于 0.08 才算"确定闭眼"，无需个人校准）；c) 两者取与。
   用 frames.csv 的 skip 比例与实测准确率对比后定。
5. **回归验收（每次模型变动必做）**：`test_gaze_live.py` 协议法——总准确率不降、
   混淆矩阵不出现 up↔down / left↔right 互窜、门控 skip 比例回落到 <30%。

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
| 8 | Model B training (n 93.62% → upgraded to s **95.62%** after capacity test) | ✅ |
| 9 | Model B personalization (awaiting phone footage → auto-label → fine-tune) | ⏸ waiting |
| 10 | Camera machine: live tooling (`live_detect.py` + `test_gaze_live.py`) + CPU env + two-machine sync | 🔶 paths verified (CPU 15 FPS, real face snapshots) — **real-person run pending** |

**Stage 8 results** (see [docs/training/summary_gaze.md](docs/training/summary_gaze.md)):
overall accuracy **93.62%** (look_center 96.86% / look_down 96.00% / look_up 88.00%),
mAP50-95=**0.9828**; weights `models/gaze_yolo26s.pt` (upgraded n→s after capacity test).
Key finding: down is never misread as up (0 cases) — **forward/backward never swap**;
main error is up→center (small +10° amplitude), harmless for control semantics.

## 3. Environment

| Item | Value |
|---|---|
| conda env | `yolo`, Python 3.10. Training machine: `C:\Users\Guards\.conda\envs\yolo`; camera machine (no GPU): `D:\Anaconda\envs\yolo` |
| PyTorch | 2.11.0. Training machine: +cu128 (required by RTX 50-series); camera machine: **CPU build** — same version number and API, just slower |
| ultralytics | 8.4.164 (YOLO26 shipped in v8.4.0) |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell, sm_120, CUDA 12.8) |
| MediaPipe | for the comparison probe and upcoming auto-labeling (CPU) |
| Others | opencv-python, matplotlib, pandas, numpy |

> The training machine has **no camera** — all experiments are dataset-based and offline.
> On a camera-equipped machine the live scripts run directly (see "11. On-Machine Live Tools").
> Both machines pin the same dependency versions, so the code is portable between them.

## 4. Repository Layout

```
yolo_model/
├─ README.md                        this file: architecture, progress, metrics (updated per stage)
├─ COMMANDS.md                      command reference organized by stage
├─ requirements.txt                 dependencies
├─ .gitignore                       datasets / videos stay out of git; deliverable weights in models/ ARE committed
│
├─ configs/
│  └─ mrl_eye.yaml                  Model-A YOLO data config (paths, 2 class names)
│
├─ models/
│  ├─ face_landmarker.task          MediaPipe 478-pt face model (copied from
│  │                                   EyeWheelchairProject; reused for probing & auto-labeling)
│  ├─ eye_yolo26n.pt                Model-A deliverable weights (2-class, 94.33%)
│  ├─ gaze_yolo26s.pt               Model-B deliverable weights (3-class, 95.62%, primary)
│  └─ gaze5_yolo26s.pt              Model-B extended weights (5-class w/ left-right, 80.90%)
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
│  ├─ auto_label_phone_videos.py    ⑨ phone-video auto-labeling (frame sampling + eye crops
│  │                                    + iris-ordering QC; e2e verified on synthetic videos)
│  ├─ live_detect.py                ⑩ live webcam detection: MediaPipe locate + both models,
│  │                                    overlay eye boxes & verdict panel (Q quit / S snapshot)
│  └─ test_gaze_live.py             ⑪ live gaze benchmark: on-screen target-dot protocol →
│                                       accuracy + confusion-matrix report (closed-eye frames
│                                       gated by Model A, not scored)
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
│  ├─ results_explained.md          results walkthrough: what each model & figure means
│  ├─ training_journey.md           full journey retrospective: datasets→training→eval, problems→fixes
│  ├─ training/                     Model-A record: summary.md, loss/metric curves, results.csv
│  └─ comparison/                   benchmark record: report, confusion matrix, accuracy bars
│
├─ datasets/                        (not committed) MRL zip, extracted tree, YOLO-format dataset
├─ gaze_captures/                   (not committed) self-captured gaze footage
└─ runs/                            (not committed) training output & curves; deliverable weights copied to models/
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

- Weights: `models/eye_yolo26n.pt` (~5 MB, committed; raw training output stays in runs/, reproducible)
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
- **5-class extended** (`models/gaze5_yolo26s.pt`, 80.90%): adds look_left/look_right
  (dominant-axis labeling + flip-swap augmentation); center is structurally weak (42%) —
  figure-by-figure walkthrough in [docs/results_explained.md](docs/results_explained.md)

## 7. Quick Start

```bash
conda activate yolo
cd <repo path on this machine>   # e.g. C:\Users\AAF12\Desktop\Yolo_model

python data/download_datasets.py --dataset all --extract  # ⓪ download datasets
python scripts/verify_gpu.py             # ① verify GPU environment
python scripts/prepare_mrl_dataset.py    # ② dataset conversion (idempotent)
python scripts/train_eye.py              # ③ train Model A (reproduce)
python scripts/plot_training.py          # ④ plot loss curves
python scripts/eval_eye_accuracy.py      # ⑤ accuracy eval + comparison charts
python predict.py --image some_face.jpg  # ⑥ one-command prediction (weights committed)
# python scripts/collect_gaze_video.py   # ⑦ (camera required) capture gaze footage
python scripts/live_detect.py            # ⑩ live webcam detection (Q quit / S snapshot)
python scripts/test_gaze_live.py         # ⑪ live gaze benchmark (follow the green dot)
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

**Pre-commit checklist (agreed 2026-09-29; run through every commit)**:

1. Do the **features/scripts/data** in this change appear in the README trees & progress table?
2. Any **new file types or directories** → update `.gitignore` (add or whitelist);
3. Any new **commands or scripts** → sync `COMMANDS.md`;
4. Did any **disk-policy clause** trigger (≥1GB writes/deletes, cross-drive, env installs) → log in `log.md`;
5. Any new **measured results or figures** → archived under `docs/` (summary/results_explained etc.)

## 10. Known Limitations

- No camera on the training machine, so live scripts cannot be tested there (offline interfaces
  fully verified); on the camera-equipped machine `live_detect.py` passed a headless smoke test and
  was verified on real face snapshots, while `test_gaze_live.py`'s statistics and protocol path are
  verified — **real-person accuracy still to be measured**
- MRL filename labels carry minor noise (~90–97% agreement reported); accuracy is a lower bound
- Model A occasionally says closed_eye on open-but-downcast eyes (domain gap: MRL closed class
  includes downcast samples); observed & documented in the predict.py snapshot test — fixed by personalization
- MRL is mostly non-Asian faces; Model B's self-collected data will cover our user population
- `datasets/`, `runs/`, and videos are not committed; weights and datasets are fully
  reproducible from the scripts

## 11. On-Machine Live Tools (camera required)

| Script | Purpose | Output |
|---|---|---|
| `scripts/live_detect.py` | Live webcam detection: MediaPipe locates 478 points → crops both eyes → Model A (open/closed) + Model B (up/center/down) → overlays eye boxes, verdict panel and FPS | Window; press `S` to save into `runs/live_snapshots/` |
| `scripts/test_gaze_live.py` | **Live gaze benchmark**: a green target dot appears (up/center/down), you follow it → per-frame predictions → accuracy + confusion matrix | `docs/gaze_test/`: `frames.csv`, `report.md`, `gaze_confusion_matrix.png` |

```bash
conda activate yolo
python scripts/live_detect.py                             # Q quit, S snapshot
python scripts/live_detect.py --camera 1 --conf 0.4       # other camera / looser threshold
python scripts/test_gaze_live.py                          # 3 directions × 2 rounds ≈ 45 s, Q aborts
python scripts/test_gaze_live.py --rounds 3 --look-sec 8  # longer run
```

**Measured (CPU-only machine, no discrete GPU)**:

- Full pipeline (MediaPipe locate + two models per eye) runs at **66 ms/frame ≈ 15 FPS** — plenty for verification
- `live_detect.py`: headless 8-frame smoke test passed; verified on real face snapshots (face found,
  2 eye boxes, both models produced verdicts)
- `test_gaze_live.py`: statistics verified with synthetic predictions (accuracy/confusion counts correct),
  full protocol path runs; **real-person accuracy still to be measured** — run it in front of the camera
  and the report is generated automatically

**Gating design**: gaze direction is meaningless while the eyes are closed, so Model B's output is gated by
Model A in both scripts — `test_gaze_live.py` excludes closed-eye frames from scoring and reports the count
separately; `live_detect.py` still shows the value but it must not be trusted when the eyes are closed.

**Terminal announcements (edge-triggered)**: `test_gaze_live.py` prints **one line per direction change**,
never per frame:

```
[方向变更] look_center → look_up  conf=0.80  动作=FORWARD   @ 12.3s
```

- A new direction must hold for **2 consecutive frames** to be announced (single-frame flicker ignored);
  the same direction is never repeated; closed-eye / no-face frames do not participate
- Each line includes the mapped action (`look_up→FORWARD` / `look_center→STOP` / `look_down→BACKWARD`)
- Note: the **3-class primary model** (`gaze_yolo26s.pt`, used by test_gaze_live.py) only has up/center/down;
  the **5-class experimental model** (`gaze5_yolo26s.pt`, wired into live_detect.py) adds left/right

**Code convention**: `eye_boxes/eye_crop`, `top_pred` and `combine` mirror `predict.py` **exactly** —
keep the copies in sync (noted in each script's header).

## 12. Model Improvement Roadmap (after the first real-person run, 2026-09-29)

**What the first live run showed**: the `test_gaze_live.py` pipeline itself works — 418 frames, ~11 FPS,
a face detected throughout, both models producing output. But **93% of frames were gated out as "closed"
by Model A**, so Model B was never really tested; of the 20 frames that passed, 19 predicted look_up.
One input-quality fact: the webcam's eye crops are only about **77×28 pixels** and blurry (a large domain
gap vs. the MRL 83×83 / Columbia high-res crops).

Do the steps in order; each has its own acceptance check:

1. **Gate diagnosis (next step, zero code changes)**:
   `python scripts/test_gaze_live.py --no-gate --dump-crops 30`
   - `--no-gate`: closed-eye frames are scored too, giving Model B's full predictions (report says "gate off")
   - `--dump-crops 30`: saves an eye-crop pair every 30 frames into `docs/gaze_test/crops/` (gitignored, privacy)
   - Reading the results: if frames.csv shows eye_pred mostly closed while the crops clearly show open eyes →
     Model A's cross-person domain gap is confirmed, go to step 3; if the crops really show closed or
     unreadably tiny eyes → do step 2 first
2. **Input quality and geometry**: raise the camera to eye level; raise capture resolution if needed —
   target eye crops ≥ 120×60 (from ~77×28); check focus and lighting
3. **Personalization fine-tune (stage 9, the real fix)**: fine-tune both models on the **test subject's own** data —
   - Capture: `scripts/collect_gaze_video.py` (works on this machine) or phone footage (`phone_videos/`,
     guide in docs/gaze_data_collection_guide.md)
   - Label: `scripts/auto_label_phone_videos.py` (iris-ordering QC)
   - Train: the `scripts/train_eye.py` flow; **flip augmentation must swap labels** (5-class lesson:
     a flipped look-left image must be re-labeled look-right, otherwise left/right poison each other)
   - Volume: ≥ 200 crops per person per direction to start, open/closed classes balanced
4. **Gate strategy evolution**: a) keep Model A as gate after fine-tuning on the user's data;
   b) switch to a conservative EAR gate (MediaPipe already gives EAR; below 0.08 counts as "definitely
   closed", no personal calibration needed); c) AND of both. Decide from skip ratios and accuracy in frames.csv.
5. **Regression acceptance (mandatory for every model change)**: the `test_gaze_live.py` protocol —
   overall accuracy must not drop, no up↔down / left↔right swapping in the confusion matrix,
   and the gate's skip ratio back under 30%.
