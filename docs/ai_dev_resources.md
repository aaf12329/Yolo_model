# AI 开发常用网站与工具 / AI Dev Resources

> 面向刚入门的开发者。按用途分类，标 ✅ 表示本项目已实际用到；其余是同赛道常用替代，
> 需要时再查。相关文档：[dataset_sources.md](dataset_sources.md)（找数据）、
> [yolo_tutorial.md](yolo_tutorial.md)（原理入门）、[COMMANDS.md](../COMMANDS.md)（本仓库指令）。

---

## 1. 官方文档（必收藏）✅

| 网站 | 地址 | 用途 | 本项目使用处 |
|---|---|---|---|
| Ultralytics 文档 | docs.ultralytics.com | YOLO 系列训练/评估/导出全流程，`models/yolo26` 页是 YOLO26 权威说明 | ✅ 全部脚本 |
| PyTorch | pytorch.org / docs.pytorch.org | 深度学习框架；`get started` 页选版本装 GPU 版 | ✅ cu128 安装 |
| MediaPipe | developers.google.com/mediapipe | Google 端侧视觉方案（人脸 478 点等），Tasks API 手册 | ✅ 对比实验/将来自动打标 |
| OpenCV | docs.opencv.org | 摄像头/视频读写、图像处理（cv2） | ✅ 采集与评估脚本 |
| Hugging Face | huggingface.co | 模型库 + 数据集库 + 在线 demo（Spaces），AI 界的 GitHub | ✅ 找数据时搜过 |
| conda | docs.conda.io | 环境管理（建环境/装包/换 Python 版本互不干扰） | ✅ `yolo` 环境 |

## 2. 代码与论文

| 网站 | 地址 | 说明 |
|---|---|---|
| GitHub | github.com | 全世界的代码。看 issue 区能解决大半报错（如 ultralytics 仓库） |
| arXiv | arxiv.org | 论文预印本；配 ar5iv.labs.arxiv.org 可看网页版 |
| Hugging Face Papers | huggingface.co/papers | 论文速递 + 社区讨论（Papers with Code 已于 2025 年归档，用这个替代） |
| Google Scholar | scholar.google.com | 搜论文引用情况 |

## 3. 数据集站点 ✅

详见 [dataset_sources.md](dataset_sources.md)（MRL、Columbia Gaze 直链与许可）。通用入口：

| 网站 | 地址 | 特点 |
|---|---|---|
| Kaggle Datasets | kaggle.com/datasets | 传统 ML 数据集大本营，要免费账号 |
| Roboflow Universe | universe.roboflow.com | 大量现成 **YOLO 框格式**数据集，要免费 API key |
| Google Dataset Search | datasetsearch.research.google.com | 按关键词搜全网数据集 |

## 4. 本机开发工具（你电脑上已有的）✅

| 工具 | 说明 | 本项目使用处 |
|---|---|---|
| **Anaconda/conda**（D:\Anaconda） | Python 环境管理器；`conda create -n yolo python=3.10` 就靠它 | ✅ |
| **VS Code**（D:\Microsoft VS Code） | 编辑器；装 Python 扩展后可选中 conda 环境、断点调试、直接跑 Jupyter cell | 推荐 |
| **ZCode** | AI 编程助手（本仓库的环境搭建、训练、文档都是它协作完成的） | ✅ |
| **Git** | 版本管理；本项目每阶段双语 commit | ✅ |
| nvidia-smi | 显卡状态命令行（显存/利用率），Git Bash 直接可用 | ✅ 训练监控 |
| Netron | netron.app 网页版：把 .pt/.onnx 拖进去可视化网络结构 | 推荐 |
| VLC / PotPlayer | 播放采集视频（mp4v 编码 Windows 自带播放器可能放不了） | ✅ 采集指南里提过 |

> 新手常见误区：**pip 装的 GPU 版 PyTorch 自带 CUDA 运行时，不需要再单独安装
> CUDA Toolkit**（装了也不冲突，但不是必需）。需要本机编译自定义算子时才装。

## 5. 在线算力（学习期白嫖 GPU）

| 平台 | 地址 | 免费额度 | 备注 |
|---|---|---|---|
| Google Colab | colab.research.google.com | T4 显卡，每天数小时 | 需科学上网；ultralytics 官方笔记本可直接跑 |
| Kaggle Notebooks | kaggle.com/code | 每周约 30h GPU（P100/T4×2） | 要账号；数据集一键挂载 |
| 百度飞桨 AI Studio | aistudio.baidu.com | 每日免费算力点（V100 等） | 国内直连，中文文档 |
| AutoDL | autodl.com | 按小时租卡（4090 约 ¥1.5/h） | 国内，学生党常用，按量计费非免费 |
| Hugging Face Spaces | huggingface.co/spaces | 免费 CPU demo 部署 | 训练好的模型放上去给队友演示 |

## 6. 国内访问加速（实测有效）

| 资源 | 方案 |
|---|---|
| Hugging Face 下载 | 镜像站 `hf-mirror.com`：`set HF_ENDPOINT=https://hf-mirror.com` 后 huggingface-cli/pip 自动走镜像 |
| pip 安装慢 | 清华镜像：`pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple`（torch 的 cu128 版仍需官方源，见 requirements.txt） |
| GitHub 慢 | 加速器/镜像代理，或用 gitee 镜像仓库；clone 大仓库用 `--depth 1` |
| 论文/文档 | arXiv 可用镜像，官方文档一般直连可开 |

## 7. 标注工具（手工标注备用）

本项目用 MediaPipe **自动打标**，不需要手工画框；但下面这些迟早用得上：

| 工具 | 地址 | 特点 |
|---|---|---|
| Label Studio | labelstud.io | 万能标注（图/文/音视频），Web 界面 |
| CVAT | cvat.ai | 工业级视频标注，支持插值 |
| X-AnyLabeling | github.com/CVHub520/X-AnyLabeling | 桌面版，内置 YOLO 模型**半自动预标注** |
| Roboflow Annotate | roboflow.com | 在线标注+自动转 YOLO 格式+托管 |

## 8. 模型部署路线（本项目未来的"出货"方式）

| 技术 | 定位 | 一句话 |
|---|---|---|
| TorchScript / ONNX | 通用中间格式 | `yolo export format=onnx`，脱离 Python 也能推理 |
| ONNX Runtime | CPU/跨平台推理 | 没有独显的机器上跑 ONNX 的标准解 |
| TensorRT | NVIDIA GPU 极致加速 | 有 N 卡时的性能上限 |
| OpenVINO | Intel CPU/核显加速 | 老笔记本友好 |
| LiteRT (TFLite) | 手机/嵌入式 | YOLO26 官方重点支持端到端导出 |

> 学习路径建议：先用 ultralytics 默认的 .pt 跑通业务（本项目现状）→
> 部署阶段再导出 ONNX（`yolo export`）→ 有性能瓶颈再上 TensorRT。

## 9. 学习资源（中文友好优先）

| 资源 | 地址 | 说明 |
|---|---|---|
| 李沐《动手学深度学习》 | zh.d2l.ai | 免费中文教材 + B 站配套视频，理论入门首选 |
| 3Blue1Brown 神经网络系列 | B 站搜索搬运 | 可视化直觉，4 集看懂神经网络/反向传播/梯度下降 |
| 吴恩达 Machine Learning | Coursera / B 站搬运 | 经典开山课，直观不深究数学 |
| PyTorch 官方 60min 入门 | pytorch.org/tutorials | 动手写第一个网络 |
| Ultralytics 官方 Colab | docs.ultralytics.com/quickstart | 零环境直接在 Colab 跑 YOLO |
| B 站"YOLO 论文精读"（李沐团队） | B 站搜"李沐 YOLO" | 逐行讲 v1 论文，理解设计动机 |

---

*由 ZCode 整理（2026-09-28）；✅ = 本项目已实际使用。链接长期有效性以官网为准，
死链处理经验见 [dataset_sources.md](dataset_sources.md) 的 Wayback Machine 一节。*
