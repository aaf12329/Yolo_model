# 数据集查找渠道记录 / Dataset Sources

> 记录本项目找数据用过的网站与方法，下次找数据直接从这里出发。
> Websites and methods used to find datasets in this project, for future reuse.

## 磁盘策略 / Disk policy（2026-09-28 约定）

- **解压后预计 >10GB 的数据集放到 D 盘**：统一放 `D:\yolo_datasets\<数据集名>\`，
  zip 压缩包也一并放 D 盘；C 盘 `datasets/` 只放 10GB 以下的小数据集。
  / Datasets that expand beyond 10 GB go to `D:\yolo_datasets\<name>\`
  (archives included); C: `datasets/` is only for small ones.

## 实际用过的源 / Sources actually used

| 数据集 | 用途 | 官方页 | 直链 | 状态 |
|---|---|---|---|---|
| MRL Eye Dataset (mrlEyes_2018_01) | 模型 A：睁/闭眼，84,898 张特写 | <https://mrl.cs.vsb.cz/eyedataset.html> | <http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip>（326MB） | ✅ 已用 |
| Columbia Gaze Data Set | 模型 B：注视方向，5,880 张（56 人 × 5 头姿 × 7 水平 × 3 垂直方向） | <https://cave.cs.columbia.edu/repository/ColumbiaGazeDataSet> | <https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze_data_set.zip>（2.2GB） | ✅ 已用 |
| **MPIIGaze**（DaRUS 官方归档） | 模型 B 扩充：**笔记本摄像头域**，15 人日常使用 213,659 张眼部图+3D 视线向量 | <https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3230> | <https://darus.uni-stuttgart.de/api/access/datafile/165887>（2.16GB，GET 可直下，HEAD 会 403） | ✅ 下载中 |
| **GazeCapture HF 镜像**（RafeiKAr） | 模型 B 扩充：**手机前置摄像头域**，35 人 19,990 张实拍+屏幕注视坐标（labels.csv 全量 34.7 万行/262 人，图片为其子集） | <https://huggingface.co/datasets/RafeiKAr/eye_tracking_gazecapture> | 逐文件或 `huggingface_hub.snapshot_download`（约1.5GB） | ✅ 下载中 |

注意事项 / caveats:

- Columbia 的**老地址** `www.cs.columbia.edu/CAVE/databases/...` 已 404，必须从新
  Repository 页进（纯前端页面，curl 抓不到正文，要用浏览器打开）。
  / The legacy CAVE URL is dead; use the new Repository page (a JS SPA — open in a
  browser, curl can't render it).
- Columbia Gaze 许可：**仅非商用**，使用需引用 Smith et al., UIST 2013
  （"Gaze Locking: Passive Eye Contact Detection for Human-Object Interaction"）。
  / Non-commercial only; cite the UIST 2013 paper.
- MRL 文件名第 5 字段 = 眼睛状态（0 睁 1 闭）；Columbia Gaze 文件名含注视角度
  （垂直 0/±10°）。两者标签都在文件名里，无需人工标注。

⚠️ 新源使用注意 / usage caveats:
- MPIIGaze：研究用途许可，需引用 Zhang et al. 2015；眼图 36×60 灰度小图+3D 视线向量，
  转 5 类需把向量转角度（符号约定同样必须目视/数值验证）
- GazeCapture：原始许可为研究用途（MIT 发布），HF 镜像为其子集；标签 x/y 是
  **多设备混合的屏幕像素坐标**（范围 40~984），换算方向类前需按设备/人归一化
- osama6/gazecapture（5.37GB part1.tar）：来源文档不明，未采用

## 通用搜索渠道 / General search channels

| 渠道 | 地址 | 特点 | 是否要账号 |
|---|---|---|---|
| Hugging Face | <https://huggingface.co/datasets> （API：`/api/datasets?search=关键词`） | AI 社区数据集，API 可编程搜索；gaze/eye 方向目前无可用的方向分类集 | 公开集不需要 |
| Roboflow Universe | <https://universe.roboflow.com> | 大量社区标注的 **YOLO 框格式**数据集，质量参差 | 要（免费 API key） |
| Kaggle Datasets | <https://www.kaggle.com/datasets> | 传统 ML 数据集大本营（MRL 镜像、疲劳驾驶等） | 要（免费） |
| 官方实验室页 | 各数据集论文里的"available at"链接 | 最权威、标签最规范；注意链接会搬家（CAVE 就搬过） | 视情况 |
| Wayback Machine | <https://web.archive.org> | 挖死链：老页面被存档，能找出原始直链再试探新服务器 | 不需要 |

## 搜索经验 / Search notes

- 关键词组合：`gaze direction`、`gaze zone`、`eye state open closed`、
  `drowsiness`、`looking direction`、`dataset download`。
- HF 的 API 搜索比网页好编程：`curl "https://huggingface.co/api/datasets?search=gaze&limit=20"`，
  返回 JSON 带 downloads 数可初筛热度。
- 找到数据先看三件事：**标签在文件名还是标注文件里、单张分辨率（决定解压体积）、
  许可证（商用/引用要求）**。
- 本项目仓库内相关文档：[gaze_data_collection_guide.md](gaze_data_collection_guide.md)
  （自采协议）、[yolo_tutorial.md](yolo_tutorial.md)（标签格式）。
