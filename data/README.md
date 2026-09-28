# 数据来源与下载 / Data Sources & Downloads

> 本文件夹集中管理数据集的**来源、许可、下载指令与下载脚本**。
> 通用搜索渠道见 [docs/dataset_sources.md](../docs/dataset_sources.md)；
> 磁盘策略（解压 >10GB 放 `D:\yolo_datasets\`、大动作报备、查 C 盘）见
> [log.md](../log.md)。/ This folder centralizes dataset sources, licenses,
> download commands and the downloader script.

---

## 1. 模型 A：MRL Eye Dataset（睁/闭眼）✅ 已用

| 项 | 值 |
|---|---|
| 官方页 | <https://mrl.cs.vsb.cz/eyedataset.html> |
| 直链 | <http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip> |
| 体积 | zip 326MB → 解压 ~496MB（C 盘 `datasets/`） |
| 内容 | 84,898 张眼部特写灰度图（约 83×83），83 人 |
| 标签 | 文件名第 5 字段：`0`=睁眼，`1`=闭眼（共 8 段，已目视验证） |
| 许可 | 公开学术数据集，使用请注明来源（MRL, VSB-TU Ostrava） |

## 2. 模型 B：Columbia Gaze Data Set（注视方向）✅ 已用

| 项 | 值 |
|---|---|
| 官方页 | <https://cave.cs.columbia.edu/repository/ColumbiaGazeDataSet> |
| 直链 | <https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze_data_set.zip> |
| 体积 | zip 2.2GB → 解压 2.45GB（<10GB，按约定留 C 盘 `datasets/`） |
| 内容 | 5,880 张 JPG（5184×3456）：56 人 × 5 头姿(0/±15/±30°) × 7 水平(0/±5/±10/±15°) × 3 垂直(0/±10°) |
| 标签 | 文件名共 5 段：`受试者_距离_头姿P_垂直V_水平H`，如 `0001_2m_-15P_-10V_-10H.jpg` |
| 许可 | **仅非商用**，引用 Smith et al., UIST 2013（"Gaze Locking"） |

⚠️ **符号约定（已抽样目视验证，反直觉）**：垂直 V 字段 **`-10V` = 往上看，
`10V`（正号省略）= 往下看，`0V` = 平视**——该数据集把"水平线以下"记为正。
水平 H 的符号未逐一验证，模型 B 初版只用垂直方向。

⚠️ **地址会搬家**：老地址 `www.cs.columbia.edu/CAVE/...` 已 404；新站是纯前端
页面（curl 抓不到正文），要用浏览器打开 Repository 页找直链。

## 3. 模型 B 补充：自采手机视频（个人化微调）⏸ 采集中

- 拍摄说明：`phone_videos/转发给拍摄的人.txt`（可直接转发微信群）
- 详细协议与归档规则：[docs/gaze_data_collection_guide.md](../docs/gaze_data_collection_guide.md)
- 归档位置：`phone_videos/一人一个文件夹/文件名带上中下字.mp4`

## 4. 下载指令 / Download commands

### 方式一：一键脚本（推荐，带断点续传 + 体积校验 + C 盘检查）

```bash
conda activate yolo
cd C:\Users\Guards\Desktop\yolo_model

python data/download_datasets.py                     # 下载全部（MRL + Columbia Gaze）
python data/download_datasets.py --dataset mrl       # 只下模型A数据
python data/download_datasets.py --dataset columbia --extract
#                                                     # 顺便解压（跳过 macOS 垃圾文件）
```

脚本行为：下载前打印 C 盘剩余空间（制度要求）；已下载部分支持断点续传
（服务器不支持 Range 时自动整段重下）；完成后按登记体积校验。

### 方式二：手动 curl

```bash
# 模型 A / Model A（~326MB）
curl -L -o datasets/mrlEyes_2018_01.zip http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip

# 模型 B / Model B（~2.2GB）
curl -L -o datasets/columbia_gaze/columbia_gaze_data_set.zip \
  https://cave.cs.columbia.edu/old/databases/columbia_gaze/columbia_gaze_data_set.zip
```

## 5. 下载之后 / After download

```bash
python scripts/prepare_mrl_dataset.py            # 模型A：转 YOLO 格式
python scripts/prepare_columbia_gaze_dataset.py  # 模型B：MediaPipe 裁眼 → 3类 YOLO 格式（~20分钟）
python scripts/train_eye.py --data configs/columbia_gaze.yaml --name gaze_yolo26n  # 模型B训练
```
