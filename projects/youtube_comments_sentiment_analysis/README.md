# YouTube 评论情感探索

基于《大数据与舆论分析导论》的[技术路线](../../references/情感探索技术路线.md)，完成从原始评论到可视化解释的快速实践。**直接打开 [HTML 报告](report/index.html)**：图表、交互脚本和浏览数据均内嵌，无需启动服务器、联网或运行 Notebook。

使用预训练模型推理，不训练／微调神经网络，不进行人工标注或正式性能评估。BERTopic 运行无监督降维、聚类与主题表示。按照最终技术路线包含抽样方面分析；点赞数和回复数量不作为权重或模型输入。

## 实际流程

1. 读取 1,196 条记录，保留主评论／回复身份、原文和稳定 ID；自动筛选出 1,082 条英文记录。
2. Twitter-RoBERTa 输出整体正／中／负概率：正面 135、中性 335、负面 612。负面表达占 56.6%，不能解释为对中国电动车的反对率。
3. MiniLM → 五维 UMAP → HDBSCAN → c-TF-IDF → KeyBERTInspired，得到 9 个主题及 341 条未归类记录。主题标题依据关键词与代表原句概括，存在交叉。
4. 使用同一批向量生成固定二维 UMAP 坐标，支持主题／情感配色切换、悬停与点击看原文。
5. 固定随机抽取 200 条英文记录，切句后通过 BART-MNLI 检测六个对象／方面；对检测到且未标记上下文待定的组合调用 DeBERTa ABSA，按评论—方面聚合。
6. 从缓存表生成 7 组 PNG／SVG、交互地图、案例和离线 HTML。完整结果以报告与运行记录为准。

## 环境与运行

PowerShell 中进入本项目目录后运行：

```powershell
conda activate d2l
python scripts/run.py --offline
python -m ty check
```

当前四个模型已下载，`--offline` 使用本地权重。阶段成功后自动缓存，重复运行会跳过推理；图表每次重新生成。只修改展示时运行 `python scripts/render_report.py`。

本次实际使用 Python 3.13、CPU 版 PyTorch，在 CPU 完成全部推理。电脑具备 RTX 4060，但本作业不依赖 CUDA，也未为此调整显卡驱动。环境路径为 `C:\Anaconda3\envs\d2l\python.exe`；无法激活时可用绝对路径调用，并以 `python -m ty check --python C:\Anaconda3\envs\d2l\python.exe` 明确检查环境。

重新安装主要依赖：

```powershell
conda activate d2l
python -m pip install -r requirements.txt
```

`requirements.txt` 固定本次实际安装的主要运行包；运行版本写入 `data/processed/run_config.json`。报告浏览器检查额外使用 Playwright 和本机 Edge，读报告无需安装它们。

## 从原始文件重新运行

```powershell
New-Item -ItemType Directory -Force data/raw
Copy-Item ../../references/dataset_youtube-comments.csv data/raw/
python scripts/run.py
```

下载脚本使用 `config.json` 的代理 `http://127.0.0.1:7897`，并按本次已授权设置 `verify_ssl=false`。这些设置只作用于该下载进程，不修改系统环境。模型版本固定在 `model_lock.json`，下载清单记录每个文件的 SHA256；推理仅加载本地模型。

`config.json` 集中保存种子、阈值、候选方面、截断上限和批量大小。改变数据或配置时，先将 `data/processed` 改名留档，再完整重跑。需要单独重做某个阶段可使用 `python scripts/analyze.py --stage topics --force` 等命令；随后重做依赖该阶段的后续阶段，避免混用缓存。当前主题标题和案例选择针对本次数据，更换数据后应一并更新 `topic_names.json` 与 `case_selection.json`。

## 文件组织

| 路径 | 内容 |
| --- | --- |
| `scripts/analyze.py` | 预处理、整体情感、主题、提及检测、ABSA、汇总与基本一致性检查 |
| `scripts/render_report.py` | 从结果表绘图并生成报告，不重新调用模型 |
| `config.json` / `model_lock.json` | 参数与固定模型版本 |
| `topic_names.json` / `case_selection.json` | 展示标题与真实案例选择；不作为训练标签 |
| `data/raw/` | 原始 CSV，本地保留、Git 忽略 |
| `data/processed/` | 结果表、句向量、抽样 ID、各阶段状态，本地保留、Git 忽略 |
| `model/` | 四个模型与 BERTopic 拟合产物，Git 忽略 |
| `report/index.html` | 自包含离线报告，作为交付文件保留 |
| `report/semantic_map.html` | 可单独打开的交互语义地图 |
| `report/figures/` | 7 组 300 dpi PNG 和 SVG |

核心结果表为 `comments_analysis.csv`、`topics_summary.csv`、`mention_scores.csv`、`aspect_evidence.csv`、`comment_aspects.csv`、`aspect_summary.csv`。所有关联都使用 `comment_id`，不按文本或主题名称连接。详细预处理口径见 [预处理说明](预处理说明.md)。

数据来自单个视频评论区，英文筛选、回复上下文缺失、讽刺与跨领域模型误判都会影响结果。本次只展示模型观察到的表达与主题分布，不推断评论者身份或总体公众态度。
