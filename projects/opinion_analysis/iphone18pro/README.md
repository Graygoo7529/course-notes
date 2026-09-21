# iPhone 18 Pro 国际新闻媒体研究

本项目对应《大数据与舆论分析导论》的数据收集与存储作业，核心问题是：**发布前后，国际媒体的关注强度与议题侧重如何变化，不同来源国家是否存在差异？**

已按课程作业规模完成：**布尔词设计与对照检索 → 原文核读 → 有限媒体案例分析 → 3 张可视化**。当前结果见 [课程作业报告](课程作业报告.md)。全时间窗采集保留为扩展，不作为本次完成条件。

## 阅读入口与当前结果

- [课程作业报告](课程作业报告.md)：当前主文，含关键词验证、媒体反响结论和三张图。
- [研究方案与执行计划](研究方案与执行计划.md)：已确认问题、检索逻辑及可选扩展；第 9 节为当前完成标准。
- [初步实验记录](初步实验记录.md)：真实请求结果、可复现证据与备用路线边界。
- [查询配置](config/study.json)：基础产品词、6 类议题、5 个来源范围、2 类敏感性查询。
- [人工编码模板](templates/coding.csv)：相关性、主次议题、评价主体、立场、引文与重复簇。

DOC 试采集两次返回 HTTP 429，未取得有效文章或时间序列。课程版采用 4 个固定 Web NGrams 分钟文件：4,774 条目录记录中得到 65 个不同候选 URL。选择 16 例检查，7 例有效、4 例排除、5 例正文不可得。当前结论属于小样本媒体案例分析，没有将分钟文件命中数写成每日热度。

当前输出目录为 `data/coursework/20260921T133415_219021Z/`，可直接查看 [检索注册表](data/coursework/20260921T133415_219021Z/query_registry.csv)、[候选表](data/coursework/20260921T133415_219021Z/documents.csv)、[核读表](data/coursework/20260921T133415_219021Z/reviews.csv)、[统计摘要](data/coursework/20260921T133415_219021Z/summary.json)。11 项单元测试与类型检查均通过。旧 [45 条 DOC 查询](data/runs/20260921T111729_882976Z/queries.csv) 是接口计划，不是本地布尔对照的执行结果。

## 课程版离线复现

在下述 `d2l` 环境配置完成后运行：

```powershell
python -X utf8 scripts/coursework_analysis.py
python -X utf8 scripts/plot_coursework.py data/coursework/20260921T133415_219021Z
```

第一条读取已经存档的四对原始文件并验证哈希，创建新的 UTC 运行目录。要绘制新运行结果，将第二条路径替换为第一条输出的目录；以上路径也可直接重画本次报告的图。`data/coursework_reviews.json` 保存逐篇核读结果，分析程序把该文件和词典、采样配置一并快照。绘图提供 PNG/SVG 及输入、输出哈希清单。

`summary.json` 的 `topic_same_frame` 才是图 2 的同分母词典对照（40 个英／西／法语 URL）；`pooled_counts` 中英文词全语言扫描与本地补词的范围不同，不应直接相减。`stage` 是文件所属阶段，核读表中的 `article_stage` 才按页面标注日期分组。

## 环境

在项目目录运行以下 PowerShell 命令：

```powershell
conda activate d2l
$env:PYTHONPATH = (Resolve-Path '.tools').Path
python -X utf8 scripts/gdelt_probe.py plan
```

采集只用 Python 标准库。当前机器已有项目本地 `.tools`，包含 `ty` 与 `matplotlib`。如需重新配置工具，可在 `d2l` 下运行：

```powershell
python -m pip install --target .tools --only-binary=:all: -r requirements-tools.txt
```

若 `conda activate/run` 因本机配置权限失败，可直接调用同一环境的解释器：

```powershell
& 'C:\Anaconda3\envs\d2l\python.exe' -X utf8 scripts/gdelt_probe.py plan
```

`-X utf8` 避免 Windows 终端输出多语种标题时出现 GBK 编码错误。不要把此类输出错误误记成下载失败。

## 复现实验

### 1. 离线生成 DOC 查询计划

```powershell
python -X utf8 scripts/gdelt_probe.py plan
```

每次建立一个新的 `data/runs/<UTC运行编号>/`。其中 `queries.csv` 是全部 45 条组合；`tasks.json` 是本次安排的 9 项探测任务。`plan` 不联网，`requests.csv` 只有表头属于正常结果。

### 2. 有限联网试检索

以下命令供接口恢复后执行。连续 429 时保留日志并停止；当前建议先使用已经下载的 NGrams 文件推进质量检查。

```powershell
python -X utf8 scripts/gdelt_probe.py pilot --max-tasks 1
python -X utf8 scripts/gdelt_probe.py one --query-id base__global --mode TimelineVolRaw --start 2026-08-26T00:00:00Z --end 2026-09-21T00:00:00Z
```

`--end` 是不包含的结束点。文章列表默认上限 25 条，只用于字段和检索质量检查；达到上限会标记 `limit_reached`。程序尚未实现递归切片补抓，也没有自动恢复队列。失败后可使用 `one` 重新运行原参数，结果另存到新的目录。

每个运行目录保存：

| 文件 | 用途 |
| --- | --- |
| `config_snapshot.json`、`tasks.json`、`runtime.json` | 配置、请求计划、解释器与脚本指纹；早期运行可能没有 `runtime.json` |
| `raw/*.body`、`raw/*.meta.json` | 原始响应，包括错误正文，以及校验哈希 |
| `pilot.sqlite` | 查询、请求、文章、查询命中关系和时间序列的关联表 |
| `queries.csv`、`requests.csv`、`articles.csv`、`article_hits.csv`、`timeline_points.csv` | 对应表的 UTF-8 BOM 导出 |
| `summary.json` | 实际尝试次数、状态、成功入库量与停止原因 |

数据库的文章数为 0，只有在请求成功且明确返回空列表时，才具有“该查询返回零条”的含义。HTTP 429、超时和无效载荷均属于缺失证据。

### 3. Web NGrams 分钟探针

```powershell
python -X utf8 scripts/ngrams_probe.py --file-stamp 20260909201600
```

这会下载两个公开压缩文件，各自最多读取 8 MiB，解压上限各为 64 MiB。已有成功归档可直接离线分析，无须重复下载：

```powershell
python -X utf8 scripts/summarize_probe.py data/ngrams_probes/20260921T080106_733713Z --plot
```

离线汇总会核对原始压缩文件的 SHA-256、重做短语匹配，并核对已有 `matches.json`；随后在新的 `data/probe_reports/<UTC运行编号>/` 内导出候选文章 CSV、标题/语言计数与质量诊断图。所有结果保留来源目录和匹配规则，历史原始文件不会覆盖。

该探针匹配规则是 `\biphone\s*18\s*pro\b`，并不等价于 DOC 的完整布尔查询。特别是国家过滤、议题组合和 Apple 域名排除尚未应用到这 61 条候选。文件中的 `lang` 不能代替媒体来源国，文件时间也不当作原网页发布时间。

### 4. 趋势图与验证

取得成功的全球 `TimelineVolRaw` 数据后，将对应运行目录传入：

```powershell
python -X utf8 scripts/plot_timeline.py data/runs/<成功运行编号>
```

无有效时间序列时程序退出，不生成虚假的全零趋势图。NGrams 单分钟诊断图同样不能替代全窗关注度、议题演变或国家差异图。

```powershell
python -X utf8 -m unittest discover -s tests -v
python -m ty check scripts tests --extra-search-path .tools --extra-search-path scripts --python 'C:\Anaconda3\envs\d2l\python.exe'
```

## 后续可选扩展

课程版已完成候选核读、四文件布尔对照和案例图表。若后续需要更强的阶段或国家结论，可增加各阶段可访问的有效原文，并独立复核编码；需要全窗趋势时再验证 DOC 的有效响应及覆盖。两条路线的总体和分母不同，不拼接为一条时间曲线。
