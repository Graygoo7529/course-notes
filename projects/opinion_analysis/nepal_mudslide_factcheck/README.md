# 2026 年中尼边境泥石流灾害谣言核查与信息真空实证

《大数据与舆论分析导论》第 10 讲（谣言传播与事实核查）课程实践项目。

## 核心研究脉络

本项目采用 **“7 步标准事实核查（独立一手证据链） + 轻量计算舆论实证（灾后 72h 信息真空与双模传播网络）”** 双层架构，拆解 2026 年 8 月 26 日中尼边境（朗塘里荣峰—吉隆/拉苏瓦）泥石流灾害后海外社媒流传的“西藏大坝溃坝致洪与 291 名外国游客失联”虚假因果链。

- **离线交互展示**：直接打开 [report/index.html](report/index.html)（单文件自包含，内嵌矢量图表、双模交互网络与 274 条原始记录检索表）。
- **正式核查报告**：阅读 [report/事实核查与舆论演化报告.md](report/事实核查与舆论演化报告.md)。

## 目录与文件职责

```text
nepal_mudslide_factcheck/
├── config.json                                 # 事件物理零点 t0、时间窗口、双层布尔词典与网络参数
├── pyproject.toml                              # ty 静态类型检查配置
├── evidence/
│   └── factcheck_knowledge.json                # 四维一手结构化证据库、原子声明矩阵、高程节点与 CRAAP 评分
├── notes/                                      # 项目方法论、数据口径与核查过程笔记
│   ├── 01-选题定位与事实核查流程.md
│   ├── 02-数据架构与预处理口径.md
│   ├── 03-量化指标与双模网络原理.md
│   ├── 04-可视化呈现与复现反思.md
│   ├── 05-一手核查证据详录.md
│   └── 附录-Grok_X数据采集任务单.md
├── scripts/                                    # 采集、清洗计算与报告渲染脚本（全部通过 ty check）
│   ├── collect_gdelt.py                        # GDELT DOC 2.0 优先 + Web NGrams 静态桶兜底采集
│   ├── analyze.py                              # 数据校验、V(t)/H(t) 指标、白盒槽位与双模网络构建
│   ├── render_report.py                        # 生成 6 组 300 dpi PNG/SVG、Markdown 报告与离线 HTML
│   ├── report.css                              # 离线交互报告样式
│   └── report.js                               # 双模 SVG 交互网络、一键聚光灯与证据检索脚本
├── data/                                       # 本地原始与中间数据目录（.gitignore 已忽略数据文件）
│   ├── raw/                                    # 放置 grok_x_posts.json、google_trends*.csv 与 gdelt 快照
│   └── processed/                              # analyze.py 与 collect_gdelt.py 生成的中间分析表
└── report/                                     # 最终交付产物
    ├── index.html                              # 单文件离线交互报告
    ├── 事实核查与舆论演化报告.md                 # 正式 Markdown 报告
    └── figures/                                # 6 组 300 dpi PNG/SVG 静态图 + 4 张 Google Trends 截图
```

## 复现流程

> 出于仓库体积与原始平台数据合规考虑，`data/raw/` 与 `data/processed/` 下的本地数据文件已在 `.gitignore` 中忽略。如需从零复算，请按 [notes/02-数据架构与预处理口径.md](notes/02-数据架构与预处理口径.md) 与 [notes/附录-Grok_X数据采集任务单.md](notes/附录-Grok_X数据采集任务单.md) 准备好 `data/raw/` 文件后，在项目虚拟环境中依次运行：

```powershell
python scripts/collect_gdelt.py --mode auto   # GDELT 双通道采集（DOC 熔断 + Web NGrams 静态桶）
python scripts/analyze.py                     # 清洗校验、V(t)/H 指标与双模社交网络构建
python scripts/render_report.py               # 渲染 6 组 PNG/SVG 图、Markdown 报告与离线 HTML
python -m ty check scripts                    # 静态类型检查
```
