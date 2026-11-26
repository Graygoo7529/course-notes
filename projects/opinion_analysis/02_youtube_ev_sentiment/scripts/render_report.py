"""Render PNG/SVG figures and an offline HTML report using only saved result tables."""
import base64
import html
import json
import textwrap

from common import CONFIG, DATA, REPORT, ROOT, setup, write_json

setup()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

LABELS = ["positive", "neutral", "negative"]
CN = {"positive": "正面", "neutral": "中性", "negative": "负面", "mixed": "混合", "failed": "失败"}
COLORS = {"positive": "#238B8E", "neutral": "#9CA3AF", "negative": "#D76A4A", "mixed": "#8B6BB1"}
PALETTE = ["#4B6EAF", "#B18E44", "#687D43", "#8D6CA8", "#567F91", "#BD8CA1", "#6F9D85", "#9B805F", "#5F819F", "#857BA6"]
FIGURES = REPORT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.family": ["Microsoft YaHei", "Segoe UI Emoji", "DejaVu Sans"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.edgecolor": "#DFE4E8", "text.color": "#253440",
                     "axes.labelcolor": "#536471", "xtick.color": "#536471", "ytick.color": "#253440",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def picture(name: str, caption: str) -> str:
    encoded = base64.b64encode((FIGURES / f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def stacked(frame: pd.DataFrame, name_col: str, count_col: str, title: str, name: str, mixed: bool = False) -> None:
    labels = LABELS + (["mixed"] if mixed else [])
    fig, ax = plt.subplots(figsize=(11.4, max(3.8, len(frame) * 0.57 + 1.6)))
    y = np.arange(len(frame))
    left = np.zeros(len(frame))
    for label in labels:
        values = pd.to_numeric(frame[f"{label}_ratio"], errors="coerce").fillna(0).to_numpy() * 100
        ax.barh(y, values, left=left, color=COLORS[label], label=CN[label], height=0.62)
        for i, value in enumerate(values):
            if value >= 9:
                ax.text(left[i] + value / 2, i, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=10)
        left += values
    for i, count in enumerate(frame[count_col]):
        ax.text(102, i, f"n={count}" if count else "无记录", va="center", fontsize=10)
    ax.set(yticks=y, yticklabels=frame[name_col], xlim=(0, 113), xlabel="各组内情感占比（%）", xticks=[0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", pad=45, fontweight="bold")
    ax.legend(ncol=len(labels), loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    fig.tight_layout()
    save(fig, name)


def build_figures(comments, topics, aspects, cases, run) -> None:
    valid = comments[comments.status == "ok"]
    counts = valid.sentiment_label.value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(11.4, 3.8))
    ax.barh([CN[x] for x in LABELS], counts.values, color=[COLORS[x] for x in LABELS], height=0.58)
    for i, count in enumerate(counts):
        ax.text(count + 10, i, f"{count} 条 · {count / len(valid):.1%}", va="center")
    ax.invert_yaxis()
    ax.set(xlim=(0, counts.max() * 1.3), xlabel="评论记录数（含主评论与回复）")
    ax.set_title(f"整体极性 · 英文分析子集 n={len(valid):,}", loc="left", pad=18, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_overall")

    fig, ax = plt.subplots(figsize=(11.4, max(4, len(topics) * 0.7 + 1.4)))
    ax.barh(range(len(topics)), topics.comment_count, color=[topic_color(t) for t in topics.topic_id], height=0.55)
    for i, row in enumerate(topics.itertuples()):
        words = " · ".join(row.keywords.split(" | ")[:4])
        ax.text(row.comment_count + 6, i, f"{row.comment_count}  |  {words}", va="center", fontsize=9)
    ax.set(yticks=range(len(topics)), yticklabels=topics.topic_name, xlabel="评论记录数", xlim=(0, topics.comment_count.max() * 2.15))
    ax.invert_yaxis()
    ax.set_title("BERTopic 主题概览 · 关键词为语义重排结果", loc="left", pad=18, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_topics")
    stacked(topics, "topic_name", "valid_sentiment_count", "同一话题内，表达如何分布？", "03_topic_sentiment")

    included = comments[comments.included]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for topic in topics.itertuples():
        group = included[included.topic_id.astype(str).isin([str(topic.topic_id), str(float(topic.topic_id))])]
        axes[0].scatter(group.umap_x.astype(float), group.umap_y.astype(float), c=topic_color(topic.topic_id), s=12, alpha=0.7, label=topic.topic_name)
    for label in LABELS:
        group = included[included.sentiment_label == label]
        axes[1].scatter(group.umap_x.astype(float), group.umap_y.astype(float), c=COLORS[label], s=12, alpha=0.65, label=CN[label])
    for ax, title in zip(axes, ["按主题配色", "按整体情感配色"]):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set(xticks=[], yticks=[], xlabel="同一组二维 UMAP 展示坐标")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "04_semantic_map")
    stacked(aspects, "aspect_name", "mentioned_comment_count", f"对象／方面情感 · 固定随机样本 n={run['stages']['mentions']['sample_size']}", "05_aspects", True)

    fig, axes = plt.subplots(len(cases), 1, figsize=(12, len(cases) * 2.25))
    for ax, case in zip(np.atleast_1d(axes), cases):
        ax.axis("off")
        ax.text(0, 0.98, f"{case['comment_id']} · {case['kind']} · 整体{CN[case['sentiment_label']]} · {case['topic_name']}", va="top", fontweight="bold", transform=ax.transAxes)
        ax.text(0, 0.73, textwrap.fill(case["text_raw"][:390], width=106) + (" …" if len(case["text_raw"]) > 390 else ""), va="top", fontsize=9.5, transform=ax.transAxes)
        note = textwrap.fill(case["note"][:100], width=53) + ("…" if len(case["note"]) > 100 else "")
        ax.text(0, 0.32, note, va="top", fontsize=8.5, color="#536471", transform=ax.transAxes)
        ax.text(0, 0.01, case["aspect_text"] or "不在方面样本内，或没有可用方面结果。", va="bottom", fontsize=9, color="#536471", transform=ax.transAxes)
    fig.tight_layout(h_pad=2)
    save(fig, "06_cases")

    type_rows = []
    for kind, group in valid.groupby("type"):
        type_rows.append({"name": "主评论" if kind == "comment" else "回复", "n": len(group),
                          **{f"{label}_ratio": (group.sentiment_label == label).mean() for label in LABELS}})
    stacked(pd.DataFrame(type_rows), "name", "n", "保留评论／回复身份 · 按记录计数", "07_record_types")


def topic_color(topic_id) -> str:
    number = int(topic_id)
    return "#BBC2C9" if number == -1 else PALETTE[number % len(PALETTE)]


def choose_cases(comments, aggregated, evidence) -> list[dict]:
    valid = comments[comments.status == "ok"].copy()
    aspect_names = {x["id"]: x["name"] for x in CONFIG["aspects"]}
    selections = []
    # Optional editorial choices are real record IDs, never gold labels.
    selection_path = ROOT / "case_selection.json"
    if selection_path.exists():
        selections = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        for comment_id, group in aggregated.groupby("comment_id"):
            labels = set(group.sentiment_label)
            if "mixed" in labels or {"positive", "negative"}.issubset(labels):
                selections.append({"id": comment_id, "kind": "不同方面的褒贬"})
                break
        for label, title in [("positive", "清晰的正面表达"), ("negative", "负面表达的对象")]:
            eligible = valid[(valid.sentiment_label == label) & valid.comment_id.isin(aggregated.comment_id)]
            if len(eligible):
                row = eligible.sort_values("confidence", ascending=False).iloc[0]
                selections.append({"id": row.comment_id, "kind": title})
        replies = valid[(valid.type == "reply") & (valid.text_raw.str.len() < 90)]
        if len(replies):
            selections.append({"id": replies.sort_values("confidence").iloc[0].comment_id, "kind": "短回复与上下文"})
    cases = []
    for choice in selections[:5]:
        row = valid[valid.comment_id == choice["id"]].iloc[0].to_dict()
        group = aggregated[aggregated.comment_id == choice["id"]]
        row.update(kind=choice["kind"], note=choice.get("note", ""),
                   aspect_text="；".join(f"{aspect_names[x.aspect_id]}：{CN[x.sentiment_label]}" for x in group.itertuples()),
                   evidence=evidence[evidence.comment_id == choice["id"]].to_dict("records"))
        cases.append(row)
    write_json(DATA / "cases.json", cases)
    return cases


def map_html(comments, topics) -> str:
    frame = comments[comments.included].copy()
    fig = go.Figure()
    for mode in ["topic", "sentiment"]:
        groups = [(row.topic_name, frame[frame.topic_name == row.topic_name], topic_color(row.topic_id)) for row in topics.itertuples()] if mode == "topic" else [(CN[label], frame[frame.sentiment_label == label], COLORS[label]) for label in LABELS]
        for name, group, color in groups:
            hover = []
            for row in group.itertuples():
                quote = "<br>".join(html.escape(part) for part in textwrap.wrap(row.text_raw[:400], 65))
                hover.append(f"{row.comment_id} · {'主评论' if row.type == 'comment' else '回复'}<br>{html.escape(row.topic_name)} · {CN.get(row.sentiment_label, '失败')}<br>{quote}")
            fig.add_trace(go.Scatter(x=group.umap_x.astype(float), y=group.umap_y.astype(float), mode="markers",
                                     marker={"color": color, "size": 7, "opacity": 0.72}, name=name,
                                     text=hover, customdata=group.comment_id.tolist(), hovertemplate="%{text}<extra></extra>", visible=mode == "topic"))
    total = len(fig.data)
    buttons = [{"label": label, "method": "update", "args": [{"visible": [i < len(topics) if mode == "topic" else i >= len(topics) for i in range(total)]}]} for label, mode in [("主题配色", "topic"), ("情感配色", "sentiment")]]
    fig.update_layout(template="plotly_white", height=650, margin={"l": 10, "r": 10, "t": 70, "b": 15},
                      font={"family": "Microsoft YaHei, sans-serif", "color": "#253440"},
                      updatemenus=[{"type": "buttons", "direction": "right", "x": 0, "y": 1.1, "buttons": buttons}],
                      legend={"orientation": "h", "y": -0.04}, xaxis={"visible": False}, yaxis={"visible": False},
                      hoverlabel={"bgcolor": "white", "font_size": 12})
    standalone = pio.to_html(fig, include_plotlyjs=True, full_html=True)
    (REPORT / "semantic_map.html").write_text(standalone, encoding="utf-8")
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="semantic-map", config={"responsive": True, "displaylogo": False})


def main() -> None:
    comments, topics, aspects = read("comments_analysis"), read("topics_summary"), read("aspect_summary")
    evidence, aggregated = read("aspect_evidence"), read("comment_aspects")
    run = json.loads((DATA / "run_config.json").read_text(encoding="utf-8"))
    run["sample_ids"] = json.loads((DATA / "sample_ids.json").read_text(encoding="utf-8"))
    write_json(REPORT / "run_config.json", run)
    cases = choose_cases(comments, aggregated, evidence)
    build_figures(comments, topics, aspects, cases, run)
    semantic = map_html(comments, topics)
    valid = comments[comments.status == "ok"]
    n = len(valid)
    neg = int((valid.sentiment_label == "negative").sum())
    prep, top, mention, absa = [run["stages"][key] for key in ["prepare", "topics", "mentions", "aspects"]]
    aspect_lookup = aspects.set_index("aspect_id")
    competition = aspect_lookup.loc["us_competition"]
    technology = aspect_lookup.loc["cn_technology"]
    counts = valid.sentiment_label.value_counts()
    top_names = "、".join(topics.loc[topics.topic_id != -1, "topic_name"].head(3))
    intro = f"纳入 {n:,} 条英文评论与回复。模型将其中 {neg} 条（{neg/n:.1%}）判为负面表达；主题聚类得到 {top['topic_count']} 个主题，另有 {top['unassigned']} 条未归类。负面表达涉及不同对象，不能直接解释成对中国电动车的反对率。"
    stats = "".join(f'<div><strong>{value}</strong><span>{label}</span></div>' for value, label in [(f"{len(comments):,}", "原始记录"), (f"{n:,}", "英文分析记录"), (str(top['topic_count']), "自动发现主题"), (str(mention['sample_size']), "方面分析随机样本")])
    exclusions = {"cjk_or_mixed": "汉字／混合语言", "short_uncertain": "过短、语言不确定", "other_language": "检测为其他语言", "language_uncertain": "英文检测分数不足", "empty": "去除提及后为空"}
    exclusion_rows = "".join(f"<tr><td>{label}</td><td>{prep['exclusion_reasons'].get(key, 0)}</td></tr>" for key, label in exclusions.items())
    topic_cards = "".join(f'<article class="topic"><span class="eyebrow">TOPIC {row["topic_id"]:02d} · {row["comment_count"]} 条</span><h3>{html.escape(row["topic_name"])}</h3><p>{html.escape(row["keywords"])}</p><small>代表记录：{row["representative_ids"].replace("|", " · ")}</small></article>' for row in topics.to_dict("records"))
    case_cards = []
    for case in cases:
        ev = "".join(f'<li><b>{html.escape(x["aspect"])}</b> · {CN.get(x["sentiment_label"], "失败")}<br><q>{html.escape(x["evidence"])}</q><br><small>提及分数 {x["mention_score"]:.2f} · 情感最大概率 {float(x["confidence"]):.2f}</small></li>' for x in case["evidence"])
        case_cards.append(f'<article class="case"><div class="eyebrow">{case["comment_id"]} · {case["kind"]} · {"主评论" if case["type"] == "comment" else "回复"}</div><h3>整体{CN[case["sentiment_label"]]} <span class="muted">／ {html.escape(case["topic_name"])}</span></h3><blockquote>{html.escape(case["text_raw"])}</blockquote><p>{html.escape(case["note"])}</p><details><summary>查看方面证据（{len(case["evidence"])} 条）</summary><ul>{ev or "该记录不在方面样本内，或没有可用方面结果；这不等于中性。"}</ul></details></article>')
    model_rows = "".join(f'<tr><td>{html.escape(key)}</td><td><a href="https://huggingface.co/{item["repo"]}">{html.escape(item["repo"])}</a></td><td><code>{item["revision"][:12]}</code></td></tr>' for key, item in run["models"].items())
    timings = "".join(f'<tr><td>{name}</td><td>{stage["status"]}</td><td>{stage.get("seconds", 0):.1f} s</td></tr>' for name, stage in run["stages"].items())
    aspect_defs = "".join(f'<li><b>{a["name"]}</b><br><small>{a["hypothesis"]}</small></li>' for a in CONFIG["aspects"])
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>中国电动车视频评论 · 情感探索</title><style>{(ROOT / 'scripts/report.css').read_text(encoding='utf-8')}</style><script>{get_plotlyjs()}</script></head><body>
<header><a class="brand" href="#">MYCOURSE <span> / 舆论分析</span></a><nav><a href="#method">技术路线</a><a href="#overall">整体</a><a href="#topics">主题</a><a href="#map">语义地图</a><a href="#aspects">方面</a><a href="#cases">案例</a></nav></header>
<main><section class="hero"><div class="eyebrow">大数据与舆论分析 · 快速实践 · 2026.10</div><h1>中国电动车，<br>在评论区被怎样谈论？</h1><p class="lede">从整体情感，到讨论主题，再到评价对象。<br>一次基于预训练模型与 BERTopic 的评论探索。</p><p class="intro">{intro}</p><div class="stats">{stats}</div><a class="button" href="#overall">阅读结果 ↓</a><span class="hero-note">全流程已实际运行 · 无神经网络训练或微调</span></section>
<section id="data"><div class="section-heading"><span>01 / 数据</span><h2>保留原文，让每个结果都能追溯。</h2></div><div class="two"><div><p>来源为 YouTube 视频 <b>Americans Crave Low-Cost Chinese EVs</b> 下的评论 CSV，共 699 条主评论、497 条回复。以记录为分析单位，不把评论条数当作人数，不使用点赞或回复数量加权。</p><p>给每条记录分配稳定 ID，保留原文、情感输入、主题输入。清理空白与不可见字符；情感输入保留否定、标点、表情，用户名与链接替换为占位符；主题输入去除用户名与链接。</p><p><b>语言筛选：</b>langdetect 固定种子 42；英文候选分数 ≥ 0.8、至少三个拉丁词，且不含汉字。保留 {prep['english_count']} 条，排除 {prep['excluded_count']} 条。这是自动筛选口径，短英文也可能被排除。</p><p>两组重复原文保留并标记。纳入的英文子集包含 {prep['included_types']['comment']} 条主评论和 {prep['included_types']['reply']} 条回复。</p></div><div class="panel"><h3>未进入英文主分析的记录</h3><table><thead><tr><th>原因</th><th>条数</th></tr></thead><tbody>{exclusion_rows}</tbody></table><p class="muted">不翻译、不凭空补全回复上下文。没有时间戳和父评论 ID，因此不绘制时间趋势或回复网络。</p></div></div></section>
<section id="method"><div class="section-heading"><span>02 / 方法</span><h2>四种模型，各自回答一个问题。</h2></div><div class="flow"><div>CSV → 轻量清理 → 英文子集</div><div class="branches"><div>RoBERTa<br><small>整体正／中／负</small></div><div>MiniLM → UMAP → HDBSCAN<br><small>BERTopic 主题与关键词</small></div></div><div>按 comment_id 合并 → 主题 × 情感 → 语义地图</div><div>固定随机抽样 → 切句 → BART-MNLI 提及检测 → DeBERTa ABSA</div><div>按评论 × 方面聚合 → 方面图与证据案例</div></div><div class="method-grid"><article><span class="eyebrow">A / 整体极性</span><h3>Twitter-RoBERTa</h3><p>Tokenizer → Transformer → 分类头 → softmax。取最大概率的类别；辅助分数 <code>s = p(positive) − p(negative)</code> 用于排序，不代表真实心理强度。</p><small>max_length=512 · batch=8 · eval + inference_mode · 本次截断 {run['stages']['sentiment']['truncated']} 条</small></article><article><span class="eyebrow">B / 主题发现</span><h3>BERTopic</h3><p>MiniLM 产生 384 维语义向量；UMAP 降为 5 维后由 HDBSCAN 聚类；c-TF-IDF 提取候选关键词，KeyBERTInspired 用语义相似度重排。</p><small>邻居数 15 · 最小簇 20 · min_samples=5 · 不强制主题数 · 不改变 Transformer 权重</small></article><article><span class="eyebrow">C / 提及检测</span><h3>BART-MNLI</h3><p>原句是 premise，对象／方面描述是假设。每个候选独立计算 <code>p(entailment) / [p(entailment)+p(contradiction)]</code>，达到 {CONFIG['mention_threshold']:.2f} 才进入方面情感。</p><small>多标签 · 最多 200 条随机评论 · 切句保留 ID · 无法明确指代时暂缓归属</small></article><article><span class="eyebrow">D / 方面情感</span><h3>DeBERTa ABSA</h3><p>输入原句与英文方面的文本对，输出正／中／负。按“评论 × 方面”聚合：正负并存为混合，正或负与中性并存保留有方向的类别。</p><small>max_length=256 · 未检测到 ≠ 中性 · 混合为聚合标签，不是模型新增类别</small></article></div><p class="callout">主题与情感分别推理，按 ID 汇合。主题内的负面比例描述表达的组成，不等于对主题对象的反对率。</p></section>
<section id="overall"><div class="section-heading"><span>03 / 整体</span><h2>负面表达占 {neg/n:.1%}，但评价对象并不单一。</h2></div><p>在 {n:,} 条有效英文记录中，正面 {counts.get('positive',0)} 条、中性 {counts.get('neutral',0)} 条、负面 {neg} 条。模型最大概率用于描述输出把握，不作为准确率。</p>{picture('01_overall', f'图 1 · 全部英文子集，n={n:,}；每条记录计一次。')}{picture('07_record_types', '补充图 · 主评论与回复分开统计；各组百分比使用自己的有效记录数。')}</section>
<section id="topics"><div class="section-heading"><span>04 / 内容</span><h2>情感之外，先看大家在谈什么。</h2></div><p>主要主题包括 {html.escape(top_names)}。主题标题由助手依据本次关键词和代表评论概括；这些主题存在内容交叉；保留 topic_id，不改变原始聚类归属。未归类记录占 {top['unassigned']/n:.1%}，单独展示。</p>{picture('02_topics', f'图 2 · {top["topic_count"]} 个聚类主题及未归类记录；关键词分数为语义相似度，不是出现次数。')}<div class="topic-grid">{topic_cards}</div>{picture('03_topic_sentiment', '图 3 · 每个主题内部的整体情感占比；右侧 n 为该主题有效情感记录数。')}</section>
<section id="map"><div class="section-heading"><span>05 / 语义空间</span><h2>同一张地图，两种阅读方式。</h2></div><p>切换主题／情感颜色，悬停查看原文；单击某个点，在下方打开完整记录。两个视图共用固定二维坐标，聚类仍来自五维表示。二维图上的距离不能当作严格的语义度量。</p><div class="map-box">{semantic}</div><div id="selected-comment" class="callout">点击地图上的一条评论，查看完整文本与模型结果。</div><details><summary>查看可导出的静态语义地图</summary>{picture('04_semantic_map', '图 4 · 同一组二维 UMAP 坐标，分别按主题与整体情感配色。')}</details></section>
<section id="aspects"><div class="section-heading"><span>06 / 对象与方面</span><h2>把“说得负面”和“在批评谁”分开。</h2></div><p>在固定随机抽取的 {mention['sample_size']} 条英文评论中，切出 {mention['sentence_count']} 个句子，对六类对象／方面分别做提及检测。{mention['detected_pairs']} 个句子—方面组合达到阈值；其中 {mention['context_deferred']} 个触发启发式指代标记而暂缓归属。最终得到 {absa['comment_aspect_records']} 条唯一评论—方面记录，覆盖 {absa['matched_comments']} 条评论；另有 {mention['sample_size'] - absa['matched_comments']} 条没有可用方面结果。</p><p>美国车企竞争力是自动检测记录最多的方面（{competition["mentioned_comment_count"]} 条），其中负面 {competition["negative_count"]} 条、混合 {competition["mixed_count"]} 条；中国电动车技术方面有 {technology["mentioned_comment_count"]} 条，其中正面 {technology["positive_count"]} 条。这些数字描述本次模型输出，受到提及误扩展影响。</p><details><summary>查看六个对象／方面候选</summary><ul class="aspect-defs">{aspect_defs}</ul></details>{picture('05_aspects', f'图 5 · 随机子样本 n={mention["sample_size"]}；每条横条的分母是该方面被检测到且推理成功的独立评论数。')}<p class="callout">同一评论可提到多个方面，所以各横条 n 的总和可以超过样本量。未检测到、对象待定、推理失败都不计为中性。政策方面的情感也不自动等于支持／反对立场。固定检查的前 8 个句子没有出现全部候选普遍被选中的情况，保留启动阈值 0.60；但发现笼统赞美被扩展为价格／技术，详见下方案例。</p></section>
<section id="cases"><div class="section-heading"><span>07 / 证据</span><h2>回到原句，看见标签压缩掉的细节。</h2></div><p>以下为实际结果中选取的说明性案例，不是人工标注测试集。模型可能误解讽刺、指代或多对象比较；保留原文便于辨认这些边界。</p><div class="case-grid">{''.join(case_cards)}</div><details><summary>查看案例汇总图</summary>{picture('06_cases', '图 6 · 案例图中较长原文节选，完整文本见上方案例卡与明细表。')}</details></section>
<section id="explore"><div class="section-heading"><span>08 / 浏览记录</span><h2>搜索原句，核对每个输出。</h2></div><div class="filters"><input id="search" placeholder="搜索文本或评论 ID" aria-label="搜索评论"><select id="type-filter" aria-label="记录类型"><option value="">全部类型</option><option value="comment">主评论</option><option value="reply">回复</option></select><select id="sentiment-filter" aria-label="情感"><option value="">全部情感</option><option value="positive">正面</option><option value="neutral">中性</option><option value="negative">负面</option></select><select id="topic-filter" aria-label="主题"><option value="">全部主题</option></select></div><p id="search-count" class="muted"></p><div id="records"></div><button id="more">显示更多</button></section>
<section id="reproduce"><div class="section-heading"><span>09 / 运行记录</span><h2>保存结果，也保存产生结果的方式。</h2></div><p>这是一个单视频评论区的快速探索。自动英文筛选、短回复缺少上下文、模型领域差异和无人工评估共同限制了结论。本次没有训练或微调神经网络；BERTopic 只拟合降维和聚类，未报告准确率、F1 或显著性检验。</p><div class="two"><div><h3>模型版本</h3><div class="table-scroll"><table><thead><tr><th>阶段</th><th>检查点</th><th>revision</th></tr></thead><tbody>{model_rows}</tbody></table></div></div><div><h3>实际运行</h3><table><thead><tr><th>阶段</th><th>状态</th><th>用时</th></tr></thead><tbody>{timings}</tbody></table></div></div><p>随机种子 42；运行设备 {run['device']}。整体情感失败 {run['stages']['sentiment']['failed']} 条，NLI 失败 {mention['failed']} 对，ABSA 失败 {absa['failed']} 对。MiniLM 按自身上限 {top['embedding_max_length']} token 编码，截断 {top['embedding_truncated']} 条；NLI／ABSA 截断分别为 {mention['truncated_pairs']}／{absa['truncated']} 对。数量、ID 唯一性与情感概率和检查已通过。</p><details><summary>复现命令与文件说明</summary><pre>conda activate d2l
python scripts/download_models.py
python scripts/analyze.py --stage all
python scripts/render_report.py
python -m ty check</pre><p>已有阶段自动复用缓存。只重画图时运行最后一条 Python 脚本。更换数据／配置需先将 data/processed 另存后重跑。原始与结果 CSV 位于 data/，模型在 model/，均忽略；report/ 保存可交付 HTML 和 300 dpi PNG／SVG。</p><p>详细参数、包版本、抽样 ID 与实际模型 revision：<a href="run_config.json">运行记录 JSON</a>（另保存在 data/processed/）。原始 CSV SHA256：<code>{run['input_sha256']}</code>。</p></details><p class="sources">方法资料：<a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest">Twitter-RoBERTa</a> · <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">MiniLM</a> · <a href="https://maartengr.github.io/BERTopic/algorithm/algorithm.html">BERTopic</a> · <a href="https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#keybertinspired">KeyBERTInspired</a> · <a href="https://huggingface.co/facebook/bart-large-mnli">BART-MNLI</a> · <a href="https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1">DeBERTa ABSA</a></p></section>
</main><footer>MYCOURSE / 情感探索实践 · 本页面图表、交互与数据已内嵌，可离线打开。</footer><script id="comment-data" type="application/json">{valid.to_json(orient='records',force_ascii=False).replace('</', '<\\/')}</script><script>{(ROOT / 'scripts/report.js').read_text(encoding='utf-8')}</script></body></html>'''
    (REPORT / "index.html").write_text(page, encoding="utf-8")
    (REPORT / "结果摘要.md").write_text(f"# 中国电动车视频评论情感探索\n\n{intro}\n\n主要主题：{top_names}。\n\n方面随机样本 {mention['sample_size']} 条，得到 {absa['comment_aspect_records']} 条评论—方面记录，覆盖 {absa['matched_comments']} 条评论。未检测到不等于中性。\n\n打开 [交互报告](index.html) 查看完整流程、图表、原句和方法说明；figures/ 提供 300 dpi PNG 与 SVG。\n", encoding="utf-8")
    print(f"Report ready: {REPORT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
