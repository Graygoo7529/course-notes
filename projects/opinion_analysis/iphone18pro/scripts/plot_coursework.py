"""Export coursework figures from one audited offline analysis run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from gdelt_probe import write_json

TOPICS = {"camera": "影像", "ai": "AI", "performance_battery": "性能与续航",
          "price_value": "价格与价值", "market_competition": "市场竞争", "availability": "预购与供货"}
INK, BLUE, TEAL, MUTED = "#18354a", "#2775b6", "#07988c", "#637786"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    reviews = json.loads((run / "reviews.json").read_text(encoding="utf-8"))
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#ced8df",
                         "figure.facecolor": "#fafcfe", "axes.facecolor": "#fafcfe"})
    output = run / "figures"
    output.mkdir(exist_ok=True)
    files = []

    def save(fig, stem: str) -> None:
        for extension in ("png", "svg"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
            files.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    queries = [("broad", "宽口径：iPhone 18"), ("core", "主检索：Pro 产品词"),
               ("single_phrase", '单短语："iPhone 18 Pro"'), ("core_and_launch", "主检索 AND 英文发布词"),
               ("core_not_rumor", "主检索 NOT 英文传闻词"), ("core_not_foldable", "主检索 NOT 折叠机词"),
               ("core_in_title", "主检索 + 标题含产品词")]
    values = [summary["pooled_counts"][key] for key, _ in queries]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.31, right=.94, top=.79, bottom=.18)
    ax.barh(range(len(queries)), values, height=.58, color=["#9db2c2", TEAL, "#9db2c2", BLUE, BLUE, BLUE, BLUE])
    ax.set_yticks(range(len(queries)), [label for _, label in queries])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel("固定语料中命中的不同 URL 数")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e6edf2")
    ax.tick_params(axis="y", length=0)
    for i, value in enumerate(values):
        ax.text(value + .8, i, str(value), va="center", fontsize=12, fontweight="bold")
    fig.text(.06, .93, "01  布尔条件如何改变候选集合", fontsize=21, fontweight="bold")
    fig.text(.06, .86, "各规则独立对照；命中少并不自动意味着质量更高", color=MUTED, fontsize=12)
    fig.text(.06, .055, "数据：4 个预先固定的 GDELT 分钟文件，共 65 个候选 URL。\n英文发布词与传闻词存在语言偏差；此图不是逐层筛选漏斗，也不是每日热度。", color=MUTED, fontsize=10)
    save(fig, "01_boolean_comparison")

    frame = summary["topic_same_frame"]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.20, right=.94, top=.78, bottom=.20)
    for i, topic in enumerate(TOPICS):
        counts = frame["counts"][topic]
        ax.barh(i - .16, counts["en"], .29, color=BLUE)
        ax.barh(i + .16, counts["local"], .29, color=TEAL)
        for offset, key in ((-.16, "en"), (.16, "local")):
            ax.text(counts[key] + .45, i + offset, str(counts[key]), va="center", fontsize=10)
    ax.set_yticks(range(len(TOPICS)), list(TOPICS.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, frame["frame_unique_urls"])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e6edf2")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("议题词命中的不同 URL 数（尚未等同于有效议题）")
    fig.legend(handles=[Patch(color=BLUE, label="英文词典"), Patch(color=TEAL, label="英文 + 对应法语／西语补词")],
               loc="upper left", bbox_to_anchor=(.20, .83), ncol=2, frameon=False, fontsize=10)
    fig.text(.06, .93, "02  多语种词典补足了哪些表达", fontsize=21, fontweight="bold")
    fig.text(.06, .86, f'两组均只比较同一批 {frame["frame_unique_urls"]} 个英／西／法语 URL', color=MUTED, fontsize=12)
    fig.text(.06, .055, "补词来自本次原文核读，属于样本内修订检验。其他语言未评估；各议题允许重叠。\n更多共现需要逐篇检查对象与语境，不能直接解释为更多支持或批评。", color=MUTED, fontsize=10)
    save(fig, "02_language_dictionary")

    relevant = sorted([row for row in reviews if row["relevance"] == "relevant"],
                      key=lambda row: (row["article_date"], row["country_label"], row["review_id"]))
    stance_codes = {"neutral": 1, "positive": 2, "negative": 3, "mixed": 4}
    colors = ["#eaf0f4", "#8295a6", "#07988c", "#d66d58", "#705ead"]
    labels = {"neutral": "信息陈述", "positive": "具体肯定", "negative": "具体批评", "mixed": "权衡／保留"}
    # Cells show topic focus only. Stance applies to the stated article target,
    # never automatically to each tagged topic or to the product as a whole.
    matrix = [[int(topic in row["main_topics"]) for topic in TOPICS] for row in relevant]
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.subplots_adjust(left=.36, right=.81, top=.73, bottom=.18)
    ax.imshow(matrix, cmap=ListedColormap([colors[0], BLUE]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TOPICS)), list(TOPICS.values()), fontsize=10)
    ax.xaxis.tick_top()
    row_labels = [f'{row["review_id"]}  {row["article_date"][5:]}  {row["outlet"]}｜{row["country_label"]}\n{row["short_label"]}' for row in relevant]
    ax.set_yticks(range(len(relevant)), row_labels, fontsize=10)
    ax.tick_params(axis="both", length=0, pad=9)
    for i, row in enumerate(relevant):
        for j, topic in enumerate(TOPICS):
            if topic in row["main_topics"]:
                ax.text(j, i, "●", ha="center", va="center", color="white", fontsize=13)
        ax.text(1.03, 1 - (i + .5) / len(relevant), labels[row["stance"]], transform=ax.transAxes,
                va="center", color=colors[stance_codes[row["stance"]]], fontsize=11, fontweight="bold")
    ax.set_xticks([x - .5 for x in range(len(TOPICS) + 1)], minor=True)
    ax.set_yticks([x - .5 for x in range(len(relevant) + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.035, .94, "03  原文中的议题与评价：7 篇有效案例", fontsize=21, fontweight="bold")
    fig.text(.035, .88, "发布期 09-09—09-11：6 例；首发期 09-19：1 例。日期采用当前页面标注。\n来源国家按出版主体核实；未核实者保留未知。", color=MUTED, fontsize=12)
    fig.text(.825, .79, "对具体对象的评价", color=MUTED, fontsize=10)
    fig.text(.035, .055, "● = 核读后有实质讨论；空格 = 未编码为实质议题。右侧评价针对各篇注明的对象，不是每个议题的情绪。\n目的性选择 16 例：7 例有效、4 例排除、5 例正文不可得。仅作案例比较，不估计国家立场或总体趋势。", color=MUTED, fontsize=10)
    save(fig, "03_reviewed_cases")
    write_json(output / "manifest.json", {"input_run": str(run), "files": files,
               "summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
               "reviews_sha256": hashlib.sha256((run / "reviews.json").read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(output)


if __name__ == "__main__":
    main()
