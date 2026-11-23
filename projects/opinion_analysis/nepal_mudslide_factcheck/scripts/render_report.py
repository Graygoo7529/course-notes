"""Render 300-dpi PNG/SVG figures, an offline interactive HTML report, and a formal Markdown report.

Reads only processed tables (data/processed/*.csv, *.json), raw Google Trends CSVs,
and the structured primary evidence file (evidence/factcheck_knowledge.json).
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
EVIDENCE_PATH = ROOT / "evidence" / "factcheck_knowledge.json"
SCRIPTS_DIR = ROOT / "scripts"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"

STANCE_COLORS: dict[str, str] = {
    "dam_collapse_rumor": "#D75A4A",
    "geopolitical_blame_other": "#D98A3C",
    "earthquake_misread": "#9B6BB1",
    "breaking_disaster_report": "#7F8FA0",
    "science_glacier_collapse": "#238B8E",
    "factcheck_debunk": "#2D6DA3",
    "external_referenced": "#B8C4CC",
}

STANCE_CN: dict[str, str] = {
    "dam_collapse_rumor": "中国大坝溃坝谣言",
    "geopolitical_blame_other": "基建/预警延误归责",
    "earthquake_misread": "构造地震触发误读",
    "breaking_disaster_report": "纯灾情与应急通报",
    "science_glacier_collapse": "冰岩崩塌科学解释",
    "factcheck_debunk": "事实核查与辟谣",
}

SLOT_LABELS: list[tuple[str, str]] = [
    ("chinese_dam", "大坝溃坝词"),
    ("hedging_reportedly", "reportedly"),
    ("tourists_291", "291外籍"),
    ("indians_105", "105印度人"),
    ("us_inflated", "美籍夸大"),
    ("delayed_warning", "延误/武器"),
    ("quake_m44_m52", "M4.4/5.2"),
    ("cctv_67584ms", "67584ms视频"),
]

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CFD9DC",
        "text.color": "#1F2F39",
        "axes.labelcolor": "#4E606B",
        "xtick.color": "#4E606B",
        "ytick.color": "#1F2F39",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def encode_image_data_uri(
    path: Path,
    max_width: int = 2400,
    fmt: str = "WEBP_LOSSLESS",
    quality: int = 94,
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """Encode raster image with lossless/high-fidelity compression for crisp offline HTML embedding."""
    with Image.open(path) as img:
        w, h = img.size
        if w > max_width:
            new_h = max(1, int(round(h * (max_width / w))))
            img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        fmt_upper = fmt.upper()
        if fmt_upper == "JPEG":
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                canvas = Image.new("RGB", rgba.size, bg_rgb)
                canvas.paste(rgba, mask=rgba.getchannel("A"))
                out_img = canvas
            else:
                out_img = img.convert("RGB")
            out_img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
            mime = "image/jpeg"
        elif fmt_upper == "WEBP_LOSSLESS":
            out_img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
            out_img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
            mime = "image/webp"
        elif fmt_upper == "WEBP":
            out_img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
            out_img.save(buf, format="WEBP", quality=quality, method=6)
            mime = "image/webp"
        else:
            img.save(buf, format="PNG", optimize=True, compress_level=9)
            mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def img_tag(name: str, caption: str) -> str:
    svg_path = FIG_DIR / f"{name}.svg"
    if svg_path.exists():
        b64 = base64.b64encode(svg_path.read_bytes()).decode("ascii")
        uri = f"data:image/svg+xml;base64,{b64}"
    else:
        uri = encode_image_data_uri(FIG_DIR / f"{name}.png", max_width=2400, fmt="WEBP_LOSSLESS")
    return (
        f'<figure><img src="{uri}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def plot_01_info_void(
    bins: list[dict[str, str]],
    phases: list[dict[str, str]],
    gdelt_slices: list[dict[str, str]],
) -> None:
    fig = plt.figure(figsize=(12.2, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.38, wspace=0.26)

    # Top panel: 6h stacked narrative bars + V_en(t) & V_all(t)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(bins))
    labels = [r["bin_label"] for r in bins]
    order = [
        "dam_collapse_rumor",
        "geopolitical_blame_other",
        "earthquake_misread",
        "breaking_disaster_report",
        "science_glacier_collapse",
        "factcheck_debunk",
    ]
    bottom = np.zeros(len(bins))
    for st in order:
        vals = np.array([int(r[st]) for r in bins], dtype=float)
        ax1.bar(x, vals, bottom=bottom, width=0.62, color=STANCE_COLORS[st], label=STANCE_CN[st], alpha=0.92)
        bottom += vals

    # Highlight 0-24h information void window
    ax1.axvspan(-0.5, 3.5, color="#D75A4A", alpha=0.065, zorder=0)
    ax1.text(1.5, 25.8, "0–24h 高危信息真空期（英语澄清滞后）", ha="center", va="bottom",
             fontsize=9.6, fontweight="bold", color="#A9392B")

    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.spines["right"].set_color("#C5D2D6")
    v_en = [float(r["deficit_ratio_en"]) for r in bins]
    v_all = [float(r["deficit_ratio_all"]) for r in bins]
    ax1_r.plot(x, v_en, color="#9C1F14", marker="o", linewidth=2.2, label="英语信息赤字比 V_en(t)")
    ax1_r.plot(x, v_all, color="#238B8E", marker="s", linestyle="--", linewidth=1.8, label="全语种信息赤字比 V_all(t)")
    ax1_r.annotate(
        "英语赤字峰值 V_en = 20.0\n（+6–12h 英语科学供给 = 0）",
        xy=(1, 20.0), xytext=(3.8, 16.5),
        arrowprops={"arrowstyle": "->", "color": "#9C1F14", "lw": 1.3},
        fontsize=9.2, fontweight="bold", color="#9C1F14",
        bbox={"boxstyle": "round,pad=0.38", "fc": "#FFF5F3", "ec": "#D75A4A", "lw": 0.9},
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(0, 29.5)
    ax1_r.set_ylim(0, 23.5)
    ax1.set_ylabel("X 平台各立场帖数（条）")
    ax1_r.set_ylabel("信息赤字比 V(t) = D / (S + 0.05)", color="#7A241B")
    ax1.set_title("A. 灾后 66 小时 X 平台叙事分布与跨语种信息赤字比 V(t) 演化（基准零点 t0 = 08-26 02:52 UTC）",
                  loc="left", fontweight="bold", pad=44)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=4,
               frameon=False, fontsize=8.6)

    # Bottom-left: Phase attribution share + Shannon Entropy H
    ax2 = fig.add_subplot(gs[1, 0])
    px = np.arange(len(phases))
    p_labels = [p["phase"].replace(" ", "\n") for p in phases]
    rumor_pct = [float(p["rumor_share"]) * 100 for p in phases]
    sci_pct = [float(p["science_debunk_share"]) * 100 for p in phases]
    ent = [float(p["attribution_entropy_bits"]) for p in phases]

    ax2.bar(px - 0.17, rumor_pct, width=0.32, color="#D75A4A", label="谣言与归责占比（%）")
    ax2.bar(px + 0.17, sci_pct, width=0.32, color="#238B8E", label="科学解释与辟谣占比（%）")
    for i, (rv, sv) in enumerate(zip(rumor_pct, sci_pct)):
        if rv > 0:
            ax2.text(i - 0.17, rv + 2, f"{rv:.1f}%", ha="center", fontsize=8.8, color="#A83628", fontweight="bold")
        ax2.text(i + 0.17, sv + 2, f"{sv:.1f}%", ha="center", fontsize=8.8, color="#196A6D", fontweight="bold")

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.plot(px, ent, color="#2D6DA3", marker="D", linewidth=2.2, markersize=6.5, label="归因分布熵 H（bits）")
    ent_offsets = [(0.0, 0.14), (0.0, 0.14), (-0.22, 0.16)]
    for i, ev in enumerate(ent):
        dx_e, dy_e = ent_offsets[i]
        ax2_r.text(
            i + dx_e, ev + dy_e, f"H={ev:.2f}", ha="center", fontsize=8.6,
            color="#2D6DA3", fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    ax2.set_xticks(px)
    ax2.set_xticklabels(p_labels, fontsize=9)
    ax2.set_ylim(0, 118)
    ax2_r.set_ylim(0, 2.8)
    ax2.set_ylabel("立场占比（%）")
    ax2_r.set_ylabel("香农熵 H（bits）", color="#2D6DA3")
    ax2.set_title("B. 三阶段归因占比切换与归因分布熵 H", loc="left", fontweight="bold", pad=12)
    h2a, l2a = ax2.get_legend_handles_labels()
    h2b, l2b = ax2_r.get_legend_handles_labels()
    ax2.legend(h2a + h2b, l2a + l2b, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8.5)

    # Bottom-right: GDELT stratified global news TOC hits
    ax3 = fig.add_subplot(gs[1, 1])
    ok_slices = [s for s in gdelt_slices if s["status"] == "ok"]
    gx = np.arange(len(ok_slices))
    g_labels = [f"{s['file_stamp'][6:8]}日\n{s['file_stamp'][8:10]}Z" for s in ok_slices]
    g_base = [int(s["base"]) for s in ok_slices]
    g_sci = [int(s["T3_glacier_science"]) for s in ok_slices]
    ax3.fill_between(gx, g_base, color="#238B8E", alpha=0.16)
    ax3.plot(gx, g_base, color="#238B8E", marker="o", linewidth=2.1, label="GDELT 灾害总命中（篇/片）")
    ax3.bar(gx, g_sci, width=0.45, color="#2D6DA3", alpha=0.85, label="标题明确含冰川/崩塌（T3）")
    ax3.axvspan(-0.4, 2.4, color="#D75A4A", alpha=0.07)
    ax3.annotate("前 15h 主流新闻稀缺\n(0 → 5 → 9 篇)", xy=(0.1, 1.0), xytext=(0.0, 34.0),
                 arrowprops={"arrowstyle": "->", "color": "#A9392B"}, fontsize=8.5, color="#A9392B",
                 bbox={"boxstyle": "round,pad=0.25", "fc": "#FFF5F3", "ec": "#E2A69E", "lw": 0.8})
    ax3.set_xticks(gx[::2])
    ax3.set_xticklabels(g_labels[::2], fontsize=8.5)
    ax3.set_ylim(0, 46)
    ax3.set_ylabel("每 6h 目录切片命中篇数")
    ax3.set_title("C. GDELT 全球新闻分层切片命中（13 片共 220 篇）", loc="left", fontweight="bold", pad=12)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8.5)

    save_fig(fig, "01_info_void_timeline")


def plot_02_elevation_corridor(knowledge: dict[str, Any]) -> None:
    corridor = knowledge["elevation_corridor"]
    dists = np.array([float(c["distance_km"]) for c in corridor])
    elevs = np.array([float(c["elevation_m"]) for c in corridor])

    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    # Smooth terrain profile curve
    x_fine = np.linspace(0, 48, 300)
    y_fine = np.interp(x_fine, dists, elevs)
    # Add realistic canyon concavity
    y_curve = y_fine - 140 * np.sin(np.pi * x_fine / 48.0) ** 1.3
    y_curve[0] = elevs[0]
    for d, e in zip(dists, elevs):
        idx = int(np.argmin(np.abs(x_fine - d)))
        y_curve[idx] = e

    ax.fill_between(x_fine, 800, y_curve, color="#E4ECEA", alpha=0.85)
    ax.plot(x_fine, y_curve, color="#2D5A62", linewidth=2.6)

    # Border vertical marker near km 21.5
    ax.axvline(21.5, color="#D75A4A", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.text(21.0, 4950, "← 尼泊尔高海拔源区\n（非西藏境内大坝）", ha="right", va="top",
            fontsize=9.5, fontweight="bold", color="#238B8E")
    ax.text(22.2, 4950, "中尼边境走廊 →\n吉隆口岸 / Rasuwagadhi 受灾段", ha="left", va="top",
            fontsize=9.5, fontweight="bold", color="#A9392B")

    colors = ["#238B8E", "#2D6DA3", "#D75A4A", "#7F8FA0"]
    offsets = [(1.2, 180), (1.4, 340), (1.3, 450), (-14.5, 420)]
    for i, c in enumerate(corridor):
        d, e = float(c["distance_km"]), float(c["elevation_m"])
        ax.scatter([d], [e], s=110, color=colors[i], edgecolor="white", linewidth=2, zorder=5)
        ox, oy = offsets[i]
        ha = "left" if ox >= 0 else "left"
        ax.annotate(
            f"{c['stage']}：{c['name']}\n海拔 ~{int(e):,} m\n{c['role']}",
            xy=(d, e), xytext=(d + ox, e + oy),
            arrowprops={"arrowstyle": "->", "color": colors[i], "lw": 1.3},
            fontsize=8.6, ha=ha, va="bottom",
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": colors[i], "lw": 1.1, "alpha": 0.96},
        )

    # Callout box for hydraulic engineering refutation
    ax.text(
        26.2, 3400,
        "水工物理证伪核心：\n• 111 MW Rasuwagadhi 及吉隆小水电均为径流引水式电站\n• 无大型蓄水库容，物理上不可能蓄积亿方级洪水\n• 电站地下厂房与吉隆口岸本身系被冲毁受灾体",
        fontsize=8.8, color="#30414B",
        bbox={"boxstyle": "round,pad=0.55", "fc": "#FDF6F4", "ec": "#D75A4A", "lw": 1.0},
    )

    ax.set_xlim(-2, 52)
    ax.set_ylim(800, 5850)
    ax.set_xlabel("沿伦德河（Lhende Kholā）— 博特科西河（Bhote Koshi）下泄路径近似距离（km）")
    ax.set_ylabel("高程（m）")
    ax.set_title("OSINT 空间地形与流域高程剖面核验：朗塘里荣峰 1.32 亿 m³ 冰岩崩塌演进路径",
                 loc="left", fontweight="bold", pad=16)
    save_fig(fig, "02_elevation_corridor")


def plot_03_trends_and_regions() -> None:
    t_main = read_csv_rows(RAW_DIR / "google_trends.csv")
    t_dam = read_csv_rows(RAW_DIR / "google_trends_dam.csv")
    r_main = read_csv_rows(RAW_DIR / "by_region.csv")
    r_dam = read_csv_rows(RAW_DIR / "by_region_dam.csv")

    fig = plt.figure(figsize=(12.2, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)

    # Top-left: main 4 terms
    ax1 = fig.add_subplot(gs[0, 0])
    dates = [r["Time"][5:] for r in t_main]
    x = np.arange(len(dates))
    series_cfg = [
        ("Nepal flood", "#238B8E", "o", "Nepal flood（总关注）"),
        ("Rasuwa", "#2D6DA3", "s", "Rasuwa（核心地名）"),
        ("glacier collapse", "#687D43", "^", "glacier collapse（自然成因）"),
        ("Tibet dam", "#D75A4A", "D", "Tibet dam（谣言词）"),
    ]
    for col, color, mk, lab in series_cfg:
        vals = [int(r[col]) for r in t_main]
        ax1.plot(x, vals, color=color, marker=mk, linewidth=2.0, markersize=5, label=lab)
    ax1.axvline(2, color="#D75A4A", linestyle=":", linewidth=1.2)
    ax1.text(1.85, 84, "08-26\n灾害爆发", fontsize=8.3, color="#A9392B", ha="right")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(dates[::2], fontsize=8.8)
    ax1.set_ylim(-4, 112)
    ax1.set_ylabel("相对搜索热度 (0–100)")
    ax1.set_title("A. Google Trends 全球主事件 4 词对比（日粒度）", loc="left", fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E1E8EA", fontsize=8.2)

    # Bottom-left: dam 2 terms isolated
    ax2 = fig.add_subplot(gs[1, 0])
    d_dates = [r["Time"][5:] for r in t_dam]
    dx = np.arange(len(d_dates))
    v_td = [int(r["Tibet dam"]) for r in t_dam]
    v_cd = [int(r["China dam collapse"]) for r in t_dam]
    ax2.fill_between(dx, v_td, color="#D98A3C", alpha=0.15)
    ax2.fill_between(dx, v_cd, color="#D75A4A", alpha=0.20)
    ax2.plot(dx, v_td, color="#D98A3C", marker="o", linewidth=2.1, label="Tibet dam（28日达峰=100）")
    ax2.plot(dx, v_cd, color="#D75A4A", marker="s", linewidth=2.1, label="China dam collapse（26–27日冲顶=31）")
    ax2.annotate("08-28 二次 AI 假溃坝视频\n推高 Tibet dam 搜索", xy=(4, 98), xytext=(5.6, 52),
                 arrowprops={"arrowstyle": "->", "color": "#B3681E", "lw": 1.2}, fontsize=8.3, color="#8D4F12",
                 bbox={"boxstyle": "round,pad=0.28", "fc": "#FFF9F2", "ec": "#E5BE8A", "lw": 0.8})
    ax2.set_xticks(dx[::2])
    ax2.set_xticklabels(d_dates[::2], fontsize=8.8)
    ax2.set_ylim(-4, 115)
    ax2.set_ylabel("独立标尺热度 (0–100)")
    ax2.set_title("B. 独立观测：溃坝谣言双词脉冲（消除主词压制）", loc="left", fontweight="bold", pad=10)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E1E8EA", fontsize=8.2)

    # Right full height: Regional divergence (Nepal local vs global dam-collapse share)
    ax3 = fig.add_subplot(gs[:, 1])
    dam_sorted = sorted(r_dam, key=lambda r: int(r["China dam collapse"]), reverse=True)[:10]
    reg_names = [r["Region"] for r in dam_sorted]
    cd_pct = np.array([int(r["China dam collapse"]) for r in dam_sorted], dtype=float)
    td_pct = np.array([int(r["Tibet dam"]) for r in dam_sorted], dtype=float)
    y = np.arange(len(reg_names))

    ax3.barh(y, cd_pct, color="#D75A4A", height=0.62, label="China dam collapse 占比（%）")
    ax3.barh(y, td_pct, left=cd_pct, color="#E7B783", height=0.62, label="Tibet dam 占比（%）")
    for i, (c, t) in enumerate(zip(cd_pct, td_pct)):
        if c >= 12:
            ax3.text(c / 2, i, f"{int(c)}%", ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
        if t >= 15:
            ax3.text(c + t / 2, i, f"{int(t)}%", ha="center", va="center", color="#4D3114", fontsize=8.5)

    ax3.set_yticks(y)
    ax3.set_yticklabels(reg_names, fontsize=9.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 102)
    ax3.set_xlabel("同国家/地区内双谣言词相对占比（%）")
    ax3.set_title("C. 溃坝词地域分化：跨国英语圈与南亚占比最高", loc="left", fontweight="bold", pad=12)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8.5)

    # Callout comparing Nepal main search focus from by_region.csv
    nepal_row = next((r for r in r_main if r["Region"] == "尼泊尔"), None)
    if nepal_row:
        ax3.text(
            0.02, -0.23,
            f"对照 43 国主表：尼泊尔本土搜索聚焦 Rasuwa ({nepal_row['Rasuwa']}%) 与 Nepal flood ({nepal_row['Nepal flood']}%)，Tibet dam 仅 {nepal_row['Tibet dam']}%。",
            transform=ax3.transAxes, fontsize=8.5, color="#4E606B",
        )

    save_fig(fig, "03_trends_and_regions")


def plot_04_slot_and_baba_arc(
    posts: list[dict[str, str]],
    slot_rows: list[dict[str, str]],
) -> None:
    fig = plt.figure(figsize=(12.2, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.55], hspace=0.42)

    # Top panel: @RealBababanaras 6-post narrative mutation arc
    ax1 = fig.add_subplot(gs[0])
    baba = [p for p in posts if p["author_handle"] == "@RealBababanaras"]
    bx = [float(p["hours_since_t0"]) for p in baba]
    bv = [int(p["views"]) for p in baba]
    bc = [STANCE_COLORS[p["narrative_stance"]] for p in baba]
    sizes = [max(110, min(680, math.sqrt(v) * 0.95)) for v in bv]

    ax1.axhline(0.5, color="#CFD9DC", linewidth=2.2, zorder=1)
    ax1.axvline(9.25, color="#238B8E", linestyle="--", linewidth=1.6, zorder=2)
    ax1.text(9.35, 0.96, "@NDRRMA_Nepal (+9.25h)\n发布 Planet Labs 卫星冰岩滑坡证据",
             fontsize=8.6, color="#196A6D", fontweight="bold", va="top")

    ax1.scatter(bx, [0.5] * len(bx), s=sizes, c=bc, edgecolor="white", linewidth=2, zorder=4)

    notes_baba = {
        "X01": ("+7.31h · X01\n首提“中国大坝溃坝”\n26.9万浏览", 7.05, 0.78),
        "X04": ("+7.81h · X04\n附 67584ms 口岸监控\n18.2万浏览", 7.81, 0.14),
        "X09": ("+8.29h · X09\n复用 67584ms 视频 +\n291/105/38 数字 · 45.8万", 8.45, 0.78),
        "X16": ("+10.78h · X16\n升级“大坝武器论”\n1.4万浏览", 10.78, 0.14),
        "X19": ("+11.62h · X19\n数字抬升至 400/135/54\n1.8万浏览", 11.62, 0.78),
        "X24": ("+13.59h · X24（归因漂移）\n改口冰川溃决，转指\n“中国晚通报45分钟”·17.9万", 13.59, 0.16),
    }
    for p, hx in zip(baba, bx):
        rid = p["record_id"]
        txt, tx, ty = notes_baba.get(rid, (rid, hx, 0.8))
        va = "bottom" if ty > 0.5 else "top"
        ax1.annotate(
            txt, xy=(hx, 0.5), xytext=(tx, ty),
            ha="center", va=va, fontsize=8.1,
            arrowprops={"arrowstyle": "-", "color": "#90A3AE", "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": STANCE_COLORS[p["narrative_stance"]], "lw": 0.9},
        )

    ax1.set_xlim(6.6, 14.6)
    ax1.set_ylim(-0.18, 1.15)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("距灾害物理起动零点 t0（2026-08-26 02:52:10 UTC）的小时数")
    ax1.set_title("A. 首发推手 @RealBababanaras 的 6 小时归因漂移弧（Moving the Goalposts）",
                  loc="left", fontweight="bold", pad=12)

    # Bottom panel: White-box slot reuse matrix (top 16 salient records)
    ax2 = fig.add_subplot(gs[1])
    key_ids = {
        "X01", "X02", "X04", "X08", "X09", "X10", "X11", "X14",
        "X15", "X16", "X18", "X20", "X22", "X24", "X26", "X29", "X30", "X32",
    }
    selected = [r for r in slot_rows if r["record_id"] in key_ids]
    selected.sort(key=lambda r: float(r["hours_since_t0"]))

    slot_keys = [k for k, _ in SLOT_LABELS]
    slot_names = [v for _, v in SLOT_LABELS]
    mat = np.array([[int(r[k]) for k in slot_keys] for r in selected], dtype=float)

    # Distinguish normal hit (1) vs exact 67584ms video fingerprint hit (2)
    display_mat = mat.copy()
    cctv_col = slot_keys.index("cctv_67584ms")
    for idx_r in range(len(selected)):
        if display_mat[idx_r, cctv_col] == 1:
            display_mat[idx_r, cctv_col] = 2

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#F3F6F7", "#E69B56", "#C93B2B"])
    ax2.imshow(display_mat, aspect="auto", cmap=cmap, vmin=0, vmax=2)

    for i in range(len(selected)):
        for j in range(len(slot_keys)):
            if mat[i, j] == 1:
                mark = "★67584" if j == cctv_col else "●"
                ax2.text(j, i, mark, ha="center", va="center", color="white", fontsize=8.2, fontweight="bold")

    # Highlight X09 & X11 mirror copy rows
    for idx_r, r in enumerate(selected):
        if r["record_id"] in ("X09", "X11"):
            rect = mpatches.Rectangle((-0.48, idx_r - 0.48), len(slot_keys) - 0.04, 0.96,
                                      fill=False, edgecolor="#9C1F14", linewidth=2.0)
            ax2.add_patch(rect)

    ylabels = [f"{r['record_id']}  {r['author_handle']}  (+{float(r['hours_since_t0']):.1f}h)" for r in selected]
    ax2.set_yticks(np.arange(len(selected)))
    ax2.set_yticklabels(ylabels, fontsize=8.6)
    ax2.set_xticks(np.arange(len(slot_names)))
    ax2.set_xticklabels(slot_names, fontsize=9.2, fontweight="bold")
    ax2.set_xticks(np.arange(-0.5, len(slot_names), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(selected), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=1.6)
    ax2.tick_params(which="minor", bottom=False, left=False)
    ax2.set_title("B. 8 类白盒话术与媒介指纹槽位复用矩阵（红框标注 X09 与 X11 的 6/6 全同镜像搬运）",
                  loc="left", fontweight="bold", pad=12)

    save_fig(fig, "04_slot_matrix_and_baba_arc")


def plot_05_attention_asymmetry(posts: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.3})

    # Left: views by stance
    ax1 = axes[0]
    stances = [
        "breaking_disaster_report",
        "geopolitical_blame_other",
        "dam_collapse_rumor",
        "science_glacier_collapse",
        "factcheck_debunk",
        "earthquake_misread",
    ]
    views_m = [sum(int(p["views"]) for p in posts if p["narrative_stance"] == st) / 1e6 for st in stances]
    counts = [sum(1 for p in posts if p["narrative_stance"] == st) for st in stances]
    y = np.arange(len(stances))
    ax1.barh(y, views_m, color=[STANCE_COLORS[s] for s in stances], height=0.6)
    for i, (vm, cnt) in enumerate(zip(views_m, counts)):
        ax1.text(vm + 0.5, i, f"{vm:.2f}M（{cnt} 帖）", va="center", fontsize=8.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([STANCE_CN[s] for s in stances], fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(views_m) * 1.28)
    ax1.set_xlabel("累计浏览量（百万次）")
    ax1.set_title("A. 六类立场的 X 平台累计曝光量", loc="left", fontweight="bold", pad=12)

    # Right: Cross-lingual asymmetry in verified vs rumor views
    ax2 = axes[1]
    rumor_stances = {"dam_collapse_rumor", "geopolitical_blame_other", "earthquake_misread"}
    supply_stances = {"science_glacier_collapse", "factcheck_debunk"}

    v_rumor_all = sum(int(p["views"]) for p in posts if p["narrative_stance"] in rumor_stances) / 1e6
    v_sci_ne = sum(int(p["views"]) for p in posts if p["narrative_stance"] in supply_stances and p["language"] == "ne") / 1e6
    v_sci_en = sum(int(p["views"]) for p in posts if p["narrative_stance"] in supply_stances and p["language"] == "en") / 1e6

    cats = ["谣言与衍生归责\n（26 帖，主为英语）", "尼泊尔语官方卫星定性\n（X38 等 2 帖）", "英语科学解释与辟谣\n（全部 12 帖合计）"]
    vals = [v_rumor_all, v_sci_ne, v_sci_en]
    cols = ["#D75A4A", "#238B8E", "#2D6DA3"]
    bx = np.arange(len(cats))
    ax2.bar(bx, vals, width=0.52, color=cols)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.16, f"{v:.2f}M", ha="center", fontweight="bold", fontsize=9.8, color=cols[i])
    ratio = v_rumor_all / max(v_sci_en, 1e-6)
    ax2.annotate(
        f"英语圈曝光落差 ≈ {ratio:.1f} 倍！",
        xy=(2, v_sci_en + 0.52), xytext=(1.15, 4.7),
        arrowprops={"arrowstyle": "->", "color": "#A9392B", "lw": 1.4},
        fontsize=9.5, fontweight="bold", color="#A9392B",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF5F3", "ec": "#D75A4A"},
    )
    ax2.set_xticks(bx)
    ax2.set_xticklabels(cats, fontsize=8.8)
    ax2.set_ylim(0, 7.2)
    ax2.set_ylabel("累计浏览量（百万次）")
    ax2.set_title("B. 跨语种信息真空：谣言曝光 vs 澄清曝光", loc="left", fontweight="bold", pad=12)

    save_fig(fig, "05_attention_asymmetry")


def plot_06_dual_network_patterns(
    nodes_layout: list[dict[str, Any]],
    explicit_edges: list[dict[str, str]],
    co_edges: list[dict[str, str]],
) -> None:
    """Two-panel static diagram comparing Explicit Propagation vs Covert Slot Co-occurrence patterns."""
    node_map = {n["handle"]: n for n in nodes_layout}

    fig = plt.figure(figsize=(14.6, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.16, 1.0], wspace=0.07)

    # ---------------- Panel A: Explicit Network (14 cross-account directed edges) ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.2, 10.6)
    ax1.set_ylim(-0.3, 10.4)
    ax1.axis("off")

    pos_a: dict[str, tuple[float, float]] = {
        # ② Parasitic traffic hijacking (top-left)
        "@BlacksmithIndie": (1.45, 7.75),
        "@ravipandey2643": (4.25, 8.70),
        "@TheInsiderPaper": (4.25, 6.72),
        # Companion single-hop rumor replies (mid-left)
        "@harsh_sinh58591": (1.45, 5.18),
        "@nexta_tv": (4.25, 5.18),
        "@JMLM31474255": (1.45, 4.15),
        "@DineshRedBull": (4.25, 4.15),
        # ① Local star radiation & cross-lingual spillover (bottom-left)
        "@RealBababanaras": (2.65, 2.02),
        "@Valeria__79": (1.05, 0.72),
        "@Praveen79066956": (4.15, 0.72),
        # ③ Official & science @mention chain (top-right)
        "@DHM_FloodEWS": (6.45, 8.45),
        "@NDRRMA_Nepal": (8.18, 8.45),
        "@moha_nepal": (9.72, 8.45),
        "@PTI_News": (6.45, 6.72),
        "@icimod": (9.35, 6.72),
        "@prof_nithiya": (6.45, 5.45),
        "@TakshashilaInst": (9.35, 5.45),
        # ④ Single-hop targeted debunks (bottom-right)
        "@F1ght4dream": (6.45, 3.45),
        "@tiraspeaks": (9.35, 3.45),
        "@RushAnn1234": (6.45, 2.20),
        "@JeanMarcNoelB": (9.35, 2.20),
        "@Corrine4o": (6.45, 0.95),
        "@RadharamnDas": (9.35, 0.95),
    }

    # Subtle semantic group backdrop boxes
    groups_a = [
        (0.12, 6.02, 5.18, 3.88, "#FDF4F2", "#E3A79E", "② 寄生式流量劫持（低粉号搭车千万级爆帖）", "#A43426"),
        (0.12, 0.02, 5.18, 3.45, "#FFF6F4", "#DC8F84", "① 局部星型辐射与跨语种外溢（EN -> IT）", "#A43426"),
        (5.52, 4.75, 4.95, 5.15, "#F2F9F8", "#9ACBC8", "③ 官方应急与科研协作链（@提及背书）", "#196A6D"),
        (5.52, 0.02, 4.95, 4.38, "#F2F6FB", "#9DB9D8", "④ 单跳靶向辟谣（民间核查点名驳斥）", "#225786"),
    ]
    for gx, gy, gw, gh, fc, ec, title, tc in groups_a:
        box = mpatches.FancyBboxPatch(
            (gx, gy), gw, gh, boxstyle="round,pad=0.14,rounding_size=0.25",
            facecolor=fc, edgecolor=ec, linestyle="--", linewidth=1.0, zorder=0,
        )
        ax1.add_patch(box)
        ax1.text(gx + 0.16, gy + gh - 0.16, title, fontsize=8.3, fontweight="bold", color=tc, va="top", zorder=1)

    # Deduplicate parallel directed edges for clean arrow rendering while tracking count
    pair_count: dict[tuple[str, str], int] = {}
    for e in explicit_edges:
        k = (e["source"], e["target"])
        pair_count[k] = pair_count.get(k, 0) + 1

    drawn_pairs: set[tuple[str, str]] = set()
    for e in explicit_edges:
        src, dst = e["source"], e["target"]
        if (src, dst) in drawn_pairs or src not in pos_a or dst not in pos_a:
            continue
        drawn_pairs.add((src, dst))
        x1, y1 = pos_a[src]
        x2, y2 = pos_a[dst]
        is_mention = e["edge_type"] == "mention"
        is_debunk = e["source_stance"] == "factcheck_debunk"
        color = "#238B8E" if is_mention else ("#2D6DA3" if is_debunk else "#677886")
        ls = ":" if is_mention else "-"
        lw = 1.9 if is_mention else 2.1
        rad = 0.14 if (src == "@BlacksmithIndie" or dst == "@RealBababanaras") else 0.0
        arrow = mpatches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=6.5,head_width=3.8",
            shrinkA=12, shrinkB=13,
            color=color, linestyle=ls, linewidth=lw, alpha=0.88, zorder=2,
        )
        ax1.add_patch(arrow)

        # Edge micro-annotation for key roles
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if src == "@BlacksmithIndie" and dst == "@ravipandey2643":
            ax1.text(mx - 0.12, my + 0.34, "Quote 搭车", fontsize=7.4, color="#A43426", ha="center", fontweight="bold")
        elif src == "@BlacksmithIndie" and dst == "@TheInsiderPaper":
            ax1.text(mx - 0.15, my - 0.56, "Reply 首评植入", fontsize=7.4, color="#A43426", ha="center", fontweight="bold")
        elif src == "@Valeria__79":
            ax1.text(mx - 0.42, my + 0.14, "意语引用", fontsize=7.4, color="#A43426", ha="center")
        elif src == "@DHM_FloodEWS" and dst == "@NDRRMA_Nepal":
            ax1.text(mx, my - 0.34, "x2 @提及", fontsize=7.2, color="#196A6D", ha="center")

    for handle, (px, py) in pos_a.items():
        n = node_map.get(handle)
        stance = n["primary_stance"] if n else "external_referenced"
        col = STANCE_COLORS.get(stance, "#B8C4CC")
        sz = 180 if handle in ("@RealBababanaras", "@NDRRMA_Nepal", "@TheInsiderPaper", "@ravipandey2643") else 110
        ax1.scatter([px], [py], s=sz, color=col, edgecolor="white", linewidth=1.7, zorder=4)
        extra = ""
        if handle == "@TheInsiderPaper":
            extra = "\n(25.56M 浏览)"
        elif handle == "@ravipandey2643":
            extra = "\n(6.53M 浏览)"
        elif handle == "@NDRRMA_Nepal":
            extra = "\n(2.38M)"
        elif handle == "@RealBababanaras":
            extra = "\n(1.12M 枢纽)"
        dy = -0.34 if handle in ("@Valeria__79", "@Praveen79066956", "@Corrine4o", "@RadharamnDas") else 0.28
        va = "top" if dy < 0 else "bottom"
        ax1.text(
            px, py + dy, f"{handle}{extra}", ha="center", va=va, fontsize=7.5,
            fontweight="bold" if extra else "normal", color="#1F2F39",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85}, zorder=5,
        )

    ax1.set_title("A. 明线传播网：显式引用、回复与 @提及（14 条跨账号有向边）", loc="left", fontweight="bold", fontsize=11.0, pad=10)

    # ---------------- Panel B: Covert Co-occurrence Network (17 weighted undirected edges) ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-0.2, 10.2)
    ax2.set_ylim(-0.3, 10.4)
    ax2.axis("off")

    pos_b: dict[str, tuple[float, float]] = {
        "@TicTocTick": (1.25, 8.20),
        "@RealBababanaras": (4.25, 6.85),
        "@Ujwalthapa12": (8.35, 6.85),
        "@DailyExplorerX": (2.15, 4.18),
        "@ferozwala": (4.55, 2.52),
        "@NewsAlgebraIND": (7.35, 2.75),
        "@SanjeevCrime": (8.65, 4.55),
    }

    # Backdrop highlight zones in Panel B
    mirror_box = mpatches.FancyBboxPatch(
        (2.85, 5.52), 6.85, 3.08, boxstyle="round,pad=0.14,rounding_size=0.25",
        facecolor="#FDEEEB", edgecolor="#D75A4A", linestyle="-", linewidth=1.4, zorder=0,
    )
    ax2.add_patch(mirror_box)
    ax2.text(
        6.28, 8.48, "① 隐蔽“伪原创”镜像搬运（明线 0 连边 -> 暗线第 1）",
        ha="center", va="top", fontsize=8.1, fontweight="bold", color="#9C1F14", zorder=1,
    )

    clique_box = mpatches.FancyBboxPatch(
        (0.55, 0.12), 9.15, 5.02, boxstyle="round,pad=0.14,rounding_size=0.25",
        facecolor="#FFF9F1", edgecolor="#E0A96D", linestyle="--", linewidth=1.1, zorder=0,
    )
    ax2.add_patch(clique_box)
    ax2.text(
        5.12, 0.30,
        "② 同源真实数字分叉污染团块（共享 291 外籍 / 105 印度人槽位，1–4h 同步爆发）\n同一份旅游局排查名单迅速分叉为：纯灾情（灰）· 地震误读（紫）· 大坝谣言（红）",
        ha="center", va="bottom", fontsize=7.8, color="#7D4A13", zorder=1,
    )

    for e in co_edges:
        a, b = e["author_a"], e["author_b"]
        if a not in pos_b or b not in pos_b:
            continue
        x1, y1 = pos_b[a]
        x2, y2 = pos_b[b]
        is_video = int(e["same_cctv_67584ms"]) == 1
        w = float(e["weight"])
        if is_video:
            ax2.plot([x1, x2], [y1, y2], color="#C93323", linewidth=4.6, linestyle="-", alpha=0.96, zorder=3)
            ax2.text(
                (x1 + x2) / 2, y1 + 0.24,
                "★ 39 min · 6/6 槽位全同 · 67584ms 视频指纹\nw = 0.896 · TF-IDF = 0.761",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#9C1F14",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D75A4A", "lw": 1.0}, zorder=6,
            )
        else:
            lw = max(1.1, w * 3.4)
            ax2.plot([x1, x2], [y1, y2], color="#D98A3C", linewidth=lw, linestyle="--", alpha=0.58, zorder=2)

    # Callout for @TicTocTick snowball relay
    ax2.annotate(
        "③ 美籍数字雪球接力\n延迟 8h 复用 3 槽位\n（美籍 38 -> 60 人）",
        xy=(1.25, 8.50), xytext=(1.40, 9.62), ha="center", va="center", fontsize=7.8,
        color="#8D3024", fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#D75A4A", "lw": 1.1},
        bbox={"boxstyle": "round,pad=0.28", "fc": "#FFF5F3", "ec": "#D75A4A", "lw": 0.9}, zorder=6,
    )

    sub_labels_b = {
        "@RealBababanaras": "X09 (+8.29h)\n大坝谣言首发枢纽",
        "@Ujwalthapa12": "X11 (+8.95h)\n无引用“伪原创”搬运",
        "@TicTocTick": "X26 (+16.83h)\n大坝+美籍60人",
        "@DailyExplorerX": "X08 (+8.25h)\n纯灾情通报",
        "@ferozwala": "X02 (+7.47h)\n纯灾情通报",
        "@NewsAlgebraIND": "X18 (+11.15h)\n纯灾情通报",
        "@SanjeevCrime": "X22 (+12.54h)\n地震致洪误读",
    }
    label_offsets_b = {
        "@RealBababanaras": (0.0, -0.34),
        "@Ujwalthapa12": (0.0, -0.34),
        "@TicTocTick": (0.0, -0.38),
        "@DailyExplorerX": (-0.12, -0.34),
        "@ferozwala": (0.0, -0.34),
        "@NewsAlgebraIND": (0.05, -0.34),
        "@SanjeevCrime": (0.05, -0.34),
    }

    for handle, (px, py) in pos_b.items():
        n = node_map.get(handle)
        stance = n["primary_stance"] if n else "dam_collapse_rumor"
        col = STANCE_COLORS.get(stance, "#7F8FA0")
        sz = 260 if handle in ("@RealBababanaras", "@Ujwalthapa12") else 180
        ax2.scatter([px], [py], s=sz, color=col, edgecolor="white", linewidth=2.0, zorder=4)
        dx, dy = label_offsets_b[handle]
        ax2.text(
            px + dx, py + dy, f"{handle}\n{sub_labels_b[handle]}",
            ha="center", va="top", fontsize=7.6,
            fontweight="bold" if handle in ("@RealBababanaras", "@Ujwalthapa12") else "normal",
            color="#1F2F39",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.85}, zorder=5,
        )

    ax2.set_title("B. 暗线共现网：白盒槽位与 67584ms 视频指纹（17 条加权无向边）", loc="left", fontweight="bold", fontsize=11.0, pad=10)

    # Shared compact bottom legend
    legend_handles = [
        mpatches.Patch(color=STANCE_COLORS["dam_collapse_rumor"], label="大坝溃坝谣言"),
        mpatches.Patch(color=STANCE_COLORS["earthquake_misread"], label="地震触发误读"),
        mpatches.Patch(color=STANCE_COLORS["breaking_disaster_report"], label="纯灾情通报"),
        mpatches.Patch(color=STANCE_COLORS["science_glacier_collapse"], label="科学冰岩解释"),
        mpatches.Patch(color=STANCE_COLORS["factcheck_debunk"], label="事实核查辟谣"),
        mpatches.Patch(color=STANCE_COLORS["external_referenced"], label="外部被提及节点"),
        plt.Line2D([0], [0], color="#C93323", lw=3.8, label="67584ms 同源视频镜像边"),
        plt.Line2D([0], [0], color="#D98A3C", lw=2.2, ls="--", label="白盒槽位时间共现边"),
        plt.Line2D([0], [0], color="#677886", lw=2.0, label="显式引用/回复"),
        plt.Line2D([0], [0], color="#2D6DA3", lw=2.0, label="单跳靶向辟谣"),
        plt.Line2D([0], [0], color="#238B8E", lw=2.0, ls=":", label="官方/科研 @提及"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=6, frameon=False, fontsize=8.6, handlelength=2.2, columnspacing=1.2,
    )

    save_fig(fig, "06_dual_network_patterns")


def compute_network_layout(
    nodes: list[dict[str, str]],
    explicit_edges: list[dict[str, str]],
    co_edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Deterministic, collision-free 2D coordinates (1000 x 520 SVG viewBox)."""
    # Hand-tuned semantic clusters so the dual-mode switch stays spatial-stable & legible
    preset: dict[str, tuple[float, float]] = {
        # Left-center: Co-occurrence rumor/slot clique around @RealBababanaras
        "@RealBababanaras": (300, 265),
        "@Ujwalthapa12": (165, 190),
        "@DailyExplorerX": (315, 125),
        "@ferozwala": (180, 355),
        "@NewsAlgebraIND": (445, 200),
        "@SanjeevCrime": (455, 345),
        "@TicTocTick": (145, 285),
        # Explicit quote/reply children of @RealBababanaras & dam spreaders
        "@Valeria__79": (245, 435),
        "@Praveen79066956": (375, 445),
        "@BlacksmithIndie": (585, 120),
        "@ravipandey2643": (725, 85),
        "@TheInsiderPaper": (735, 165),
        "@harsh_sinh58591": (590, 235),
        "@nexta_tv": (730, 235),
        "@JMLM31474255": (590, 305),
        "@DineshRedBull": (730, 305),
        # Right-bottom: Fact-check / debunk confrontation pairs
        "@F1ght4dream": (580, 395),
        "@tiraspeaks": (720, 395),
        "@RushAnn1234": (580, 460),
        "@JeanMarcNoelB": (720, 460),
        "@Corrine4o": (845, 395),
        "@RadharamnDas": (930, 455),
        # Top-right: Official Nepali agencies & science mention network
        "@DHM_FloodEWS": (835, 95),
        "@NDRRMA_Nepal": (895, 185),
        "@moha_nepal": (940, 105),
        "@PTI_News": (835, 265),
        "@icimod": (940, 265),
        "@prof_nithiya": (835, 330),
        "@TakshashilaInst": (940, 330),
    }

    active: set[str] = set()
    for e in explicit_edges:
        active.add(e["source"])
        active.add(e["target"])
    for e in co_edges:
        active.add(e["author_a"])
        active.add(e["author_b"])

    out: list[dict[str, Any]] = []
    fallback_idx = 0
    for n in nodes:
        h = n["handle"]
        if h not in active:
            continue
        if h in preset:
            px, py = preset[h]
        else:
            angle = 2 * math.pi * fallback_idx / 8.0
            px, py = 500 + 200 * math.cos(angle), 260 + 160 * math.sin(angle)
            fallback_idx += 1
        out.append(
            {
                "handle": h,
                "in_dataset": int(n["in_dataset"]),
                "post_count": int(n["post_count"]),
                "author_category": n["author_category"],
                "primary_stance": n["primary_stance"],
                "primary_stance_cn": n["primary_stance_cn"],
                "earliest_hours_since_t0": float(n["earliest_hours_since_t0"]),
                "total_views": int(n["total_views"]),
                "total_engagement": int(n["total_engagement"]),
                "explicit_in_degree": int(n["explicit_in_degree"]),
                "explicit_out_degree": int(n["explicit_out_degree"]),
                "cooccurrence_degree": int(n["cooccurrence_degree"]),
                "slots": n["slots"],
                "x": round(px, 1),
                "y": round(py, 1),
            }
        )
    return out


PARAM_CN: dict[str, str] = {
    "utc_time": "物理零点",
    "coordinates": "震源坐标",
    "depth_km": "震源深度",
    "classification": "波形定性",
    "source_zone": "崩塌源区",
    "collapse_volume": "崩塌方量",
    "insar_precursor": "灾前雷达",
    "hazard_mechanism": "成灾机理",
    "initial_tally": "初期统计",
    "traveler_profile": "人员构成",
    "plant_types": "电站类型",
    "storage_constraint": "库容约束",
    "actual_damage": "实测受损",
}


def img_tag_path(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    uri = encode_image_data_uri(path, max_width=2400, fmt="WEBP_LOSSLESS")
    return (
        f'<figure><img src="{uri}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def build_html_report(
    knowledge: dict[str, Any],
    summary: dict[str, Any],
    posts: list[dict[str, str]],
    phases: list[dict[str, str]],
    explicit_edges: list[dict[str, str]],
    co_edges: list[dict[str, str]],
    nodes_layout: list[dict[str, Any]],
    gdelt_matches: list[dict[str, str]],
) -> None:
    css = (SCRIPTS_DIR / "report.css").read_text(encoding="utf-8")
    js = (SCRIPTS_DIR / "report.js").read_text(encoding="utf-8")

    xstat = summary["x_dataset"]
    gdelt_toc = (summary.get("gdelt") or {}).get("toc") or {}
    gdelt_hits = int(gdelt_toc.get("matched_docs", len(gdelt_matches)))

    rumor_views = int(xstat["rumor_views"])
    en_sci_views = sum(
        int(p["views"])
        for p in posts
        if p["narrative_stance"] in ("science_glacier_collapse", "factcheck_debunk") and p["language"] == "en"
    )
    asym_ratio = rumor_views / max(en_sci_views, 1)

    claims_html = "".join(
        f'<article class="card claim-card {"partial" if c["id"] != "C3" else ""}">'
        f'<span class="claim-tag">{html.escape(c["id"])} · {html.escape(c["rating"])}</span>'
        f'<h3>{html.escape(c["dimension"])}</h3>'
        f'<p style="font-size:13.8px"><b>待核查断言：</b>{html.escape(c["claim_text"])}</p>'
        f'<p class="muted" style="font-size:13px;margin-bottom:0">{html.escape(c["summary"])}</p>'
        f"</article>"
        for c in knowledge["atomic_claims"]
    )

    ev_cards = []
    for src in knowledge["primary_sources"]:
        items = "".join(
            f'<li><span class="kv-key">{html.escape(PARAM_CN.get(k, k))}</span>'
            f'<span>{html.escape(str(v) + (" km（地表崩塌）" if k == "depth_km" else ""))}</span></li>'
            for k, v in src["key_parameters"].items()
        )
        ev_cards.append(
            f'<article class="evidence-card">'
            f'<div class="eyebrow">{html.escape(src["evidence_tier"])}</div>'
            f'<h3>{html.escape(src["agency"])}</h3>'
            f'<ul class="kv-list">{items}</ul>'
            f'<p class="muted" style="font-size:12.6px;margin:10px 0 0"><b>证伪逻辑：</b>{html.escape(src["verification_role"])}</p>'
            f"</article>"
        )

    craap_rows = "".join(
        f"<tr><td><b>{html.escape(r['criterion'])}</b></td><td>{html.escape(r['rumor_account'])}</td><td><code>{html.escape(r['score'])}</code></td></tr>"
        for r in knowledge["craap_assessment"]
    )

    phase_rows_html = "".join(
        f"<tr><td><b>{html.escape(r['phase'])}</b></td><td>{r['total_posts']}</td>"
        f"<td>{r['rumor_and_blame_posts']}（{float(r['rumor_share']):.1%}）</td>"
        f"<td>{r['science_and_debunk_posts']}（{float(r['science_debunk_share']):.1%}）</td>"
        f"<td><b>{float(r['attribution_entropy_bits']):.2f} bits</b></td></tr>"
        for r in phases
    )

    override_rows_html = "".join(
        f"<tr><td><code>{html.escape(p['record_id'])}</code></td><td>{html.escape(p['author_handle'])}</td>"
        f"<td><code>{html.escape(p['claimed_cause_raw'])}</code> → <b>{html.escape(p['stance_cn'])}</b></td>"
        f"<td>{html.escape(p['stance_correction_reason'])}</td></tr>"
        for p in posts
        if p["stance_correction_reason"]
    )

    raw_rows_html = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td><td>{info['bytes']:,} B</td><td><code>{html.escape(info['sha256_12'])}</code></td></tr>"
        for name, info in summary["raw_files"].items()
    )

    gt_shots_html = "".join(
        [
            img_tag_path(FIG_DIR / "gt_02_dam_trend.png", "Google Trends 独立检索截图：Tibet dam 与 China dam collapse 的 12 天脉冲走势"),
            img_tag_path(FIG_DIR / "gt_03_dam_worldmap.png", "Google Trends 全球热力图：加拿大（红色）China dam collapse 占比达 52%，欧美与南亚高度聚集"),
            img_tag_path(FIG_DIR / "gt_01_main_trend.png", "Google Trends 主事件 4 词走势原始截图（Nepal flood 27日达峰=100）"),
            img_tag_path(FIG_DIR / "gt_04_rising_queries.png", "Google Trends 飙升关联查询：flash flood nepal、tibet flood 与“西藏自治区 / 8月26日”同步暴增"),
        ]
    )

    payload = {
        "network": {
            "nodes": nodes_layout,
            "explicit_edges": explicit_edges,
            "cooccurrence_edges": co_edges,
        },
        "x_posts": [
            {
                "record_id": p["record_id"],
                "author_handle": p["author_handle"],
                "created_at_utc": p["created_at_utc"],
                "hours_since_t0": float(p["hours_since_t0"]),
                "phase": p["phase"],
                "language": p["language"],
                "narrative_stance": p["narrative_stance"],
                "stance_cn": p["stance_cn"],
                "claimed_cause_raw": p["claimed_cause_raw"],
                "stance_correction_reason": p["stance_correction_reason"],
                "slots": p["slots"],
                "views": int(p["views"]),
                "post_url": p["post_url"],
                "text_verbatim": p["text_verbatim"],
            }
            for p in posts
        ],
        "gdelt_matches": gdelt_matches,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    legend_html = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{color}"></span>{label}</span>'
        for key, label in [
            ("dam_collapse_rumor", "大坝溃坝谣言"),
            ("geopolitical_blame_other", "基建/延误归责"),
            ("earthquake_misread", "地震误读"),
            ("breaking_disaster_report", "灾情通报"),
            ("science_glacier_collapse", "科学冰岩解释"),
            ("factcheck_debunk", "辟谣核查"),
            ("external_referenced", "外部提及节点"),
        ]
        for color in [STANCE_COLORS[key]]
    )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>2026 中尼边境泥石流谣言核查与信息真空实证</title>
  <style>{css}</style>
</head>
<body>
<header>
  <a class="brand" href="#">MYCOURSE <span>/ 事实核查与舆论演化</span></a>
  <nav>
    <a href="#architecture">一、问题与架构</a>
    <a href="#factcheck">二、7 步事实核查</a>
    <a href="#empirical">三、72h 信息真空实证</a>
    <a href="#network">双模交互网络</a>
    <a href="#appendix">附录 · 证据库</a>
  </nav>
</header>

<main>
  <section class="hero">
    <div class="eyebrow">大数据与舆论分析导论 · 事实核查与计算传播实践 · 2026.11</div>
    <h1>24 小时信息真空里的<br>虚假因果链是如何被编织的？</h1>
    <div class="verdict-banner"><span class="verdict-dot"></span>核查综合评级：{html.escape(knowledge["target_claim"]["overall_verdict"])}</div>
    <p class="lede">以 2026 年 8 月 26 日中尼边境（朗塘里荣峰—吉隆/拉苏瓦）冰岩崩塌泥石流为案例，走通“标准 7 步事实核查 + 72 小时信息真空量化验证”双层闭环。</p>
    <p class="intro">灾后 7.8 小时，海外社交账号将<strong>吉隆边检站真实受灾监控</strong>、<strong>尼泊尔旅游局初期 291 名外籍失联名单</strong>与虚构的<strong>“中国西藏大坝溃坝”</strong>拼接成极具迷惑性的虚假因果链。跨平台实证表明：英语科学澄清滞后超过 6 小时，使定向核心样本在 <code>+6–12h</code> 窗口的英语信息赤字比冲至 <b>20.0</b>，谣言与衍生归责累计曝光达 <b>{rumor_views / 1e6:.2f}M</b>（为同期全部英语科学/辟谣帖的 <b>{asym_ratio:.1f} 倍</b>）；借助白盒槽位与精确到毫秒的 <code>67584 ms</code> 视频指纹，还能直接锁定伪装成原创的镜像搬运账号。</p>
    <div class="stats">
      <div><strong>54</strong><span>X 定向核心样本（40 账号 · 精确时间戳）</span></div>
      <div><strong>{gdelt_hits}</strong><span>GDELT 13 切片全球新闻命中</span></div>
      <div><strong>20.0</strong><span>+6–12h 英语信息赤字峰值 V_en</span></div>
      <div><strong>{asym_ratio:.1f}×</strong><span>定向样本英语谣言 vs 澄清曝光落差</span></div>
    </div>
    <a class="button" href="#architecture">进入报告演示 ↓</a>
    <span class="hero-note">单文件离线自包含 · 图表、双模网络与 274 条原始记录全部内嵌</span>
  </section>

  <section id="architecture">
    <div class="section-heading"><span>01 / 问题界定与三层数据架构</span><h2>一套“新闻核查 + 多源定量”的轻量闭环设计。</h2></div>
    <p>既有新闻核查报道（如 <i>The Real Fact Hunter</i>）已对首发账号的两条推文做过定性辟谣，但尚未对灾后 72 小时的信息真空、跨语种澄清时滞与隐蔽搬运结构进行定量验证。为此，本研究以 USGS 精确记录的崩塌起动时刻 <code>t0 = 2026-08-26 02:52:10 UTC</code>（北京时间 10:52:10）为统一时间零点，组合四路互补数据，各司其职：</p>
    <div class="four">
      <article class="arch-card">
        <span class="eyebrow">层 1 · 社媒微观演变</span>
        <strong>X 平台 54 条定向核心样本</strong>
        <p>按配额检索构建（谣言扩散 32 条 + 官方/科学/辟谣 22 条），保留原链、逐字正文、精确到秒的 UTC 时间戳与视频毫秒时长。</p>
      </article>
      <article class="arch-card">
        <span class="eyebrow">层 2 · 国际新闻旁证</span>
        <strong>GDELT 13 切片 220 篇</strong>
        <p>DOC 2.0 遇 HTTP 429 自动熔断，切换无频控 Web NGrams 静态目录每 6 小时采样，以双层布尔式扫描 24,131 条新闻。</p>
      </article>
      <article class="arch-card">
        <span class="eyebrow">层 3 · 公众搜索基线</span>
        <strong>Google Trends 4 份表</strong>
        <p>导出 12 天主事件 4 词与溃坝双词独立曲线，及 43 国/13 国地区分布，消除主词量级压制并对照跨国搜索分化。</p>
      </article>
      <article class="arch-card">
        <span class="eyebrow">层 4 · 一手物理锚点</span>
        <strong>四大独立权威证据</strong>
        <p>交叉调取 USGS 震源波形、ICIMOD 遥感与 InSAR 形变、尼泊尔 NTB/NDRRMA 通报及流域水电站工程参数。</p>
      </article>
    </div>
    <div class="callout">
      <b>两层证据口径区分：</b>① <b>不依赖抽样配额的硬证据</b>——包括物理时间滞后（<code>+7.31h</code> 谣言首发 → <code>+9.25h</code> 尼语官方卫星定性 → <code>+15.15h</code> 英语科学首帖）、GDELT 13 片 24,131 条全量扫描、Google Trends 全球搜索指数、首发号 6 帖漂移与 <code>67584 ms</code> 搬运物证；② <b>依托定向核心样本的指标演示</b>——信息赤字比 <code>V(t)</code>、三阶段归因熵 <code>H</code> 与头部传播节点的 <b>{asym_ratio:.1f}×</b> 曝光落差，用于刻画代表性高影响力节点的供需结构，不直接外推为 X 全站发帖绝对比例。
    </div>
    <details class="drawer">
      <summary>展开查看预处理审计明细（6 条人工立场纠偏 + 原始文件 SHA-256 指纹）</summary>
      <p class="muted" style="font-size:13px;margin-top:0">清洗脚本 <code>analyze.py</code> 保留原始字段不覆盖，剔除 <code>X02</code> 自引用环与辟谣引述干扰，对 6 条自动标注偏差的推文执行可追溯人工复核：</p>
      <div class="table-scroll">
        <table><thead><tr><th>帖号</th><th>账号</th><th>立场校准</th><th>人工核读依据</th></tr></thead><tbody>{override_rows_html}</tbody></table>
      </div>
      <div class="table-scroll" style="margin-top:14px">
        <table><thead><tr><th>原始只读文件（data/raw/）</th><th>大小</th><th>SHA-256 前 12 位</th></tr></thead><tbody>{raw_rows_html}</tbody></table>
      </div>
    </details>
  </section>

  <section id="factcheck">
    <div class="section-heading"><span>02 / 标准 7 步事实核查</span><h2>拆开“真实画面 + 真实名单 + 虚构因果”的拼接结构。</h2></div>
    <div class="step-bar">
      <div class="step-item"><b>① 锁定对象</b>存档 08-26 溃坝热帖</div>
      <div class="step-item"><b>② 声明解绑</b>拆为 C1 / C2 / C3</div>
      <div class="step-item"><b>③ OSINT 溯源</b>监控与高程时空核验</div>
      <div class="step-item"><b>④ 多源交叉</b>USGS / ICIMOD / NTB</div>
      <div class="step-item"><b>⑤ 信源评估</b>证据分级 + CRAAP</div>
      <div class="step-item"><b>⑥ 综合评级</b>虚假 / 严重误导</div>
      <div class="step-item"><b>⑦ 透明归档</b>代码与原链可复现</div>
    </div>
    <div class="callout warn">
      <b>核心待核查声明（@RealBababanaras，北京时间 2026-08-26 18:10 / 18:41 / 19:09 连发）：</b><br>
      “Nepal after flash floods reportedly originating from Tibet, following the collapse of a Chinese dam... 291 foreign tourists are missing, including 105 Indians, 38 U.S. citizens &amp; 14 Britishers.”
    </div>
    <div class="causal-compare">
      <div class="chain-row real">
        <span class="chain-badge">真实物理因果链</span>
        <span class="chain-node">朗塘里荣峰 5,200 m<br><small class="muted">Sentinel-1 灾前 40 cm 形变</small></span>
        <span class="chain-arrow">→</span>
        <span class="chain-node">1.32 亿 m³ 冰岩崩塌<br><small class="muted">USGS 02:52Z · 深度 0.0 km</small></span>
        <span class="chain-arrow">→</span>
        <span class="chain-node">伦德河 &gt;160 km/h 泥石流<br><small class="muted">冲毁吉隆口岸与径流水电站</small></span>
        <span class="chain-arrow">→</span>
        <span class="chain-node">通信中断 384 人暂失联<br><small class="muted">含 291 外籍 / 105 印度香客</small></span>
      </div>
      <div class="chain-row fake">
        <span class="chain-badge">伪造拼接因果链</span>
        <span class="chain-node forged">虚构“西藏中国大坝溃坝”<br><small>C3 完全虚假 · 流域无大库容大坝</small></span>
        <span class="chain-arrow bad">✕ 强行嫁接</span>
        <span class="chain-node kept">保留真实受灾监控画面<br><small>C1 抹水印换音频 · 吉隆边检站母本</small></span>
        <span class="chain-arrow bad">✕ 夸大错配</span>
        <span class="chain-node kept">保留 291 外籍 / 105 印度人<br><small>C2 断网排查表变“溃坝失踪”· 美籍夸大</small></span>
      </div>
    </div>
    <div class="three">{claims_html}</div>

    <div class="sub-block">
      <h3>四维独立一手证据与 OSINT 空间高程核验</h3>
      <p>逐一核对地球物理台网、山地遥感、当事国旅游应急统计与水电站工程参数，四类互不隶属的一手记录共同指向同一物理事实：灾害起动于尼泊尔境内朗塘里荣峰高海拔冰岩崩塌，下游径流引水式水电与边境口岸均系受灾体。</p>
      <div class="evidence-grid">{''.join(ev_cards)}</div>
      {img_tag("02_elevation_corridor", "图 1 · OSINT 流域高程与空间核验图：源头精确位于尼泊尔境内朗塘里荣峰约 5,200 m 高海拔处，111 MW 拉苏瓦嘎迪与吉隆小水电系下游无大库容径流引水式电站。")}
      <details class="drawer">
        <summary>展开查看首发账号 CRAAP 五维信源评估表</summary>
        <div class="table-scroll">
          <table><thead><tr><th>CRAAP 维度</th><th>首发造谣账号表现诊断</th><th>评估结果</th></tr></thead><tbody>{craap_rows}</tbody></table>
        </div>
      </details>
    </div>
  </section>

  <section id="empirical">
    <div class="section-heading"><span>03 / 72 小时信息真空与计算舆论实证</span><h2>用可计算的指标，看清谣言如何抢跑、漂移与共振。</h2></div>
    <p>为将“信息真空”“叙事收敛”与“隐蔽搬运”落到可检验的数字上，本研究定义三项核心计算指标，并在全部样本上逐窗实测：</p>
    <div class="formula-grid">
      <article class="formula-card">
        <span class="eyebrow">指标 1 · 真空度</span>
        <h3>信息赤字比 V(t)</h3>
        <div class="formula-expr">V(t) = D̃(t) / (S̃(t) + 0.05)</div>
        <p>衡量归一化追问/谣言需求 D̃(t) 相对权威科学供给 S̃(t) 的缺口。供给归零时峰值封顶 20.0，用于精准识别跨语种信息真空窗口。</p>
      </article>
      <article class="formula-card">
        <span class="eyebrow">指标 2 · 分散度</span>
        <h3>归因分布熵 H</h3>
        <div class="formula-expr">H = − Σ p<sub>k</sub> log<sub>2</sub> p<sub>k</sub></div>
        <p>刻画不同成因解释的混杂程度：真空期单一谣言主导（熵较低）→ 对冲期多说交锋（熵达峰）→ 澄清期收敛回科学共识（熵骤降）。</p>
      </article>
      <article class="formula-card">
        <span class="eyebrow">指标 3 · 暗线共现</span>
        <h3>槽位时间共现权 w<sub>uv</sub></h3>
        <div class="formula-expr">w<sub>uv</sub> = J*(u,v) · exp(−|Δt| / 6h)</div>
        <p>以 8 类白盒话术及 67584ms 视频指纹的 Jaccard 重合度乘以时间衰减核，排除辟谣引述帖，避免稠密句向量带来的“主题混淆”。</p>
      </article>
    </div>

    <div class="sub-block">
      <h3>3.1 跨语种信息真空与 {asym_ratio:.1f} 倍曝光落差</h3>
      <div class="two">
        <div>
          <p>尼泊尔官方 <b>@NDRRMA_Nepal</b> 早在灾后 <code>+9.25h</code> 就以<b>尼泊尔语</b>发布了获 235 万阅读的 Planet Labs 卫星冰岩滑坡证据（<code>X38</code>），但面向全球的英语科学澄清（<code>X41 / X42</code>）滞后了 6 小时以上。这一语言屏障使 <code>+6–12h</code> 窗口的英语信息赤字比直冲理论峰值 <b>20.0</b>（全语种为 2.61），最终造成英语圈谣言/归责曝光（<b>{rumor_views / 1e6:.2f}M</b>）达到全部英语科学辟谣帖（<b>{en_sci_views / 1e6:.2f}M</b>）的 <b>{asym_ratio:.1f} 倍</b>。</p>
          <p>GDELT 全球新闻静态切片（13 片共 220 篇命中）提供了独立旁证：灾后 <code>+3.4h</code> 全球主流新闻为 <b>0 篇</b>，<code>+9.1h</code> 仅 <b>5 篇</b>，次日才跃升至 <b>21–34 篇/片</b>。经逐条核读，220 篇中 15 篇标题指向冰川崩塌，仅 1 篇印地语报道在 8 月 28 日受社媒外溢误引福建永安水电站视频；另 1 条含 <code>dam</code> 标题实为自然堰塞坝（<i>temporary debris dam</i>），印证了关键词筛选必须配合人工核读去噪。</p>
        </div>
        <div class="panel">
          <h3>三阶段归因切换与香农熵 H</h3>
          <table><thead><tr><th>阶段</th><th>帖数</th><th>谣言/归责</th><th>科学/辟谣</th><th>熵 H</th></tr></thead><tbody>{phase_rows_html}</tbody></table>
          <p class="muted" style="font-size:12.3px;margin:10px 0 0">清晰的“1.81 → 2.25 → 0.97 bits”演化曲线：先被溃坝谣言抢占锚点，经次日剧烈对冲后收敛回科学共识。</p>
        </div>
      </div>
      {img_tag("01_info_void_timeline", "图 2 · 灾后 66 小时 X 平台叙事分箱、跨语种信息赤字比 V(t)、三阶段归因熵 H 与 GDELT 全球新闻切片命中对照。")}
      {img_tag("05_attention_asymmetry", f"图 3 · 曝光不对称实证：谣言与衍生归责累计 {rumor_views / 1e6:.2f}M 浏览，全部英语科学解释与辟谣帖合计仅 {en_sci_views / 1e6:.2f}M（{asym_ratio:.1f} 倍落差）。")}
    </div>

    <div class="sub-block">
      <h3>3.2 Google Trends 宏观搜索基线与跨国地域分化</h3>
      <p>Google Trends 12 天曲线显示，主词 <code>Nepal flood</code> 于 8 月 27 日达峰（100）。消除主词压制后单独观测双谣言词可见：<code>China dam collapse</code> 于 26–27 日冲顶（30–31）后回落，而 <code>Tibet dam</code> 在 8 月 28 日冲至 100，恰好呼应当日社媒流传的 AI 合成假溃坝视频（见 <code>X52 / X53</code> 辟谣）。地域结构反差鲜明：尼泊尔本土搜索聚焦救灾地名 <code>Rasuwa (45%)</code>，而 <code>China dam collapse</code> 相对占比最高的集中在加拿大（52%）、菲律宾（41%）、尼泊尔（40%）、美国（37%）、荷兰（25%）与印度（23%）。</p>
      {img_tag("03_trends_and_regions", "图 4 · Google Trends 12 天主事件与谣言双词走势，以及跨国地域搜索分化。")}
      <details class="drawer">
        <summary>展开查看 Google Trends 原始界面采集截图（全球热力地图与飙升查询）</summary>
        <div class="shot-grid">{gt_shots_html}</div>
      </details>
    </div>

    <div class="sub-block">
      <h3>3.3 首发号 6 帖“归因漂移弧”与 67584ms 视频指纹</h3>
      <p>精确时间戳序列揭示了比原新闻报道更深的操纵链条：<code>@RealBababanaras</code> 在 6.3 小时内连续发出 <b>6 帖</b>（<code>X01 → X04 → X09 → X16 → X19 → X24</code>），从初提溃坝、叠加 <code>67584 ms</code> 口岸监控与 <code>291 / 105 / 38</code> 失联数字，一路升级至“大坝武器论”；待尼泊尔官方卫星证据（<code>X38</code>）公布后，该账号于 <code>X24</code>（<code>+13.59h</code>）果断抛弃溃坝说，无缝转进为新指控——“中国延误 45 分钟通报冰川溃决”（典型的<b>移动球门柱 / 归因漂移</b>）。</p>
      {img_tag("04_slot_matrix_and_baba_arc", "图 5 · 上：@RealBababanaras 6 小时归因漂移时间轴；下：8 类白盒话术与 67584ms 视频指纹槽位复用矩阵，红框标出 X09 与 X11 的 6/6 全同镜像搬运。")}
    </div>

    <div class="sub-block" id="network">
      <h3>3.4 明线断链、暗线成团：双模交互社交网络与五类传播机制</h3>
      <p>同一批账号在不同连边口径下暴露出截然不同的机制：<b>明线</b>看可见互动（引用、回复与 @提及），<b>暗线</b>看不可见的话术槽位与 <code>67584 ms</code> 视频指纹同步。点击下方视图按钮或<b>一键聚光灯</b>，可聚焦观察每一类识别出的隐蔽传播模式；<b>点击任意节点</b>即可在下方展开原帖全文：</p>
      <div class="network-shell">
        <div class="network-toolbar">
          <div class="mode-tabs">
            <button class="mode-btn active" data-mode="cooccurrence">暗线 · 槽位与视频指纹共现网（17 边）</button>
            <button class="mode-btn" data-mode="explicit">明线 · 显式引用与提及传播网（14 边）</button>
            <button class="mode-btn" data-mode="both">全视角叠加对照</button>
          </div>
          <div class="legend-bar">{legend_html}</div>
        </div>
        <div class="edge-legend">
          <span class="edge-key"><span class="line-swatch red-solid"></span>暗线：67584ms 同源视频镜像搬运</span>
          <span class="edge-key"><span class="line-swatch amber-dash"></span>暗线：白盒槽位时间共现</span>
          <span class="edge-key"><span class="line-swatch"></span>明线：引用 / 回复传播</span>
          <span class="edge-key"><span class="line-swatch blue-solid"></span>明线：单跳靶向辟谣</span>
          <span class="edge-key"><span class="line-swatch teal-dash"></span>明线：官方 / 科研 @提及</span>
        </div>
        <div class="spotlight-bar">
          <span class="spotlight-label">一键聚光灯：</span>
          <button class="spot-btn" data-spot="mirror">① 镜像搬运 (X09 ↔ X11)</button>
          <button class="spot-btn" data-spot="hijack">② 寄生式流量劫持 (25.56M)</button>
          <button class="spot-btn" data-spot="clique">③ 同源数字分叉团块 (291/105)</button>
          <button class="spot-btn" data-spot="debunk">④ 跨语种外溢 · 靶向辟谣 · 官方链</button>
          <button class="spot-btn reset" id="spot-reset">重置高亮</button>
        </div>
        <div class="svg-stage">
          <svg id="network-svg" viewBox="0 0 1000 520" preserveAspectRatio="xMidYMid meet"></svg>
          <div id="net-tooltip" class="net-tooltip"></div>
        </div>
        <p id="net-mode-desc" class="muted" style="font-size:12.8px;margin:12px 0 0"></p>
        <div id="node-inspector" class="node-inspector">点击上方任意节点查看完整推文与核查详情。</div>
      </div>

      <div class="pattern-grid">
        <article class="pattern-card">
          <span class="eyebrow">明线机制 1 · 局部星型辐射与跨语种外溢</span>
          <h4>以 @RealBababanaras 为枢纽向外溢出至欧洲小语种</h4>
          <p>在明线图中，<code>@RealBababanaras</code> 形成局部星型辐射中心，意大利语账号 <code>@Valeria__79</code>（<code>X21</code>）与 <code>@Praveen79066956</code>（<code>X23</code>）直接引用其溃坝推文，将原本发源于南亚与英语圈的虚假因果链进一步翻译外溢至欧洲语境。</p>
        </article>
        <article class="pattern-card">
          <span class="eyebrow">明线机制 2 · 寄生式流量劫持</span>
          <h4>低粉丝号精准搭车 2,556 万与 653 万中立灾情爆帖</h4>
          <p><code>@BlacksmithIndie</code> 自身浏览仅数百次，却精准瞄准浏览量高达 <b>25.56M</b>（<code>@TheInsiderPaper</code>）与 <b>6.53M</b>（<code>@ravipandey2643</code>）的中立纯灾情爆帖，通过引用与高排位首评植入简短情绪词“Chinese dam break 💔”，借助公域爆款流量池低成本扩散虚假归因。</p>
        </article>
        <article class="pattern-card">
          <span class="eyebrow">明线机制 3 · 单跳靶向辟谣与官方协作链</span>
          <h4>点名驳斥精准但孤立，官方背书局限于同温层</h4>
          <p>民间核查账号呈精准“单跳点名驳斥”（<code>@F1ght4dream → @tiraspeaks</code>、<code>@RushAnn1234 → @JeanMarcNoelB</code>、<code>@Corrine4o → @RadharamnDas</code>）；官方与科研节点则沿 <code>@DHM_FloodEWS → @NDRRMA_Nepal → @moha_nepal</code> 与 <code>@PTI_News → @icimod</code> 协作背书，但缺乏穿透跨语种谣言圈的二次转发扩散。</p>
        </article>
        <article class="pattern-card">
          <span class="eyebrow">暗线机制 4 &amp; 5 · 分叉污染团块与 67584ms 镜像搬运</span>
          <h4>真实数字锚点分叉变异，无引用“伪原创”现原形</h4>
          <p>1–4 小时内，<code>@ferozwala</code>、<code>@DailyExplorerX</code>、<code>@NewsAlgebraIND</code>、<code>@SanjeevCrime</code> 与 <code>@RealBababanaras</code> 同步抓取 <code>291/105</code> 真实排查名单构成暗线高密团块，随即分叉为纯灾情、地震误读与溃坝谣言，8 小时后 <code>@TicTocTick</code> 继续将美籍数字雪球推高至 60 人；更隐蔽的是 <code>@Ujwalthapa12</code>（<code>X11</code>）以无引用原创帖在明线隐身，却在暗线凭 <b>39 分钟时差、6/6 全槽位重合与 67584 ms 同源视频指纹</b>（<code>w=0.896, TF-IDF=0.761</code>）被彻底锁定为镜像搬运。</p>
        </article>
      </div>
    </div>
  </section>

  <section id="appendix">
    <div class="section-heading"><span>04 / 附录：原始证据检索与方法边界</span><h2>每一条统计结果，都能回溯到原句与原链。</h2></div>
    <div class="tab-row">
      <button class="tab-btn active" data-tab="x">X 平台定向核心样本（54 条）</button>
      <button class="tab-btn" data-tab="gdelt">GDELT 全球新闻命中（{len(gdelt_matches)} 篇）</button>
    </div>
    <div class="filters">
      <input id="search-input" placeholder="搜索账号、关键词或帖号（如 67584 / RealBababanaras / 291 / glacier）" aria-label="搜索记录">
      <select id="stance-filter" aria-label="立场筛选">
        <option value="">全部 6 类立场</option>
        <option value="dam_collapse_rumor">中国大坝溃坝谣言</option>
        <option value="geopolitical_blame_other">基建/预警延误归责</option>
        <option value="earthquake_misread">构造地震触发误读</option>
        <option value="breaking_disaster_report">纯灾情与应急通报</option>
        <option value="science_glacier_collapse">冰岩崩塌科学解释</option>
        <option value="factcheck_debunk">事实核查与辟谣</option>
      </select>
      <select id="phase-filter" aria-label="阶段筛选">
        <option value="">全部阶段</option>
        <option value="0-24h 信息真空期">0-24h 信息真空期</option>
        <option value="24-48h 核实对冲期">24-48h 核实对冲期</option>
        <option value="48-72h 二次谣言与澄清期">48-72h 二次谣言与澄清期</option>
      </select>
      <select id="lang-filter" aria-label="语种筛选">
        <option value="">全部语种</option>
        <option value="en">英语 (EN)</option>
        <option value="ne">尼泊尔语 (NE)</option>
        <option value="it">意大利语 (IT)</option>
      </select>
    </div>
    <p id="search-count" class="muted" style="font-size:12.8px"></p>
    <div id="records"></div>
    <button id="more" style="margin-top:14px">加载更多</button>

    <details class="drawer" style="margin-top:26px">
      <summary>展开查看方法边界与一键复现命令</summary>
      <p style="font-size:13.3px"><b>口径说明：</b>54 条 X 样本属按两批配额构建的定向核心代表集（谣言扩散 32 条 + 官方/科学/辟谣 22 条），带精确到秒的 UTC 时间戳，用以还原节点时序、头部曝光落差与隐蔽搬运结构，不直接外推为全站发帖绝对总量；Google Trends 历史回溯为日粒度相对指数，地域表为同国内相对占比；GDELT 采用 13 个 6 小时静态目录切片规避 DOC 2.0 限流，作为主流媒体跟进节奏的独立旁证。</p>
      <pre>python scripts/collect_gdelt.py --mode auto   # GDELT 双通道采集（DOC 熔断 + 静态桶兜底）
python scripts/analyze.py                     # 清洗校验、V(t)/H(t) 指标与双模网络构建
python scripts/render_report.py               # 生成 6 组 300dpi 图、交互 HTML 与 Markdown 报告
python -m ty check                            # 静态类型检查（All checks passed）</pre>
    </details>
  </section>
</main>

<footer>MYCOURSE / 《大数据与舆论分析导论》第 10 讲课程实践 · 图表、交互网络与 274 条原始证据均已离线内嵌。</footer>
<script id="report-payload" type="application/json">{payload_json}</script>
<script>{js}</script>
</body>
</html>
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "index.html").write_text(page, encoding="utf-8")


def build_markdown_report(
    knowledge: dict[str, Any],
    summary: dict[str, Any],
    posts: list[dict[str, str]],
    phases: list[dict[str, str]],
    co_edges: list[dict[str, str]],
    gdelt_matches: list[dict[str, str]],
) -> None:
    xstat = summary["x_dataset"]
    rumor_views = int(xstat["rumor_views"])
    en_sci_views = sum(
        int(p["views"])
        for p in posts
        if p["narrative_stance"] in ("science_glacier_collapse", "factcheck_debunk") and p["language"] == "en"
    )
    asym_ratio = rumor_views / max(en_sci_views, 1)
    top_co = co_edges[0]

    phase_table = "\n".join(
        f"| {r['phase']} | {r['total_posts']} | {r['rumor_and_blame_posts']}（{float(r['rumor_share']):.1%}） | "
        f"{r['science_and_debunk_posts']}（{float(r['science_debunk_share']):.1%}） | {float(r['attribution_entropy_bits']):.2f} |"
        for r in phases
    )

    md = f"""# 2026 年中尼边境泥石流灾害“西藏大坝溃坝”谣言：事实核查与 72 小时信息真空实证报告

> **课程**：《大数据与舆论分析导论》第 10 讲（谣言传播与事实核查）  
> **核查综合评级**：**{knowledge["target_claim"]["overall_verdict"]}**  
> **配套交互展示**：同目录 [index.html](index.html) 内嵌高清静态图、双模交互网络与 274 条原始证据检索表，可离线直接打开。

---

## 一、核查背景、问题界定与研究架构

2026 年 8 月 26 日上午，喜马拉雅山脉中尼边境的尼泊尔拉苏瓦县（Rasuwa）与中国西藏吉隆口岸走廊突发特大山洪泥石流。猛烈的碎屑流迅速摧毁沿河道路、桥梁与通信基站，致使数百名跨境旅客与当地居民一度失联。在灾后最初的 24 小时里，权威勘测与跨国信息通报存在天然时滞，海外社交平台 X 上以 `@RealBababanaras` 为代表的多组账号迅速抢占叙事窗口，将边境口岸受灾监控与失联游客名单拼接到一起，抛出“中国西藏一侧大坝垮塌引发尼泊尔山洪”的说法，在英语与南亚舆论场引发数百万次浏览。

此前已有事实核查网站（如 *The Real Fact Hunter*）围绕 `@RealBababanaras` 的两条代表性推文做过定性辟谣，但新闻调查式的文本核查尚未对灾后 72 小时的“信息真空”做出定量测量，也未系统追踪首发账号后续的叙事转向以及隐蔽搬运账号之间的协同结构。因此，本报告将工作定位为**“独立证据链核验 + 轻量计算舆论实证”**：前半部分严格遵循标准七步核查流程，交叉调取国际地球物理台网、卫星遥感、当事国行政通报与流域水工参数，对谣言的核心环节逐一证伪；后半部分则从数据出发，通过形式化指标与双模社交网络，定量刻画这场跨境灾害中的信息真空、跨语种澄清时滞与模板化搬运特征。

### 为什么采用三层互补的数据采集架构

在突发跨境灾害研究中，单一平台往往只能看到舆论生态的一个切面：社交媒体更新极快但噪声高、主流新闻报道严谨但存在数小时滞后、搜索引擎则反映了普通公众在产生困惑后的主动追问。为了让三类信号相互校准，同时避开商业接口的高昂成本，我们以美国地质调查局（USGS）记录到的崩塌震动起动时刻 `2026-08-26 02:52:10 UTC`（北京时间 `10:52:10`）作为统一物理零点 $t_0$，构建了三层互补的轻量数据采集与预处理流水线。

第一层是 **X 平台带精确时间戳的定向核心样本**。针对官方免费接口无法回溯历史推文的限制，我们编写了带严格反幻觉约束的结构化任务单，委派 Grok 按“谣言与早期扩散（Batch 1，32 条）”与“官方、科学与事实核查（Batch 2，22 条）”两批配额检索，要求逐字保留原文、原帖 URL、精确到秒的 UTC 时间、引用对象以及附件视频的毫秒级时长。合并校验后共获得 **54 条唯一 URL、40 个唯一账号**，精确覆盖灾后 $+1.64\\text{{h}}$ 至 $+63.85\\text{{h}}$。需要说明的是，这批样本属于**目的性配额抽样（purposive core sample）**而非全站随机普查，因此后续推论严格区分两层证据：物理时间滞后、GDELT 24,131 条全量扫描、Google Trends 宏观曲线以及单一账号的 6 帖漂移和 `67584 ms` 搬运物证属于**不依赖抽样配额的硬证据**；而基于 54 条样本计算出的信息赤字比 $V(t)$、归因分布熵 $H$ 与英语曝光倍数，则用于刻画代表性高影响力节点的供需结构，不直接外推为平台全站发帖的绝对占比。在预处理阶段，我们保留模型原始分类 `claimed_cause_raw`，并对 6 条明显归类偏差做逐条人工纠偏（例如将原标为冰川崩塌、实则指责中国基建或延误预警的 `X06` 与 `X24`，同 `X12 / X13 / X28 / X31` 一并归入“基建/预警延误衍生归责”），最终形成 6 类互斥叙事立场；同时剔除了 `X02`（`@ferozwala`）的自引用环，并将正文语种标定为英语（44 条）、尼泊尔语（9 条）与意大利语（1 条）。

第二层是 **GDELT 全球新闻目录切片**。为了给社交平台的谣言爆发建立主流媒体参照系，采集脚本 `collect_gdelt.py` 采用双层布尔检索式 $Q_k = B \\cap T_k$，先以基础事件词 $B$ 锁定中尼边境地名与洪灾/泥石流词汇，再分别与溃坝谣言（$T_1$）、地震误读（$T_2$）、冰川崩塌科学（$T_3$）及权威机构（$T_4$）求交集。实测中 GDELT DOC 2.0 动态接口连续两次返回 `HTTP 429` 限流，脚本随即触发自动熔断，无缝回退至无频控的 Web NGrams TOC 静态存储桶，在 `08-26 06:16Z` 至 `08-29 12:16Z` 间按每 6 小时抽取 14 个目录切片（13/14 成功），共扫描 **24,131 条**全球新闻目录，精准命中 **220 篇**灾害相关报道。

第三层是 **Google Trends 宏观搜索基线与跨国地域分布**。我们在全球范围导出 `2026-08-24` 至 `2026-09-04`（含灾前两天零基线）的主事件 4 词走势（`Nepal flood / Rasuwa / Tibet dam / glacier collapse`）与 43 国地区表。由于主词 `Nepal flood` 的搜索体量远超具体谣言词、在合图中会压扁细词波动，我们又单独导出了一组溃坝双词对照（`Tibet dam / China dam collapse`）及 13 国地区表，从而同时掌握宏观关注走势与谣言词的独立脉冲。如下图所示，灾后 Google 搜索中 `flash flood nepal`、`tibet flood` 以及“暴洪”“西藏自治区”“8月26日”等关联主题均出现爆发式增长，直观印证了公众在突发灾害初期的强烈解释需求。

![Google Trends 灾后飙升关联查询截图](figures/gt_04_rising_queries.png)

---

## 二、标准 7 步事实核查实操

### 1. 核查对象锁定与原子声明解绑（步骤 1–2）

高迷惑性谣言之所以容易在危机时刻大行其道，往往源于它保留了受众一眼可见的真实碎片，仅在因果归因环节偷梁换柱。本案中，`@RealBababanaras` 于北京时间 8 月 26 日 `18:10`、`18:41` 与 `19:09`（即 $t_0$ 后 `7.31h`、`7.81h`、`8.29h`）连续发布三条推文，将汹涌的洪水监控视频、291 名外国游客失联名单与“中国西藏大坝垮塌”紧密捆绑。若对整段话笼统判断真假，很容易陷入“画面和数字明明是真的”的认知陷阱。因此，核查的第一要务是将复合叙事拆解为三项相互独立的原子命题：

| 声明编号 | 核查维度 | 网传断言摘要 | 分项核查评级 |
| :--- | :--- | :--- | :--- |
| **C1** | 视觉影像与空间归属 | 所配监控视频展示“中国大坝溃坝冲毁尼泊尔”的现场 | **移花接木 / 剥离语境** |
| **C2** | 失联人数与国籍构成 | 溃坝造成 291 名外国游客失联（含 105 名印度人、38 名美国人等） | **数据嫁接与局部夸大** |
| **C3** | 核心物理致灾成因 | 山洪泥石流由西藏一侧中国大坝垮塌（或构造地震）引发 | **完全虚假** |

### 2. OSINT 时空溯源与四维一手独立证据交叉验证（步骤 3–5）

围绕拆解出的三个命题，我们按开源情报（OSINT）“关键帧与反向检索 → 地理定位 → 时间核验”三步法，交叉比对了四类互不隶属的一手权威记录，逐环斩断虚假因果链。

**第一维证据来自美国地质调查局（USGS）全球地震台网的一手物理监测。** 台网在 `2026-08-26 02:52:10 UTC` 捕捉到强烈地面震动信号，震中坐标定位于 `28.271°N, 85.515°E`，震源深度为 **`0.0 km`**。这一组参数同时提供了两个关键证伪锚点：空间上，该坐标精确落在**尼泊尔境内**朗塘国家公园的朗塘里荣峰（Langtang Lirung）北侧高海拔冰川带，而非西藏境内所谓水库大坝；物理机制上，零公里深度与长周期面波特征表明这是一次地表巨量物质崩塌。虽然自动速报系统最初曾短暂将其标为 `M4.4` 地震，但 USGS 经波形重分析后正式更正为**等效 `M5.2` 滑坡与冰岩崩塌事件**——这一澄清既否定了人工溃坝说，也解释了同期另一则“构造地震引发洪水”谣言为何属于典型的倒果为因。

**第二维证据来自国际山地综合开发中心（ICIMOD）的卫星遥感与冰冻圈专题调查。** ICIMOD 发布的综合报告（*Understanding the Gyirong–Rasuwa disaster of 26 August 2026: A preliminary synthesis*）还原了完整的自然成灾链条：崩塌始于朗塘里荣峰北翼海拔约 **5,200 m** 的高寒斜坡，约 **1.32 亿 $\\mathrm{{m}}^3$** 基岩与冰川冰体混合脱离山体，坠入伦德河（Lhende Kholā）峡谷短暂堵江后迅即溃决，转化为时速超过 **160 km/h** 的超高速泥石流山洪。尤为有力的是欧空局 Sentinel-1 雷达干涉测量（InSAR）的前兆记录——在灾前数周（`2026-07-07` 至 `08-24`），该失稳山体已呈现出高达 **40 cm** 的视线向渐进形变，源区灾前甚至不存在大型冰湖（排除了传统冰湖溃决洪水 GLOF），从而从孕灾前兆到物质方量完整锁定了自然地质成因。

**第三维证据来自尼泊尔旅游局（NTB）与国家减灾管理局（NDRRMA）的官方通报。** 核查原始行政口径可知，推文中反复出现的“291 名外国游客”出自 NTB 在灾后断网初期发布的紧急联络排查表：受公路冲毁与基站断电影响，共有 **384 名旅客暂时失联**，由 **93 名尼泊尔公民**和 **291 名外国公民**（含 105 名赴冈仁波齐朝圣的印度香客）构成。这份名单的性质是“断网期待联络人员”，随着军方直升机救援与通信抢通，大批旅客陆续恢复联络并获救。造谣帖不仅把自然灾害下的断网失联表强行绑定到“溃坝后果”上，还把原本极少数的美籍失联人数层层夸大错配为 38 人（后续变体甚至升至 47、54、60 人），借此拉高英语舆论圈的恐慌与关注度。

**第四维证据来自流域水电工程参数与监控视频母本溯源。** 从空间高程与水工物理来看，受灾河谷走廊内的主要水电设施——包括尼泊尔侧装机 **111 MW 的拉苏瓦嘎迪水电站（Rasuwagadhi Hydropower Project）**、下游 Chilime 与 Trishuli-3A 电站，以及中方吉隆口岸配套小水电——全系**径流引水式电站（run-of-the-river / diversion-type）**。这类工程仅靠低矮拦河闸引水入洞发电，缺乏大型调节库容，在流体体积守恒上根本无法蓄积并释放亿方级的跨境毁灭性山洪。恰恰相反，刚从 2025 年洪灾修复投产的拉苏瓦嘎迪电站地下厂房被本次泥石流灌满泥沙，吉隆口岸联检区域严重被淹；而造谣者抹去时间水印、替换音频后当作“溃坝证据”传播的视频，其母本正是**吉隆出入境边防检查站**记录下游口岸受灾的官方监控画面。

![OSINT 流域高程与空间核验图](figures/02_elevation_corridor.png)

### 3. 证据分级、CRAAP 信源评估与核查结论（步骤 6–7）

按“一手物理监测与卫星遥感 > 当事国减灾主管机关通报 > 主流媒体现场核实 > 匿名社媒转述”的证据金字塔衡量，上述四维一手证据构成了严密的交叉闭环。反观首发账号 `@RealBababanaras` 的五维 CRAAP 表现：其既无现场记者或地质水文资质（权威性极低），又在正文反复使用 `reportedly` 规避核实责任，同时抹除视频水印、错配国籍数字，且在尼泊尔官方公布卫星滑坡证据后拒不更正，反而迅速切换新指控继续煽动（准确性极低、目的具有明显议程操纵色彩）。综合三项原子命题的分项核验，本研究给出明确评级：**虚假 / 具有严重误导性（False / Severely Misleading）**。

---

## 三、轻量计算舆论实证：从信息真空到双模传播结构

完成单案证伪后，我们进一步回到《大数据与舆论分析导论》的核心追问：为什么这样一条在物理和工程上漏洞百出的虚假因果链，却能在灾后短时间内席卷数百万浏览？借助多源时序数据与三项形式化指标，我们对灾后 72 小时的传播动力学做出了实证还原。

### 1. 信息赤字比与归因分布熵：跨语种信息真空与 {asym_ratio:.1f} 倍曝光落差

突发事件中的“信息真空”并非指平台鸦雀无声，而是公众对成因的追问需求远大于权威可核实信息的供给。将 54 条 X 样本按相对 $t_0$ 的 6 小时时间窗分箱，记窗口 $t$ 内归一化后的谣言与待解灾情需求强度为 $\\tilde{{D}}(t) \\in [0, 1]$，归一化后的科学解释与辟谣供给强度为 $\\tilde{{S}}(t) \\in [0, 1]$，定义**信息赤字比**（Information Deficit Ratio）：

$$
V(t) = \\frac{{\\tilde{{D}}(t)}}{{\\tilde{{S}}(t) + \\varepsilon}}, \\qquad \\varepsilon = 0.05.
$$

平滑常数 $\\varepsilon=0.05$ 既防止供给归零时分母发散，也使无权威供给时的理论峰值封顶于 $20.0$。同时，为了观察舆论在不同成因假设间的聚集与分化，设某阶段内 5 类明确归因与辟谣立场占比为 $p_1, \\dots, p_5$，定义**归因分布熵**（Attribution Entropy）：

$$
H = -\\sum_{{k=1}}^{{5}} p_k \\log_2 p_k.
$$

按阶段汇总三项观测结果如下表所示：

| 阶段 | 样本帖数 | 谣言与归责占比 | 科学解释与辟谣占比 | 归因分布熵 $H$（bits） |
| :--- | ---: | ---: | ---: | ---: |
{phase_table}

数据揭示了一个极易被忽视的关键机制——**跨语种信息真空**。实际上，尼泊尔国家减灾管理局 `@NDRRMA_Nepal` 的响应相当迅速，早在灾后 **`+9.25h`**（`12:07 UTC`）就发布了获得 235 万浏览的 Planet Labs 卫星冰岩滑坡定性帖（`X38`），使全语种信息赤字比在 `+6–12h` 窗口控制在 $V_{{\\text{{all}}}}=2.61$。然而这条官方权威通报使用的是**尼泊尔语**；同一时段面向全球传播的英语科学澄清供给恰好为 $0$，导致**英语信息赤字比 $V_{{\\text{{en}}}}$ 直接飙升至上限 `20.0`**。直到 `+15.15h` 以后，`@theinformant_x`（`X41`）与美联社 `@AP`（`X42`）等英语科普与权威报道才陆续进场。正是在这超过 6 小时的语言断层里，定向核心样本中的英语谣言与衍生归责（26 帖）抢占了全部先验锚点，累计拿下 **`{rumor_views / 1e6:.2f}M`** 浏览，而同期收录的全部 12 条英语科学解释与辟谣帖合计仅获 **`{en_sci_views / 1e6:.2f}M`** 浏览，头部节点之间形成了高达 **`{asym_ratio:.1f}` 倍**的曝光不对称。三阶段归因分布熵也精准勾勒出“真空期溃坝论抢跑（`1.81 bits`，其中 `+6–12h` 单窗低至 `1.09 bits`）→ 次日多方解释正面碰撞（`2.25 bits` 达峰）→ 第三日向科学共识与辟谣收敛（`0.97 bits`）”的经典演化节律。

GDELT 全球新闻静态切片从主流媒体侧给出了高度一致的交叉旁证。在 13 个成功获取的 6 小时切片（24,131 条全球新闻目录、220 篇灾害命中）中，灾后 `+3.4h`（`06:16Z`）全球主流新闻命中为 **0 篇**，`+9.1h` 仅 **5 篇**，`+15.4h` 为 **9 篇**，直至次日 `+27h` 至 `+33h` 才冲高至 **21–34 篇/片**，印证了灾后首日主流媒体确处于高度供给稀缺状态。更值得强调的是对 220 篇命中标题的人工核读：共有 15 篇标题直接指出冰川或冰岩崩塌，而命中 `dam` 词的仅有 2 篇——其中 8 月 28 日的 1 篇印地语报道确实受到社交平台谣言外溢，误将福建永安水电站视频嫁接进尼泊尔洪灾报道；另一篇英文标题（*“Nepal faces secondary flash flood risk as debris creates temporary dam”*）所指的则是泥石流堵江形成的自然堰塞坝，属于典型的关键词误匹配。这一细节再次提醒我们，自动布尔检索适合做规模筛查，但结论定性必须回归原文核读。

![灾后 66 小时信息真空与叙事演化图](figures/01_info_void_timeline.png)

![跨语种曝光不对称图](figures/05_attention_asymmetry.png)

### 2. Google Trends 宏观基线与跨国搜索错位

将目光从社交平台与新闻室转向更广泛的普通网民，Google Trends 12 天曲线（含灾前 2 天零基线）清晰勾勒了公众注意力的潮起潮落。主事件词 `Nepal flood` 于 8 月 27 日攀至峰值 100；在消除主词压制的独立双谣言词对照中，`China dam collapse` 在 8 月 26 日至 27 日迅速冲顶（30–31）后快速回落，而 `Tibet dam` 则在 8 月 28 日走出二次峰值（100）——把时间轴与 X 样本对齐便会发现，28 日恰逢一段早先被 Facebook 标为 AI 生成的假溃坝视频被重新包装成“另一座中国大坝垮塌”在 X 上疯传（见 `X52`、`X53` 辟谣记录），可见二次谣言直接牵动了搜索端的次生脉冲。

地域维度的反差同样耐人寻味。对比 43 国主事件表与 13 国溃坝词表可以看出，身处震中地带的尼泊尔本土网民搜索高度聚焦于核心受灾地名 `Rasuwa`（占比 45%）与 `Nepal flood`（54%），`Tibet dam` 占比近乎为零，关注的首先是切实的灾情与救援；反倒是远离现场的海外英语圈与南亚受众成为“中国大坝溃坝”词条的主要检索群体，在双谣言词内部，`China dam collapse` 相对占比最高的前列依次为加拿大（52%）、菲律宾（41%）、尼泊尔（40%）、美国（37%）、荷兰（25%）与印度（23%）。

![Google Trends 时序与跨国地域分化图](figures/03_trends_and_regions.png)

![Google Trends 溃坝双词走势原始截图](figures/gt_02_dam_trend.png)

![Google Trends 溃坝双词全球地域热力图](figures/gt_03_dam_worldmap.png)

### 3. 归因漂移弧、白盒槽位指纹与双模传播网络

深入到账号与文本微观层面，这批数据进一步暴露了“首发号归因漂移”与“双模社交网络五类隐蔽机制”。首先是首发推手的**“6 小时归因漂移弧”（Moving the Goalposts）**。原新闻核查报道仅记录了 `@RealBababanaras` 的两条推文，而我们的精确时间戳序列完整还原了该账号在 `+7.31h` 至 `+13.59h` 之间连发 **6 帖**（`X01 → X04 → X09 → X16 → X19 → X24`）的演变轨迹：起初抛出溃坝指控（`X01`，26.9 万浏览），半小时后贴出 `67584 ms` 口岸监控（`X04`）并嫁接 `291 / 105 / 38` 失联数字（`X09`，45.8 万浏览），随后升级为“大坝能否成为战争武器”（`X16`）并抬高伤亡数字至 400 人（`X19`）；耐人寻味的是，当尼泊尔官方在 `+9.25h` 抛出 Planet Labs 卫星滑坡证据后，该账号并未为之前的溃坝假消息致歉删帖，而是在 `+13.59h` 的 `X24` 中顺滑改口承认“冰川溃决”，却立刻换上新的地缘指控——声称“中国晚了 45 分钟通知尼泊尔”。这种随着证据变化不断更换归责靶心、却始终维系对抗性情绪框架的做法，正是高阶计算宣传的典型手法。

其次，为了透视账号之间的显式传播与隐蔽搬运结构，我们构建了包含 47 个节点的**双模社交网络（Dual-Mode Social Network）**。其中，**明线传播网**仅保留跨账号的可见互动（引用 `quote`、直接回复 `reply` 与正文 `@mention`），共 14 条有向边；**暗线共现网**则专门用于捕捉无转发标注的“伪原创”同步行为。这里我们有意避开了预训练稠密句向量（Dense Embedding），原因在于同一灾害下的造谣帖与辟谣帖共享高度重合的实体词，稠密句向量余弦相似度常达 0.80 以上，容易将驳斥谣言的核查者与造谣大 V 误连成团。相反，我们剔除科学解释与辟谣帖后，仅用确定性正则提取 8 类白盒话术与媒介指纹槽位（大坝溃坝词、`reportedly` 免责词、`291` 外籍、`105` 印度人、美籍夸大数字、延迟预警指控、`M4.4/5.2` 地震词，以及 **`67584 ms` 同源监控视频指纹**），结合发帖时间差 $|\\Delta t|$ 构造完全可解释的暗线共现边权：

$$
w_{{uv}} = J^*(u, v) \\cdot \\exp\\!\\left(-\\frac{{|t_u - t_v|}}{{\\tau}}\\right), \\qquad \\tau = 6\\ \\text{{h}},
$$

其中 $J^*(u,v)$ 为两账号槽位集合的 Jaccard 重合度（若同时命中精确到毫秒的 `67584 ms` 视频指纹则叠加 $+0.35$ 媒介加权并截断于 $1$），并辅以词级 TF-IDF 余弦相似度验证字面复用程度。

把明线与暗线并置对照（见下图），五类极具代表性的传播与操纵模式清晰浮现：

1. **局部星型辐射与跨语种外溢（明线）**：`@RealBababanaras` 在明线图中充当局部星型辐射中心，意大利语账号 `@Valeria__79`（`X21`）与 `@Praveen79066956`（`X23`）直接引用其溃坝推文，将原本发源于南亚与英语舆论场的虚假因果链翻译并外溢至欧洲语境。
2. **寄生式流量劫持（明线）**：低粉丝、低浏览账号 `@BlacksmithIndie`（单帖仅数百浏览）并未单打独斗，而是精准瞄准两条中立纯灾情爆帖——`@TheInsiderPaper`（**2,556 万浏览**）与 `@ravipandey2643`（**653 万浏览**），分别以回复首评（`X17`）与引用转发（`X10`）强行植入“Chinese dam break 💔”，通过“寄生搭车”把虚假归因低成本注入千万级公域流量池。
3. **单跳靶向辟谣与官方协作链（明线）**：民间核查账号呈现精准的“单跳点名驳斥”（`@F1ght4dream → @tiraspeaks`、`@RushAnn1234 → @JeanMarcNoelB`、`@Corrine4o → @RadharamnDas`）；尼泊尔水文气象局与减灾局则沿 `@DHM_FloodEWS → @NDRRMA_Nepal → @moha_nepal` 及 `@PTI_News → @icimod` 通过 `@提及` 形成规范的官方/科研背书链。但二者多停留在单跳互动或同温层内部，缺乏穿透跨语种谣言圈的二次爆发链条。
4. **同源真实数字的分叉污染团块与雪球接力（暗线）**：暗线图中，`@ferozwala`（`X02`）、`@DailyExplorerX`（`X08`）、`@NewsAlgebraIND`（`X18`）、`@SanjeevCrime`（`X22`）与 `@RealBababanaras`（`X09`）在 1–4 小时内因共同抓取尼泊尔旅游局的 `291 名外国游客 / 105 名印度人` 排查名单，形成高度致密的橙色虚线团块。同一份真实信息源在短时间内迅速分叉为三条路径：纯报灾情（灰节点）、构造地震致洪误读（紫节点）以及中国大坝溃坝谣言（红节点）；8 小时后，`@TicTocTick`（`X26`）再次复用 `大坝 + reportedly + 美籍夸大` 三个槽位，将美籍失联人数从 38 人雪球式推高至 60 人。
5. **“明线断链、暗线锁死”的隐蔽镜像搬运账号（暗线）**：全场权重排名第一的暗线连边精准锁定了 **`{top_co['author_a']} ({top_co['record_a']})` 与 `{top_co['author_b']} ({top_co['record_b']})`**——`X11` 在明线图上不带任何转发或引用标注，完全伪装成“独立原创帖”；但在暗线共现网中，三重独立物证将其彻底钉死：与 `X09` 相隔仅 **`{top_co['delta_hours']}h`（39 分钟）**、**6/6 特征槽位完全相同**、正文连同末尾截断片段高度照搬（TF-IDF 相似度 `{top_co['tfidf_sim']}`），且附件视频时长精确同为 **`67584 ms`**，最终以 **`w = {top_co['weight']}`** 的最高权重现形。交互式双模切换与“一键聚光灯”聚焦探索可在配套页面 [index.html](index.html) 中直接操作。

![归因漂移弧与槽位复用矩阵](figures/04_slot_matrix_and_baba_arc.png)

![双模社交网络模式识别对照图](figures/06_dual_network_patterns.png)

---

## 四、方法边界与复现说明

回顾整个研究过程，三项口径边界需要明确交代：首先，54 条 X 样本是基于两批配额（谣言扩散 32 条 + 官方/科学/辟谣 22 条）构建的定向核心样本，带精确到秒的 UTC 时间戳，能够扎实支撑时间线重建、槽位指纹追踪、头部节点曝光对比与典型网络剖析，但样本内部占比与曝光倍数刻画的是代表集结构，不直接外推为平台全站发帖绝对总数；其次，Google Trends 历史回溯返回的是日粒度相对热度，地域表反映的是同一国家内各词的相对份额，与 X 精确时间戳序列属于不同观测尺度，彼此印证而不混算；最后，GDELT 全球新闻采用 13 个分层 6 小时目录切片以规避 DOC 2.0 限流，作为国际主流媒体报道节奏的独立参照。

全部原始数据保存在只读目录 `data/raw/` 并附 SHA-256 哈希校验，预处理清洗脚本（`scripts/analyze.py`）、GDELT 双通道采集脚本（`scripts/collect_gdelt.py`）与图表报告渲染脚本（`scripts/render_report.py`）均已在独立 `uv` 环境中运行并通过 `python -m ty check` 静态类型检查，任一读者均可按步骤完整复算全部 6 组图表与报告。
"""
    (REPORT_DIR / "事实核查与舆论演化报告.md").write_text(md, encoding="utf-8")


def main() -> None:
    knowledge = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    summary = json.loads((PROC_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    posts = read_csv_rows(PROC_DIR / "x_posts_cleaned.csv")
    bins = read_csv_rows(PROC_DIR / "info_void_6h.csv")
    phases = read_csv_rows(PROC_DIR / "phase_summary.csv")
    slot_rows = read_csv_rows(PROC_DIR / "slot_matrix.csv")
    explicit_edges = read_csv_rows(PROC_DIR / "network_explicit_edges.csv")
    co_edges = read_csv_rows(PROC_DIR / "network_cooccurrence_edges.csv")
    nodes = read_csv_rows(PROC_DIR / "network_nodes.csv")
    gdelt_slices = read_csv_rows(PROC_DIR / "gdelt_toc_slices.csv")
    gdelt_matches = read_csv_rows(PROC_DIR / "gdelt_toc_matches.csv")

    plot_01_info_void(bins, phases, gdelt_slices)
    plot_02_elevation_corridor(knowledge)
    plot_03_trends_and_regions()
    plot_04_slot_and_baba_arc(posts, slot_rows)
    plot_05_attention_asymmetry(posts)

    nodes_layout = compute_network_layout(nodes, explicit_edges, co_edges)
    plot_06_dual_network_patterns(nodes_layout, explicit_edges, co_edges)
    build_html_report(knowledge, summary, posts, phases, explicit_edges, co_edges, nodes_layout, gdelt_matches)
    build_markdown_report(knowledge, summary, posts, phases, co_edges, gdelt_matches)
    print(f"Rendered 6 figures + HTML report ({REPORT_DIR / 'index.html'}) + Markdown report.", flush=True)


if __name__ == "__main__":
    main()
