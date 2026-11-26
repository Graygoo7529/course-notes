"""Clean, audit, and analyze collected X posts, Google Trends, and GDELT data.

Pure Python standard library implementation (zero third-party dependencies).
Reads read-only raw files from data/raw/ and writes deterministic CSV/JSON
outputs to data/processed/ for downstream plotting and HTML report rendering.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EVIDENCE_PATH = ROOT / "evidence" / "factcheck_knowledge.json"

# Manual, auditable stance corrections for Grok raw auto-labels.
# Every correction preserves the original `claimed_cause_raw` alongside the reason.
STANCE_OVERRIDES: dict[str, tuple[str, str]] = {
    "X06": (
        "geopolitical_blame_other",
        "原文主张中国喜马拉雅小康村基建引发雪崩，属地缘基建归责而非科学冰川崩塌解释",
    ),
    "X12": (
        "geopolitical_blame_other",
        "原文直指中国喜马拉雅工程导致洪水，属地缘基建归责",
    ),
    "X13": (
        "geopolitical_blame_other",
        "原文声称西藏500座大坝构成威胁并指控蓄意惩罚尼泊尔，属地缘阴谋归责",
    ),
    "X24": (
        "geopolitical_blame_other",
        "首发造谣账号从溃坝论转向指责中国延误45分钟通报冰川溃决（归因漂移）",
    ),
    "X28": (
        "geopolitical_blame_other",
        "原文聚焦“中国未提前2小时预警”指控，属预警延误归责",
    ),
    "X31": (
        "geopolitical_blame_other",
        "原文借施工爆破视频将洪水归咎于中国水电建设与西藏建坝",
    ),
}

RAW_TO_STANCE: dict[str, str] = {
    "chinese_dam_collapse": "dam_collapse_rumor",
    "earthquake": "earthquake_misread",
    "glacier_rock_collapse": "science_glacier_collapse",
    "debunk_clarification": "factcheck_debunk",
    "unattributed_report": "breaking_disaster_report",
}

STANCE_LABELS_CN: dict[str, str] = {
    "dam_collapse_rumor": "中国大坝溃坝谣言",
    "geopolitical_blame_other": "基建/预警延误衍生归责",
    "earthquake_misread": "构造地震触发误读",
    "breaking_disaster_report": "纯灾情与应急通报",
    "science_glacier_collapse": "冰岩崩塌科学解释",
    "factcheck_debunk": "事实核查与辟谣澄清",
}

# Deterministic white-box slot definitions (focused on rumor templates & media fingerprints,
# deliberately excluding generic geography like 'Tibet' or 'Rasuwa').
SLOT_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "chinese_dam",
        "中国大坝溃坝归因",
        re.compile(r"(chinese\s+dam|china[’']?s?\s+dam|china\s+dam|dam\s+collaps|dam\s+burst|dam\s+bust|dam\s+break|collasso\s+di\s+una\s+diga|bad\s+quality\s+of\s+dam|low\s+quality\s+dams)", re.IGNORECASE),
    ),
    (
        "hedging_reportedly",
        "免责模糊话术(reportedly)",
        re.compile(r"\b(reportedly|is\s+reported\s+to|are\s+said\s+to|locals.*saying|reports\s+also\s+allege)\b", re.IGNORECASE),
    ),
    (
        "tourists_291",
        "291名外籍游客失联槽位",
        re.compile(r"\b291\b", re.IGNORECASE),
    ),
    (
        "indians_105",
        "105名印度公民槽位",
        re.compile(r"\b105\s+indian", re.IGNORECASE),
    ),
    (
        "us_inflated",
        "美籍失联夸大槽位(38/47/54/60)",
        re.compile(r"\b(38\s+u\.?s\.?|47\s+americans|54\s+americans|60\s+americans)\b", re.IGNORECASE),
    ),
    (
        "delayed_warning",
        "预警延误/蓄意归责槽位",
        re.compile(r"(45\s+minutes\s+late|didn[’']t\s+inform|not\s+warned\s+earlier|weapons\s+of\s+war|punished\s+nepal|no\s+information\s+was\s+provided)", re.IGNORECASE),
    ),
    (
        "quake_m44_m52",
        "M4.4/5.2地震信号槽位",
        re.compile(r"\b(m4\.4|4\.4-magnitude|magnitude\s+4\.4|m5\.2|magnitude\s+5\.2|earthquake\s+triggers)\b", re.IGNORECASE),
    ),
]

CCT_VIDEO_MS = "67584"


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_language(text: str) -> str:
    # Devanagari Unicode block for Nepali official tweets
    if re.search(r"[\u0900-\u097F]", text):
        return "ne"
    if re.search(r"\b(ecco|collasso|diga)\b", text.lower()):
        return "it"
    return "en"


def extract_video_durations_ms(media_note: str | None) -> list[str]:
    if not media_note:
        return []
    # Match patterns like 'duration 67584 ms' or 'durations 74496, 39729, 69265, and 67477 ms'
    found: list[str] = []
    for m in re.finditer(r"duration[s]?\s+([0-9,\s]+(?:and\s+\d+)?)\s*ms", media_note, re.IGNORECASE):
        found.extend(re.findall(r"\d{4,7}", m.group(1)))
    return found


def extract_mentions(text: str) -> list[str]:
    handles = re.findall(r"@[A-Za-z0-9_]{2,20}", text)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in handles:
        low = h.lower()
        if low not in seen:
            seen.add(low)
            out.append(h)
    return out


def assign_phase(hours_since_t0: float) -> str:
    if hours_since_t0 < 24.0:
        return "0-24h 信息真空期"
    if hours_since_t0 < 48.0:
        return "24-48h 核实对冲期"
    return "48-72h 二次谣言与澄清期"


def shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return round(h, 4)


def tokenize_for_tfidf(text: str) -> list[str]:
    clean = re.sub(r"https?://\S+", " ", text.lower())
    clean = re.sub(r"@[a-z0-9_]+", " ", clean)
    words = re.findall(r"[a-z0-9]{2,}", clean)
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have", "has",
        "are", "was", "were", "been", "into", "after", "near", "about", "there",
        "they", "their", "them", "its", "not", "but", "all", "can", "may",
        "more", "also", "other", "over", "some", "out", "now", "one", "two",
    }
    return [w for w in words if w not in stop]


def build_tfidf_vectors(docs: dict[str, str]) -> dict[str, dict[str, float]]:
    tokenized = {k: tokenize_for_tfidf(v) for k, v in docs.items()}
    df: Counter[str] = Counter()
    for tokens in tokenized.values():
        for tok in set(tokens):
            df[tok] += 1
    n_docs = max(len(docs), 1)
    vecs: dict[str, dict[str, float]] = {}
    for doc_id, tokens in tokenized.items():
        tf = Counter(tokens)
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for tok, cnt in tf.items():
            weight = (1.0 + math.log(cnt)) * (math.log((n_docs + 1.0) / (df[tok] + 1.0)) + 1.0)
            vec[tok] = weight
            norm_sq += weight * weight
        norm = math.sqrt(norm_sq) or 1.0
        vecs[doc_id] = {t: w / norm for t, w in vec.items()}
    return vecs


def cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(w * v2.get(t, 0.0) for t, w in v1.items())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_trends_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned: dict[str, Any] = {}
            for k, v in row.items():
                if k is None:
                    continue
                val = (v or "").strip()
                if k in ("Time", "Region"):
                    cleaned[k] = val
                else:
                    cleaned[k] = 0.5 if val == "<1" else int(val) if val.isdigit() else 0
            rows.append(cleaned)
    return rows


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    t0 = parse_utc(str(cfg["event"]["t0_utc"]))
    bin_hours = int(cfg["event"].get("bin_hours", 6))
    epsilon = float(cfg["metrics"].get("deficit_epsilon", 0.05))
    tau_hours = float(cfg["metrics"].get("cooccurrence_tau_hours", 6.0))

    raw_posts_path = RAW_DIR / "grok_x_posts.json"
    raw_posts: list[dict[str, Any]] = json.loads(raw_posts_path.read_text(encoding="utf-8"))

    # 1. Integrity check & clean preprocessing
    seen_urls: set[str] = set()
    cleaned_posts: list[dict[str, Any]] = []

    for item in raw_posts:
        url = str(item["post_url"])
        if url in seen_urls:
            raise ValueError(f"Duplicate post_url detected: {url}")
        seen_urls.add(url)

        rec_id = str(item["record_id"])
        created = parse_utc(str(item["created_at_utc"]))
        hours = round((created - t0).total_seconds() / 3600.0, 2)
        text = str(item.get("text_verbatim") or "")
        media_note = str(item.get("media_note") or "")
        raw_cause = str(item.get("claimed_cause") or "unattributed_report")

        if rec_id in STANCE_OVERRIDES:
            stance, reason = STANCE_OVERRIDES[rec_id]
        else:
            stance = RAW_TO_STANCE.get(raw_cause, "breaking_disaster_report")
            reason = ""

        # White-box slot extraction
        slots: list[str] = []
        for slot_id, _, pat in SLOT_RULES:
            if pat.search(text):
                slots.append(slot_id)
        durations = extract_video_durations_ms(media_note)
        has_cctv_67584 = CCT_VIDEO_MS in durations
        if has_cctv_67584:
            slots.append("cctv_67584ms")

        metrics = item.get("metrics") or {}
        likes = int(metrics.get("likes") or 0)
        reposts = int(metrics.get("reposts") or 0)
        replies = int(metrics.get("replies") or 0)
        views = int(metrics.get("views") or 0)
        engagement = likes + reposts + replies

        author = str(item["author_handle"])
        ref_raw = item.get("referenced_handle")
        ref_handle = str(ref_raw) if ref_raw else ""
        is_self_quote = bool(ref_handle and ref_handle.lower() == author.lower())
        mentions = [m for m in extract_mentions(text) if m.lower() != author.lower()]

        cleaned_posts.append(
            {
                "record_id": rec_id,
                "batch": str(item.get("batch") or ""),
                "post_url": url,
                "author_handle": author,
                "author_category": str(item.get("author_category") or ""),
                "created_at_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hours_since_t0": hours,
                "bin_6h": int(max(0, math.floor(hours / bin_hours))) * bin_hours,
                "phase": assign_phase(hours),
                "language": detect_language(text),
                "post_type": str(item.get("post_type") or "original"),
                "referenced_handle": "" if is_self_quote else ref_handle,
                "is_self_quote": int(is_self_quote),
                "mentions": "|".join(mentions),
                "claimed_cause_raw": raw_cause,
                "narrative_stance": stance,
                "stance_cn": STANCE_LABELS_CN[stance],
                "stance_correction_reason": reason,
                "slots": "|".join(slots),
                "slot_count": len(slots),
                "has_cctv_67584ms": int(has_cctv_67584),
                "video_durations_ms": "|".join(durations),
                "has_media": int(bool(item.get("has_video_or_image"))),
                "likes": likes,
                "reposts": reposts,
                "replies": replies,
                "views": views,
                "engagement": engagement,
                "text_verbatim": text,
                "media_note": media_note,
            }
        )

    cleaned_posts.sort(key=lambda r: float(r["hours_since_t0"]))
    write_csv(PROCESSED_DIR / "x_posts_cleaned.csv", list(cleaned_posts[0].keys()), cleaned_posts)

    # 2. Time-window Information Deficit Ratio V(t) & Attribution Entropy H(t)
    # 6-hour bins from 0h to 66h (covering +1.64h to +63.85h)
    max_bin = 60
    bin_starts = list(range(0, max_bin + 1, bin_hours))
    bin_rows: list[dict[str, Any]] = []

    raw_demand_posts: list[float] = []
    raw_demand_logviews: list[float] = []
    raw_supply_all: list[float] = []
    raw_supply_en: list[float] = []

    rumor_stances = {"dam_collapse_rumor", "geopolitical_blame_other", "earthquake_misread"}
    supply_stances = {"science_glacier_collapse", "factcheck_debunk"}

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cleaned_posts:
        grouped[int(row["bin_6h"])].append(row)

    for b in bin_starts:
        items = grouped.get(b, [])
        cnt = Counter(str(r["narrative_stance"]) for r in items)
        d_posts = sum(cnt[s] for s in rumor_stances) + 0.5 * cnt["breaking_disaster_report"]
        d_logv = sum(math.log10(int(r["views"]) + 10) for r in items if str(r["narrative_stance"]) in rumor_stances | {"breaking_disaster_report"})
        s_all = sum(cnt[s] for s in supply_stances)
        s_en = sum(1 for r in items if str(r["narrative_stance"]) in supply_stances and str(r["language"]) == "en")

        raw_demand_posts.append(d_posts)
        raw_demand_logviews.append(d_logv)
        raw_supply_all.append(float(s_all))
        raw_supply_en.append(float(s_en))

        bin_rows.append(
            {
                "bin_start_h": b,
                "bin_end_h": b + bin_hours,
                "bin_label": f"+{b}–{b + bin_hours}h",
                "total_posts": len(items),
                "dam_collapse_rumor": cnt["dam_collapse_rumor"],
                "geopolitical_blame_other": cnt["geopolitical_blame_other"],
                "earthquake_misread": cnt["earthquake_misread"],
                "breaking_disaster_report": cnt["breaking_disaster_report"],
                "science_glacier_collapse": cnt["science_glacier_collapse"],
                "factcheck_debunk": cnt["factcheck_debunk"],
                "rumor_total": sum(cnt[s] for s in rumor_stances),
                "verified_supply_all": s_all,
                "verified_supply_en": s_en,
                "total_views": sum(int(r["views"]) for r in items),
                "rumor_views": sum(int(r["views"]) for r in items if str(r["narrative_stance"]) in rumor_stances),
                "verified_views": sum(int(r["views"]) for r in items if str(r["narrative_stance"]) in supply_stances),
                "attribution_entropy_bits": shannon_entropy([cnt[s] for s in rumor_stances | supply_stances]),
            }
        )

    max_d = max(raw_demand_posts) or 1.0
    max_d_lv = max(raw_demand_logviews) or 1.0
    max_s_all = max(raw_supply_all) or 1.0
    max_s_en = max(raw_supply_en) or 1.0

    for i, brow in enumerate(bin_rows):
        d_norm = round(raw_demand_posts[i] / max_d, 4)
        d_lv_norm = round(raw_demand_logviews[i] / max_d_lv, 4)
        s_all_norm = round(raw_supply_all[i] / max_s_all, 4)
        s_en_norm = round(raw_supply_en[i] / max_s_en, 4)
        brow["demand_norm"] = d_norm
        brow["demand_logviews_norm"] = d_lv_norm
        brow["supply_all_norm"] = s_all_norm
        brow["supply_en_norm"] = s_en_norm
        brow["deficit_ratio_all"] = round(d_norm / (s_all_norm + epsilon), 3)
        brow["deficit_ratio_en"] = round(d_norm / (s_en_norm + epsilon), 3)

    write_csv(PROCESSED_DIR / "info_void_6h.csv", list(bin_rows[0].keys()), bin_rows)

    # Phase-level attribution entropy & stance summary
    phases = ["0-24h 信息真空期", "24-48h 核实对冲期", "48-72h 二次谣言与澄清期"]
    phase_rows: list[dict[str, Any]] = []
    for ph in phases:
        sub = [r for r in cleaned_posts if r["phase"] == ph]
        cnt = Counter(str(r["narrative_stance"]) for r in sub)
        attr_counts = [
            cnt["dam_collapse_rumor"],
            cnt["geopolitical_blame_other"],
            cnt["earthquake_misread"],
            cnt["science_glacier_collapse"],
            cnt["factcheck_debunk"],
        ]
        rumor_n = cnt["dam_collapse_rumor"] + cnt["geopolitical_blame_other"] + cnt["earthquake_misread"]
        sci_n = cnt["science_glacier_collapse"] + cnt["factcheck_debunk"]
        phase_rows.append(
            {
                "phase": ph,
                "total_posts": len(sub),
                "rumor_and_blame_posts": rumor_n,
                "science_and_debunk_posts": sci_n,
                "breaking_posts": cnt["breaking_disaster_report"],
                "dam_collapse_rumor": cnt["dam_collapse_rumor"],
                "geopolitical_blame_other": cnt["geopolitical_blame_other"],
                "earthquake_misread": cnt["earthquake_misread"],
                "science_glacier_collapse": cnt["science_glacier_collapse"],
                "factcheck_debunk": cnt["factcheck_debunk"],
                "rumor_share": round(rumor_n / max(len(sub), 1), 4),
                "science_debunk_share": round(sci_n / max(len(sub), 1), 4),
                "attribution_entropy_bits": shannon_entropy(attr_counts),
                "rumor_views": sum(int(r["views"]) for r in sub if str(r["narrative_stance"]) in rumor_stances),
                "science_debunk_views": sum(int(r["views"]) for r in sub if str(r["narrative_stance"]) in supply_stances),
            }
        )
    write_csv(PROCESSED_DIR / "phase_summary.csv", list(phase_rows[0].keys()), phase_rows)

    # 3. Dual-mode Social Network Construction (Explicit Propagation + Implicit Slot Co-occurrence)
    # 3a. Explicit edges (cross-account quote/reply + @mentions)
    explicit_edges: list[dict[str, Any]] = []
    for row in cleaned_posts:
        src = str(row["author_handle"])
        ref = str(row["referenced_handle"])
        if ref:
            explicit_edges.append(
                {
                    "source": src,
                    "target": ref,
                    "edge_type": str(row["post_type"]),
                    "record_id": str(row["record_id"]),
                    "hours_since_t0": row["hours_since_t0"],
                    "source_stance": str(row["narrative_stance"]),
                    "weight": 1.0,
                }
            )
        for m in [x for x in str(row["mentions"]).split("|") if x]:
            if m.lower() == ref.lower():
                continue
            explicit_edges.append(
                {
                    "source": src,
                    "target": m,
                    "edge_type": "mention",
                    "record_id": str(row["record_id"]),
                    "hours_since_t0": row["hours_since_t0"],
                    "source_stance": str(row["narrative_stance"]),
                    "weight": 0.5,
                }
            )

    write_csv(
        PROCESSED_DIR / "network_explicit_edges.csv",
        ["source", "target", "edge_type", "record_id", "hours_since_t0", "source_stance", "weight"],
        explicit_edges,
    )

    # 3b. Implicit slot + video fingerprint + time-decay co-occurrence edges
    # Strictly exclude factcheck_debunk & science_glacier_collapse so debunk quotes never cluster with rumors
    eligible = [
        r for r in cleaned_posts
        if str(r["narrative_stance"]) not in supply_stances and int(r["slot_count"]) >= 1
    ]
    tfidf_vecs = build_tfidf_vectors({str(r["record_id"]): str(r["text_verbatim"]) for r in eligible})

    co_edges_raw: list[dict[str, Any]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            r1, r2 = eligible[i], eligible[j]
            if str(r1["author_handle"]).lower() == str(r2["author_handle"]).lower():
                continue
            s1 = {x for x in str(r1["slots"]).split("|") if x}
            s2 = {x for x in str(r2["slots"]).split("|") if x}
            shared = sorted(s1 & s2)
            union = s1 | s2
            same_cctv = "cctv_67584ms" in shared
            if len(shared) < 2 and not same_cctv:
                continue
            dt_h = abs(float(r1["hours_since_t0"]) - float(r2["hours_since_t0"]))
            jaccard = len(shared) / max(len(union), 1)
            if same_cctv:
                jaccard = min(1.0, jaccard + 0.35)
            time_kernel = math.exp(-dt_h / tau_hours)
            weight = round(jaccard * time_kernel, 4)
            if weight < 0.12 and not same_cctv:
                continue
            tsim = round(cosine_sim(tfidf_vecs[str(r1["record_id"])], tfidf_vecs[str(r2["record_id"])]), 4)
            co_edges_raw.append(
                {
                    "author_a": str(r1["author_handle"]),
                    "author_b": str(r2["author_handle"]),
                    "record_a": str(r1["record_id"]),
                    "record_b": str(r2["record_id"]),
                    "shared_slots": "|".join(shared),
                    "shared_count": len(shared),
                    "same_cctv_67584ms": int(same_cctv),
                    "delta_hours": round(dt_h, 2),
                    "jaccard": round(jaccard, 4),
                    "time_kernel": round(time_kernel, 4),
                    "weight": weight,
                    "tfidf_sim": tsim,
                }
            )

    # Deduplicate author pairs keeping strongest co-occurrence link
    best_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in co_edges_raw:
        pair_key = tuple(sorted([edge["author_a"], edge["author_b"]]))
        prev = best_pair.get(pair_key)
        if prev is None or float(edge["weight"]) > float(prev["weight"]):
            best_pair[pair_key] = edge

    co_edges = sorted(best_pair.values(), key=lambda x: float(x["weight"]), reverse=True)
    if co_edges:
        write_csv(PROCESSED_DIR / "network_cooccurrence_edges.csv", list(co_edges[0].keys()), co_edges)

    # 3c. Unified node table (all authors in dataset + external referenced handles)
    author_posts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cleaned_posts:
        author_posts[str(row["author_handle"])].append(row)

    all_handles: set[str] = set(author_posts.keys())
    for e in explicit_edges:
        all_handles.add(str(e["source"]))
        all_handles.add(str(e["target"]))

    exp_in: Counter[str] = Counter(str(e["target"]) for e in explicit_edges)
    exp_out: Counter[str] = Counter(str(e["source"]) for e in explicit_edges)
    co_deg: Counter[str] = Counter()
    for e in co_edges:
        co_deg[str(e["author_a"])] += 1
        co_deg[str(e["author_b"])] += 1

    node_rows: list[dict[str, Any]] = []
    for handle in sorted(all_handles):
        posts = author_posts.get(handle, [])
        if posts:
            primary_stance = Counter(str(p["narrative_stance"]) for p in posts).most_common(1)[0][0]
            earliest_h = min(float(p["hours_since_t0"]) for p in posts)
            total_v = sum(int(p["views"]) for p in posts)
            total_eng = sum(int(p["engagement"]) for p in posts)
            category = str(posts[0]["author_category"])
            all_slots = sorted({s for p in posts for s in str(p["slots"]).split("|") if s})
            rep_post = max(posts, key=lambda p: int(p["views"]))
            sample_id = str(rep_post["record_id"])
            sample_text = str(rep_post["text_verbatim"])
            in_dataset = 1
        else:
            primary_stance = "external_referenced"
            earliest_h = -1.0
            total_v = 0
            total_eng = 0
            category = "referenced_target"
            all_slots = []
            sample_id = ""
            sample_text = "（数据集外部被引用/提及账号）"
            in_dataset = 0

        node_rows.append(
            {
                "handle": handle,
                "in_dataset": in_dataset,
                "post_count": len(posts),
                "author_category": category,
                "primary_stance": primary_stance,
                "primary_stance_cn": STANCE_LABELS_CN.get(primary_stance, "外部被提及节点"),
                "earliest_hours_since_t0": earliest_h,
                "total_views": total_v,
                "total_engagement": total_eng,
                "explicit_in_degree": exp_in[handle],
                "explicit_out_degree": exp_out[handle],
                "cooccurrence_degree": co_deg[handle],
                "slots": "|".join(all_slots),
                "rep_record_id": sample_id,
                "rep_text": sample_text,
            }
        )

    write_csv(PROCESSED_DIR / "network_nodes.csv", list(node_rows[0].keys()), node_rows)

    # 4. Slot matrix table for heatmap visualization (rumor / early posts with >= 1 slot)
    slot_ids = [s[0] for s in SLOT_RULES] + ["cctv_67584ms"]
    slot_matrix_rows: list[dict[str, Any]] = []
    for row in cleaned_posts:
        r_slots = {x for x in str(row["slots"]).split("|") if x}
        if not r_slots:
            continue
        entry: dict[str, Any] = {
            "record_id": row["record_id"],
            "author_handle": row["author_handle"],
            "hours_since_t0": row["hours_since_t0"],
            "narrative_stance": row["narrative_stance"],
            "stance_cn": row["stance_cn"],
            "views": row["views"],
        }
        for sid in slot_ids:
            entry[sid] = int(sid in r_slots)
        slot_matrix_rows.append(entry)
    write_csv(PROCESSED_DIR / "slot_matrix.csv", list(slot_matrix_rows[0].keys()), slot_matrix_rows)

    # 5. Google Trends & GDELT summary integration
    trends_main = read_trends_csv(RAW_DIR / "google_trends.csv")
    trends_dam = read_trends_csv(RAW_DIR / "google_trends_dam.csv")
    region_main = read_trends_csv(RAW_DIR / "by_region.csv")
    region_dam = read_trends_csv(RAW_DIR / "by_region_dam.csv")

    gdelt_summary_path = PROCESSED_DIR / "gdelt_latest_summary.json"
    gdelt_summary = json.loads(gdelt_summary_path.read_text(encoding="utf-8")) if gdelt_summary_path.exists() else None

    # Key summary JSON
    baba_posts = [r for r in cleaned_posts if str(r["author_handle"]) == "@RealBababanaras"]
    cctv_posts = [r for r in cleaned_posts if int(r["has_cctv_67584ms"]) == 1]

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "t0_utc": cfg["event"]["t0_utc"],
        "raw_files": {
            f.name: {"bytes": f.stat().st_size, "sha256_12": sha256_file(f)[:12]}
            for f in sorted(RAW_DIR.iterdir())
            if f.is_file() and f.name != ".gitkeep"
        },
        "x_dataset": {
            "total_posts": len(cleaned_posts),
            "unique_authors": len(author_posts),
            "stance_counts": dict(Counter(str(r["narrative_stance"]) for r in cleaned_posts)),
            "language_counts": dict(Counter(str(r["language"]) for r in cleaned_posts)),
            "stance_overrides_applied": len(STANCE_OVERRIDES),
            "time_span_hours": [cleaned_posts[0]["hours_since_t0"], cleaned_posts[-1]["hours_since_t0"]],
            "total_views": sum(int(r["views"]) for r in cleaned_posts),
            "rumor_views": sum(int(r["views"]) for r in cleaned_posts if str(r["narrative_stance"]) in rumor_stances),
            "science_debunk_views": sum(int(r["views"]) for r in cleaned_posts if str(r["narrative_stance"]) in supply_stances),
            "baba_arc_count": len(baba_posts),
            "baba_arc_records": [r["record_id"] for r in baba_posts],
            "cctv_67584ms_records": [r["record_id"] for r in cctv_posts],
            "first_official_science_ne": next(
                (r["record_id"] + f" (+{r['hours_since_t0']}h)" for r in cleaned_posts if r["narrative_stance"] == "science_glacier_collapse" and r["language"] == "ne"),
                "",
            ),
            "first_official_science_en": next(
                (r["record_id"] + f" (+{r['hours_since_t0']}h)" for r in cleaned_posts if r["narrative_stance"] == "science_glacier_collapse" and r["language"] == "en"),
                "",
            ),
        },
        "networks": {
            "total_nodes": len(node_rows),
            "explicit_edges": len(explicit_edges),
            "explicit_quote_reply_edges": sum(1 for e in explicit_edges if e["edge_type"] in ("quote", "reply")),
            "cooccurrence_edges": len(co_edges),
            "top_cooccurrence_pairs": co_edges[:8],
        },
        "phases": phase_rows,
        "google_trends": {
            "days_main": len(trends_main),
            "days_dam": len(trends_dam),
            "regions_main": len(region_main),
            "regions_dam": len(region_dam),
        },
        "gdelt": gdelt_summary,
    }
    (PROCESSED_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["x_dataset"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["networks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
