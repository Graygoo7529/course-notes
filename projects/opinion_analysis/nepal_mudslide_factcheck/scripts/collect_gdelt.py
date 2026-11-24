"""Auditable GDELT news probe for the 2026 Nepal-China border mudslide study.

Uses only the Python standard library. Archives raw responses, HTTP statuses,
and SHA-256 digests so every result or rate-limit failure is reproducible.

Channels:
  1. GDELT DOC 2.0 API (TimelineVolRaw + ArtList) for base + attribution queries.
  2. GDELT Web NGrams TOC static fallback (https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/)
     when DOC 2.0 returns HTTP 429 or is throttled.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
RAW_DIR = ROOT / "data" / "raw" / "gdelt"
PROCESSED_DIR = ROOT / "data" / "processed"
DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
NGRAMS_BASE = "https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams"
USER_AGENT = "MyCourse-FactCheck-Probe/1.0 (academic coursework)"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_bytes(url: str, timeout: int = 25, max_bytes: int = 12 * 1024 * 1024) -> tuple[str, int | None, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = int(resp.status)
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                return "size_limit", status_code, data[:max_bytes], f"Exceeded {max_bytes} bytes"
            return "ok", status_code, data, ""
    except urllib.error.HTTPError as exc:
        body = exc.read(8192)
        return "http_error", int(exc.code), body, str(exc)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return "network_error", None, b"", f"{type(exc).__name__}: {exc}"


def build_doc_queries(cfg: dict[str, Any]) -> list[dict[str, str]]:
    base = str(cfg["queries"]["base"])
    items: list[dict[str, str]] = [{"query_id": "base", "query": base}]
    for key, clause in cfg["queries"]["subqueries"].items():
        items.append({"query_id": str(key), "query": f"{base} {clause}"})
    return items


def run_doc_channel(cfg: dict[str, Any], out_dir: Path, sleep_sec: float = 5.5) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = build_doc_queries(cfg)
    start_dt = datetime(2026, 8, 25, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 2, 0, 0, 0, tzinfo=timezone.utc)
    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str = (end_dt - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")

    tasks: list[tuple[str, str, str]] = []
    for q in queries:
        tasks.append((q["query_id"], "TimelineVolRaw", q["query"]))
    for qid in ("base", "T1_dam_rumor", "T3_glacier_science"):
        qtext = next(x["query"] for x in queries if x["query_id"] == qid)
        tasks.append((qid, "ArtList", qtext))

    logs: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    article_rows: list[dict[str, Any]] = []
    consecutive_429 = 0
    ok_count = 0

    for idx, (qid, mode, query) in enumerate(tasks, start=1):
        params: dict[str, str] = {
            "query": query,
            "mode": mode,
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
        }
        if mode == "ArtList":
            params["maxrecords"] = "50"
            params["sort"] = "DateAsc"
        else:
            params["timelinesmooth"] = "0"

        url = f"{DOC_ENDPOINT}?{urllib.parse.urlencode(params)}"
        t0 = time.monotonic()
        status, http_status, body, err = fetch_bytes(url)
        elapsed = round(time.monotonic() - t0, 3)

        raw_name = f"doc_{idx:02d}_{qid}_{mode}.body"
        (out_dir / raw_name).write_bytes(body)

        item_count = 0
        if status == "ok":
            try:
                parsed = json.loads(body.decode("utf-8"))
                if mode == "TimelineVolRaw":
                    series_list = parsed.get("timeline", [])
                    for series in series_list:
                        series_name = str(series.get("series", ""))
                        for pt in series.get("data", []):
                            timeline_rows.append(
                                {
                                    "query_id": qid,
                                    "series": series_name,
                                    "date_raw": str(pt.get("date", "")),
                                    "value": pt.get("value", 0),
                                    "norm": pt.get("norm", ""),
                                }
                            )
                            item_count += 1
                else:
                    articles = parsed.get("articles", [])
                    for art in articles:
                        article_rows.append(
                            {
                                "query_id": qid,
                                "seendate": str(art.get("seendate", "")),
                                "title": str(art.get("title", "")),
                                "url": str(art.get("url", "")),
                                "domain": str(art.get("domain", "")),
                                "language": str(art.get("language", "")),
                                "sourcecountry": str(art.get("sourcecountry", "")),
                            }
                        )
                        item_count += 1
                ok_count += 1
                consecutive_429 = 0
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                status = "invalid_json"
                err = f"{type(exc).__name__}: {exc}"
        elif http_status == 429:
            consecutive_429 += 1

        entry = {
            "request_id": f"doc_{idx:02d}",
            "query_id": qid,
            "mode": mode,
            "requested_at_utc": utc_now_iso(),
            "status": status,
            "http_status": http_status,
            "elapsed_seconds": elapsed,
            "item_count": item_count,
            "bytes_saved": len(body),
            "sha256_12": sha256_hex(body)[:12],
            "raw_file": raw_name,
            "error": err,
            "url": url,
        }
        logs.append(entry)
        print(f"[DOC] {entry['request_id']} {qid} {mode}: {status} http={http_status} items={item_count}", flush=True)

        if consecutive_429 >= 2:
            print("[DOC] Two consecutive HTTP 429 responses; stopping DOC pilot early.", flush=True)
            break
        if idx < len(tasks):
            time.sleep(sleep_sec)

    write_json(out_dir / "doc_requests.json", logs)
    if timeline_rows:
        write_csv(
            PROCESSED_DIR / "gdelt_doc_timeline.csv",
            ["query_id", "series", "date_raw", "value", "norm"],
            timeline_rows,
        )
    if article_rows:
        write_csv(
            PROCESSED_DIR / "gdelt_doc_articles.csv",
            ["query_id", "seendate", "title", "url", "domain", "language", "sourcecountry"],
            article_rows,
        )

    return {
        "channel": "doc_v2",
        "attempted": len(logs),
        "succeeded": ok_count,
        "timeline_points": len(timeline_rows),
        "articles": len(article_rows),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_title(title: str) -> dict[str, bool]:
    low = title.lower()
    place = bool(re.search(r"\b(nepal|tibet|xizang|gyirong|rasuwa|rasuwagadhi|lhende|bhotekoshi|bhote koshi|langtang)\b", low))
    hazard = bool(re.search(r"\b(flood|floods|flooding|mudslide|landslide|debris|avalanche|glacier|glacial)\b", low))
    base_hit = place and hazard
    return {
        "base": base_hit,
        "T1_dam_rumor": base_hit and bool(re.search(r"\b(dam|dams|dam burst|dam collapse|dam break)\b", low)),
        "T2_seismic_misread": base_hit and bool(re.search(r"\b(earthquake|quake|seismic|tremor|m4\.4|m5\.2)\b", low)),
        "T3_glacier_science": base_hit and bool(re.search(r"\b(glacier|glacial|ice-rock|ice rock|avalanche|langtang|icimod)\b", low)),
        "T4_official_supply": base_hit and bool(re.search(r"\b(usgs|ndrrma|icimod|tourism board|ministry|official|officials|police|army|satellite)\b", low)),
    }


def generate_ngram_stamps() -> list[str]:
    """Stratified 6-hour slices from 2026-08-26 06:16Z to 2026-08-29 12:16Z (14 slices)."""
    cur = datetime(2026, 8, 26, 6, 16, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 29, 12, 16, 0, tzinfo=timezone.utc)
    stamps: list[str] = []
    while cur <= end:
        stamps.append(cur.strftime("%Y%m%d%H%M%S"))
        cur += timedelta(hours=6)
    return stamps


def run_ngrams_toc_channel(out_dir: Path, sleep_sec: float = 1.0) -> dict[str, Any]:
    """Download lightweight stratified Web NGrams TOC slices (~230 KB each) and match locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamps = generate_ngram_stamps()
    logs: list[dict[str, Any]] = []
    slice_summary: list[dict[str, Any]] = []
    matched_docs: list[dict[str, Any]] = []
    ok_slices = 0

    for idx, stamp_str in enumerate(stamps, start=1):
        # Try :16 first, fallback to :01 or :00 if missing
        candidates = [stamp_str, stamp_str[:10] + "0100", stamp_str[:10] + "0000"]
        used_stamp = stamp_str
        status, http_status, body, err = "not_tried", None, b"", ""
        url = ""

        for cand in candidates:
            used_stamp = cand
            url = f"{NGRAMS_BASE}/{cand}.toc.json.gz"
            status, http_status, body, err = fetch_bytes(url, timeout=20, max_bytes=4 * 1024 * 1024)
            if status == "ok":
                break
            if http_status != 404:
                break

        raw_name = f"toc_{used_stamp}.json.gz"
        if body:
            (out_dir / raw_name).write_bytes(body)

        total_docs = 0
        counts = {"base": 0, "T1_dam_rumor": 0, "T2_seismic_misread": 0, "T3_glacier_science": 0, "T4_official_supply": 0}
        if status == "ok":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
                    plain = gz.read(32 * 1024 * 1024).decode("utf-8", errors="replace")
                for line in plain.splitlines():
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    total_docs += 1
                    title = str(doc.get("title", "") or "")
                    flags = classify_title(title)
                    if flags["base"]:
                        for k, v in flags.items():
                            if v:
                                counts[k] += 1
                        matched_docs.append(
                            {
                                "file_stamp": used_stamp,
                                "doc_id": str(doc.get("ID", "")),
                                "date": str(doc.get("date", "")),
                                "lang": str(doc.get("lang", "")),
                                "title": title,
                                "url": str(doc.get("url", "")),
                                "T1_dam_rumor": int(flags["T1_dam_rumor"]),
                                "T2_seismic_misread": int(flags["T2_seismic_misread"]),
                                "T3_glacier_science": int(flags["T3_glacier_science"]),
                                "T4_official_supply": int(flags["T4_official_supply"]),
                            }
                        )
                ok_slices += 1
            except (OSError, json.JSONDecodeError) as exc:
                status = "parse_error"
                err = f"{type(exc).__name__}: {exc}"

        slice_summary.append(
            {
                "file_stamp": used_stamp,
                "status": status,
                "http_status": http_status or "",
                "total_toc_docs": total_docs,
                **counts,
            }
        )
        log_item = {
            "slice_idx": idx,
            "file_stamp": used_stamp,
            "requested_at_utc": utc_now_iso(),
            "status": status,
            "http_status": http_status,
            "bytes_saved": len(body),
            "sha256_12": sha256_hex(body)[:12] if body else "",
            "total_toc_docs": total_docs,
            "base_hits": counts["base"],
            "error": err,
            "url": url,
        }
        logs.append(log_item)
        print(
            f"[TOC] {idx:02d}/{len(stamps)} {used_stamp}: {status} http={http_status} "
            f"docs={total_docs} hits={counts['base']} (T1={counts['T1_dam_rumor']}, T2={counts['T2_seismic_misread']}, T3={counts['T3_glacier_science']})",
            flush=True,
        )
        if idx < len(stamps):
            time.sleep(sleep_sec)

    write_json(out_dir / "toc_requests.json", logs)
    write_csv(
        PROCESSED_DIR / "gdelt_toc_slices.csv",
        ["file_stamp", "status", "http_status", "total_toc_docs", "base", "T1_dam_rumor", "T2_seismic_misread", "T3_glacier_science", "T4_official_supply"],
        slice_summary,
    )
    write_csv(
        PROCESSED_DIR / "gdelt_toc_matches.csv",
        ["file_stamp", "doc_id", "date", "lang", "title", "url", "T1_dam_rumor", "T2_seismic_misread", "T3_glacier_science", "T4_official_supply"],
        matched_docs,
    )
    return {
        "channel": "web_ngrams_toc",
        "slices_attempted": len(stamps),
        "slices_succeeded": ok_slices,
        "matched_docs": len(matched_docs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "doc", "toc"], default="auto")
    args = parser.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RAW_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"run_id": run_id, "mode": args.mode, "started_at_utc": utc_now_iso()}
    if args.mode in ("auto", "doc"):
        doc_res = run_doc_channel(cfg, out_dir)
        summary["doc"] = doc_res
        if args.mode == "auto" and doc_res["succeeded"] < 3:
            print("[AUTO] DOC channel incomplete/throttled; activating Web NGrams TOC fallback...", flush=True)
            summary["toc"] = run_ngrams_toc_channel(out_dir)
    else:
        summary["toc"] = run_ngrams_toc_channel(out_dir)

    summary["finished_at_utc"] = utc_now_iso()
    write_json(out_dir / "summary.json", summary)
    write_json(PROCESSED_DIR / "gdelt_latest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
