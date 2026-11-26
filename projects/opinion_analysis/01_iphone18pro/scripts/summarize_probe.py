"""Replay and audit an archived minute probe; export candidates and diagnostic plots offline."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from gdelt_probe import ROOT, digest, stamp, write_json
from ngrams_probe import PATTERN, extract_matches


def replay(run: Path) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    logs = json.loads((run / "requests.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    expected = {"toc.json.gz.response", "ngrams.txt.gz.response"}
    if len(logs) != 2 or {row["raw_path"] for row in logs} != expected:
        raise ValueError("Expected exactly the two archived minute files")
    if len({row["file_stamp"] for row in logs}) != 1:
        raise ValueError("Cannot join DOCIDs from different minute files")
    for log in logs:
        path = run / log["raw_path"]
        if log["status"] != "ok" or log["http_status"] != 200:
            raise ValueError(f"Unsuccessful source request: {path.name}")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Compressed source exceeds the probe budget")
        body = path.read_bytes()
        if len(body) != log["bytes_saved"] or hashlib.sha256(body).hexdigest() != log["raw_sha256"]:
            raise ValueError(f"Archive size/hash mismatch: {path.name}")
        with gzip.open(path, "rb") as archive:
            plain = archive.read(64 * 1024 * 1024 + 1)
        if len(plain) > 64 * 1024 * 1024:
            raise ValueError("Decompressed source exceeds the probe budget")
        texts[path.name] = plain.decode("utf-8")
    total, hits = extract_matches(texts["toc.json.gz.response"], texts["ngrams.txt.gz.response"],
                                  logs[0]["file_stamp"])
    archived_hits = json.loads((run / "matches.json").read_text(encoding="utf-8"))
    if hits != archived_hits:
        raise ValueError("Replay differs from archived matches; inspect before reporting")
    return total, hits, logs


def export_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(output: Path, summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained", width_ratios=(1, 1.4))
    languages = summary["language_counts"]
    groups = [("URLs", summary["unique_urls"]), ("Exact titles", summary["unique_titles"])]
    axes[0].barh(list(languages), list(languages.values()), color="#2463a6")
    axes[0].invert_yaxis()
    axes[0].set_title("Language labels in the matched documents")
    for index, value in enumerate(languages.values()):
        axes[0].text(value + 0.2, index, str(value), va="center")
    axes[0].set_xlim(0, max(languages.values(), default=1) * 1.15)
    axes[0].set_xlabel("Matched documents (not source countries)")
    axes[1].barh([key for key, _ in groups], [value for _, value in groups], color=["#177e70", "#d99035"])
    axes[1].invert_yaxis()
    for index, (_, value) in enumerate(groups):
        axes[1].text(value + 0.5, index, str(value), va="center")
    axes[1].set_xlim(0, max((value for _, value in groups), default=1) * 1.2)
    axes[1].set_xlabel("Distinct values (exact titles are not verified content clusters)")
    axes[1].set_title("Different URLs can share the same headline")
    notes = "\n\n".join(
        textwrap.fill(f'{row["count"]} URLs: {row["title"]}', width=62)
        for row in summary["repeated_titles"][:2]
    )
    axes[1].text(0.03, 0.97, notes, transform=axes[1].transAxes, va="top", fontsize=9,
                 bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    # Keep room for the title examples above the bars.
    axes[1].set_ylim(2, -2)
    fig.suptitle(f'Single-minute feasibility probe: {summary["file_stamp"]} UTC | '
                 f'{summary["matched_documents"]} candidates; relevance not yet reviewed', fontsize=12)
    fig.supxlabel("Diagnostic sample only. No event trend, country comparison, or public sentiment inferred.", fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(output / f"probe_diagnostics.{extension}", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        total, hits, logs = replay(args.run)
        for hit in hits:
            metadata = hit["metadata"]
            if not isinstance(metadata.get("url"), str) or not metadata["url"].startswith(("http://", "https://")):
                raise ValueError("Matched document has no valid URL")
    except (ValueError, OSError, EOFError) as exc:
        parser.exit(2, f"Probe audit failed: {exc}\n")
    titles = Counter(hit["metadata"].get("title", "") for hit in hits)
    languages = Counter(hit["metadata"].get("lang", "unknown") for hit in hits)
    candidates = []
    for hit in hits:
        metadata = hit["metadata"]
        title, url = metadata.get("title", ""), metadata["url"]
        candidates.append({
            "article_id": digest(url), "file_stamp": hit["file_stamp"], "doc_id": hit["doc_id"],
            "url": url, "hostname": urlsplit(url).hostname, "title": title,
            "metadata_date_raw": metadata.get("date", ""), "language_raw": metadata.get("lang", ""),
            "matched_quadgrams": json.dumps(hit["matched_quadgrams"], ensure_ascii=False),
            "exact_title_group_id": digest(title) if title else "",
            "exact_title_group_size": titles[title] if title else "",
            "sourcecountry_verified": "", "relevant": "", "product_stance": "",
        })
    summary = {
        "source_run": str(args.run.resolve()), "generated_at": stamp(),
        "file_stamp": logs[0]["file_stamp"], "match_pattern": PATTERN,
        "toc_documents": total, "matched_documents": len(hits),
        "unique_urls": len({hit["metadata"]["url"] for hit in hits}),
        "unique_titles": len({title for title in titles if title}),
        "missing_titles": titles.get("", 0), "language_counts": dict(languages.most_common()),
        "repeated_titles": [{"title": title, "count": count} for title, count in titles.most_common()
                            if title and count > 1],
        "source_sha256": {log["raw_path"]: log["raw_sha256"] for log in logs},
        "source_hashes_verified": True, "archived_matches_reproduced": True,
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "scripts_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                                       for name in ("summarize_probe.py", "ngrams_probe.py")}},
        "note": "One-minute candidate diagnostic only; no country, stance, content deduplication or time trend inferred.",
    }
    output = ROOT / "data" / "probe_reports" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    fields = ["article_id", "file_stamp", "doc_id", "url", "hostname", "title", "metadata_date_raw",
              "language_raw", "matched_quadgrams", "exact_title_group_id", "exact_title_group_size",
              "sourcecountry_verified", "relevant", "product_stance"]
    export_csv(output / "candidates.csv", fields, candidates)
    export_csv(output / "title_counts.csv", ["title", "count"],
               [{"title": title, "count": count} for title, count in titles.most_common()])
    export_csv(output / "language_counts.csv", ["language_raw", "count"],
               [{"language_raw": language, "count": count} for language, count in languages.most_common()])
    write_json(output / "summary.json", summary)
    if args.plot:
        plot_diagnostics(output, summary)
    print(f"Verified archive and replay: {total} TOC documents, {len(hits)} candidates")
    print(output)


if __name__ == "__main__":
    main()
