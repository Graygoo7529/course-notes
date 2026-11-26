"""Small, offline Boolean search comparison over predeclared GDELT snapshots."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from gdelt_probe import ROOT, digest, write_json
from summarize_probe import export_csv


def terms(expression: str) -> list[str]:
    return [part.strip().strip('"') for part in expression.strip().strip("()").split(" OR ")]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def matcher(words: list[str]) -> re.Pattern[str]:
    alternatives = [re.escape(normalize(word)).replace(r"\ ", r"\s+") for word in words]
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


def excluded(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in ("apple.com", "apple.com.cn"))


def query_registry(config: dict[str, Any], lexicon: dict[str, Any]) -> list[dict[str, str]]:
    """Logical expressions for this local evaluator, not DOC API requests."""
    product, broad = config["core"], config["broad_core"]
    domain_filter = "NOT host_in(apple.com, apple.com.cn; include subdomains)"
    base = f"{product} AND {domain_filter}"
    combine = lambda words: "(" + " OR ".join(json.dumps(word, ensure_ascii=False) for word in words) + ")"
    expressions = {
        "broad_raw": broad, "broad": f"{broad} AND {domain_filter}",
        "core_raw": product, "core": base,
        "single_phrase": f'"iphone 18 pro" AND {domain_filter}',
        "core_and_launch": f'{base} AND {combine(lexicon["launch_terms"])}',
        "core_not_rumor": f'{base} AND NOT {combine(lexicon["rumor_terms"])}',
        "core_not_foldable": f'{base} AND NOT {combine(lexicon["comparison_terms"])}',
        "core_in_title": f"{base} AND title_contains({product})",
    }
    result = [{"query_id": key, "language_scope": "all", "expression": value,
               "engine": "local_ngram_document_boolean"} for key, value in expressions.items()]
    for topic, expression in config["topics"].items():
        result.append({"query_id": topic + "__en", "language_scope": "all", "expression": f"{base} AND {expression}",
                       "engine": "local_ngram_document_boolean"})
        for language in lexicon["supported_languages"]:
            words = terms(expression) + lexicon["additions"].get(language, {}).get(topic, [])
            result.append({"query_id": topic + "__local", "language_scope": language,
                           "expression": f"{base} AND {combine(words)} AND lang={language}",
                           "engine": "local_ngram_document_boolean"})
    return result


def topic_comparison(documents: list[dict[str, Any]], config: dict[str, Any],
                     lexicon: dict[str, Any]) -> dict[str, Any]:
    frame = [row for row in documents if "core" in row["queries"]
             and row["language"] in lexicon["supported_languages"]]
    return {"frame_unique_urls": len({row["url"] for row in frame}),
            "languages": lexicon["supported_languages"],
            "counts": {topic: {suffix: len({row["url"] for row in frame if topic + "__" + suffix in row["queries"]})
                               for suffix in ("en", "local")} for topic in config["topics"]}}


def analyze(toc: dict[str, dict[str, Any]], ngrams: str, config: dict[str, Any],
            lexicon: dict[str, Any]) -> list[dict[str, Any]]:
    rules = {"broad": matcher(terms(config["broad_core"])), "core": matcher(terms(config["core"])),
             "single_phrase": matcher(["iphone 18 pro"]), "launch": matcher(lexicon["launch_terms"]),
             "rumor": matcher(lexicon["rumor_terms"]), "foldable": matcher(lexicon["comparison_terms"])}
    for topic, expression in config["topics"].items():
        rules[topic + "__en"] = matcher(terms(expression))
        for language in lexicon["supported_languages"]:
            extra = lexicon["additions"].get(language, {}).get(topic, [])
            rules[topic + "__" + language + "_local"] = matcher(terms(expression) + extra)
    # First identify product candidates, then inspect every quadgram for those
    # documents. Product and topic need not appear in the same quadgram.
    rows = list(csv.reader(io.StringIO(ngrams), delimiter="\t", quoting=csv.QUOTE_NONE))
    candidate_ids = set()
    for row in rows:
        if len(row) != 3:
            raise ValueError("Malformed quadgram row")
        text = normalize(row[1])
        if rules["broad"].search(text) or rules["core"].search(text):
            candidate_ids.add(row[0])
    evidence: dict[str, dict[str, str]] = {document_id: {} for document_id in candidate_ids}
    for document_id, quadgram, _ in rows:
        if document_id not in candidate_ids:
            continue
        if document_id not in toc:
            raise ValueError("Matched document is missing from TOC")
        text = normalize(quadgram)
        language = toc[document_id].get("lang", "")
        for key, regex in rules.items():
            if key.endswith("_local") and not key.endswith("__" + language + "_local"):
                continue
            if key not in evidence[document_id] and regex.search(text):
                evidence[document_id][key] = quadgram
    results = []
    for document_id in sorted(candidate_ids, key=int):
        metadata, hits = toc[document_id], evidence[document_id]
        url, language = metadata["url"], metadata.get("lang", "")
        core = "core" in hits and not excluded(url)
        queries = []
        if "broad" in hits:
            queries.append("broad_raw")
            if not excluded(url):
                queries.append("broad")
        if "core" in hits:
            queries.append("core_raw")
        if core:
            queries.append("core")
            if "single_phrase" in hits:
                queries.append("single_phrase")
            if "launch" in hits:
                queries.append("core_and_launch")
            if "rumor" not in hits:
                queries.append("core_not_rumor")
            if "foldable" not in hits:
                queries.append("core_not_foldable")
            if rules["core"].search(normalize(metadata.get("title", ""))):
                queries.append("core_in_title")
            for topic in config["topics"]:
                if topic + "__en" in hits:
                    queries.append(topic + "__en")
                if topic + "__" + language + "_local" in hits:
                    queries.append(topic + "__local")
        results.append({"doc_id": document_id, "article_id": digest(url), "url": url,
                        "title": metadata.get("title", ""), "language": language,
                        "metadata_date_raw": metadata.get("date", ""),
                        "excluded_domain": excluded(url), "queries": queries, "evidence": hits})
    return results


def load_snapshot(spec: dict[str, Any]) -> tuple[Path, dict[str, dict[str, Any]], str, list[dict[str, Any]]]:
    if spec.get("existing_run"):
        run = ROOT / spec["existing_run"]
    else:
        matches = []
        for folder in sorted((ROOT / "data" / "ngrams_probes").iterdir()):
            if not (folder / "summary.json").exists():
                continue
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary.get("file_stamp") == spec["file_stamp"]:
                matches.append(folder)
        if not matches:
            raise ValueError(f'No successful archive for {spec["file_stamp"]}')
        run = matches[-1]
    logs = json.loads((run / "requests.json").read_text(encoding="utf-8"))
    texts = {}
    for log in logs:
        if log["status"] != "ok" or log["file_stamp"] != spec["file_stamp"]:
            raise ValueError("Snapshot has a failed or mismatched request")
        filename = log["raw_path"]
        if filename not in ("toc.json.gz.response", "ngrams.txt.gz.response"):
            raise ValueError("Unexpected archive filename")
        path = run / filename
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Compressed archive exceeds probe budget")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != log["raw_sha256"] or len(body) != log["bytes_saved"]:
            raise ValueError("Snapshot archive hash/size mismatch")
        with gzip.open(path, "rb") as source:
            plain = source.read(64 * 1024 * 1024 + 1)
        if len(plain) > 64 * 1024 * 1024:
            raise ValueError("Expanded archive exceeds probe budget")
        texts[filename] = plain.decode("utf-8")
    toc = {}
    for line in texts["toc.json.gz.response"].splitlines():
        document = json.loads(line)
        key = str(document["ID"])
        if key in toc:
            raise ValueError("Duplicate TOC ID")
        toc[key] = document
    return run, toc, texts["ngrams.txt.gz.response"], logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=ROOT / "data" / "coursework_reviews.json")
    args = parser.parse_args()
    config = json.loads((ROOT / "config" / "study.json").read_text(encoding="utf-8"))
    lexicon = json.loads((ROOT / "config" / "coursework_lexicon.json").read_text(encoding="utf-8"))
    sampling = json.loads((ROOT / "config" / "coursework_sampling.json").read_text(encoding="utf-8"))
    documents, snapshots, query_rows = [], [], []
    query_ids = ["broad_raw", "broad", "core_raw", "core", "single_phrase", "core_and_launch",
                 "core_not_rumor", "core_not_foldable", "core_in_title"]
    query_ids += [topic + suffix for topic in config["topics"] for suffix in ("__en", "__local")]
    for spec in sampling["snapshots"]:
        run, toc, ngrams, logs = load_snapshot(spec)
        found = analyze(toc, ngrams, config, lexicon)
        for row in found:
            row.update(file_stamp=spec["file_stamp"], stage=spec["stage"], source_run=str(run.relative_to(ROOT)))
        documents.extend(found)
        counts = {query: len({row["url"] for row in found if query in row["queries"]}) for query in query_ids}
        snapshots.append({**spec, "source_run": str(run.relative_to(ROOT)), "toc_documents": len(toc),
                          "counts": counts, "source_hashes": {log["raw_path"]: log["raw_sha256"] for log in logs}})
        query_rows.extend({"file_stamp": spec["file_stamp"], "stage": spec["stage"], "query_id": query,
                           "unique_urls": count} for query, count in counts.items())
    output = ROOT / "data" / "coursework" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    write_json(output / "snapshots.json", snapshots)
    write_json(output / "documents.json", documents)
    write_json(output / "config_snapshot.json", {"study": config, "lexicon": lexicon, "sampling": sampling})
    query_specs = query_registry(config, lexicon)
    export_csv(output / "query_registry.csv", list(query_specs[0]), query_specs)
    export_csv(output / "query_counts.csv", ["file_stamp", "stage", "query_id", "unique_urls"], query_rows)
    rows = [{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
             for key, value in row.items()} for row in documents]
    if rows:
        export_csv(output / "documents.csv", list(rows[0]), rows)
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews.exists() else []
    known = {row["url"] for row in documents}
    if any(review["url"] not in known for review in reviews):
        raise ValueError("A reviewed URL is not in the collected candidate frame")
    if len({review["url"] for review in reviews}) != len(reviews):
        raise ValueError("Duplicate reviewed URL")
    for review in reviews:
        if review["relevance"] not in ("relevant", "excluded", "unknown"):
            raise ValueError("Invalid review relevance")
        if review["relevance"] != "relevant" and (review["main_topics"] or review["stance"] not in ("unknown", "not_applicable")):
            raise ValueError("Unconfirmed or excluded articles cannot supply opinion findings")
    write_json(output / "reviews.json", reviews)
    if reviews:
        review_rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()} for row in reviews]
        fields = list(dict.fromkeys(key for row in review_rows for key in row))
        export_csv(output / "reviews.csv", fields, review_rows)
        audit = [{"review_id": review["review_id"], "url": review["url"], "relevance": review["relevance"],
                  "query_id": query, "hit": any(query in row["queries"] for row in documents if row["url"] == review["url"])}
                 for review in reviews for query in query_ids]
        export_csv(output / "review_query_audit.csv", list(audit[0]), audit)
    with sqlite3.connect(output / "coursework.sqlite") as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE articles(article_id TEXT PRIMARY KEY, url TEXT UNIQUE, title TEXT, language TEXT);
            CREATE TABLE occurrences(file_stamp TEXT, doc_id TEXT, article_id TEXT REFERENCES articles,
                stage TEXT, source_run TEXT, evidence_json TEXT, PRIMARY KEY(file_stamp,doc_id));
            CREATE TABLE hits(file_stamp TEXT, doc_id TEXT, query_id TEXT,
                FOREIGN KEY(file_stamp,doc_id) REFERENCES occurrences, PRIMARY KEY(file_stamp,doc_id,query_id));
            CREATE TABLE reviews(article_id TEXT PRIMARY KEY REFERENCES articles, review_json TEXT);
        """)
        for row in documents:
            connection.execute("INSERT OR IGNORE INTO articles VALUES (?,?,?,?)",
                               (row["article_id"], row["url"], row["title"], row["language"]))
            connection.execute("INSERT INTO occurrences VALUES (?,?,?,?,?,?)",
                               (row["file_stamp"], row["doc_id"], row["article_id"], row["stage"],
                                row["source_run"], json.dumps(row["evidence"], ensure_ascii=False)))
            connection.executemany("INSERT INTO hits VALUES (?,?,?)",
                                   [(row["file_stamp"], row["doc_id"], query) for query in row["queries"]])
        for review in reviews:
            connection.execute("INSERT INTO reviews VALUES (?,?)", (digest(review["url"]), json.dumps(review, ensure_ascii=False)))
    summary = {"snapshots": len(snapshots), "toc_documents": sum(item["toc_documents"] for item in snapshots),
               "unique_candidate_urls": len(known),
               "pooled_counts": {query: len({row["url"] for row in documents if query in row["queries"]}) for query in query_ids},
               "core_language_counts": dict(Counter(row["language"] for row in documents if "core" in row["queries"])),
               "topic_same_frame": topic_comparison(documents, config, lexicon),
               "review_outcomes": dict(Counter(review["relevance"] for review in reviews)),
               "reviewed_cases": len(reviews), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "note": "Convenience snapshots only. Counts compare Boolean rules in a fixed corpus, not daily news totals. Topic language coverage is en/es/fr; other languages are unassessed."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
