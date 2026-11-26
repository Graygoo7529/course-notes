"""Small, auditable GDELT DOC pilot; no third-party dependencies.

The pilot is deliberately capped and is NOT a complete article collector.
All requests, including HTTP errors and invalid payloads, are archived.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODES = ("ArtList", "TimelineVolRaw", "TimelineTone")


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    return result.astimezone(timezone.utc)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def registry(config: dict[str, Any]) -> list[dict[str, str]]:
    core = config["core"]
    exclusions = config["exclusions"]
    definitions = {"base": core}
    definitions.update({key: f"{core} {terms}" for key, terms in config["topics"].items()})
    definitions["broad_sensitivity"] = config["broad_core"]
    definitions["no_rumor_sensitivity"] = f"{core} -rumor -rumors -rumour -rumours -leak -leaks"
    return [
        {
            "query_id": f"{name}__{country}",
            "version": config["query_version"],
            "topic": name,
            "country": country,
            "query": " ".join(part for part in (expression, exclusions, country_filter) if part),
        }
        for name, expression in definitions.items()
        for country, country_filter in config["countries"].items()
    ]


def task(config: dict[str, Any], query_id: str, mode: str, start: str, end: str,
         limit: int = 25) -> dict[str, Any]:
    # Internally [start, end); API uses a last-second end time.
    begin, stop = utc(start), utc(end)
    if stop <= begin or stop > datetime.now(timezone.utc):
        raise ValueError("Window must be positive and must not extend into the future")
    if stop - begin < timedelta(minutes=15):
        raise ValueError("GDELT requests need at least a 15-minute window")
    queries = {row["query_id"]: row for row in registry(config)}
    params = {
        "query": queries[query_id]["query"], "mode": mode, "format": "json",
        "startdatetime": begin.strftime("%Y%m%d%H%M%S"),
        "enddatetime": (stop - timedelta(seconds=1)).strftime("%Y%m%d%H%M%S"),
    }
    if mode == "ArtList":
        params.update(maxrecords=str(limit), sort="DateAsc")
    else:
        params["timelinesmooth"] = "0"
    url = config["endpoint"] + "?" + urllib.parse.urlencode(params)
    return {"query_id": query_id, "mode": mode, "start_utc": start,
            "end_exclusive_utc": end, "params": params, "url": url,
            "request_key": digest(url)}


def pilot_tasks(config: dict[str, Any], limit: int = 25) -> list[dict[str, Any]]:
    begin, end = config["window_start"], config["snapshot_end_exclusive"]
    launch_start, launch_end = "2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z"
    definitions = [
        ("base__global", "ArtList", launch_start, launch_end),
        ("base__global", "TimelineVolRaw", begin, end),
        ("price_value__global", "ArtList", launch_start, launch_end),
        ("base__us", "ArtList", launch_start, launch_end),
        ("base__uk", "ArtList", launch_start, launch_end),
        ("base__jp", "ArtList", launch_start, launch_end),
        ("base__cn", "ArtList", launch_start, launch_end),
        ("broad_sensitivity__global", "ArtList", launch_start, launch_end),
        ("base__global", "TimelineTone", begin, end),
    ]
    return [task(config, *definition, limit=limit) for definition in definitions]


def database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE queries (
            query_id TEXT PRIMARY KEY, version TEXT, topic TEXT, country TEXT, query TEXT);
        CREATE TABLE requests (
            request_id TEXT PRIMARY KEY, request_key TEXT, query_id TEXT REFERENCES queries,
            mode TEXT, start_utc TEXT, end_exclusive_utc TEXT, requested_at TEXT,
            attempt INTEGER, status TEXT, http_status INTEGER, elapsed_seconds REAL,
            item_count INTEGER, limit_reached INTEGER, url TEXT, raw_path TEXT,
            raw_sha256 TEXT, error TEXT);
        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY, url TEXT UNIQUE, title TEXT, seendate_raw TEXT,
            domain TEXT, language TEXT, sourcecountry_raw TEXT, url_mobile TEXT);
        CREATE TABLE article_hits (
            request_id TEXT REFERENCES requests, article_id TEXT REFERENCES articles,
            query_id TEXT REFERENCES queries, result_rank INTEGER,
            PRIMARY KEY (request_id, article_id));
        CREATE TABLE timeline_points (
            request_id TEXT REFERENCES requests, query_id TEXT REFERENCES queries,
            mode TEXT, series TEXT, date_raw TEXT, value REAL, norm REAL,
            PRIMARY KEY (request_id, series, date_raw));
    """)
    return connection


def payload_items(payload: Any, mode: str) -> list[dict[str, Any]]:
    """Validate required fields; an error page must never become a zero count."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    key = "articles" if mode == "ArtList" else "timeline"
    items = payload.get(key)
    if not isinstance(items, list):
        raise ValueError(f"Missing list field: {key}; keys={list(payload)}")
    seen_points: set[tuple[str, datetime]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Expected object entries")
        if mode == "ArtList":
            if not isinstance(item.get("url"), str) or not item["url"].startswith(("http://", "https://")):
                raise ValueError("Missing or invalid article URL")
        else:
            if not isinstance(item.get("series"), str) or not isinstance(item.get("data"), list):
                raise ValueError("Malformed timeline series")
            for point in item["data"]:
                if not isinstance(point, dict) or not isinstance(point.get("date"), str):
                    raise ValueError("Malformed timeline point")
                point_key = (item["series"], utc(point["date"]))
                if point_key in seen_points:
                    raise ValueError("Duplicate timestamp within a timeline series")
                seen_points.add(point_key)
                for field in ("value", "norm"):
                    value = point.get(field)
                    if field == "norm" and value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Non-finite, non-numeric or missing timeline {field}")
                    if (field == "norm" or mode == "TimelineVolRaw") and value < 0:
                        raise ValueError(f"Negative volume or denominator: {field}")
    return items


def ingest(connection: sqlite3.Connection, request_id: str, spec: dict[str, Any],
           items: list[dict[str, Any]]) -> None:
    if spec["mode"] == "ArtList":
        for rank, article in enumerate(items, 1):
            # Only exact-URL deduplication. Content-level duplicates need review.
            article_id = digest(article["url"])
            connection.execute(
                "INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (article_id, article["url"], article.get("title"), article.get("seendate"),
                 article.get("domain"), article.get("language"), article.get("sourcecountry"),
                 article.get("url_mobile")),
            )
            connection.execute("INSERT OR IGNORE INTO article_hits VALUES (?, ?, ?, ?)",
                               (request_id, article_id, spec["query_id"], rank))
    else:
        for series in items:
            for point in series["data"]:
                connection.execute("INSERT INTO timeline_points VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (request_id, spec["query_id"], spec["mode"], series["series"],
                                    point["date"], point["value"], point.get("norm")))


def fetch(connection: sqlite3.Connection, config: dict[str, Any], spec: dict[str, Any],
          run: Path, sequence: int, attempt: int) -> tuple[str, int | None, float]:
    request_id = f"r{sequence:03d}_a{attempt}"
    raw_path = run / "raw" / f"{request_id}.body"
    requested_at, started = stamp(), time.monotonic()
    body, http_status, error, retry_after = b"", None, "", 0.0
    content_type = ""
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "CourseResearch-GDELT-Pilot/0.1"})
        with urllib.request.urlopen(req, timeout=config["timeout_seconds"]) as response:
            http_status = response.status
            content_type = response.headers.get("Content-Type", "")
            body = response.read()
    except urllib.error.HTTPError as exc:
        http_status, body = exc.code, exc.read()
        error = str(exc)
        content_type = exc.headers.get("Content-Type", "")
        try:
            retry_after = float(exc.headers.get("Retry-After", "0"))
        except ValueError:
            # Absolute Retry-After dates are handled conservatively by stopping.
            retry_after = 61.0
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.monotonic() - started, 3)
    raw_path.write_bytes(body)
    status, items = "network_error", []
    if http_status is not None:
        status = "http_error"
        if http_status == 200:
            try:
                items = payload_items(json.loads(body), spec["mode"])
                status = "ok" if items else "empty"
            except (ValueError, UnicodeError) as exc:
                status, error = "invalid_payload", str(exc)
    item_count = None
    limit_reached = None
    if status in ("ok", "empty"):
        item_count = len(items) if spec["mode"] == "ArtList" else sum(len(s["data"]) for s in items)
        limit_reached = int(item_count >= int(spec["params"]["maxrecords"])) if spec["mode"] == "ArtList" else None
    record = {
        "request_id": request_id, "request_key": spec["request_key"],
        "query_id": spec["query_id"], "mode": spec["mode"],
        "start_utc": spec["start_utc"], "end_exclusive_utc": spec["end_exclusive_utc"],
        "requested_at": requested_at, "attempt": attempt, "status": status,
        "http_status": http_status, "elapsed_seconds": elapsed, "item_count": item_count,
        "limit_reached": limit_reached, "url": spec["url"],
        "raw_path": str(raw_path.relative_to(run)), "raw_sha256": hashlib.sha256(body).hexdigest(),
        "error": error,
    }
    connection.execute("INSERT INTO requests VALUES (" + ",".join("?" for _ in record) + ")",
                       tuple(record.values()))
    if status in ("ok", "empty"):
        # Keep the request audit even if metadata cannot be inserted. Never
        # retain only the first part of a failed response as successful data.
        connection.execute("SAVEPOINT response_items")
        try:
            ingest(connection, request_id, spec, items)
        except sqlite3.Error as exc:
            connection.execute("ROLLBACK TO response_items")
            status, error = "ingest_error", f"{type(exc).__name__}: {exc}"
            record.update(status=status, error=error, item_count=None, limit_reached=None)
            connection.execute(
                "UPDATE requests SET status=?, error=?, item_count=NULL, limit_reached=NULL WHERE request_id=?",
                (status, error, request_id),
            )
            item_count = None
        finally:
            connection.execute("RELEASE response_items")
    connection.commit()
    write_json(run / "raw" / f"{request_id}.meta.json",
               {**record, "content_type": content_type, "retry_after_seconds": retry_after})
    print(f"{request_id} {spec['query_id']} {spec['mode']}: {status}, HTTP={http_status}, items={item_count}", flush=True)
    return status, http_status, retry_after


def export_tables(connection: sqlite3.Connection, run: Path) -> None:
    for table in ("queries", "requests", "articles", "article_hits", "timeline_points"):
        cursor = connection.execute(f"SELECT * FROM {table}")
        with (run / f"{table}.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow([column[0] for column in cursor.description])
            writer.writerows(cursor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "pilot", "one"))
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "study.json")
    parser.add_argument("--query-id", default="base__global")
    parser.add_argument("--mode", choices=MODES, default="ArtList")
    parser.add_argument("--start", default="2026-09-09T00:00:00Z")
    parser.add_argument("--end", default="2026-09-10T00:00:00Z", help="Exclusive UTC end")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-tasks", type=int, default=9)
    args = parser.parse_args()
    if not 1 <= args.limit <= 250 or args.max_tasks < 1:
        parser.error("limit must be 1..250; max-tasks must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    queries = registry(config)
    tasks = ([task(config, args.query_id, args.mode, args.start, args.end, args.limit)]
             if args.command == "one" else pilot_tasks(config, args.limit)[:args.max_tasks])
    run = ROOT / "data" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    (run / "raw").mkdir(parents=True)
    write_json(run / "config_snapshot.json", config)
    write_json(run / "tasks.json", tasks)
    write_json(run / "runtime.json", {"python": sys.version, "executable": sys.executable,
                                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(f"Run: {run}", flush=True)
    with database(run / "pilot.sqlite") as connection:
        connection.executemany("INSERT INTO queries VALUES (?, ?, ?, ?, ?)",
                               [tuple(row.values()) for row in queries])
        completed, interrupted = 0, ""
        try:
            if args.command != "plan":
                for sequence, spec in enumerate(tasks, 1):
                    if sequence > 1:
                        time.sleep(config["min_interval_seconds"])
                    for attempt in range(1, config["max_attempts"] + 1):
                        status, http_status, retry_after = fetch(connection, config, spec, run, sequence, attempt)
                        retryable = status == "network_error" or http_status in (429, 500, 502, 503, 504)
                        if retry_after > 60:
                            interrupted = "Server requests a longer cooldown; stopped without further requests."
                            break
                        if not retryable or attempt == config["max_attempts"]:
                            break
                        time.sleep(max(config["min_interval_seconds"], retry_after, 15 * attempt))
                    completed += 1
                    if interrupted or status not in ("ok", "empty"):
                        interrupted = interrupted or f"Stopped after failed task: {spec['query_id']} ({status})."
                        break
        except KeyboardInterrupt:
            interrupted = "Interrupted by user; completed requests retained."
        finally:
            export_tables(connection, run)
            counts = dict(connection.execute("SELECT status, COUNT(*) FROM requests GROUP BY status"))
            summary = {"command": args.command, "finished_at": stamp(), "planned_tasks": len(tasks),
                       "attempted_tasks": completed, "request_status_counts": counts,
                       "unique_articles": connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
                       "timeline_points": connection.execute("SELECT COUNT(*) FROM timeline_points").fetchone()[0],
                       "note": interrupted or ("Offline plan; no requests made." if args.command == "plan"
                                               else "Pilot only; article results are capped and not population counts.")}
            write_json(run / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    if interrupted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
