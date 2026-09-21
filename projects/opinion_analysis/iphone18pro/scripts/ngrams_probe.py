"""Bounded one-minute feasibility probe of GDELT's published Web NGrams files.

This does not establish coverage of the event, country labels, or API equivalence.
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
import urllib.request
from datetime import datetime, timezone
from typing import Any

from gdelt_probe import ROOT, stamp, write_json

PATTERN = r"\biphone\s*18\s*pro\b"


def extract_matches(toc_text: str, ngrams_text: str, file_stamp: str) -> tuple[int, list[dict[str, Any]]]:
    """Join within one file only; raw quadgrams are TSV, not CSV-quoted text."""
    toc = {}
    for line in toc_text.splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        if not isinstance(document, dict) or "ID" not in document:
            raise ValueError("Invalid TOC document")
        document_id = str(document["ID"])
        if document_id in toc:
            raise ValueError(f"Duplicate TOC ID: {document_id}")
        toc[document_id] = document
    pattern = re.compile(PATTERN, re.IGNORECASE)
    matches: dict[str, list[str]] = {}
    for row in csv.reader(io.StringIO(ngrams_text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if not row:
            continue
        if len(row) != 3 or not row[2].isdigit() or int(row[2]) <= 0:
            raise ValueError("Expected DOCID, QUADGRAM and positive COUNT")
        if pattern.search(row[1]):
            if row[0] not in toc:
                raise ValueError(f"Matched document missing from TOC: {row[0]}")
            matches.setdefault(row[0], []).append(row[1])
    hits = [{"file_stamp": file_stamp, "doc_id": document_id,
             "matched_quadgrams": phrases, "metadata": toc[document_id]}
            for document_id, phrases in matches.items()]
    return len(toc), hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-stamp", default="20260909201600")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{12}00", args.file_stamp):
        parser.error("file-stamp must be YYYYMMDDHHMM00")
    requested_minute = datetime.strptime(args.file_stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if requested_minute >= datetime.now(timezone.utc):
        parser.error("Only historical files can be requested")
    run = ROOT / "data" / "ngrams_probes" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True)
    logs, texts = [], {}
    for suffix in ("toc.json.gz", "ngrams.txt.gz"):
        url = f"https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/{args.file_stamp}.{suffix}"
        status, http_status, error, body = "network_error", None, "", b""
        log = {"file_stamp": args.file_stamp, "url": url, "requested_at": stamp()}
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                http_status = response.status
                # Bound both the transfer and the decompression.
                body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                status, error = "size_limit", "Transfer exceeds the 8 MiB pilot budget; prefix only"
            else:
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as archive:
                    plain = archive.read(64 * 1024 * 1024 + 1)
                if len(plain) > 64 * 1024 * 1024:
                    status, error = "size_limit", "Decompressed file exceeds 64 MiB pilot budget"
                else:
                    texts[suffix] = plain.decode("utf-8")
                    status = "ok"
        except urllib.error.HTTPError as exc:
            http_status, body, error, status = exc.code, exc.read(8192), str(exc), "http_error"
        except (urllib.error.URLError, OSError, EOFError, UnicodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        filename = f"{suffix}.response"
        (run / filename).write_bytes(body)
        log.update(status=status, http_status=http_status, bytes_saved=len(body), error=error,
                   raw_path=filename, raw_sha256=hashlib.sha256(body).hexdigest())
        logs.append(log)
        write_json(run / "requests.json", logs)
        print(json.dumps(log, ensure_ascii=True), flush=True)
        if status != "ok":
            raise SystemExit(2)
        if suffix == "toc.json.gz":
            time.sleep(6)
    toc_documents, hits = extract_matches(texts["toc.json.gz"], texts["ngrams.txt.gz"], args.file_stamp)
    write_json(run / "matches.json", hits)
    write_json(run / "summary.json", {"file_stamp": args.file_stamp, "toc_documents": toc_documents,
                                      "match_pattern": PATTERN,
                                      "matched_documents": len(hits),
                                      "note": "One-minute probe only; no country or tone inferred."})
    print(f"Run: {run}; TOC documents={toc_documents}, matching documents={len(hits)}", flush=True)


if __name__ == "__main__":
    main()
