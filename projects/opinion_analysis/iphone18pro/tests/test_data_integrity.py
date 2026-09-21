"""Synthetic checks only. These fixtures are not study observations."""

import json
import gzip
import hashlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gdelt_probe import ROOT, database, fetch, ingest, payload_items, pilot_tasks, registry, task
from ngrams_probe import extract_matches
from plot_timeline import read_daily
from summarize_probe import replay


class DataIntegrityTests(unittest.TestCase):
    def test_invalid_response_is_not_zero_articles(self) -> None:
        self.assertEqual(payload_items({"articles": []}, "ArtList"), [])
        for invalid in ({"error": "Rate limited"}, {}, [], {"articles": [{"title": "no URL"}]}):
            with self.assertRaises(ValueError):
                payload_items(invalid, "ArtList")

    def test_query_logic_and_end_boundary(self) -> None:
        config = json.loads((ROOT / "config" / "study.json").read_text(encoding="utf-8"))
        queries = {row["query_id"]: row for row in registry(config)}
        self.assertEqual(len(queries), 45)
        base = queries["base__global"]["query"]
        self.assertNotIn("-rumor", base)
        self.assertNotIn("launch OR", base)
        self.assertTrue(queries["price_value__us"]["query"].endswith("sourcecountry:unitedstates"))
        request = task(config, "base__global", "ArtList", "2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z")
        self.assertEqual(request["params"]["enddatetime"], "20260909235959")
        self.assertEqual(pilot_tasks(config, limit=5)[0]["params"]["maxrecords"], "5")

    def test_bad_timeline_is_rejected_before_storage(self) -> None:
        def response(value: object, norm: object = 100) -> dict:
            return {"timeline": [{"series": "Volume", "data": [
                {"date": "20260909T000000Z", "value": value, "norm": norm}]}]}

        for value in (True, float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                payload_items(response(value), "TimelineVolRaw")
        for norm in (True, float("inf"), -1):
            with self.subTest(norm=norm), self.assertRaises(ValueError):
                payload_items(response(1, norm), "TimelineVolRaw")
        repeated = response(1)
        repeated["timeline"][0]["data"].append({"date": "2026-09-09T00:00:00Z", "value": 2})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            payload_items(repeated, "TimelineVolRaw")
        self.assertEqual(len(payload_items(response(-3, None), "TimelineTone")), 1)

    def test_ingest_failure_keeps_audit_and_rolls_back_all_items(self) -> None:
        config = json.loads((ROOT / "config" / "study.json").read_text(encoding="utf-8"))
        spec = task(config, "base__global", "ArtList", "2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z")
        response = MagicMock()
        response.status, response.headers = 200, {}
        response.read.return_value = json.dumps({"articles": [
            {"url": "https://example.invalid/good", "title": "Synthetic"},
            {"url": "https://example.invalid/bad", "title": {"unexpected": "object"}},
        ]}).encode()
        response.__enter__.return_value = response
        connection = database(Path(":memory:"))
        try:
            connection.execute("INSERT INTO queries(query_id) VALUES ('base__global')")
            with tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                (run / "raw").mkdir()
                with patch("gdelt_probe.urllib.request.urlopen", return_value=response):
                    status, _, _ = fetch(connection, config, spec, run, 1, 1)
                self.assertEqual(status, "ingest_error")
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM article_hits").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT status, item_count FROM requests").fetchone(),
                                 ("ingest_error", None))
                meta = json.loads((run / "raw" / "r001_a1.meta.json").read_text())
                self.assertEqual(meta["status"], "ingest_error")
                self.assertEqual((run / "raw" / "r001_a1.body").read_bytes(), response.read.return_value)
        finally:
            connection.close()

    def test_quadgram_quotes_do_not_merge_tsv_rows(self) -> None:
        toc = '\n'.join(json.dumps({"ID": number}) for number in (1, 2))
        grams = '1\t"the iPhone 18 Pro\t1\n2\tiPhone 18 Pro models\t2\n'
        total, hits = extract_matches(toc, grams, "20260909201600")
        self.assertEqual(total, 2)
        self.assertEqual([hit["doc_id"] for hit in hits], ["1", "2"])
        with self.assertRaisesRegex(ValueError, "missing from TOC"):
            extract_matches(toc, "3\tiPhone 18 Pro models\t1\n", "20260909201600")
        with self.assertRaisesRegex(ValueError, "Duplicate TOC"):
            extract_matches(toc + '\n{"ID": 1}', grams, "20260909201600")

    def test_replay_rejects_corrupt_archive(self) -> None:
        file_stamp = "20260909201600"
        toc = '{"ID": 1, "url": "https://example.invalid/synthetic"}\n'
        grams = "1\tiPhone 18 Pro models\t1\n"
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            logs = []
            for name, text in (("toc.json.gz.response", toc), ("ngrams.txt.gz.response", grams)):
                body = gzip.compress(text.encode())
                (run / name).write_bytes(body)
                logs.append({"raw_path": name, "file_stamp": file_stamp, "status": "ok", "http_status": 200,
                             "bytes_saved": len(body), "raw_sha256": hashlib.sha256(body).hexdigest()})
            (run / "requests.json").write_text(json.dumps(logs), encoding="utf-8")
            _, hits = extract_matches(toc, grams, file_stamp)
            (run / "matches.json").write_text(json.dumps(hits), encoding="utf-8")
            self.assertEqual(len(replay(run)[1]), 1)
            path = run / "ngrams.txt.gz.response"
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 1
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                replay(run)

    def test_article_can_match_multiple_queries_without_duplicate_article(self) -> None:
        # In-memory database: no synthetic records are written to data/runs.
        connection = database(Path(":memory:"))
        try:
            article = {"url": "https://example.invalid/synthetic", "title": "Synthetic fixture"}
            for number, query_id in enumerate(("base__global", "ai__global"), 1):
                request_id = str(number)
                connection.execute("INSERT INTO queries(query_id) VALUES (?)", (query_id,))
                connection.execute("INSERT INTO requests(request_id, query_id) VALUES (?, ?)", (request_id, query_id))
                ingest(connection, request_id, {"query_id": query_id, "mode": "ArtList"}, [article])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM article_hits").fetchone()[0], 2)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO article_hits VALUES ('unknown', 'unknown', 'unknown', 1)")
        finally:
            connection.close()

    def test_failed_real_run_cannot_generate_a_volume_chart(self) -> None:
        run = ROOT / "data" / "runs" / "20260921T075025_117613Z"
        if not run.exists():
            self.skipTest("The archived pilot run is not available")
        with self.assertRaisesRegex(ValueError, "No successful"):
            read_daily(run)


if __name__ == "__main__":
    unittest.main()
