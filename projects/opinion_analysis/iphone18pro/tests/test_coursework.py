"""Boolean semantics fixtures; none of these records are research data."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coursework_analysis import analyze, excluded, matcher, normalize, topic_comparison
from gdelt_probe import ROOT


class CourseworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "config/study.json").read_text(encoding="utf-8"))
        self.lexicon = json.loads((ROOT / "config/coursework_lexicon.json").read_text(encoding="utf-8"))

    def test_and_across_quadgrams_but_never_across_documents(self) -> None:
        toc = {str(i): {"url": f"https://example.invalid/{i}", "lang": "en", "title": "Fixture"}
               for i in (1, 2, 3)}
        grams = ("1\tiPhone 18 Pro Max\t1\n1\tbattery life is better\t1\n"
                 "2\tiPhone 18 Pro Max\t1\n3\tbattery life is better\t1\n")
        found = {row["doc_id"]: row for row in analyze(toc, grams, self.config, self.lexicon)}
        self.assertIn("performance_battery__en", found["1"]["queries"])
        self.assertNotIn("performance_battery__en", found["2"]["queries"])
        self.assertNotIn("3", found)

    def test_normalization_phrase_boundaries_and_exact_domain_exclusion(self) -> None:
        regex = matcher(["iphone 18 pro"])
        self.assertIsNotNone(regex.search(normalize("ＩＰＨＯＮＥ  18 Pro Max")))
        self.assertIsNone(regex.search(normalize("iphone 18 prototype")))
        self.assertTrue(excluded("https://newsroom.apple.com/story"))
        self.assertTrue(excluded("https://apple.com.cn/story"))
        self.assertFalse(excluded("https://notapple.com/story"))
        self.assertFalse(excluded("https://apple.com.example.invalid/story"))

    def test_local_comparison_uses_same_language_frame_and_unique_urls(self) -> None:
        toc = {"1": {"url": "https://example.invalid/es", "lang": "es", "title": "Fixture"},
               "2": {"url": "https://example.invalid/de", "lang": "de", "title": "Fixture"}}
        grams = ("1\tiPhone 18 Pro Max\t1\n1\tel precio es alto\t1\n"
                 "2\tiPhone 18 Pro Max\t1\n2\tprice is too high\t1\n")
        found = analyze(toc, grams, self.config, self.lexicon)
        comparison = topic_comparison(found + found, self.config, self.lexicon)
        self.assertEqual(comparison["frame_unique_urls"], 1)
        self.assertEqual(comparison["counts"]["price_value"], {"en": 0, "local": 1})
        self.assertNotIn("price_value__local", found[1]["queries"])


if __name__ == "__main__":
    unittest.main()
