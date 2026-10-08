from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "cnn_cartoon_portrait" / ".deps"))

from cartoon_portrait_v2.artifacts import export_run_bundle, save_run
from cartoon_portrait_v2.comic import V2Config, cartoonize
from cartoon_portrait_v2.features import extract_features
from cartoon_portrait_v2.pipeline import masked_gaussian


class V2Tests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:32, :40]
        self.mask = np.ones((32, 40), bool)
        self.image = np.stack([x / 39, y / 31, np.full_like(x, .35)], axis=-1).astype(np.float32)

    def test_masked_gaussian_does_not_pull_background_into_foreground(self):
        x = np.zeros((9, 9), np.float32)
        m = np.zeros_like(x, bool)
        x[4, 4], m[4, 4] = 1, True
        got = masked_gaussian(x, 1.0, m)
        self.assertAlmostEqual(float(got[4, 4]), 1.0, places=4)
        self.assertEqual(float(got[0, 0]), 0.0)

    def test_structure_tensor_is_finite(self):
        f = extract_features(self.image, self.mask)
        self.assertTrue(np.isfinite(f.edge).all())
        self.assertTrue(np.all((f.coherence >= 0) & (f.coherence <= 1)))
        self.assertEqual(f.gx.shape[1:], self.image.shape)
        self.assertEqual(len(f.scales), f.edges.shape[0])

    def test_end_to_end_and_manifest(self):
        result = cartoonize(self.image, self.mask, V2Config())
        self.assertEqual(result.rendered.rgba.shape[-1], 4)
        self.assertTrue(np.isfinite(result.rendered.rgb).all())
        with tempfile.TemporaryDirectory() as d:
            root = save_run(result, d, V2Config(), "input.png", "mask.png")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input"], "input.png")
            self.assertEqual(manifest["mask"], "mask.png")
            self.assertEqual([n["id"] for n in manifest["architecture"]["nodes"]],
                             ["features", "lines", "regions", "tone", "compose"])
            self.assertTrue((root / "index.html").exists())
            (root / "previous.zip").write_bytes(b"old export")
            (root / "unrelated.txt").write_text("not part of this run")
            archive = export_run_bundle(root)
            self.assertTrue(archive.exists())
            # Repeated export must not recursively include archives or stale files.
            archive = export_run_bundle(root)
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                portable = json.loads(bundle.read("manifest.json"))
                self.assertFalse(portable["bundle"]["available"])
                self.assertNotIn("source", portable)
                self.assertIsNone(bundle.testzip())
                with tempfile.TemporaryDirectory(prefix="观察室 transferred ") as moved:
                    bundle.extractall(moved)
                    for mapping in portable["images"].values():
                        for name in mapping.values():
                            self.assertTrue((Path(moved) / name).is_file(), name)
                    for name in ("input", "mask"):
                        self.assertTrue((Path(moved) / portable[name]).is_file())
            self.assertIn("index.html", names)
            self.assertIn("images/compose__cartoon.png", names)
            self.assertNotIn("observer_bundle.zip", names)
            self.assertNotIn("previous.zip", names)
            self.assertNotIn("unrelated.txt", names)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["bundle"]["available"])
            old_archive = archive.read_bytes()
            (root / manifest["images"]["features"]["input"]).unlink()
            with self.assertRaises(FileNotFoundError):
                export_run_bundle(root)
            self.assertEqual(old_archive, archive.read_bytes())


if __name__ == "__main__":
    unittest.main()
