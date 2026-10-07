"""观察页的数字必须与真正运行的算子一致，包括边界与有符号特征。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / "src")]

import cv2
import numpy as np

from cartoon_portrait.comic import ComicConfig, cartoonize
from cartoon_portrait.comic_artifacts import export_comic
from cartoon_portrait.feature_bank import BankConfig
from cartoon_portrait.features import channel_gradients
from cartoon_portrait.inspection import _bool_value, architecture_description, export_inspection, gaussian_stages, gradient_kernel, main, tensor_stages
from cartoon_portrait.pipeline import _extend_foreground, masked_gaussian


class InspectionTests(unittest.TestCase):
    def test_architecture_and_view_boolean_contract(self) -> None:
        self.assertTrue(_bool_value("true"))
        self.assertFalse(_bool_value("false"))
        self.assertEqual(_bool_value("ON"), True)
        with self.assertRaises(Exception):
            _bool_value("maybe")
        arrays = {"mask": np.ones((2, 3), bool), "guides": np.zeros((1, 2, 3, 3), np.float32),
                  "gx": np.zeros((1, 2, 3, 3), np.float32), "edges": np.zeros((1, 2, 3), np.float32),
                  "dog": np.zeros((1, 2, 3), np.float32), "theta": np.zeros((1, 2, 3), np.float32),
                  "lab": np.zeros((2, 3, 3), np.float32), "chroma_edge": np.zeros((2, 3), np.float32)}
        arrays["gy"] = arrays["gx"]
        arrays["coherence"] = arrays["theta"]
        record = {"config": {"features": {"scales": [0.6]}}}
        stats = {"rgb_magnitude_correlations": [[None] * 3] * 3,
                 "tensor_weighted_gray_strength_correlation": None,
                 "ridge_removed_pixels": 0, "ridge_removed_above_low": 0}
        architecture = cast(dict[str, Any], architecture_description(arrays, cast(dict[str, object], record),
                                                                     cast(dict[str, object], stats)))
        self.assertEqual([node["id"] for node in architecture["nodes"]],
                         ["input", "bank", "line", "color", "render"])
        self.assertEqual(len(architecture["edges"]), 5)

    def test_gaussian_factorization_and_local_kernel_at_boundaries(self) -> None:
        rng = np.random.default_rng(11)
        image = rng.random((11, 13, 3), dtype=np.float32)
        mask = np.ones((11, 13), bool)
        mask[3:6, 4:8] = False
        for sigma in (.6, 1.2, 2.4):
            trace = gaussian_stages(image, mask, sigma)
            np.testing.assert_allclose(trace["guide"], masked_gaussian(image, mask, sigma), atol=2e-7)
        for operator in ("sobel", "scharr"):
            gx, gy = channel_gradients(image, mask, operator)
            pad = cv2.copyMakeBorder(_extend_foreground(image, mask), 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
            for y, x in ((0, 0), (10, 12), (3, 3), (6, 5)):
                for axis, stored in (("x", gx), ("y", gy)):
                    products = pad[y:y + 3, x:x + 3] * gradient_kernel(operator, axis)[..., None]
                    np.testing.assert_allclose(products.sum(axis=(0, 1)), stored[y, x], atol=7e-8)

    def test_tensor_preserves_opposite_channel_gradients_and_signed_xy(self) -> None:
        gx = np.zeros((1, 3, 5, 3), np.float32)
        gy = np.zeros_like(gx)
        gx[..., 0], gy[..., 0] = 1, -2
        gx[..., 1], gy[..., 1] = -1, 2
        terms, raw, smoothed = tensor_stages(gx, gy, np.ones((3, 5), bool), BankConfig())
        expected = np.broadcast_to([2 / 3, -4 / 3, 8 / 3], raw.shape)
        np.testing.assert_allclose(raw, expected, atol=2e-7)
        np.testing.assert_allclose(smoothed, expected, atol=5e-7)
        self.assertTrue((terms[..., 0, 1] < 0).all())

    def test_export_can_replay_crop_and_refuses_overwrite_or_changed_history(self) -> None:
        image = np.full((17, 21, 3), 190 / 255, np.float32)
        image[5:13, 8:11] = 25 / 255
        mask = np.ones((17, 21), bool)
        config = ComicConfig()
        result = cartoonize(image, mask, config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, output = root / "原始结果", root / "观察"
            export_comic(run, image, mask, result, config, {"pipeline": "comic"})
            report = export_inspection(run, output, (4, 3, 12, 11))
            self.assertTrue(report.exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["crop_xywh"], [4, 3, 12, 11])
            self.assertEqual(max(manifest["statistics"]["replay_max_errors"].values()), 0)
            self.assertTrue(all((output / m["file"]).is_file() for m in manifest["maps"]))
            with np.load(output / "trace.npz", allow_pickle=False) as data:
                self.assertEqual(data["tensor_terms"].shape, (3, 17, 21, 3, 3))
                np.testing.assert_array_equal(data["thinning_removed"] | data["path_removed"], result.strokes.rejected)
                self.assertFalse((data["thinning_removed"] & data["path_removed"]).any())
            original = report.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--run", str(run), "--output", str(output)]), 2)
            self.assertEqual(report.read_bytes(), original)
            with np.load(run / "features.npz", allow_pickle=False) as saved:
                changed = {key: saved[key] for key in saved.files}
            changed["line_score"][8, 8] += .1
            np.savez_compressed(run / "features.npz", allow_pickle=False, **changed)
            with self.assertRaisesRegex(ValueError, "不一致"):
                export_inspection(run, root / "不应生成")
            self.assertFalse((root / "不应生成").exists())


if __name__ == "__main__":
    unittest.main()
