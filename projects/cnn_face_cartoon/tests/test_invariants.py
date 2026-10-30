"""Small correctness checks for the failure-prone geometric/numeric operations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial import KDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from face_cartoon.cli import cache_prepared, load_prepared
from face_cartoon.features import feature_difference
from face_cartoon.preprocess import FaceGeometry, normalize, prepare, unpad
from face_cartoon.rendering import color_layer
from face_cartoon.strokes import resample, select_lines, smooth_path, trace_paths


class PipelineInvariants(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        (ROOT / ".cache").mkdir(exist_ok=True)

    def test_normalization_ignores_padding_and_handles_constants(self):
        field = np.arange(100, dtype=np.float32).reshape(10, 10)
        mask = np.zeros_like(field, dtype=bool)
        mask[2:8, 2:8] = True
        expected = normalize(field, mask)
        field[~mask] = 1e8
        np.testing.assert_array_equal(expected, normalize(field, mask))
        self.assertFalse(normalize(np.ones_like(field), mask).any())
        self.assertFalse(normalize(field, np.zeros_like(mask)).any())

    def test_feature_difference_uses_direction_not_vector_magnitude(self):
        torch.manual_seed(42)
        features = torch.randn(1, 8, 12, 10)
        amplitude = torch.rand(1, 1, 12, 10) * 10 + 0.1
        np.testing.assert_allclose(feature_difference(features), feature_difference(features * amplitude), atol=1e-6)
        self.assertFalse(feature_difference(torch.ones(1, 8, 12, 10)).any())

    def test_letterbox_and_disabled_geometry_are_honest(self):
        rgb = np.full((90, 50, 3), 150, dtype=np.uint8)
        prepared = prepare(rgb, self.config["preprocess"], FaceGeometry(Path("missing.task"), enabled=False))
        w, h = prepared.transform.resized_size
        self.assertEqual(unpad(prepared.original_rgb, prepared.transform).shape, (h, w, 3))
        self.assertEqual(int(prepared.valid_mask.sum()), w * h)
        self.assertFalse(prepared.edge.any())
        self.assertFalse(prepared.face_groups)
        self.assertEqual(prepared.face_status, "disabled")
        self.assertTrue(all(not mask.any() for mask in prepared.masks.values()))
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            folder = Path(directory)
            cache_prepared(folder, prepared, "valid")
            restored = load_prepared(folder, "valid")
            self.assertIsNotNone(restored)
            assert restored is not None
            np.testing.assert_array_equal(restored.edge, prepared.edge)
            self.assertIsNone(load_prepared(folder, "changed_input"))

    def test_paths_preserve_branch_edges_and_closed_loop(self):
        cross = np.zeros((9, 9), bool)
        cross[4, 1:8] = True
        cross[1:8, 4] = True
        paths = trace_paths(cross)
        self.assertEqual(len(paths), 4)
        self.assertEqual(sum(len(p) - 1 for p, _ in paths), 12)
        ring = np.zeros((9, 9), bool)
        ring[2, 2:7] = ring[6, 2:7] = True
        ring[2:7, 2] = ring[2:7, 6] = True
        loops = trace_paths(ring)
        self.assertEqual(len(loops), 1)
        points, closed = loops[0]
        self.assertTrue(closed)
        np.testing.assert_array_equal(points[0], points[-1])
        self.assertEqual(len(points) - 1, 16)

    def test_spline_bounds_against_original_and_keeps_endpoints(self):
        points = np.array([[x, 5 + x % 2] for x in range(25)], np.float32)
        for bound in [1.5, 0.2, 0.0]:
            fitted, info = smooth_path(points, False, bound)
            np.testing.assert_array_equal(fitted[[0, -1]], points[[0, -1]])
            if info["smoothed"]:
                original = resample(points, 0.2)
                distance = max(KDTree(original).query(fitted)[0].max(), KDTree(fitted).query(original)[0].max())
                self.assertLessEqual(distance, bound)
            else:
                np.testing.assert_array_equal(fitted, points)
        _, info = smooth_path(points, False, 0)
        self.assertEqual(info["reason"], "bounded_fallback")

    def test_cnn_gate_is_not_cancelled_by_threshold_rescaling(self):
        edge = np.zeros((40, 40), np.float32)
        edge[5:35, 20] = 0.6
        direction = np.zeros_like(edge)
        masks = {key: np.zeros_like(edge, dtype=bool) for key in ["features", "cheeks"]}
        strong, _ = select_lines(edge, edge, direction, masks, self.config["face"], 0.5)
        attenuated, _ = select_lines(edge * 0.25, edge, direction, masks, self.config["face"], 0.5)
        self.assertGreater(int(strong.sum()), 0)
        self.assertEqual(int(attenuated.sum()), 0)

    def test_colors_stay_in_shared_palette(self):
        image = np.random.default_rng(42).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        result = color_layer(image, self.config["style"])
        actual = {tuple(c) for c in result.reshape(-1, 3)}
        expected = {tuple(c) for c in self.config["style"]["palette_rgb"]}
        self.assertTrue(actual <= expected)


if __name__ == "__main__":
    unittest.main()
