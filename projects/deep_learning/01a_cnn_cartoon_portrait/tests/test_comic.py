"""验证新路线的特征意义、区域约束、图拓扑与绘制透明度。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / "src")]

import cv2
import numpy as np
from PIL import Image

from cartoon_portrait.cli import main
from cartoon_portrait.comic import ComicConfig, cartoonize
from cartoon_portrait.feature_bank import BankConfig, extract_feature_bank
from cartoon_portrait.regions import RegionConfig, design_fills, fuse_region_features, organize_regions
from cartoon_portrait.rendering import RenderConfig, rasterize_regions, srgb_to_linear, linear_to_srgb
from cartoon_portrait.strokes import StrokeConfig, fuse_line_features, skeletonize, trace_paths, _fit


class BankTests(unittest.TestCase):
    def test_constant_mask_and_background_independence(self) -> None:
        rng = np.random.default_rng(4)
        image = rng.random((21, 25, 3)).astype(np.float32)
        mask = np.zeros((21, 25), bool)
        mask[3:18, 4:21] = True
        mask[10:12, 12:14] = False
        image[mask] = [.6, .3, .2]
        original = image.copy()
        bank = extract_feature_bank(image, mask)
        self.assertEqual(bank.gx.shape, (3, 21, 25, 3))
        self.assertLess(float(bank.edges.max()), 2e-6)
        self.assertLess(float(np.abs(bank.dog).max()), 2e-6)
        altered = image.copy()
        altered[~mask] = 1 - altered[~mask]
        other = extract_feature_bank(altered, mask)
        np.testing.assert_array_equal(bank.edges, other.edges)
        np.testing.assert_array_equal(bank.dog, other.dog)
        np.testing.assert_array_equal(image, original)

    def test_dark_bar_has_center_response_and_tangent(self) -> None:
        image = np.full((35, 37, 3), .8, np.float32)
        image[4:31, 17:20] = .1
        mask = np.ones((35, 37), bool)
        bank = extract_feature_bank(image, mask)
        self.assertGreater(float(bank.dog[0, 17, 18]), 0.04)
        self.assertLess(float(bank.dog[0, 17, 13]), 0)
        self.assertLess(abs(float(np.sin(bank.theta[0, 17, 18]))), .01)
        self.assertGreater(float(bank.coherence[0, 17, 18]), .9)
        # 梯度在边缘，暗结构在内部，验证它们没有被直接视为同一特征。
        self.assertLess(float(bank.edges[0, 17, 18]), float(bank.edges[0, 17, 16]))

    def test_scale_configuration_validation(self) -> None:
        for scales in ((), (1, .6), (0, 1), (.6, float("nan"))):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                BankConfig(scales=scales).validate()


class RegionTests(unittest.TestCase):
    def test_two_colors_remain_separated_and_connected(self) -> None:
        image = np.full((24, 30, 3), [.7, .3, .2], np.float32)
        image[:, 15:] = [.15, .5, .7]
        mask = np.ones((24, 30), bool)
        bank = extract_feature_bank(image, mask)
        cfg = RegionConfig(colors=2, merge_area=3)
        fields = fuse_region_features(bank, mask, cfg)
        self.assertLess(float(np.mean(fields.right[:, 14])), .15)
        scene = organize_regions(fields, mask, np.zeros_like(mask), cfg)
        self.assertNotEqual(scene.labels[12, 3], scene.labels[12, 26])
        for label in np.unique(scene.labels):
            if label:
                self.assertEqual(cv2.connectedComponents((scene.labels == label).astype(np.uint8), connectivity=4)[0], 2)

    def test_flat_region_has_no_forced_shadow_and_palette_override(self) -> None:
        image = np.full((20, 24, 3), [.65, .42, .3], np.float32)
        mask = np.ones((20, 24), bool)
        custom = ((.2, .6, .7),)
        cfg = ComicConfig(regions=RegionConfig(palette=custom), render=RenderConfig(scale=1))
        result = cartoonize(image, mask, cfg)
        self.assertFalse(result.fills.shadow.any())
        np.testing.assert_allclose(result.fills.flat, np.broadcast_to(custom[0], image.shape), atol=1e-6)
        self.assertEqual(len(result.strokes.strokes), 0)

    def test_shadow_disabled_and_base_detail_reconstruct(self) -> None:
        ramp = np.linspace(.25, .8, 40, dtype=np.float32)
        image = np.tile(ramp[None, :, None], (28, 1, 3))
        mask = np.ones((28, 40), bool)
        bank = extract_feature_bank(image, mask)
        cfg = RegionConfig(colors=2, shadow_depth=0)
        features = fuse_region_features(bank, mask, cfg)
        scene = organize_regions(features, mask, np.zeros_like(mask), cfg)
        fills = design_fills(bank, scene, mask, np.zeros_like(mask), cfg)
        self.assertFalse(fills.shadow.any())
        np.testing.assert_allclose(fills.base + fills.detail, bank.lightness, atol=1e-7)
        self.assertLessEqual(fills.solver_residual, 1e-5)


class StrokeTests(unittest.TestCase):
    def test_skeleton_keeps_small_components_and_loop(self) -> None:
        source = np.zeros((20, 25), bool)
        source[2:4, 2:4] = True
        source[7:17, 10:22] = True
        source[10:14, 13:19] = False
        result = skeletonize(source)
        self.assertTrue(result[2:4, 2:4].any())
        self.assertFalse(np.any(result & ~source))
        self.assertEqual(cv2.connectedComponents(source.astype(np.uint8), connectivity=8)[0],
                         cv2.connectedComponents(result.astype(np.uint8), connectivity=8)[0])
        self.assertTrue(any(closed for _, closed, _, _ in trace_paths(result)))

    def test_graph_branch_edges_are_not_duplicated(self) -> None:
        skeleton = np.zeros((11, 13), bool)
        skeleton[2:9, 6] = True
        skeleton[5, 2:11] = True
        paths = trace_paths(skeleton)
        self.assertEqual(len(paths), 4)
        segments = []
        for points, closed, _, _ in paths:
            self.assertFalse(closed)
            for a, b in zip(points.tolist(), points[1:].tolist()):
                segments.append(tuple(sorted((tuple(a), tuple(b)))))
        self.assertEqual(len(segments), len(set(segments)))
        self.assertEqual(len(segments), 14)

    def test_fit_bounds_and_endpoints(self) -> None:
        source = np.array([[1, 2], [2, 2], [3, 3], [4, 3], [5, 3]], np.float32)
        result = _fit(source, False, .3)
        self.assertLessEqual(float(np.linalg.norm(result - source, axis=1).max()), .30001)
        np.testing.assert_array_equal(result[[0, -1]], source[[0, -1]])

    def test_suppression_protection_and_hint_validation(self) -> None:
        image = np.full((25, 29, 3), .8, np.float32)
        image[5:21, 13:16] = .15
        mask = np.ones((25, 29), bool)
        bank = extract_feature_bank(image, mask)
        empty = np.zeros_like(mask)
        suppressed = np.ones_like(mask)
        lines = fuse_line_features(bank, mask, empty, suppressed, StrokeConfig())
        self.assertFalse(lines.retained.any())
        with self.assertRaises(ValueError):
            fuse_line_features(bank, mask, suppressed, suppressed, StrokeConfig())


class RenderTests(unittest.TestCase):
    def test_geometry_holes_and_background_are_preserved(self) -> None:
        labels = np.zeros((12, 15), np.int32)
        labels[2:10, 2:13] = 1
        labels[5:8, 6:10] = 2
        expanded = rasterize_regions(labels, 4)
        self.assertEqual(expanded.shape, (48, 60))
        self.assertEqual(expanded[25, 29], 2)
        self.assertEqual(expanded[12, 12], 1)
        self.assertEqual(expanded[0, 0], 0)

    def test_alpha_supersampling_and_no_dark_color_fringe(self) -> None:
        image = np.full((15, 19, 3), [.8, .2, .1], np.float32)
        mask = np.zeros((15, 19), bool)
        mask[3:12, 4:15] = True
        cfg = ComicConfig(strokes=StrokeConfig(outline_width=0), render=RenderConfig(scale=3, supersample=2))
        result = cartoonize(image, mask, cfg)
        rgba = result.rendered.rgba
        self.assertEqual(rgba.shape, (45, 57, 4))
        alpha = rgba[..., 3]
        self.assertTrue(((alpha > 0) & (alpha < 1)).any())
        self.assertTrue((rgba[alpha == 0, :3] == 0).all())
        expected = result.fills.palette[1]
        np.testing.assert_allclose(rgba[alpha > .01, :3], np.broadcast_to(expected, rgba[alpha > .01, :3].shape), atol=3e-5)

    def test_srgb_midpoint_and_roundtrip(self) -> None:
        # 线性光 50% 白对应约 .735 sRGB，而不是直接取 .5。
        self.assertAlmostEqual(float(linear_to_srgb(np.array([.5], np.float32))[0]), .735357, places=5)
        values = np.linspace(0, 1, 101, dtype=np.float32)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=2e-7)

    def test_degenerate_shapes_and_preallocation_limit(self) -> None:
        for shape in ((1, 1), (1, 7), (7, 1)):
            image = np.full((*shape, 3), .5, np.float32)
            mask = np.ones(shape, bool)
            result = cartoonize(image, mask, ComicConfig(render=RenderConfig(scale=2)))
            self.assertEqual(result.rendered.rgba.shape, (shape[0] * 2, shape[1] * 2, 4))
            self.assertTrue(np.isfinite(result.rendered.rgba).all())
        with self.assertRaises(ValueError):
            cartoonize(np.full((10, 10, 3), .5, np.float32), np.ones((10, 10), bool),
                       ComicConfig(render=RenderConfig(max_pixels=10)))


class ComicCliTests(unittest.TestCase):
    def test_default_exports_full_trace_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            image = np.full((21, 27, 3), 180, np.uint8)
            image[6:16, 12:15] = 30
            Image.fromarray(image).save(root / "输入.png")
            Image.fromarray(np.full((21, 27), 255, np.uint8)).save(root / "遮罩.png")
            output = root / "结果"
            args = ["--input", str(root / "输入.png"), "--mask", str(root / "遮罩.png"), "--output", str(output)]
            self.assertEqual(main(args), 0)
            with Image.open(output / "cartoon.png") as opened:
                self.assertEqual(opened.size, (108, 84))
                self.assertEqual(opened.mode, "RGBA")
            with np.load(output / "features.npz", allow_pickle=False) as data:
                self.assertEqual(data["gx"].shape, (3, 21, 27, 3))
                self.assertTrue(np.isfinite(data["dog"]).all())
                self.assertIn("regions", data.files)
            for name in ("report.html", "strokes.json", "palette.json", "lines_overview.png", "shadow.png", "render.npz"):
                self.assertTrue((output / name).is_file(), name)
            record = json.loads((output / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(record["pipeline"], "comic")
            prior = (output / "cartoon.png").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
                self.assertEqual(main(args[:-1] + [str(root / "bad"), "--amount", "1"]), 2)
            self.assertEqual(prior, (output / "cartoon.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
