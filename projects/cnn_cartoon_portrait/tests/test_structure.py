"""新版的数值、边界、交互坐标和端到端契约。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".deps").is_dir():
    sys.path.insert(0, str(ROOT / ".deps"))

import cv2
import numpy as np

from cartoon_portrait.cli import main
from cartoon_portrait.colors import ColorConfig, build_boundary_weights, edge_aware_smooth, map_lightness
from cartoon_portrait.features import (
    ChannelFeatures, FeatureConfig, FusionConfig, comparison_strengths,
    extract_channel_features, fuse_features,
)
from cartoon_portrait.lines import hysteresis, map_ink, outline_mask, thin_edges
from cartoon_portrait.pipeline import masked_gaussian
from cartoon_portrait.selection import ViewConfig, collect_selection, make_preview, view_rect_to_image, view_to_image
from cartoon_portrait.workflow import StyleConfig, stylize


class FeatureTests(unittest.TestCase):
    def test_signed_masked_gaussian_matches_full_2d_kernel(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(17, 19, 2)).astype(np.float32)
        mask = rng.random((17, 19)) > .3
        before = source.copy()
        kernel = cv2.getGaussianKernel(7, 1, cv2.CV_32F)
        numerator = cv2.filter2D(source * mask[..., None], -1, kernel @ kernel.T, borderType=cv2.BORDER_REFLECT_101)
        denominator = cv2.filter2D(mask.astype(np.float32), -1, kernel @ kernel.T, borderType=cv2.BORDER_REFLECT_101)
        expected = numerator[mask] / denominator[mask, None]
        actual = masked_gaussian(source, mask, 1)
        np.testing.assert_allclose(actual[mask], expected, atol=3e-7)
        np.testing.assert_array_equal(actual[~mask], 0)
        np.testing.assert_array_equal(source, before)
        self.assertLess(actual.min(), 0)

    def test_equal_gray_color_edge_survives_fusion(self) -> None:
        image = np.full((15, 21, 3), .5, dtype=np.float32)
        image[:, 10:] = [.7, .5 - .2 * .299 / .587, .5]
        mask = np.ones(image.shape[:2], dtype=bool)
        features = extract_channel_features(image, mask, FeatureConfig(sigma=0))
        fusion = fuse_features(features)
        controls = comparison_strengths(features, mask, FeatureConfig(sigma=0), FusionConfig())
        self.assertLess(float(controls["gray_strength"].max()), 1e-7)
        self.assertGreater(float(fusion.strength.max()), .06)
        np.testing.assert_allclose(fusion.theta[:, 9:11], 0, atol=1e-6)

    def test_same_channels_reduce_to_scalar_and_rotate_consistently(self) -> None:
        ramp = np.linspace(0, 1, 21, dtype=np.float32)[None, :, None]
        image = np.tile(ramp, (15, 1, 3))
        mask = np.ones(image.shape[:2], dtype=bool)
        features = extract_channel_features(image, mask, FeatureConfig(sigma=0))
        structure = fuse_features(features)
        np.testing.assert_allclose(structure.strength, features.channel_strength[..., 0], atol=1e-7)
        rotated = fuse_features(extract_channel_features(
            np.rot90(image).copy(), np.rot90(mask).copy(), FeatureConfig(sigma=0),
        ))
        np.testing.assert_allclose(rotated.jxx, np.rot90(structure.jyy), atol=1e-7)
        np.testing.assert_allclose(rotated.jyy, np.rot90(structure.jxx), atol=1e-7)
        np.testing.assert_allclose(rotated.strength, np.rot90(structure.strength), atol=1e-7)

    def test_degenerate_directions_and_opposite_gradients(self) -> None:
        guide = np.zeros((2, 2, 3), dtype=np.float32)
        gx, gy = guide.copy(), guide.copy()
        empty = fuse_features(ChannelFeatures(guide, gx, gy, guide.copy()))
        self.assertFalse(empty.direction_valid.any())
        self.assertTrue(np.isfinite(empty.theta).all())
        gx[..., 0], gy[..., 1] = 1, 1
        crossing = fuse_features(ChannelFeatures(guide, gx, gy, np.hypot(gx, gy)))
        np.testing.assert_allclose(crossing.coherence, 0)
        self.assertFalse(crossing.direction_valid.any())
        gx[..., 1], gy[..., 1] = -1, 0
        opposite = fuse_features(ChannelFeatures(guide, gx, gy, np.hypot(gx, gy)))
        np.testing.assert_allclose(opposite.strength, np.sqrt(2 / 3), rtol=1e-6)
        self.assertTrue(opposite.direction_valid.all())

    def test_constant_foreground_and_background_changes_do_not_create_edges(self) -> None:
        image = np.random.default_rng(3).random((17, 23, 3)).astype(np.float32)
        mask = np.zeros((17, 23), dtype=bool)
        mask[3:, 4:20] = True
        mask[9:11, 8:10] = False
        image[mask] = [.7, .4, .2]
        features = extract_channel_features(image, mask)
        self.assertLess(float(fuse_features(features).strength.max()), 1e-6)
        changed = image.copy()
        changed[~mask] = 1 - changed[~mask]
        np.testing.assert_array_equal(extract_channel_features(changed, mask).gx, features.gx)
        for weights in ((0, 0, 0), (1, 1, 1), (-1, 1, 1), (np.nan, 0, 1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fuse_features(features, FusionConfig(weights))


class ColorTests(unittest.TestCase):
    def test_wls_matches_independent_dense_solve(self) -> None:
        rng = np.random.default_rng(2)
        height, width = 4, 5
        mask = np.ones((height, width), dtype=bool)
        source = rng.normal(size=mask.shape).astype(np.float32)
        right = rng.random((height, width - 1)).astype(np.float32)
        down = rng.random((height - 1, width)).astype(np.float32)
        smoothness = 3.2
        matrix = np.eye(height * width)
        for y in range(height):
            for x in range(width):
                p = y * width + x
                neighbors = []
                if x + 1 < width:
                    neighbors.append((p + 1, right[y, x]))
                if y + 1 < height:
                    neighbors.append((p + width, down[y, x]))
                for q, weight in neighbors:
                    matrix[p, p] += smoothness * weight
                    matrix[q, q] += smoothness * weight
                    matrix[p, q] -= smoothness * weight
                    matrix[q, p] -= smoothness * weight
        expected = np.linalg.solve(matrix, source.ravel()).reshape(mask.shape)
        actual, residual, _ = edge_aware_smooth(source, right, down, mask, smoothness, tolerance=2e-7)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertLessEqual(residual, 2e-7)
        self.assertLess(np.linalg.norm(matrix @ actual.ravel() - source.ravel()) / np.linalg.norm(source), 1e-6)

    def test_wls_constant_components_single_pixel_and_nonconvergence(self) -> None:
        mask = np.ones((5, 7), dtype=bool)
        mask[:, 3] = False
        source = np.where(np.indices(mask.shape)[1] < 3, -.4, .8).astype(np.float32)
        right = (mask[:, :-1] & mask[:, 1:]).astype(np.float32)
        down = (mask[:-1] & mask[1:]).astype(np.float32)
        result, residual, _ = edge_aware_smooth(source, right, down, mask, 10)
        np.testing.assert_array_equal(result[mask], source[mask])
        self.assertEqual(residual, 0)
        tiny = np.array([[.4]], dtype=np.float32)
        actual, _, _ = edge_aware_smooth(tiny, np.zeros((1, 0), np.float32), np.zeros((0, 1), np.float32), np.ones((1, 1), bool), 10)
        np.testing.assert_array_equal(actual, tiny)
        noise = np.random.default_rng(4).random(mask.shape).astype(np.float32)
        with self.assertRaises(ArithmeticError):
            edge_aware_smooth(noise, right, down, mask, 100, max_iterations=1)
        right[:, 2:4] = 1
        with self.assertRaises(ValueError):
            edge_aware_smooth(source, right, down, mask, 1)

    def test_soft_mapping_monotonic_range_and_identity(self) -> None:
        ramp = np.linspace(0, 1, 10001, dtype=np.float32)[None, :]
        for levels in (2, 6, 12):
            for transition in (.05, .25, .5):
                result = map_lightness(ramp, levels, 1, transition)
                self.assertGreaterEqual(float(result.min()), 0)
                self.assertLessEqual(float(result.max()), 1 + 1e-6)
                self.assertGreaterEqual(float(np.diff(result).min()), -1e-6)
                np.testing.assert_allclose(result[:, [0, -1]], [[0, 1]], atol=1e-6)
        np.testing.assert_array_equal(map_lightness(ramp, 6, 0, .25), ramp)
        for transition in (0, .6, np.nan):
            with self.assertRaises(ValueError):
                map_lightness(ramp, 6, .5, transition)

    def test_boundary_weights_restrict_crossings_and_mask(self) -> None:
        image = np.full((10, 15, 3), .2, dtype=np.float32)
        image[:, 8:] = .8
        mask = np.ones(image.shape[:2], dtype=bool)
        mask[0, :] = False
        structure = fuse_features(extract_channel_features(image, mask, FeatureConfig(sigma=0)))
        right, down, _ = build_boundary_weights(structure.guide, structure, mask, ColorConfig())
        np.testing.assert_allclose(right[2:, 2], 1)
        self.assertLess(float(right[2:, 7].max()), .001)
        np.testing.assert_array_equal(down[0], 0)

    def test_decomposition_identity_and_no_input_mutation(self) -> None:
        image = np.random.default_rng(5).uniform(.1, .9, (12, 17, 3)).astype(np.float32)
        mask = np.ones(image.shape[:2], dtype=bool)
        mask[:2] = False
        original = image.copy()
        config = StyleConfig(colors=ColorConfig(amount=0, detail=1))
        result = stylize(image, mask, config)
        lightness = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)[..., 0] / 100
        np.testing.assert_allclose((result.color_debug.base + result.color_debug.detail)[mask], lightness[mask], atol=1e-7)
        np.testing.assert_allclose(result.color_debug.mapped[mask], lightness[mask], atol=1e-7)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(result.rgba[..., 3], mask.astype(np.float32))


class LineTests(unittest.TestCase):
    def test_connected_weak_line_is_retained_but_isolated_weak_line_is_not(self) -> None:
        field = np.zeros((8, 12), dtype=np.float32)
        field[2, 2:7] = .04
        field[2, 2] = .1
        field[6, 8:10] = .04
        retained = hysteresis(field, np.ones(field.shape, bool), .02, .08)
        self.assertTrue(retained[2, 2:7].all())
        self.assertFalse(retained[6, 8:10].any())
        ink = map_ink(field, retained, .02, .1, .8)
        self.assertGreater(ink[2, 2], ink[2, 3])
        self.assertEqual(ink[6, 8], 0)

    def test_thinning_and_ambiguous_direction_fallback(self) -> None:
        field = np.zeros((7, 11), dtype=np.float32)
        field[:, 4:6] = .5
        mask = np.ones(field.shape, bool)
        result = thin_edges(field, np.zeros_like(field), np.ones_like(field), mask)
        self.assertEqual(np.count_nonzero(result), 7)
        preserved = thin_edges(field, np.zeros_like(field), np.zeros_like(field), mask)
        np.testing.assert_array_equal(preserved, field)

    def test_outline_does_not_draw_along_image_crop(self) -> None:
        mask = np.zeros((9, 11), dtype=bool)
        mask[2:, 2:9] = True
        outline = outline_mask(mask, 1)
        self.assertEqual(outline[-1, 5], 0)
        self.assertEqual(outline[-1, 2], 1)
        np.testing.assert_array_equal(outline_mask(np.ones_like(mask), 2), 0)


class PreviewAndWorkflowTests(unittest.TestCase):
    def test_enlarged_brush_coordinates_and_dirty_mask_cannot_be_accepted(self) -> None:
        image = np.full((197, 150, 3), .5, np.float32)
        labels = np.full((197, 150), 2, np.uint8)
        _, transform = make_preview(image, labels)
        callbacks: list[object] = []
        keys = [ord("f"), 13, ord("b"), 32, 13, 32]

        def register(name: str, callback: object) -> None:
            callbacks.append(callback)

        def events(delay: int) -> int:
            callback = callbacks[0]
            assert callable(callback)
            if len(keys) == 6:
                callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
                callback(cv2.EVENT_LBUTTONUP, 5, 5, 0, None)
            if len(keys) in (5, 3):
                coordinate = 10 if len(keys) == 5 else 20
                x = transform.offset_x + 4 * coordinate + 1
                y = transform.offset_y + 4 * coordinate + 1
                callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)
            return keys.pop(0)

        expected = np.ones(image.shape[:2], bool)
        with (
            patch("cartoon_portrait.selection.cv2.namedWindow"),
            patch("cartoon_portrait.selection.cv2.setMouseCallback", side_effect=register),
            patch("cartoon_portrait.selection.cv2.imshow"),
            patch("cartoon_portrait.selection.cv2.waitKey", side_effect=events),
            patch("cartoon_portrait.selection.cv2.getWindowProperty", return_value=1),
            patch("cartoon_portrait.selection.cv2.destroyWindow"),
            patch("cartoon_portrait.selection.segment_person", return_value=expected) as segment,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result, saved = collect_selection(image)
        self.assertEqual(segment.call_count, 2)
        self.assertEqual(saved[10, 10], cv2.GC_FGD)
        self.assertEqual(saved[20, 20], cv2.GC_BGD)
        self.assertEqual(saved[0, 0], cv2.GC_PR_BGD)
        np.testing.assert_array_equal(result, expected)

    def test_preview_upscales_small_photo_and_preserves_source(self) -> None:
        image = np.full((197, 150, 3), .5, np.float32)
        labels = np.full((197, 150), 2, np.uint8)
        source = image.copy()
        canvas, transform = make_preview(image, labels)
        self.assertEqual((transform.view_width, transform.view_height), (600, 788))
        self.assertGreaterEqual(canvas.shape[1], 760)
        self.assertIsNone(view_to_image(transform, 0, 0))
        self.assertIsNone(view_to_image(transform, transform.offset_x - 1, transform.offset_y))
        np.testing.assert_array_equal(image, source)
        x, y = transform.offset_x, transform.offset_y
        self.assertEqual(view_rect_to_image(transform, (x, y), (x + 599, y + 787)), (0, 0, 150, 197))

    def test_noninteger_preview_matches_nearest_neighbor_labels(self) -> None:
        image = np.full((7, 9, 3), .5, np.float32)
        labels = np.full((7, 9), 2, np.uint8)
        _, transform = make_preview(image, labels, config=ViewConfig(max_scale=2.37))
        indexes = np.arange(63, dtype=np.uint16).reshape(7, 9)
        expected = cv2.resize(indexes, (transform.view_width, transform.view_height), interpolation=cv2.INTER_NEAREST_EXACT)
        for y in range(transform.view_height):
            for x in range(transform.view_width):
                point = view_to_image(transform, x + transform.offset_x, y + transform.offset_y)
                assert point is not None
                self.assertEqual(int(indexes[point[1], point[0]]), int(expected[y, x]))

    def test_small_constant_full_pipeline_and_scharr(self) -> None:
        for size in ((1, 1), (1, 9), (9, 1), (12, 17)):
            with self.subTest(size=size):
                image = np.full((*size, 3), .5, np.float32)
                mask = np.ones(size, bool)
                config = replace(StyleConfig(), features=FeatureConfig(operator="scharr"))
                result = stylize(image, mask, config)
                np.testing.assert_array_equal(result.lines, 0)
                self.assertEqual(result.rgba.shape, (*size, 4))
                self.assertTrue(np.isfinite(result.rgba).all())

    def test_structure_cli_exports_raw_features_and_baseline_remains_available(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            image = np.full((13, 19, 3), 120, np.uint8)
            image[4:9, 8:10] = 30
            Image.fromarray(image).save(root / "照片.png")
            Image.fromarray(np.full((13, 19), 255, np.uint8)).save(root / "mask.png")
            arguments = ["--input", str(root / "照片.png"), "--mask", str(root / "mask.png")]
            structured = root / "structure"
            self.assertEqual(main(arguments + ["--output", str(structured)]), 0)
            with np.load(structured / "features.npz", allow_pickle=False) as features:
                self.assertEqual(features["gx"].shape, (13, 19, 3))
                self.assertLess(float(features["gx"].min()), 0)
                np.testing.assert_allclose(
                    features["base_lightness"] + features["detail_lightness"],
                    cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)[..., 0] / 100,
                    atol=1e-7,
                )
            parameters = json.loads((structured / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(parameters["pipeline"], "structure")
            self.assertLessEqual(parameters["diagnostics"]["solver_max_relative_residual"], 1e-5)
            self.assertTrue((structured / "features_overview.png").is_file())
            baseline = root / "baseline"
            self.assertEqual(main(arguments + ["--pipeline", "baseline", "--output", str(baseline)]), 0)
            self.assertFalse((baseline / "features.npz").exists())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments + ["--transition", "0", "--output", str(root / "invalid")]), 2)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
