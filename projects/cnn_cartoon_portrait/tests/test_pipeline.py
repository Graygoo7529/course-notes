"""验证算子的关键不变量、透明度约定和可重复运行的文件管道。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".deps").is_dir():
    sys.path.insert(0, str(ROOT / ".deps"))

import cv2
import numpy as np
from PIL import Image

from cartoon_portrait.artifacts import load_mask, make_demo, preview
from cartoon_portrait.cli import main
from cartoon_portrait.pipeline import (
    Config, Selection, compose_rgba, extract_lines, quantize_colors, read_image,
    save_png, segment_person, smooth_foreground,
)
from cartoon_portrait.selection import ViewConfig, collect_selection, make_preview


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((31, 37), dtype=bool)
        self.mask[4:29, 7:30] = True
        self.mask[10:13, 10:13] = False  # 内孔同样不应被当作黑色前景。
        self.image = np.random.default_rng(0).random((31, 37, 3)).astype(np.float32)
        self.image[self.mask] = [0.7, 0.4, 0.2]

    def test_normalized_smoothing_preserves_foreground_constant(self) -> None:
        source = self.image.copy()
        base = smooth_foreground(source, self.mask, 2)
        expected = np.broadcast_to(np.array([0.7, 0.4, 0.2]), base[self.mask].shape)
        np.testing.assert_allclose(base[self.mask], expected, atol=2e-6)
        self.assertTrue(np.all(base[~self.mask] == 0))
        np.testing.assert_array_equal(source, self.image)
        source[~self.mask] = 1 - source[~self.mask]
        np.testing.assert_array_equal(base, smooth_foreground(source, self.mask, 2))

    def test_mask_edge_does_not_create_ink_for_constant_foreground(self) -> None:
        base = smooth_foreground(self.image, self.mask, 1.5)
        lines = extract_lines(base, self.mask, threshold=0.001)
        np.testing.assert_array_equal(lines, np.zeros_like(lines))

    def test_sobel_scale_and_internal_edge(self) -> None:
        image = np.zeros((15, 21, 3), dtype=np.float32)
        image[:, 10:] = 1
        mask = np.ones(image.shape[:2], dtype=bool)
        # 单位阶跃的 Sobel/8 最大响应为 0.5，宽度为两列。
        lines = extract_lines(image, mask, threshold=0, softness=1)
        np.testing.assert_allclose(lines[:, 9:11], 0.5, atol=1e-6)
        self.assertEqual(np.count_nonzero(lines[:, :9]), 0)
        self.assertEqual(np.count_nonzero(lines[:, 11:]), 0)
        self.assertEqual(extract_lines(image, mask, threshold=0.6).max(), 0)

    def test_quantization_controls_lightness_without_mutation(self) -> None:
        ramp = np.linspace(0.05, 0.95, 40, dtype=np.float32)
        source = np.repeat(ramp[None, :, None], 3, axis=2)
        original = source.copy()
        np.testing.assert_array_equal(quantize_colors(source, amount=0), source)
        result = quantize_colors(source, levels=5, amount=1)
        lightness = cv2.cvtColor(result, cv2.COLOR_RGB2Lab)[..., 0] / 100
        distance_to_level = abs(lightness * 4 - np.rint(lightness * 4))
        self.assertLess(float(distance_to_level.max()), 0.015)
        np.testing.assert_array_equal(source, original)

    def test_straight_alpha_and_preview(self) -> None:
        colors = np.full((4, 5, 3), 0.8, dtype=np.float32)
        lines = np.full((4, 5), 0.5, dtype=np.float32)
        alpha = np.full((4, 5), 0.25, dtype=np.float32)
        output = compose_rgba(colors, lines, alpha, strength=0.5)
        np.testing.assert_allclose(output[..., :3], 0.6)
        np.testing.assert_array_equal(output[..., 3], alpha)
        np.testing.assert_allclose(preview(output, 1), 0.9)
        np.testing.assert_allclose(colors, 0.8)
        np.testing.assert_allclose(lines, 0.5)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            smooth_foreground(self.image, np.zeros_like(self.mask))
        with self.assertRaises(ValueError):
            smooth_foreground(self.image, self.mask, np.nan)
        with self.assertRaises(ValueError):
            extract_lines(self.image, self.mask, softness=0)
        with self.assertRaises(ValueError):
            quantize_colors(self.image, levels=1)
        with self.assertRaises(ValueError):
            quantize_colors(self.image, amount=1.2)
        with self.assertRaises(ValueError):
            Config(strength=float("inf")).validate()
        broken = self.image.copy()
        broken[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            quantize_colors(broken)
        with self.assertRaises(ValueError):
            compose_rgba(self.image, self.mask.astype(np.float32), np.zeros((3, 3), dtype=np.float32))


class SegmentationTests(unittest.TestCase):
    def test_grabcut_keeps_subject_touching_bottom_border(self) -> None:
        image, truth, rect = make_demo()
        original = image.copy()
        mask = segment_person(image, Selection(rect=rect))
        iou = np.count_nonzero(mask & truth) / np.count_nonzero(mask | truth)
        self.assertGreater(iou, 0.99)
        self.assertTrue(mask[-1].any())
        np.testing.assert_array_equal(image, original)

    def test_labels_enforce_marks_and_are_not_mutated(self) -> None:
        image, truth, _ = make_demo()
        labels = np.where(truth, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        labels[250:255, 235:240] = cv2.GC_FGD
        labels[5:10, 5:10] = cv2.GC_BGD
        original = labels.copy()
        result = segment_person(image, Selection(labels=labels, iterations=2))
        self.assertTrue(result[250:255, 235:240].all())
        self.assertFalse(result[5:10, 5:10].any())
        np.testing.assert_array_equal(labels, original)

    def test_invalid_selection_is_rejected_before_opencv(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.float32)
        for selection in (
            Selection(), Selection(rect=(0, 0, 30, 20)),
            Selection(rect=(-1, 0, 20, 15)),
            Selection(rect=(10, 10, 30, 20)),
            Selection(labels=np.full((20, 30), 4, dtype=np.uint8)),
        ):
            with self.subTest(selection=selection.rect), self.assertRaises(ValueError):
                segment_person(image, selection)

    def test_interactive_coordinates_account_for_toolbar_and_accept_mask(self) -> None:
        # 模拟窗口事件，验证说明栏偏移不会让实际框选向下错位；不打开桌面窗口。
        image, _, _ = make_demo()
        callbacks: list[object] = []
        expected = np.zeros(image.shape[:2], dtype=bool)
        expected[45:560, 48:432] = True
        view_config = ViewConfig(max_scale=1)
        _, transform = make_preview(
            image, np.full(image.shape[:2], 2, dtype=np.uint8), config=view_config,
        )

        def register(name: str, callback: object) -> None:
            callbacks.append(callback)

        def key_events(delay: int) -> int:
            if len(keys) == 2:
                callback = callbacks[0]
                assert callable(callback)
                callback(cv2.EVENT_LBUTTONDOWN, 48 + transform.offset_x, 45 + transform.offset_y, 0, None)
                callback(cv2.EVENT_LBUTTONUP, 431 + transform.offset_x, 559 + transform.offset_y, 0, None)
            return keys.pop(0)

        keys = [13, 32]
        with (
            patch("cartoon_portrait.selection.cv2.namedWindow"),
            patch("cartoon_portrait.selection.cv2.setMouseCallback", side_effect=register),
            patch("cartoon_portrait.selection.cv2.imshow"),
            patch("cartoon_portrait.selection.cv2.waitKey", side_effect=key_events),
            patch("cartoon_portrait.selection.cv2.getWindowProperty", return_value=1),
            patch("cartoon_portrait.selection.cv2.destroyWindow"),
            patch("cartoon_portrait.selection.segment_person", return_value=expected),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            mask, labels = collect_selection(image, view_config=view_config)
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue((labels[expected] == cv2.GC_PR_FGD).all())
        self.assertTrue((labels[~expected] == cv2.GC_BGD).all())


class FilePipelineTests(unittest.TestCase):
    def test_unicode_png_roundtrip_and_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "中文目录" / "透明结果.png"
            rgba = np.full((6, 9, 4), 0.6, dtype=np.float32)
            rgba[..., 3] = np.linspace(0, 1, 9, dtype=np.float32)
            save_png(rgba, path)
            with Image.open(path) as opened:
                self.assertEqual(opened.mode, "RGBA")
                actual = np.asarray(opened)
            np.testing.assert_array_equal(actual, np.rint(rgba * 255).astype(np.uint8))
            # 读入透明图有明确约定：铺白底，而不是忽略 alpha。
            self.assertTrue(np.all(read_image(path)[:, 0] == 1))

    def test_exif_orientation_is_applied_before_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rotated.jpg"
            source = Image.new("RGB", (12, 8), "red")
            exif = Image.Exif()
            exif[274] = 6
            source.save(path, exif=exif)
            self.assertEqual(read_image(path).shape, (12, 8, 3))

    def test_cli_exports_reusable_mask_and_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            first, second, third = root / "first", root / "second", root / "third"
            self.assertEqual(main(["--pipeline", "structure", "--demo", "--output", str(first)]), 0)
            for name in (
                "input.png", "mask.png", "base.png", "colors.png", "lines.png",
                "cartoon.png", "preview_light.png", "preview_dark.png",
                "overview.png", "parameters.json", "selection_labels.png",
            ):
                self.assertTrue((first / name).is_file(), name)
            details = json.loads((first / "parameters.json").read_text(encoding="utf-8"))
            self.assertGreater(details["synthetic_mask_iou"], 0.99)
            # 复用已导出的规范化原图与遮罩，跳过耗时分割，仍应得到同一结果。
            with patch("cartoon_portrait.cli.segment_person", side_effect=AssertionError("不应重跑分割")):
                self.assertEqual(main(["--pipeline", "structure",
                    "--input", str(first / "input.png"), "--mask", str(first / "mask.png"),
                    "--output", str(second),
                ]), 0)
            with Image.open(second / "cartoon.png") as opened:
                rgba = np.asarray(opened)
                self.assertEqual(opened.mode, "RGBA")
            mask = load_mask(first / "mask.png", rgba.shape)
            np.testing.assert_array_equal(rgba[..., 3], mask.astype(np.uint8) * 255)
            # 两次处理的源图有一次 8 位量化，色差允许一个输出级左右。
            with Image.open(first / "cartoon.png") as opened:
                prior = np.asarray(opened).astype(np.int16)
            self.assertLessEqual(float(np.abs(prior - rgba.astype(np.int16)).mean()), 0.5)
            # 四类标签 PNG 也能重新初始化分割。
            self.assertEqual(main(["--pipeline", "structure",
                "--input", str(first / "input.png"), "--labels", str(first / "selection_labels.png"),
                "--output", str(third),
            ]), 0)
            original = (first / "cartoon.png").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--pipeline", "structure", "--demo", "--output", str(first)]), 2)
            self.assertEqual((first / "cartoon.png").read_bytes(), original)

    def test_soft_mask_is_not_silently_accepted_as_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"
            Image.new("L", (12, 8), 128).save(path)
            with self.assertRaises(ValueError):
                load_mask(path, (8, 12, 3))


if __name__ == "__main__":
    unittest.main()
