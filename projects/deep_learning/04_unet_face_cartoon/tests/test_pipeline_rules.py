from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from unet_face_cartoon.alignment import FaceAligner, NoFaceError
from unet_face_cartoon.artifacts import write_report
from unet_face_cartoon.geometry import align_crop, limit_image, resize_matrix, transform_points
from unet_face_cartoon.labels import FaceLabel
from unet_face_cartoon.pipeline import Controls, Pipeline
from unet_face_cartoon.regions import edit_regions, head_foreground, inward_weight, white_input


def landmarks():
    points = np.column_stack((np.linspace(70, 180, 68), 110 + 30 * np.sin(np.linspace(0, 6, 68))))
    points[36], points[45] = [90, 100], [150, 115]
    return points


class GeometryTests(unittest.TestCase):
    def test_eyes_align_horizontally_and_inverse_recovers_points(self) -> None:
        image = np.zeros((240, 260, 3), dtype=np.uint8)
        points = landmarks()
        original_to_working = resize_matrix(0.5, 0.5)
        portrait = align_crop(image, points, original_to_working)
        self.assertEqual(portrait.rgb.shape, (512, 512, 3))
        self.assertAlmostEqual(portrait.landmarks[36, 1], portrait.landmarks[45, 1])
        restored = transform_points(portrait.landmarks, np.linalg.inv(portrait.original_to_portrait))
        expected = transform_points(points, np.linalg.inv(original_to_working))
        np.testing.assert_allclose(restored, expected, atol=1e-10)

    def test_crop_outside_image_is_white(self) -> None:
        image = np.zeros((240, 260, 3), dtype=np.uint8)
        portrait = align_crop(image, landmarks() - [60, 60], np.eye(3))
        self.assertTrue((portrait.rgb[0, 0] == 255).all())

    def test_invalid_landmarks_stop(self) -> None:
        image = np.zeros((240, 260, 3), dtype=np.uint8)
        for points in (np.zeros((68, 2)), np.full((68, 2), np.nan), np.zeros((5, 2))):
            with self.assertRaises(ValueError):
                align_crop(image, points, np.eye(3))

    def test_resize_records_pixel_centers(self) -> None:
        image = np.zeros((2000, 1000, 3), dtype=np.uint8)
        resized, matrix = limit_image(image, 1000)
        self.assertEqual(resized.shape, (1000, 500, 3))
        np.testing.assert_allclose(transform_points(np.array([[0, 0]]), matrix), [[-0.25, -0.25]])


class RegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        self.labels = np.zeros((64, 64), dtype=np.uint8)
        self.labels[8:56, 8:56] = FaceLabel.SKIN
        self.labels[25:35, 20:44] = FaceLabel.MOUTH
        self.labels[4:16, 8:56] = FaceLabel.HAIR

    def test_zero_controls_are_exact_identity(self) -> None:
        edited, masks = edit_regions(self.rgb, self.labels, 0)
        np.testing.assert_array_equal(edited, self.rgb)
        np.testing.assert_array_equal(white_input(edited, masks["foreground"]),
                                      white_input(self.rgb, masks["foreground"]))

    def test_skin_edit_never_changes_other_labels(self) -> None:
        edited, masks = edit_regions(self.rgb, self.labels, 1)
        outside = self.labels != FaceLabel.SKIN
        np.testing.assert_array_equal(edited[outside], self.rgb[outside])
        self.assertTrue((masks["skin_weight"][outside] == 0).all())
        self.assertTrue(np.any(edited != self.rgb))

    def test_hair_edit_is_local(self) -> None:
        edited, _ = edit_regions(self.rgb, self.labels, 0, (150, 40, 100), 0.8)
        outside = self.labels != FaceLabel.HAIR
        np.testing.assert_array_equal(edited[outside], self.rgb[outside])
        self.assertTrue(np.any(edited != self.rgb))

    def test_small_eye_holes_survive(self) -> None:
        self.labels[20, 20] = FaceLabel.LEFT_EYE
        edited, masks = edit_regions(self.rgb, self.labels, 1)
        self.assertEqual(masks["skin_weight"][20, 20], 0)
        np.testing.assert_array_equal(edited[20, 20], self.rgb[20, 20])

    def test_inward_weight_has_no_outward_spill(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        result = inward_weight(mask)
        self.assertTrue((result[~mask] == 0).all())
        self.assertLess(result[5, 5], result[10, 10])

    def test_neck_cloth_excluded_and_empty_background_white(self) -> None:
        labels = np.full((64, 64), FaceLabel.CLOTH, dtype=np.uint8)
        _, masks = edit_regions(self.rgb, labels, 0)
        self.assertTrue((white_input(self.rgb, masks["foreground"]) == 255).all())

    def test_isolated_false_hair_removed_but_small_eye_preserved(self) -> None:
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[10:40, 10:40] = FaceLabel.SKIN
        labels[25, 25] = FaceLabel.LEFT_EYE
        labels[50:55, 50:55] = FaceLabel.HAIR
        before = labels.copy()
        cleaned, raw = head_foreground(labels)
        self.assertEqual(raw[52, 52], 1)
        self.assertEqual(cleaned[52, 52], 0)
        self.assertEqual(cleaned[25, 25], 1)
        np.testing.assert_array_equal(labels, before)

    def test_invalid_controls_stop(self) -> None:
        for strength in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                Controls(skin_strength=strength)
        with self.assertRaises(ValueError):
            Controls(hair_strength=0.5)


class FlowTests(unittest.TestCase):
    def test_no_face_and_multiple_faces_are_explicit(self) -> None:
        aligner = object.__new__(FaceAligner)
        aligner.net = Mock()
        for result in (None, [], [landmarks(), landmarks()]):
            aligner.net.get_landmarks_from_image.return_value = result
            with self.assertRaises(NoFaceError):
                aligner.landmarks(np.zeros((256, 256, 3), dtype=np.uint8))

    def test_zero_controls_reuse_single_generation(self) -> None:
        pipeline = object.__new__(Pipeline)
        pipeline.device = "cpu"
        pipeline.aligner = Mock()
        pipeline.aligner.landmarks.return_value = landmarks()
        pipeline.models = Mock()
        pipeline.models.parse.return_value = np.ones((512, 512), dtype=np.uint8)
        pipeline.models.cartoon.return_value = np.full((256, 256, 3), 128, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "input.png"
            Image.new("RGB", (260, 240), "pink").save(path)
            record = pipeline.process(path, root / "result", Controls(0))
            self.assertEqual(record["generator_calls"], 1)
            self.assertEqual(record["output_difference_mae"], 0)
            self.assertTrue((root / "result/comparison.png").exists())
        pipeline.models.parse.assert_called_once()
        pipeline.models.cartoon.assert_called_once()

    def test_report_escapes_error_and_uses_relative_assets(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_report(root, [{"id": "bad", "status": "error", "error": "<script>oops</script>"}], {})
            html = (root / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
