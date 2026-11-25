from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from unet_face_cartoon.image_contracts import generator_input, generator_output, read_rgb, resize_labels
from unet_face_cartoon.labels import FaceLabel, HEAD_LABELS, SKIN_EDIT_LABELS


class ImageContractTests(unittest.TestCase):
    def test_official_class_ids(self) -> None:
        self.assertEqual([int(label) for label in FaceLabel], list(range(19)))
        self.assertEqual(FaceLabel.HAIR, 13)
        self.assertEqual(FaceLabel.NOSE, 2)
        self.assertNotIn(FaceLabel.CLOTH, HEAD_LABELS)
        self.assertNotIn(FaceLabel.NECK, HEAD_LABELS)
        self.assertNotIn(FaceLabel.NOSE, SKIN_EDIT_LABELS)

    def test_nearest_resize_preserves_class_ids_and_width_height(self) -> None:
        labels = np.array([[0, 13], [2, 18]], dtype=np.uint8)
        resized = resize_labels(labels, (9, 5))
        self.assertEqual(resized.shape, (5, 9))
        self.assertEqual(set(np.unique(resized)), {0, 2, 13, 18})

    def test_invalid_labels_are_rejected(self) -> None:
        for labels in (np.array([[19]], dtype=np.uint8), np.empty((0, 0), dtype=np.uint8),
                       np.zeros((2, 2, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                resize_labels(labels, (8, 8))

    def test_generator_rgb_roundtrip(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        tensor = generator_input(rgb)
        self.assertEqual(tensor.shape, (1, 3, 256, 256))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertGreaterEqual(float(tensor.min()), -1)
        self.assertLessEqual(float(tensor.max()), 1)
        np.testing.assert_array_equal(generator_output(tensor[0]), rgb)

    def test_clip_before_uint8_conversion(self) -> None:
        output = np.zeros((3, 256, 256), dtype=np.float32)
        output[0], output[2] = -2, 2
        np.testing.assert_array_equal(generator_output(output)[0, 0], [0, 128, 255])
        output[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            generator_output(output)

    def test_wrong_input_shape_is_not_silently_resized(self) -> None:
        with self.assertRaises(ValueError):
            generator_input(np.zeros((512, 512, 3), dtype=np.uint8))

    def test_transparency_composites_on_white_and_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "透明头像.png"
            Image.new("RGBA", (2, 3), (255, 0, 0, 0)).save(path)
            rgb = read_rgb(path)
        self.assertEqual(rgb.shape, (3, 2, 3))
        self.assertTrue((rgb == 255).all())

    def test_exif_orientation_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rotated.jpg"
            image = Image.new("RGB", (2, 3), (255, 0, 0))
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_rgb(path).shape, (2, 3, 3))
