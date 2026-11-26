import hashlib
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from scripts.download_models import extract_generator, inspect_file


class DownloadCheckTests(unittest.TestCase):
    def test_small_file_is_not_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.part"
            path.write_bytes(b"failed")
            with self.assertRaises(ValueError):
                inspect_file(path, None)

    def test_large_error_page_is_not_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.part"
            path.write_bytes(b"<!DOCTYPE html>" + b" " * (1024 * 1024))
            with self.assertRaises(ValueError):
                inspect_file(path, None)

    def test_local_hash_does_not_claim_authenticity_or_inference(self) -> None:
        content = b"\x80" * (1024 * 1024)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.part"
            path.write_bytes(content)
            result = inspect_file(path, None)
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertFalse(result["matches_pinned_sha256"])
        self.assertFalse(result["inference_verified"])

    def test_pinned_hash_must_match(self) -> None:
        content = b"\x80" * (1024 * 1024)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.part"
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                inspect_file(path, "0" * 64)
            self.assertTrue(inspect_file(path, hashlib.sha256(content).hexdigest())["matches_pinned_sha256"])

    def test_extract_only_named_generator_to_fixed_path(self) -> None:
        content = b"\x80" * (1024 * 1024)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bundle.zip"
            target = root / "models" / "photo2cartoon_weights.pt"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("../../photo2cartoon_weights.pt", content)
                bundle.writestr("cartoon_data/private.jpg", b"not needed")
            extract_generator(archive, target)
            self.assertEqual(target.read_bytes(), content)
            self.assertFalse((root / "cartoon_data").exists())
            with self.assertRaises(FileExistsError):
                extract_generator(archive, target)

    def test_failed_extraction_hash_never_promotes_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bundle.zip"
            target = root / "target.pt"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("photo2cartoon_weights.pt", b"\x80" * (1024 * 1024))
            with self.assertRaises(ValueError):
                extract_generator(archive, target, "0" * 64)
            self.assertFalse(target.exists())

    def test_ambiguous_generator_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bundle.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("a/photo2cartoon_weights.pt", b"x")
                bundle.writestr("b/photo2cartoon_weights.pt", b"x")
            with self.assertRaises(ValueError):
                extract_generator(archive, root / "target.pt")
