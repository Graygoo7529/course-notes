"""Shared paths and configuration; downloads and analysis never modify the source CSV."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL = ROOT / "model"
REPORT = ROOT / "report"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def setup() -> None:
    for directory in (DATA, MODEL, REPORT, ROOT / ".cache"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL / "hub"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
