"""Run from the project directory without installing the local source package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from face_cartoon.cli import main

if __name__ == "__main__":
    main()
