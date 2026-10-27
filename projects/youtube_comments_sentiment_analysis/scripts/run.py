"""Run the complete assignment, reusing completed inference stages by default."""
import argparse
import subprocess
import sys

from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the four already downloaded model directories.")
    args = parser.parse_args()
    scripts = (["download_models.py"] if not args.offline else []) + ["analyze.py", "render_report.py"]
    for script in scripts:
        subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
