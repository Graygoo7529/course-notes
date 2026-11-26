"""Read-only import checks using this interpreter; no installs or downloads."""
import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "torch": True, "torchvision": True, "timm": True, "cv2": True,
    "numpy": True, "PIL": True, "scipy": True, "matplotlib": True,
    "huggingface_hub": True, "safetensors": True, "ty": True,
    "mediapipe": False,
}


def probe(name: str) -> dict[str, Any]:
    if importlib.util.find_spec(name) is None:
        return {"status": "missing"}
    # A separate process makes a native import error attributable to its package.
    script = (
        f"import {name} as package, json; "
        "info={'status':'ok','version':getattr(package,'__version__',None)}; "
        + ("info.update(cuda_available=package.cuda.is_available(), "
           "torch_cuda=package.version.cuda); " if name == "torch" else "")
        + "print(json.dumps(info))"
    )
    try:
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", script],
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=30)
        if result.returncode:
            return {"status": "import_error", "detail": (result.stderr or result.stdout)[-1500:]}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"status": "check_failed", "detail": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/environment.json")
    parser.add_argument("--strict", action="store_true", help="Exit 2 if a core package is unavailable.")
    args = parser.parse_args()
    modules = {}
    for name, required in MODULES.items():
        modules[name] = {**probe(name), "required_for_core": required}
        print(f"{name:18} {modules[name]['status']}", flush=True)
    ready = all(item["status"] == "ok" for item in modules.values() if item["required_for_core"])
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(),
              "core_imports_ready": ready, "models_checked": False, "modules": modules}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Import readiness: {ready}; report: {args.output}")
    return 2 if args.strict and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
