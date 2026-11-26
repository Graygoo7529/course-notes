"""Download the four CNN checkpoints and MediaPipe task; no inference or training."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
import urllib3

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
REGISTRY = ROOT / "models.json"
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def retrieve(session: requests.Session, url: str, path: Path, size: int | None = None,
             expected_hash: str | None = None) -> dict[str, Any]:
    if path.is_file() and expected_hash and digest(path) == expected_hash:
        print(f"  cached: {path.relative_to(ROOT)}", flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                with session.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    expected_size = size or int(response.headers.get("Content-Length", 0))
                    temporary = path.with_suffix(path.suffix + ".part")
                    total = 0
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            handle.write(chunk)
                            total += len(chunk)
                            print(f"\r  {path.name}: {total / 1048576:.1f} MiB", end="", flush=True)
                    print(flush=True)
                    if not total or (expected_size and total != expected_size):
                        raise ValueError(f"Incomplete download: {path.name}")
                    if expected_hash and digest(temporary) != expected_hash:
                        raise ValueError(f"SHA256 mismatch: {path.name}")
                    temporary.replace(path)
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def check_files(keys: list[str]) -> None:
    for key in keys:
        manifest_path = MODEL / ("face_landmarker.json" if key == "face" else f"{key}/download_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = ROOT / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Missing or damaged file: {path}")
        print(f"{key}: file checks passed")
    print("File integrity verified; model loading/inference is a separate check.")


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*registry, "face"], default=[*registry, "face"])
    parser.add_argument("--proxy", help="Optional proxy, e.g. http://127.0.0.1:7897; otherwise use the current environment.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for this process only.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify local files without network access.")
    mode.add_argument("--plan", action="store_true", help="Query URLs and metadata only; do not download weights or change registry.")
    args = parser.parse_args()
    if args.check:
        check_files(args.models)
        return
    with requests.Session() as session:
        if args.proxy:
            session.trust_env = False
            session.proxies = {"http": args.proxy, "https": args.proxy}
        session.verify = not args.insecure
        if args.insecure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        for key in args.models:
            if key == "face":
                if args.plan:
                    with session.head(FACE_URL, timeout=30, allow_redirects=True) as response:
                        response.raise_for_status()
                        print(f"face: HTTP {response.status_code}; {response.headers.get('Content-Length')} bytes")
                    continue
                manifest_path = MODEL / "face_landmarker.json"
                previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                expected = previous["files"][0]["sha256"] if previous.get("url") == FACE_URL else None
                info = retrieve(session, FACE_URL, MODEL / "face_landmarker.task", expected_hash=expected)
                save_json(manifest_path, {"url": FACE_URL, "version": "1", "files": [info]})
                continue
            entry = registry[key]
            repo = entry["hf_repo"]
            revision = entry.get("revision")
            api = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
            with session.get(api, params={"blobs": "true"}, timeout=30) as response:
                response.raise_for_status()
                metadata = response.json()
            revision = metadata["sha"]
            available = {item["rfilename"]: item for item in metadata["siblings"]}
            names = ["config.json", "model.safetensors", "README.md"]
            for name in names:
                if name not in available:
                    raise ValueError(f"Expected file absent: {repo}/{name}")
            print(f"{key}: {repo} @ {revision}; {sum(available[n].get('size', 0) for n in names):,} bytes", flush=True)
            if args.plan:
                continue
            entry.update(revision=revision, status="downloading")
            save_json(REGISTRY, registry)  # Pin before transfer so retries use the same commit.
            manifest_path = MODEL / key / "download_manifest.json"
            previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            hashes = {f["path"]: f["sha256"] for f in previous.get("files", [])} if previous.get("revision") == revision else {}
            files = []
            for name in names:
                item = available[name]
                path = MODEL / key / name
                expected = item.get("lfs", {}).get("sha256") or hashes.get(path.relative_to(ROOT).as_posix())
                files.append(retrieve(session, f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                                      path, item.get("size"), expected))
            save_json(manifest_path, {"repo": repo, "revision": revision, "files": files})
            entry["status"] = "downloaded"
            save_json(REGISTRY, registry)
    if not args.plan:
        print("Downloads complete. Run: python scripts/download_models.py --check")


if __name__ == "__main__":
    main()
