"""Download pinned public model files, verifying LFS hashes and reusing local copies."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3

from common import CONFIG, MODEL, ROOT, setup, sha256, write_json


def download_one(key: str) -> dict:
    repo = CONFIG["models"][key]
    destination = MODEL / key
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    if CONFIG["proxy"]:
        session.proxies = {"http": CONFIG["proxy"], "https": CONFIG["proxy"]}
    session.verify = CONFIG["verify_ssl"]
    lock_path = ROOT / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    pinned = lock.get(key, {}).get("revision")
    api = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{pinned}" if pinned else "")
    response = session.get(api, params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    revision = metadata["sha"]
    entries = metadata["siblings"]
    names = {entry["rfilename"] for entry in entries}
    weights = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    files = []
    for entry in entries:
        name = entry["rfilename"]
        if name == weights or ("/" not in name and name.endswith((".json", ".txt", ".model", ".md"))) or name == "1_Pooling/config.json":
            files.append(entry)
    print(f"{key}: {repo} @ {revision} ({len(files)} files)", flush=True)
    manifest = {"repo": repo, "revision": revision, "files": []}
    for entry in files:
        name = entry["rfilename"]
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        size = entry.get("size")
        expected = entry.get("lfs", {}).get("sha256")
        if not (path.exists() and (size is None or path.stat().st_size == size) and (not expected or sha256(path) == expected)):
            for attempt in range(3):
                try:
                    with session.get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", stream=True, timeout=(30, 120)) as result:
                        result.raise_for_status()
                        temporary = path.with_suffix(path.suffix + ".part")
                        with temporary.open("wb") as handle:
                            for chunk in result.iter_content(4 * 1024 * 1024):
                                handle.write(chunk)
                        if size is not None and temporary.stat().st_size != size:
                            raise ValueError(f"Size mismatch: {name}")
                        if expected and sha256(temporary) != expected:
                            raise ValueError(f"Hash mismatch: {name}")
                        temporary.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            print(f"  {key}/{name}: {path.stat().st_size:,} bytes", flush=True)
        manifest["files"].append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(destination / "download_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    setup()
    if not CONFIG["verify_ssl"]:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=list(CONFIG["models"]))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download_one, args.models))
