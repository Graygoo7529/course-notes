"""下载官方资源；只在显式指定时提取生成器，不执行模型。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

import gdown

ROOT = Path(__file__).resolve().parents[1]


def inspect_file(path: Path, expected_sha256: str | None) -> dict[str, object]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(1024).lstrip().lower()
        if size < 1024 * 1024 or prefix.startswith((b"<", b"<!doctype", b"{")):
            raise ValueError(f"{path.name}: too small or looks like an error page")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path.name}: SHA256 mismatch")
    return {
        "bytes": size, "sha256": digest,
        "matches_pinned_sha256": expected_sha256 is not None,
        "inference_verified": False,
    }


def extract_generator(archive: Path, target: Path, expected_sha256: str | None = None) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    with ZipFile(archive) as bundle:
        members = [
            entry for entry in bundle.infolist()
            if entry.filename.rsplit("/", 1)[-1] == "photo2cartoon_weights.pt"
            and not entry.is_dir()
        ]
        if len(members) != 1:
            raise ValueError("Expected exactly one photo2cartoon_weights.pt in archive")
        member = members[0]
        if not 1024 * 1024 <= member.file_size <= 512 * 1024 * 1024:
            raise ValueError("Unexpected generator checkpoint size")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with bundle.open(member) as source, partial.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        inspect_file(partial, expected_sha256)
        partial.rename(target)


def main() -> int:
    manifest = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(manifest))
    parser.add_argument("--check", action="store_true", help="只检查本地文件，不联网")
    parser.add_argument("--extract-generator", action="store_true")
    args = parser.parse_args()
    if args.extract_generator and args.model != "photo2cartoon-bundle":
        parser.error("--extract-generator is only for photo2cartoon-bundle")
    entry = manifest[args.model]
    target = ROOT / "models" / entry["filename"]
    partial = target.with_name(target.name + ".part")
    try:
        if not target.exists():
            if args.check:
                raise FileNotFoundError(f"Missing model: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = gdown.download(
                url=entry["url"], output=str(partial), quiet=False,
                use_cookies=False, resume=True,
            )
            if downloaded is None:
                raise RuntimeError("Download did not produce a file")
            inspect_file(partial, entry["sha256"])
            partial.rename(target)
        record = inspect_file(target, entry["sha256"])
        record.update({
            "model": args.model, "source": entry["source"], "url": entry["url"],
            "file": str(target.relative_to(ROOT)),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        if args.extract_generator:
            generator = ROOT / "models/photo2cartoon/photo2cartoon_weights.pt"
            if not generator.exists() and not args.check:
                extract_generator(target, generator, entry["generator_sha256"])
            record["generator"] = {
                "file": str(generator.relative_to(ROOT)),
                **inspect_file(generator, entry["generator_sha256"]),
            }
        text = json.dumps(record, ensure_ascii=False, indent=2)
        print(text)
        if not args.check:
            target.with_suffix(target.suffix + ".download.json").write_text(
                text + "\n", encoding="utf-8",
            )
        return 0
    except Exception as exc:
        print(f"{args.model}: {exc}\nOfficial source: {entry['source']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
