"""下载许可明确的 Commons 人像；原图与导出结果默认不进入 Git。"""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "CoursePortraitResearch/1.0 (local educational experiment)"}


def main() -> None:
    records = []
    directory = ROOT / "data/public"
    directory.mkdir(parents=True, exist_ok=True)
    for entry in json.loads((ROOT / "samples.json").read_text(encoding="utf-8")):
        target = directory / f"{entry['id']}.jpg"
        provenance = target.with_suffix(".json")
        if target.exists() and provenance.exists():
            record = json.loads(provenance.read_text(encoding="utf-8"))
            if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Sample changed: {target}")
        else:
            query = urlencode({
                "action": "query", "format": "json", "titles": entry["commons_title"],
                "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 1200,
            })
            with urlopen(Request(f"https://commons.wikimedia.org/w/api.php?{query}", headers=HEADERS),
                         timeout=40) as response:
                data = json.load(response)
            page = next(iter(data["query"]["pages"].values()))
            info = page["imageinfo"][0]
            url = info.get("thumburl", info["url"])
            with urlopen(Request(url, headers=HEADERS), timeout=60) as response:
                content = response.read()
            digest = hashlib.sha256(content).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Commons file changed; review before updating samples.json: {target}")
            partial = target.with_suffix(".part")
            partial.write_bytes(content)
            with Image.open(partial) as image:
                image.verify()
            partial.replace(target)
            record = {
                **entry, "download_url": url,
                "sha256": digest,
                "commons_metadata": info["extmetadata"],
                "changes": "Commons thumbnail; later aligned/cropped and algorithmically cartoonized",
            }
            provenance.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if record["sha256"] != entry["sha256"]:
            raise ValueError(f"Sample does not match pinned manifest: {target}")
        records.append(record)
        print(f"{entry['id']}: {target.name}, SHA256={record['sha256']}", flush=True)
    (directory / "sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
