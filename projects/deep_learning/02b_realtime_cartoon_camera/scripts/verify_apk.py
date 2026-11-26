"""Check APK payloads without a connected Android device (not a runtime test)."""
import hashlib
import json
from pathlib import Path
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "dist-android/CartoonCamera-arm64-debug.apk"
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    report = {"apk": path.name, "bytes": path.stat().st_size, "sha256": digest}
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, "Corrupt ZIP entry"
        names = archive.namelist()
        assets = {}
        for name in ["mobilenetv3_features.onnx", "face_landmarker.task", "cartoon_config.json"]:
            content = archive.read("assets/" + name)
            assert content == (ROOT / "models/android" / name).read_bytes(), f"Asset differs: {name}"
            assets[name] = hashlib.sha256(content).hexdigest()
        libraries = [name for name in names if name.startswith("lib/") and name.endswith(".so")]
        assert {name.split("/")[1] for name in libraries} == {"arm64-v8a"}
        expected = {"librealtime_cartoon_camera_lib.so", "libonnxruntime.so", "libonnxruntime4j_jni.so",
                    "libopencv_java4.so", "libmediapipe_tasks_vision_jni.so", "libc++_shared.so"}
        assert expected.issubset({Path(name).name for name in libraries})
        for name in libraries:
            with archive.open(name) as file:
                header = file.read(4096)
            assert header[:5] == b"\x7fELF\x02", f"Not ELF64: {name}"
            assert struct.unpack_from("<H", header, 18)[0] == 183, f"Not AArch64: {name}"
            offset = struct.unpack_from("<Q", header, 32)[0]
            size, count = struct.unpack_from("<HH", header, 54)
            alignment = [struct.unpack_from("<Q", header, offset + i*size + 48)[0]
                         for i in range(count) if struct.unpack_from("<I", header, offset + i*size)[0] == 1]
            assert alignment and min(alignment) >= 16384, f"Library lacks 16 KB alignment: {name}"
        dex = b"".join(archive.read(name) for name in names if name.startswith("classes") and name.endswith(".dex"))
        for cls in [b"CartoonPlugin", b"CartoonEngine", b"CartoonMath"]:
            assert b"Lcn/coursenotes/cartoon/" + cls + b";" in dex
        report.update(assets=assets, libraries=libraries, elf_16kb_alignment=True, runtime_verified=False)
    (path.parent / "payload-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
