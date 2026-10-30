"""Install the project Android SDK in a directory without spaces."""
from pathlib import Path
import hashlib
import os
import subprocess
import zipfile
import xml.etree.ElementTree as ET
import requests

SDK = Path(r"B:\tmp\realtime-android-sdk")
JAVA = Path(r"D:\IDEAS\jdk\jdk-21.0.8")


def main():
    SDK.mkdir(parents=True, exist_ok=True)
    manager = SDK / "cmdline-tools/latest/bin/sdkmanager.bat"
    session = requests.Session()
    session.trust_env = False
    session.proxies.update({"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
    if not manager.exists():
        response = session.get("https://dl.google.com/android/repository/repository2-1.xml", timeout=60)
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        package = next(e for e in tree if e.tag.endswith("remotePackage") and e.attrib.get("path") == "cmdline-tools;latest")
        archive = next(a for a in package.findall("archives/archive") if a.findtext("host-os") == "windows")
        url = archive.findtext("complete/url")
        checksum = archive.findtext("complete/checksum")
        archive_path = SDK / "command-tools.zip"
        with session.get("https://dl.google.com/android/repository/" + str(url), stream=True, timeout=120) as stream:
            stream.raise_for_status()
            with archive_path.open("wb") as file:
                for chunk in stream.iter_content(1024 * 1024): file.write(chunk)
        assert hashlib.sha1(archive_path.read_bytes()).hexdigest() == checksum, "SDK archive checksum mismatch"
        destination = SDK / "cmdline-tools/latest"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive_file:
            for entry in archive_file.infolist():
                relative = Path(entry.filename).relative_to("cmdline-tools")
                target = (destination / relative).resolve()
                if not target.is_relative_to(destination.resolve()): raise ValueError("Invalid zip path")
                if entry.is_dir(): target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive_file.read(entry))
    env = os.environ.copy()
    env.update(JAVA_HOME=str(JAVA), ANDROID_HOME=str(SDK), ANDROID_SDK_ROOT=str(SDK))
    args = [str(manager.with_name("android.exe")), f"--sdk={SDK}", "--no-metrics", "sdk", "install"]
    log = SDK / "install.log"
    with log.open("w", encoding="utf-8") as file:
        result = subprocess.run(args + ["platform-tools", "platforms/android-36", "build-tools/36.0.0", "ndk/28.2.13676358"],
                                env=env, input="y\n" * 30, text=True, stdout=file, stderr=subprocess.STDOUT)
    print(f"SDK exit={result.returncode}; log={log}", flush=True)
    if result.returncode: raise SystemExit(result.returncode)
    assert (SDK / "ndk/28.2.13676358/source.properties").exists(), "NDK installation incomplete"


if __name__ == "__main__": main()
