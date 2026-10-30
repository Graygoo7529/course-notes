"""Optional browser integration smoke; requires Playwright and running npm dev.

Uses a synthetic camera stream made from the public sample photo.
It never opens the physical camera. Results are kept in ignored .cache/qa/.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / ".cache/qa"


def main():
    QA.mkdir(parents=True, exist_ok=True)
    with Image.open(ROOT.parent / "cnn_face_cartoon/data/input/grace_hopper.jpg") as source:
        frame = np.asarray(ImageOps.pad(source.convert("RGB"), (640, 480), color="#e7e4dc"))
    yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420).tobytes()
    video = QA / "sample.y4m"
    with video.open("wb") as file:
        file.write(b"YUV4MPEG2 W640 H480 F10:1 Ip A1:1 C420jpeg\n")
        for _ in range(30):
            file.write(b"FRAME\n" + yuv)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-video-capture={video}"])
        try:
            context = browser.new_context(permissions=["camera"], viewport={"width": 1440, "height": 1050}, accept_downloads=True)
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:1420")
            page.get_by_role("button", name="开启相机", exact=True).click()
            page.wait_for_function("document.querySelector('.phase')?.textContent === '实时预览'", timeout=60000)
            page.get_by_role("button", name="运行信息", exact=True).click()
            page.wait_for_function("document.querySelector('.debug')?.textContent.includes('detected')", timeout=30000)
            page.wait_for_timeout(5000)
            info = page.locator(".debug").inner_text()
            page.screenshot(path=str(QA / "desktop.png"), full_page=True)
            page.get_by_role("slider", name="笔画密度").fill("0.8")
            page.get_by_role("button", name="柔和粉彩", exact=True).click()
            page.wait_for_timeout(500)
            page.get_by_role("button", name="暂停处理", exact=True).click()
            snapshot = page.locator("canvas").evaluate("canvas => canvas.toDataURL()")
            page.wait_for_timeout(500)
            assert snapshot == page.locator("canvas").evaluate("canvas => canvas.toDataURL()")
            page.get_by_role("button", name="继续预览", exact=True).click()
            page.get_by_role("button", name="拍照精修", exact=True).click()
            page.get_by_role("dialog", name="精修结果").wait_for(timeout=60000)
            page.locator(".still-card img").evaluate("image => image.decode()")
            assert page.locator(".still-card img").evaluate("image => [image.naturalWidth,image.naturalHeight]") == [384, 288]
            with page.expect_download() as download:
                page.get_by_role("button", name="保存 PNG", exact=True).click()
            download.value.save_as(QA / "capture.png")
            page.get_by_role("button", name="关闭精修结果", exact=True).click()
            page.get_by_role("button", name="停止相机", exact=True).click()
            page.wait_for_function("document.querySelector('video').srcObject === null")
            page.get_by_role("button", name="开启相机", exact=True).click()
            page.wait_for_function("document.querySelector('.phase')?.textContent === '实时预览'", timeout=60000)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1500)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(QA / "mobile.png"), full_page=True)
            page.get_by_role("button", name="停止相机", exact=True).click()
            assert not errors, errors
            (QA / "browser.json").write_text(json.dumps({"runtime_info": info, "page_errors": errors,
                "camera": "synthetic video from public sample image", "capture_size": [384, 288],
                "checks": ["OpenVINO preview", "face detected", "parameters", "pause", "resume", "capture", "PNG download", "stop", "restart", "390px layout"]},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(info)
            print("Browser smoke passed")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
