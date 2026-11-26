"""Mobile UI regression with fake camera + fake native bridge, not Android inference.

Run against npm dev. Simulates no aspect-ratio support and stalled video callbacks.
The probe flag records the old behavior without requiring the assertions to pass.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / ".cache/qa/mobile-regression"
BRIDGE = """
Object.defineProperty(crypto, 'randomUUID', {value: undefined});
HTMLVideoElement.prototype.requestVideoFrameCallback = () => 1;
HTMLVideoElement.prototype.cancelVideoFrameCallback = () => {};
window.__uiTest = {preview: 0, capture: 0, saved: 0};
window.__TAURI_INTERNALS__ = {invoke: async (command, {payload}) => {
  if (command.endsWith('|initialize')) return {
    backend: 'UI test double', model: 'test', inputSize: 256, device: 'test', coldStartMs: 0
  };
  if (command.endsWith('|process_frame')) {
    const meta = JSON.parse(payload.meta);
    window.__uiTest[meta.kind]++;
    await new Promise(resolve => setTimeout(resolve, 25));
    return {meta: {...meta, mime: meta.kind === 'capture' ? 'image/png' : 'image/jpeg',
      processingMs: 25, faceStatus: 'test', renderer: 'test'}, image: payload.image};
  }
  if (command.endsWith('|save_png')) {window.__uiTest.saved++; return {path: 'test/CartoonCamera'};}
  return {};
}};
"""


def snapshot(page: Page) -> dict:
    return page.evaluate("""() => ({
      ...window.__uiTest,
      previewHeight: document.querySelector('.cartoon-frame').getBoundingClientRect().height,
      originalHeight: document.querySelector('.original-view .media-frame').getBoundingClientRect().height,
      canvasWidth: document.querySelector('.cartoon-frame canvas').width,
      resultPosition: document.querySelector('.modal') ? getComputedStyle(document.querySelector('.modal')).position : null,
      scrollHeight: document.documentElement.scrollHeight, viewportHeight: innerHeight,
      overflowX: document.documentElement.scrollWidth > innerWidth
    })""")


def assert_painted(page: Page, selector: str, filename: str):
    screenshot = page.locator(selector).screenshot(path=str(QA / filename))
    pixels = np.asarray(Image.open(io.BytesIO(screenshot)).convert("RGB"))
    assert pixels.std(axis=(0, 1)).max() > 10, f"Blank {selector}"


def swipe(page: Page, start_y: int, end_y: int):
    client = page.context.new_cdp_session(page)
    client.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": 180, "y": start_y}]})
    for y in np.linspace(start_y, end_y, 10)[1:]:
        client.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": 180, "y": float(y)}]})
        page.wait_for_timeout(25)
    client.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    client.detach()
    page.wait_for_timeout(350)


def main():
    QA.mkdir(parents=True, exist_ok=True)
    # Color bars provide an unmistakable nonblank camera image without any face data.
    rgb = np.zeros((480, 360, 3), dtype=np.uint8)
    rgb[:, :120] = [230, 75, 45]; rgb[:, 120:240] = [65, 200, 105]; rgb[:, 240:] = [55, 90, 230]
    import cv2
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).tobytes()
    video = QA / "bars.y4m"
    with video.open("wb") as file:
        file.write(b"YUV4MPEG2 W360 H480 F10:1 Ip A1:1 C420jpeg\n")
        for _ in range(20): file.write(b"FRAME\n" + yuv)
    probe = "--probe" in sys.argv
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True, args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-video-capture={video}"])
        try:
            context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True,
                has_touch=True, device_scale_factor=1, permissions=["camera"],
                user_agent="Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/83.0 Mobile Safari/537.36")
            context.add_init_script(BRIDGE)
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("http://127.0.0.1:1420")
            page.add_style_tag(content="* { aspect-ratio: auto !important; }")
            page.get_by_role("button", name="开启相机", exact=True).click()
            page.wait_for_function("document.querySelector('.phase').textContent === '实时预览'")
            page.wait_for_timeout(1200)
            before_capture = snapshot(page)
            if not probe:
                assert before_capture["previewHeight"] > 300, before_capture
                assert before_capture["preview"] > 0, before_capture
                assert_painted(page, ".cartoon-frame canvas", "cartoon.png")
                page.get_by_role("button", name="原图", exact=True).click()
                assert_painted(page, ".original-view video", "original.png")
                previous = snapshot(page)["preview"]
                page.get_by_role("button", name="漫画", exact=True).click()
                page.wait_for_function("count => window.__uiTest.preview > count", arg=previous)
            page.get_by_role("button", name="拍照成片", exact=True).click()
            page.get_by_role("dialog").wait_for()
            page.locator(".still-card img").evaluate("image => image.decode()")
            after_capture = snapshot(page)
            if not probe:
                assert after_capture["resultPosition"] == "static", after_capture
                assert not after_capture["overflowX"], after_capture
                page.get_by_role("button", name="保存 PNG", exact=True).click()
                page.wait_for_function("window.__uiTest.saved === 1")
                page.locator(".still-card").scroll_into_view_if_needed()
                previous_scroll = page.evaluate("scrollY")
                swipe(page, 300, 700)
                assert page.evaluate("scrollY") < previous_scroll - 30, "Result overlay blocked page touch scrolling"
                page.screenshot(path=str(QA / "after-capture.png"), full_page=True)
                # Short landscape screens must expose both close/save actions as well.
                page.set_viewport_size({"width": 844, "height": 390})
                assert not snapshot(page)["overflowX"]
                page.get_by_role("button", name="返回相机", exact=True).click()
                page.wait_for_function("document.querySelector('.phase').textContent === '实时预览'")
                page.get_by_role("button", name="停止相机", exact=True).click()
                count = snapshot(page)["preview"]
                page.wait_for_timeout(400)
                assert snapshot(page)["preview"] == count, "Sampling continued after stop"
                assert not errors, errors
            report = {"before_capture": before_capture, "after_capture": after_capture, "errors": errors,
                      "mode": "probe" if probe else "regression", "native_inference_tested": False}
            (QA / ("before.json" if probe else "after.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        finally:
            browser.close()


if __name__ == "__main__": main()
