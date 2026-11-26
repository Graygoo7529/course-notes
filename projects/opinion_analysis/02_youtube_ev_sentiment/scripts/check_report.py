"""Optional offline browser smoke check (requires Playwright and local Edge)."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from common import REPORT, ROOT


def main() -> None:
    output = ROOT / ".cache/qa"
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    requests: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, offline=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url) if request.url.startswith("http") else None)
        page.goto((REPORT / "index.html").as_uri(), wait_until="load")
        page.wait_for_function("document.querySelectorAll('#records .record').length === 20")
        assert page.locator("#search-count").inner_text().startswith("共 1082 条")
        assert page.evaluate("[...document.images].every(i=>i.complete && i.naturalWidth>0)")
        page.screenshot(path=str(output / "desktop.png"))
        page.locator("#search").fill("YT0964")
        assert page.locator("#records .record").count() == 1
        assert "gas over $4" in page.locator("#records").inner_text()
        page.locator("#search").fill("")
        page.locator("#type-filter").select_option("reply")
        page.locator("#sentiment-filter").select_option("negative")
        assert all("回复" in text and "负面" in text for text in page.locator("#records .record").all_inner_texts())
        page.get_by_text("情感配色", exact=True).click()
        assert page.evaluate("document.getElementById('semantic-map').data.filter(t=>t.visible===true).length") == 3
        page.locator("#semantic-map .scatterlayer .point").first.click(force=True)
        assert "YT" in page.locator("#selected-comment").inner_text()
        page.locator(".map-box").screenshot(path=str(output / "map.png"))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0,0)")
        page.screenshot(path=str(output / "mobile.png"))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert not errors, errors
        assert not requests, requests
        browser.close()
    result = {"status": "passed", "offline": True, "javascript_errors": errors,
              "external_requests": requests, "checks": ["embedded figures", "search", "filters", "map toggle", "map click", "desktop/mobile overflow"]}
    (output / "browser_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
