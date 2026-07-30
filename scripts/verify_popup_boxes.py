"""spec-086: capture the Image Detail popup to confirm numbered face boxes draw
(they need the now-populated filter_scores)."""
import sys
from playwright.sync_api import sync_playwright

URL = "http://localhost:8511"
OUT = "specs/086-image-repository-over-rundb"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 1700})
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(4000)
        page.get_by_role("button", name="People & Faces").click()
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="View").first.click()
        page.wait_for_timeout(2500)
        # Click the first Detail button to open the popup dialog.
        page.get_by_role("button", name="Detail", exact=True).first.click()
        page.wait_for_timeout(2500)
        # Screenshot just the dialog if present, else full page.
        dlg = page.get_by_role("dialog")
        if dlg.count():
            dlg.first.screenshot(path=f"{OUT}/PROBE_popup_boxes.png")
        else:
            page.screenshot(path=f"{OUT}/PROBE_popup_boxes.png", full_page=True)
        print("captured popup")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
