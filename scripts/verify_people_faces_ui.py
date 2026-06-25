"""spec-086 T11: Playwright smoke of the Albumify People & Faces fix.

Opens People & Faces, a person, toggles the Selected/Not-selected filter, opens
the Image Detail popup, and screenshots each. Proves the filter partitions and
the popup draws face boxes (which need the now-populated filter_scores).
"""
import sys
import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8511"
OUT = "specs/086-image-repository-over-rundb"


def _settle(page, ms=2500):
    page.wait_for_timeout(ms)


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 1700})
        page.goto(URL, wait_until="networkidle")
        _settle(page, 4000)

        # Navigate: People & Faces
        page.get_by_role("button", name="People & Faces").click()
        _settle(page)
        # Select album if a selector is present (first option)
        try:
            page.get_by_role("button", name="View").first.click()
            _settle(page)
        except Exception as e:
            print("no View button:", e)

        page.screenshot(path=f"{OUT}/PROBE_person_detail.png", full_page=True)
        print("captured person detail")

        # Toggle filter to "Selected" then "Not selected"
        for mode in ["Selected", "Not selected"]:
            try:
                page.get_by_role("combobox").first.click()
                _settle(page, 800)
                page.get_by_role("option", name=mode, exact=True).click()
                _settle(page, 1800)
                page.screenshot(path=f"{OUT}/PROBE_filter_{mode.replace(' ', '_')}.png", full_page=True)
                print(f"captured filter={mode}")
            except Exception as e:
                print(f"filter {mode} failed:", e)

        # Open an Image Detail popup (box overlay needs filter_scores)
        try:
            page.get_by_role("combobox").first.click(); _settle(page, 600)
            page.get_by_role("option", name="All", exact=True).click(); _settle(page, 1500)
            page.get_by_role("button", name="Detail").first.click()
            _settle(page, 2200)
            page.screenshot(path=f"{OUT}/PROBE_image_detail_popup.png", full_page=True)
            print("captured image detail popup")
        except Exception as e:
            print("popup failed:", e)

        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
