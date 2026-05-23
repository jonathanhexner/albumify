"""spec-041 regression — Load the profile_1.json fixture through the UI
and assert no Streamlit widget error is raised.

This is the test that *should have* caught the StreamlitValueAboveMaxError
from yaw_max=999 vs slider max=90. We seed a profile with all the
permissive sentinels, expand the Profiles section, pick the profile,
click Load, then verify the page does NOT contain Streamlit's red
exception box.

Manual / opt-in — not part of pytest collection. Requires Streamlit
running on port 8889.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PROFILE_DIR = Path.home() / ".sim_bench" / "profiles_v2"
PROFILE_NAME = "profile_smoke.json"

# Permissive sentinels — these are the FCParams range edges that exposed
# the slider/number_input mismatch.
SMOKE_PROFILE = {
    "K": 5,
    "distance_threshold": 0.35,
    "min_cluster_size": 2,
    "yaw_max": 999.0,
    "pitch_max": 999.0,
    "roll_max": 999.0,
    "blur_min": 0.0,
    "max_faces_per_image_core": 10,
    "min_face_area": None,
    "require_pose": False,
    "det_score_min": None,
    "d10_k": 3,
    "exemplars_d10_threshold": 0.35,
    "N_exemplars_max": 10,
    "exemplar_suppression_radius": 0.2,
    "split_enabled": False,
    "split_diameter_threshold": 0.6,
    "split_distance_threshold": 0.3,
    "split_K": 3,
    "merge_enabled": True,
    "merge_candidate_threshold": 0.45,
    "merge_exemplar_threshold": 0.35,
    "use_adaptive_merge_threshold": True,
    "merge_exemplar_percentile": 90,
    "merge_global_percentile": 75,
    "merge_threshold_alpha": 1.0,
    "merge_threshold_beta": 0.5,
    "merge_use_cross_gate": True,
    "merge_cross_threshold": 0.4,
    "merge_cross_max_size": 5,
    "merge_support_frac": 0.3,
    "merge_support_min": 2,
    "merge_support_unique": False,
    "merge_margin": 0.05,
    "merge_diameter_expansion_factor": 1.5,
    "cluster_diameter_cap_enabled": True,
    "max_full_diameter": 1.2,
    "max_exemplar_diameter": 0.8,
    "attach_enabled": False,
    "attach_distance_threshold": 0.35,
    "K_attach": 5,
    "vote_min": 3,
    "margin": 0.1,
    "embed_cache_enabled": True,
}


def main() -> int:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILE_DIR / PROFILE_NAME
    profile_path.write_text(json.dumps(SMOKE_PROFILE, indent=2), encoding="utf-8")

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page()
        page.goto("http://localhost:8889", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("h1:has-text('Face Clustering')", timeout=15000)

        # Open the Profiles expander.
        page.get_by_text("Profiles", exact=False).first.click()
        page.wait_for_timeout(300)

        # Select profile_smoke.json from the dropdown.
        page.locator("[role='combobox']").first.click()
        page.wait_for_timeout(200)
        page.get_by_text(PROFILE_NAME, exact=False).first.click()
        page.wait_for_timeout(300)

        # Click Load. Quality Gate is `expanded=True` by default in run_tab.py
        # (and merge_enabled=True in our smoke profile triggers the merge
        # expander too), so the yaw_max / merge widgets are instantiated on
        # the rerun. If any of them is fed a session_state value outside its
        # widget bounds, Streamlit raises StreamlitValueAboveMaxError /
        # StreamlitValueBelowMinError and the error markdown shows up in the
        # body. No expander-clicking needed — Streamlit lazy-renders content
        # ONLY for collapsed sections; expanded ones render eagerly.
        page.get_by_role("button", name="Load").click()
        page.wait_for_timeout(2000)  # rerun + widget instantiation

        body = page.locator("body").inner_text()
        # Streamlit's red exception box has a deterministic heading.
        bad_markers = [
            "StreamlitValueAboveMaxError",
            "StreamlitValueBelowMinError",
            "StreamlitAPIException",
            "ValidationError",
            "Traceback",
        ]
        for marker in bad_markers:
            assert marker not in body, (
                f"Load raised a Streamlit error containing {marker!r}.\n"
                "This is the regression class that previously slipped: a profile "
                "with a value at the FCParams range edge crashed the widget."
            )

        # Positive assertion: the yaw_max widget label is present in the body.
        # If the widget had crashed, the label wouldn't render.
        assert "yaw_max" in body, (
            "yaw_max widget not visible after Load — Quality Gate section "
            "may not have rendered (default-expanded section content missing)."
        )

        page.screenshot(path="tests/manual/_v2_load_smoke.png", full_page=True)
        b.close()
    print("OK — profile loaded without widget error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
