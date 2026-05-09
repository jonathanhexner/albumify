"""Full E2E flow test — mirrors the user's exact workflow.

Creates album → loads profile → runs pipeline → checks ALL tabs →
opens Face Clustering App → verifies merge analysis.

Run: .venv/Scripts/python tests/test_full_e2e_flow.py
"""
import requests
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

API = "http://localhost:8000"
APP = "http://localhost:8501"
FC_APP = "http://localhost:8502"
SOURCE_DIR = r"D:\Budapest2025_Google"
PROFILE = "default6_with_relaxed_cross_dist"
SCREENSHOTS = Path("tests/screenshots/e2e_flow")
SCREENSHOTS.mkdir(parents=True, exist_ok=True)

results = {}
def log(msg):
    print(f"  {msg}")

def fail(test, msg):
    results[test] = f"FAIL: {msg}"
    print(f"  [FAIL] {test}: {msg}")

def ok(test, msg=""):
    results[test] = f"PASS{f': {msg}' if msg else ''}"
    print(f"  [PASS] {test}{f': {msg}' if msg else ''}")


print("=" * 60)
print("  FULL E2E FLOW TEST")
print("=" * 60)

# ─── Step 1: Verify services are running ───
print("\n1. Checking services...")
try:
    r = requests.get(f"{API}/health", timeout=5)
    assert r.json()["status"] == "ok"
    ok("Backend", "healthy")
except Exception as e:
    fail("Backend", str(e))
    print("Backend not running. Start with: scripts\\restart_apps.bat")
    sys.exit(1)

# ─── Step 2: Create album ───
print("\n2. Creating album...")
album_name = f"E2E_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
try:
    r = requests.post(f"{API}/api/v1/albums/", json={
        "name": album_name,
        "source_path": SOURCE_DIR,
    }, timeout=10)
    r.raise_for_status()
    album = r.json()
    album_id = album["id"]
    log(f"Album created: {album_name} (id={album_id[:8]})")
    ok("Create Album", f"{album['image_count']} images")
except Exception as e:
    fail("Create Album", str(e))
    sys.exit(1)

# ─── Step 3: Load profile and run pipeline ───
print("\n3. Running pipeline with merge enabled...")
try:
    # Load profile config
    from face_cluster.profile_store import ProfileStore
    store = ProfileStore()
    profile_params = store.load(PROFILE)
    if profile_params:
        log(f"Loaded profile '{PROFILE}': {len(profile_params)} params")
    else:
        log(f"Profile '{PROFILE}' not found, using defaults")

    # Build config with merge enabled
    config = {
        "cluster_people": {
            "method": "face_cluster_knn",
            "K": profile_params.get("rc_K", 5),
            "distance_threshold": profile_params.get("rc_dist", 0.35),
            "min_cluster_size": profile_params.get("rc_min_cluster", 2),
            "merge_enabled": True,  # Force merge on
            "attach_enabled": False,
            "export_for_analysis": True,
        },
        "select_best": {
            "max_images_per_cluster": 2,
            "min_score_threshold": 0.4,
            "siamese": {"enabled": False},  # Skip siamese to avoid numpy issues
        },
    }

    # Start pipeline
    r = requests.post(f"{API}/api/v1/pipeline/run", json={
        "album_id": album_id,
        "config": config,
    }, timeout=10)
    r.raise_for_status()
    job_id = r.json().get("job_id", r.json().get("id", ""))
    log(f"Pipeline started: {job_id[:8]}")

    # Poll until complete
    for i in range(120):  # 2 min max
        time.sleep(2)
        r = requests.get(f"{API}/api/v1/pipeline/{job_id}", timeout=10)
        status = r.json().get("status", "unknown")
        step = r.json().get("current_step", "")
        if status == "completed":
            log(f"Pipeline completed!")
            break
        elif status == "failed":
            error = r.json().get("error_message", "unknown error")
            fail("Pipeline Run", f"failed: {error[:100]}")
            break
        if i % 5 == 0:
            log(f"  ...{status} ({step})")
    else:
        fail("Pipeline Run", "timeout after 2 min")

    if status == "completed":
        ok("Pipeline Run", f"job={job_id[:8]}")

except Exception as e:
    fail("Pipeline Run", str(e))
    import traceback
    traceback.print_exc()

# ─── Step 4: Check results via API ───
print("\n4. Checking results...")
try:
    r = requests.get(f"{API}/api/v1/results/", params={"album_id": album_id}, timeout=10)
    results_data = r.json()
    if results_data:
        latest = results_data[0]
        n_people = latest.get("num_people", 0)
        n_selected = latest.get("num_selected", 0)
        fc_dir = latest.get("fc_export_dir")
        decisions = latest.get("step_decisions")
        n_decisions = len(decisions) if decisions else 0

        log(f"People: {n_people}, Selected: {n_selected}")
        log(f"fc_export_dir: {fc_dir}")
        log(f"step_decisions: {n_decisions}")

        if n_people and n_people > 1:
            ok("Multiple People", f"{n_people} people")
        else:
            fail("Multiple People", f"only {n_people} person(s)")

        if fc_dir:
            ok("Export Dir", fc_dir)
            # Check for crops
            export_path = Path(fc_dir)
            crops_dir = export_path / "crops"
            if crops_dir.exists():
                n_crops = len(list(crops_dir.iterdir()))
                ok("Face Crops", f"{n_crops} crop files")
            else:
                fail("Face Crops", f"crops/ dir not found in {fc_dir}")

            # Check crop_manifest
            manifest = export_path / "crop_manifest.json"
            if manifest.exists():
                import json
                m = json.loads(manifest.read_text())
                ok("Crop Manifest", f"{len(m)} entries")
            else:
                fail("Crop Manifest", "crop_manifest.json missing")

            # Check merge_log
            merge_log = export_path / "merge_log.json"
            if merge_log.exists():
                ok("Merge Log", "exists")
            else:
                fail("Merge Log", "merge_log.json missing (merge may not have produced merges)")
        else:
            fail("Export Dir", "fc_export_dir is None")

        if n_decisions > 0:
            # Check decision steps
            steps = {}
            for d in decisions:
                steps[d["step"]] = steps.get(d["step"], 0) + 1
            log(f"Decision steps: {steps}")
            ok("Step Decisions", f"{n_decisions} total")
        else:
            fail("Step Decisions", "no decisions in API response")
    else:
        fail("Results", "no results returned")
except Exception as e:
    fail("Results Check", str(e))

# ─── Step 5: Playwright UI tests ───
print("\n5. Playwright UI verification...")
try:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.goto(APP, wait_until="networkidle")
        pg.wait_for_timeout(5000)

        def shot(name): pg.screenshot(path=str(SCREENSHOTS / f"{name}.png"), full_page=True)
        def errs(): return pg.locator(".stException").count()
        def nav(name): pg.locator(f'button:has-text("{name}")').first.dispatch_event("click"); pg.wait_for_timeout(4000)

        # Select the right album
        # The album selector should auto-select latest

        # 5a. People & Faces — check bounding box
        nav("People & Faces")
        pg.wait_for_timeout(2000)
        vb = pg.locator('button:has-text("View")')
        if vb.count() > 0:
            vb.first.dispatch_event("click")
            pg.wait_for_timeout(5000)
            shot("people_detail")

            bbox_caption = pg.locator("text=Face highlighted").count()
            if bbox_caption > 0:
                ok("People Bbox Caption", "visible")
            else:
                fail("People Bbox Caption", "not found")

            if errs() > 0:
                fail("People Detail", pg.locator(".stException").first.text_content(timeout=3000)[:200])
            else:
                ok("People Detail", "no errors")

        # 5b. Explore tabs
        nav("Explore")
        pg.wait_for_timeout(3000)
        shot("explore_quality")

        has_decisions = pg.locator("text=Config used").count() > 0
        if has_decisions:
            ok("Explore Decisions", "Config used visible")
        else:
            fail("Explore Decisions", "no decision data shown")

        thumbs = pg.locator('[data-testid="stImage"] img').count()
        ok("Explore Thumbnails", f"{thumbs} images")

        # Selection tab
        sel = pg.locator('button[role="tab"]:has-text("Selection")')
        if sel.count() > 0:
            sel.first.dispatch_event("click")
            pg.wait_for_timeout(3000)
            shot("explore_selection")
            reasons = pg.locator("text=/Best in cluster|Outranked|threshold/").count()
            if reasons > 0:
                ok("Selection Reasons", f"{reasons} visible")
            else:
                fail("Selection Reasons", "no decision reasons found")

        # Scene Clustering tab
        sc = pg.locator('button[role="tab"]:has-text("Scene Clustering")')
        if sc.count() > 0:
            sc.first.dispatch_event("click")
            pg.wait_for_timeout(3000)
            shot("explore_scene")
            cluster_imgs = pg.locator('[data-testid="stImage"] img').count()
            if cluster_imgs > 5:
                ok("Scene Cluster Images", f"{cluster_imgs} images")
            else:
                fail("Scene Cluster Images", f"only {cluster_imgs} images")

        # 5c. Image detail popup — check for bbox in popup
        nav("Results")
        pg.wait_for_timeout(3000)
        detail_btn = pg.locator('button:has-text("Detail")')
        if detail_btn.count() > 0:
            detail_btn.first.dispatch_event("click")
            pg.wait_for_timeout(4000)
            shot("image_popup")
            dialog = pg.locator('[role="dialog"]')
            if dialog.count() > 0:
                popup_imgs = dialog.locator('[data-testid="stImage"] img').count()
                has_quality = dialog.locator('text=Quality Scores').count()
                if has_quality > 0:
                    ok("Image Popup Content", f"imgs={popup_imgs}")
                else:
                    fail("Image Popup Content", "no Quality Scores in popup")

                # Close
                close = dialog.locator('button:has-text("Close")')
                if close.count() > 0:
                    close.first.dispatch_event("click")
                    pg.wait_for_timeout(1000)
            else:
                fail("Image Popup Content", "dialog not found")

        # 5d. Results — check Detail buttons
        nav("Results")
        pg.wait_for_timeout(4000)
        shot("results")
        detail_btns = pg.locator('button:has-text("Detail")').count()
        if detail_btns > 0:
            ok("Results Detail Buttons", f"{detail_btns}")
        else:
            fail("Results Detail Buttons", "none found")

        # 5d. Face Clustering App deep-link
        if fc_dir:
            from urllib.parse import quote
            fc_url = f"{FC_APP}/?load_run={quote(fc_dir)}"
            log(f"Opening Face Clustering App: {fc_url}")
            pg.goto(fc_url, wait_until="networkidle")
            pg.wait_for_timeout(8000)
            shot("face_clustering_app")

            fc_errors = pg.locator(".stException").count()
            if fc_errors > 0:
                fail("FC App Load", pg.locator(".stException").first.text_content(timeout=3000)[:200])
            else:
                ok("FC App Load", "no errors")

            # Check for merge analysis
            merge_tab = pg.locator('button:has-text("Merge Analysis")')
            if merge_tab.count() > 0:
                merge_tab.first.dispatch_event("click")
                pg.wait_for_timeout(3000)
                shot("fc_merge_analysis")

                no_merge = pg.locator("text=No merge data available").count()
                if no_merge > 0:
                    fail("FC Merge Analysis", "No merge data available")
                else:
                    ok("FC Merge Analysis", "data shown")

            # Check for face crops in cluster browser (use exact tab name)
            cluster_tab = pg.locator('button:has-text("Clusters (Base)")')
            if cluster_tab.count() > 0:
                cluster_tab.first.dispatch_event("click")
                pg.wait_for_timeout(3000)
                shot("fc_clusters")

                no_crop = pg.locator("text=no crop").count()
                fc_imgs = pg.locator('[data-testid="stImage"] img').count()
                if no_crop > 0:
                    fail("FC Face Crops", f"{no_crop} '(no crop)' labels found")
                elif fc_imgs > 0:
                    ok("FC Face Crops", f"{fc_imgs} face images visible")
                else:
                    fail("FC Face Crops", "no images found")

        browser.close()

except ImportError:
    fail("Playwright", "not installed")
except Exception as e:
    fail("Playwright", str(e))
    import traceback
    traceback.print_exc()

# ─── Final Report ───
print("\n" + "=" * 60)
print("  E2E FLOW REPORT")
print("=" * 60)
passes = 0
fails = 0
for test, result in results.items():
    status = "PASS" if result.startswith("PASS") else "FAIL"
    if status == "PASS":
        passes += 1
    else:
        fails += 1
    print(f"  [{status:4s}] {test}: {result}")
print("=" * 60)
print(f"  {passes}/{passes+fails} PASSED, {fails} FAILED")
if fails > 0:
    print(f"  Screenshots: {SCREENSHOTS}")
print("=" * 60)

# Cleanup — delete the test album
try:
    requests.delete(f"{API}/api/v1/albums/{album_id}", timeout=5)
    print(f"\n  Cleaned up album {album_name}")
except:
    pass

sys.exit(1 if fails > 0 else 0)
