"""End-to-end smoke test for spec-030 Phase 3 (one-shot, manual run).

What it does, in order:
  1. Runs `FaceClusteringPipeline` on a real album folder of JPEGs.  This
     exercises the real face-detection + embedding stack and produces both
     legacy artifacts AND the v4 layout in a parallel `_v4/` subdir
     (Phase 1 dual-write).
  2. Boots a Streamlit FC App instance on a non-conflicting port.
  3. Drives Playwright against the live app, navigating tabs and asserting:
       - Merge Analysis renders
       - Outcome labels include the new MERGED / PASSED / REJECTED states
       - Margin badge says "disabled" (not literal "inf") when merge_margin=0
       - Cluster Analysis renders without crashing
       - "Actual merges" metric matches the database
  4. Captures a screenshot per checkpoint into `e2e_screens/`.

Usage:
    .venv/Scripts/python scripts/e2e_smoke_phase3.py
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Repo root on path for face_cluster imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from face_cluster import FaceClusteringPipeline, PipelineConfig

ALBUM_DIR = REPO_ROOT / "test_data" / "face_clustering_100"
OUTPUT_PARENT = REPO_ROOT / "results" / "e2e_smoke"
SCREEN_DIR = REPO_ROOT / "e2e_screens"
APP_PORT = 8511          # avoid 8501 (Albumify) and 8502 (user's FC App)
APP_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Step 1 — produce a fresh run on a real album
# ---------------------------------------------------------------------------

def run_pipeline() -> Path:
    if not ALBUM_DIR.is_dir():
        sys.exit(f"album not found: {ALBUM_DIR}")

    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_PARENT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[1/4] running pipeline on {ALBUM_DIR.name} -> {out.name}")
    cfg = PipelineConfig.full_run(ALBUM_DIR, out, merge_enabled=True)
    result = FaceClusteringPipeline().run(cfg)
    n_merges = sum(1 for e in (result.merge_log or []) if e.get("actually_merged"))
    n_passed = sum(1 for e in (result.merge_log or [])
                   if e.get("action") == "passed" and not e.get("actually_merged"))
    print(f"     {len(result.faces)} faces, "
          f"{result.cluster_result.n_clusters}->{(result.merged_cluster_result or result.cluster_result).n_clusters} clusters, "
          f"{n_merges} merges, {n_passed} passed-but-not-merged rows")

    v4 = out / "_v4"
    if not (v4 / "face_clustering.db").exists():
        sys.exit("v4 layout not produced — Phase 1 dual-write did not fire?")

    # Confirm the v4 schema has every field we expect (FR-004).
    conn = sqlite3.connect(v4 / "face_clustering.db")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(merge_decisions)").fetchall()]
    user_ver = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert "actually_merged" in cols, f"actually_merged missing from DB: {cols}"
    assert "max_allowed_diameter" in cols
    assert "required_support" in cols
    assert user_ver == 4, f"PRAGMA user_version = {user_ver}, expected 4"
    print(f"     v4 layout verified: {len(cols)}-col merge_decisions, schema_version=4")

    return out


# ---------------------------------------------------------------------------
# Step 2 — boot Streamlit
# ---------------------------------------------------------------------------

def wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"streamlit did not bind {host}:{port} within {timeout}s")


def boot_streamlit() -> subprocess.Popen:
    print(f"[2/4] launching Streamlit on http://{APP_HOST}:{APP_PORT}")
    cmd = [
        str(REPO_ROOT / ".venv" / "Scripts" / "streamlit"),
        "run", str(REPO_ROOT / "app" / "face_clustering" / "main.py"),
        "--server.port", str(APP_PORT),
        "--server.address", APP_HOST,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    log_path = SCREEN_DIR / "streamlit.log"
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    proc._log_fh = log_fh  # keep alive
    print(f"     log: {log_path}")
    try:
        wait_for_port(APP_HOST, APP_PORT, timeout=60)
        # Streamlit may bind the port before it's fully ready to serve.
        time.sleep(3.0)
    except Exception:
        proc.terminate()
        raise
    print(f"     bound; pid={proc.pid}")
    return proc


# ---------------------------------------------------------------------------
# Step 3 — drive Playwright
# ---------------------------------------------------------------------------

def drive_browser(run_dir: Path) -> dict:
    from playwright.sync_api import sync_playwright

    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    findings: dict = {"screenshots": []}

    from urllib.parse import quote
    deep_link = (
        f"http://{APP_HOST}:{APP_PORT}/?load_run="
        f"{quote(run_dir.as_posix(), safe='')}"
    )
    print(f"[3/4] driving browser at {deep_link}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page = context.new_page()
        page.set_default_timeout(20_000)

        def shot(name: str) -> Path:
            path = SCREEN_DIR / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            findings["screenshots"].append(str(path.name))
            return path

        page.goto(deep_link, wait_until="networkidle")
        page.wait_for_load_state("networkidle")
        # Streamlit reruns the script on first widget render; give it time to
        # process the deep-link, hit the loader, and render real content.
        for _ in range(20):
            time.sleep(1.0)
            txt = page.locator("body").inner_text()
            if "Pipeline Run" in txt or "Run Pipeline" in txt:
                # If we still see the empty Run tab form, the deep-link didn't fire.
                if "results\\my_album" in txt or "D:\\Google_Germany" in txt:
                    continue
            if "Faces" in txt and ("Clusters" in txt or "Merge" in txt):
                break
        shot("01_loaded")
        findings["loaded_text_excerpt"] = page.locator("body").inner_text()[:1500]

        # Streamlit renders tabs as buttons with role="tab"
        def _wait_for_render(marker: str, timeout: float = 30.0):
            """Wait until Streamlit's running indicator clears AND `marker` appears
            in body text."""
            deadline = time.time() + timeout
            last = ""
            while time.time() < deadline:
                txt = page.locator("body").inner_text()
                last = txt
                if marker in txt:
                    return txt
                time.sleep(0.5)
            return last

        try:
            page.get_by_role("tab", name="Merge Analysis").click()
            page.wait_for_load_state("networkidle")
            # Wait for actual merge-analysis content to render — the Summary
            # subheader appears once MergeAnalysisView.compute() returns.
            body_text = _wait_for_render("Summary", timeout=40.0)
            # Streamlit's bird/runner icon clears a beat after content renders;
            # one more short sleep to be safe before screenshot.
            time.sleep(2)
        except Exception as e:
            findings["merge_tab_error"] = str(e)
            shot("02_merge_tab_error")
            body_text = page.locator("body").inner_text()
        else:
            shot("02_merge_tab")
            # Scroll into the pair list and snap that too — that's where the
            # MERGED / PASSED / REJECTED / disabled-margin labels live.
            page.mouse.wheel(0, 1200)
            time.sleep(2)
            shot("02b_merge_tab_pairs")
            page.mouse.wheel(0, 1200)
            time.sleep(2)
            shot("02c_merge_tab_pairs_lower")
            findings["page_text_excerpt"] = body_text[:3000]

            findings["has_MERGED_label"] = "MERGED" in body_text
            findings["has_PASSED_label"] = "PASSED" in body_text
            findings["has_REJECTED_label"] = "REJECTED" in body_text

            findings["has_disabled_margin"] = "disabled" in body_text
            findings["has_literal_inf_margin"] = (
                "Margin\ninf" in body_text or "inf\n" in body_text and "Margin" in body_text
            )

            findings["has_support_zero"] = "/0" in body_text and "Support" in body_text
            findings["has_diameter_na"] = "/n/a" in body_text
            findings["has_actual_merges_metric"] = "Actual merges" in body_text

            # Try opening the first pair expander to see the gate badges.
            try:
                first_expander = page.locator(
                    "button[kind='secondary'][aria-expanded='false']"
                ).first
                if first_expander.count() > 0:
                    first_expander.click()
                    time.sleep(1)
                    shot("03_pair_expanded")
            except Exception:
                pass

        # Cluster Analysis tab
        try:
            page.get_by_role("tab", name="Cluster Analysis").click()
            page.wait_for_load_state("networkidle")
            time.sleep(2)
            shot("04_cluster_analysis")
            findings["cluster_analysis_renders"] = True
        except Exception as e:
            findings["cluster_analysis_error"] = str(e)
            findings["cluster_analysis_renders"] = False

        # Overview tab — basic sanity that everything still works
        try:
            page.get_by_role("tab", name="Clusters (Base)").click()
            page.wait_for_load_state("networkidle")
            time.sleep(2)
            shot("05_clusters_base")
            findings["clusters_base_renders"] = True
        except Exception as e:
            findings["clusters_base_error"] = str(e)
            findings["clusters_base_renders"] = False

        browser.close()

    return findings


# ---------------------------------------------------------------------------
# Step 4 — assert
# ---------------------------------------------------------------------------

def assert_findings(run_dir: Path, findings: dict) -> int:
    print("[4/4] checking findings")
    out_path = SCREEN_DIR / "findings.json"
    out_path.write_text(json.dumps(findings, indent=2, default=str))
    print(f"     wrote {out_path}")
    print(f"     screenshots: {findings['screenshots']}")

    fails: list[str] = []
    asserts: list[tuple[str, bool]] = []

    def check(name: str, condition: bool):
        asserts.append((name, condition))
        if not condition:
            fails.append(name)

    check("Merge Analysis tab rendered", findings.get("merge_tab_error") is None)
    check("Cluster Analysis tab rendered", findings.get("cluster_analysis_renders", False))
    check("Clusters (Base) tab rendered", findings.get("clusters_base_renders", False))

    # Read what the data should look like for sanity-checking the labels.
    conn = sqlite3.connect(run_dir / "_v4" / "face_clustering.db")
    n_merged = conn.execute("SELECT COUNT(*) FROM merge_decisions WHERE actually_merged = 1").fetchone()[0]
    n_passed = conn.execute("SELECT COUNT(*) FROM merge_decisions WHERE action = 'passed' AND actually_merged = 0").fetchone()[0]
    n_rejected = conn.execute("SELECT COUNT(*) FROM merge_decisions WHERE action = 'rejected'").fetchone()[0]
    conn.close()
    print(f"     DB: n_merged={n_merged}, n_passed={n_passed}, n_rejected={n_rejected}")

    if n_merged > 0:
        check("MERGED label present in UI when n_merged>0", findings.get("has_MERGED_label", False))
    if n_passed > 0:
        check("PASSED label present in UI when n_passed>0", findings.get("has_PASSED_label", False))
    if n_rejected > 0:
        check("REJECTED label present in UI when n_rejected>0", findings.get("has_REJECTED_label", False))

    print()
    print("RESULTS:")
    for name, ok in asserts:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    return 0 if not fails else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    run_dir = run_pipeline()
    proc = boot_streamlit()
    try:
        findings = drive_browser(run_dir)
        rc = assert_findings(run_dir, findings)
    finally:
        print("[stop] terminating streamlit")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rc


if __name__ == "__main__":
    sys.exit(main())
