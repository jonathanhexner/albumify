# spec-062 — Click every button: real-browser E2E gate for the v2 app

**Created**: 2026-05-29
**Status**: Draft
**Predecessors**: spec-060 (v2 E2E gold-standard gate, Phase 2 AppTest landed early). spec-061 (read-path audit).
**Trigger**: Four sightings in one week (078, 079, 080, today's face-grid crop bug), all with "tests green but user sees broken UI" as the failure mode. AppTest's "no exception" assertion was passing while real users saw stuck spinners, missing thumbnails, disabled buttons, and contradictory error messages. We need a test that asks the question the user asks: *"does this button do what it's supposed to?"*

---

## Problem

`test_v2_app_smoke.py` (spec-060 Phase 2 early) renders pages through Streamlit's `AppTest` harness and asserts `len(at.exception) == 0`. That catches **crashes** but not **broken interactions**. This week proved it:

| Bug shape | AppTest verdict | User verdict |
|---|---|---|
| Cluster Analysis stuck on "Analysing cluster…" forever | ✓ pass (no exception, just a caption) | ✗ tab unusable |
| History "Load Run" button disabled with wrong warning text | ✓ pass | ✗ can't load any run |
| Face grid renders captions but no thumbnails | ✓ pass | ✗ user sees text-only — looks broken |

AppTest also can't simulate `st.dataframe` row selection, button clicks across reruns, or assert on rendered image content. We need a real browser.

## What we build

`tests/face_clustering/test_v2_click_every_button.py` — a Playwright test suite that:

1. Starts the v2 Streamlit app on a fixed port against a seeded fixture run.
2. Enumerates every interactable element (button, selectbox, expander, tab, dataframe row) on every tab.
3. For each element: clicks it, waits for the next render, asserts on the observable result (visible widget appeared, value updated, no error overlay, expected text present).

One test per element. Failure shows a screenshot + the element it was operating on. No more "passed but doesn't work."

## What we don't build

- **No browser-vendor matrix.** Chromium only. Cross-browser is a separate concern.
- **No visual diff testing.** Screenshot-on-failure is for debugging, not the gate (pixel diffs are flaky for Streamlit).
- **No happy-path-only "user flow" tests.** Element-by-element is the contract — flows can be assembled on top later.
- **No login / auth setup.** v2 is dev-machine-local.
- **No production smoke.** This is a pre-merge gate, not monitoring.

## The element inventory (the contract)

The current v2 app, after this week's fixes, has three tabs. The element inventory below is what the test must cover. **One test per row.**

### Tab: Run

| Element | Trigger | Expected observable |
|---|---|---|
| Source dir text input | Type a valid dir path | Input echoes value; "Run pipeline" button enables when album + source are non-empty |
| Album text input | Type a name | Same as above |
| Profile selectbox | Switch profile | Selected option echoes |
| "Run pipeline" button | Click (with valid inputs) | Progress bar appears; final success/failure message renders |
| "Run pipeline" button | Click (with empty source) | Inline error, no pipeline started |

### Tab: Cluster Analysis

| Element | Trigger | Expected observable |
|---|---|---|
| Cluster picker selectbox | (no run loaded) | Friendly "No completed run available" info banner |
| Cluster picker selectbox | (run loaded) | Selectbox populated with N cluster labels |
| Cluster picker selectbox | Change selection | Metrics + grid + nearest list re-render for the new cluster |
| Face grid | (run with crops on disk) | At least one `<img>` element rendered with src under `/crops/` |
| Face grid | (run with corrupt crops) | Page does NOT crash; captions still render |
| "Nearest clusters" expander | Click to expand | Up to 10 rows of nearest-cluster info appear |
| "Go to" button on a nearest cluster | Click | Picker switches to that cluster; metrics re-render |
| "Force Merge" expander | Click to expand | Two cluster dropdowns + "Preview Merge" button appear |
| "Preview Merge" button | Click | 3 gate badges (Exemplar / Support / Diameter) appear with PASS/FAIL |
| "Confirm: merge" button | Click | Snapshot dir created; toast shows success; tab switches to the new run |
| "Graph debug" expander | Click to expand | 4-metric strip appears; heatmap renders if cluster has ≥2 faces with embeddings |

### Tab: History

| Element | Trigger | Expected observable |
|---|---|---|
| Album filter selectbox | Change | Runs table filters to matching rows |
| Date-range picker | Change | Runs table filters to matching dates |
| Free-text search | Type | Runs table filters |
| Runs dataframe | Click a row | Row-detail panel appears below; "Load into analysis tabs" button visible |
| "Load into analysis tabs" button | Click on a v2 run | Toast shows success; session_state populated; can navigate to Cluster Analysis |
| "Load into analysis tabs" button | Click on a v4 run with `_v4/` subdir | Same — backward-compatible |
| "Load into analysis tabs" button | (incomplete run) | Button disabled; clear warning message |
| Comment text input | Type + blur | Comment persists to action_log |
| Actions sub-table | Click a row | Payload JSON inspector renders |

## Locked decisions

1. **Real browser via Playwright.** AppTest is not enough — it can't simulate dataframe selection, doesn't poll background threads, doesn't render images. Chromium headless via `playwright.sync_api`.
2. **Fixed port 8889.** Test starts `streamlit run` in a subprocess on a non-default port, tears it down on test exit. One server per test session (~10s cold start).
3. **Seeded fixture run.** Reuse the `no_op_merge_run_dir` synthetic fixture from `test_v2_app_smoke.py` so the test is self-contained — no dependency on the user's real Budapest album. (A second opt-in fixture against the real album can be added later as a separate marker.)
4. **One test per element, not per flow.** Flow tests batch failures; element tests pinpoint them. A failing test names exactly the button / picker / row that broke.
5. **Failure artifact = screenshot at the point of failure.** Saved next to the test file. Cheaper than recording video; enough to debug from.
6. **Marker: `@pytest.mark.browser`.** Opt-in (browser tests are slower than AppTest). Wired into the spec-implementer agent's gate the same way spec-060 wired AppTest.
7. **No flake budget.** A flaky browser test is worse than no test — it teaches the team to ignore failures. If a test flakes, fix the wait condition or remove the test.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `tests/face_clustering/test_v2_click_every_button.py` exists with one test per row in §"element inventory" | grep |
| AC2 | `pytest -m browser tests/face_clustering/test_v2_click_every_button.py -v` exits 0 on the seeded fixture | run |
| AC3 | Failure artifact (PNG screenshot) saved under `tests/face_clustering/_failure_artifacts/` when a test fails | manual: break a button on purpose; observe artifact |
| AC4 | Cold-start time ≤ 15 s; per-test time ≤ 5 s; full suite ≤ 3 min on dev machine | wall-clock |
| AC5 | Adding a new button to the app fails CI until a corresponding row is added to §"element inventory" + a test exists | drift guard: arch test grepping the UI files for new `st.button` / `st.selectbox` calls without a matching test name |
| AC6 | spec-061 audit can re-run this test as its last validation phase | tasks.md cross-reference |
| AC7 | spec-implementer agent's Implementation gate runs this test before flipping any "considerable change" spec to Implemented | agent prompt updated |

## Risks

- **Playwright flake.** Streamlit's render-on-rerun model means widgets appear asynchronously. Mitigate: use `expect(locator).to_be_visible(timeout=5000)` rather than fixed sleeps.
- **Test maintenance burden.** N buttons → N tests. Mitigate: parametrize where the assertion shape is identical (e.g., all "Go to" buttons share one parametrized test).
- **CI environment without browsers.** Mitigate: `@pytest.mark.browser` skips when Playwright isn't installed; only runs locally + on the spec-implementer agent's machine.
- **Element selectors drift.** Streamlit's DOM structure can change between versions. Mitigate: prefer text-based selectors (`page.get_by_role("button", name="Load into analysis tabs")`) over CSS path selectors.

## Effort estimate

**~8–12 hours.** Breakdown:
- Playwright + Streamlit subprocess plumbing: ~2 h
- Element inventory tests (Run + History + Cluster Analysis): ~4–6 h
- Drift-guard arch test (AC5): ~1 h
- Hook into spec-implementer agent (AC7): ~1 h
- Documentation + close-out: ~1 h

Big yield: this becomes the load-bearing pre-merge gate for the v2 app. spec-060 Phase 1 (real-album pipeline smoke) remains a separate, longer-running gate; this one runs in 3 min on the synthetic fixture.

## Open questions

1. **Subprocess lifecycle.** Start the Streamlit server in a session-scoped fixture and tear it down at session end? Or per-test (slower but more isolated)? Recommendation: session-scoped; if tests pollute each other we'll know fast.
2. **Real album as an opt-in second matrix?** spec-060's pipeline-smoke tests already exercise the real Budapest run end-to-end; this spec sticks to the synthetic fixture for speed. A separate `@pytest.mark.browser_real` marker can be added later.
3. **What about the legacy `app/face_clustering/` app?** Out of scope. spec-040 Phase 7 retires it.
