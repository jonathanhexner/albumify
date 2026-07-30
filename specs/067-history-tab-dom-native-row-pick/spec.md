# spec-067 — Bypass dataframe row-pick in e2e via query-param seeding (resolve SIGHTING-091)

**Created**: 2026-05-31
**Status**: In Progress (started 2026-06-03 — see [tasks.md](tasks.md))
**Priority**: P2 (unblocks 5 of 6 budapest e2e scenarios)
**Predecessors**: spec-042 (v2 tab parity), spec-063/064/065 (scenarios that depend on the row-pick)
**Sighting**: SIGHTING-091
**Renamed from**: "History tab DOM-native row-pick" — see §"Decision log" below for why we flipped from a UI change to a test-harness change.

---

## Problem

The v2 History tab's run-picker uses `st.dataframe(on_select="rerun")`. That widget is canvas-rendered (glide-data-grid) and not addressable by Playwright role selectors. Budapest Scenarios B/C/D/E/F all need to click row `6437d335…` to load the reference run; all 5 time out at the same step. **Real users in a browser are unaffected** — this is purely an e2e-harness gap.

Full analysis (modules, class diagram, LOC, testability map, the three resolution options we considered): **[ANALYSIS.html](ANALYSIS.html)**.

## Decision: bypass the click, don't replace the widget

The earlier draft of this spec proposed replacing `st.dataframe` with `st.radio` per row to make the click Playwright-addressable. That's a real UI change with collateral cost (loses sort/resize, adds LOC) for a benefit only the test harness sees.

The chosen approach: **drive the same code path the Load button triggers via a query-param seed, then let Playwright continue from the next tab.**

The untestable step is ONE click. Everything after it (Cluster Analysis / Recluster / Face Analysis / Merged Clusters / Quality tab interactions) renders as normal DOM and IS fully Playwright-addressable. We bypass the click, keep all downstream coverage.

## What we build

### Production change (small + reusable)

**`app/face_clustering_v2/main.py`** — add a one-time query-param seeder that runs on first render:

```python
# Sketch — exact field set TBD during implementation.
def _seed_session_state_from_query_params() -> None:
    """Map ?key=value query params to st.session_state on first render.

    Honors a small allowlist:
      - ?current_run_dir=<path>   -> st.session_state['current_run_dir']
      - ?selected_face_id=<int>   -> st.session_state['selected_face_id']

    Runs once per session (idempotent via a session_state sentinel).
    Production benefit: shareable deep-links. Test benefit: budapest
    e2e bypasses the dataframe row-pick.
    """
```

Reuses the existing `current_run_dir` session_state contract that every analysis tab already reads. ~15 LOC. No tab-level changes.

### Test harness change

**`tests/face_clustering/e2e_budapest/conftest.py`** — new fixture `page_with_reference_run_loaded`:

```python
@pytest.fixture
def page_with_reference_run_loaded(page):
    """Skips the dataframe row-pick by seeding current_run_dir via
    query param. Equivalent to: open History tab -> click row -> click
    'Load into analysis tabs' -> wait for 'Loaded'."""
    if not REFERENCE_RUN_DIR.exists():
        pytest.skip(...)
    page.goto(f"{APP_URL}?current_run_dir={REFERENCE_RUN_DIR}")
    page.wait_for_selector("h1", state="visible")
    return page
```

### Scenario rewrites

Scenarios B/C/D/E/F replace their 3-line "open History → click row → click Load" preamble with `def test_...(page_with_reference_run_loaded):`. The rest of each test is unchanged.

Scenario A (fresh-run) is unaffected — it doesn't load a prior run.

### Coverage we keep

- Every downstream tab interaction (the 5 tabs and their components) under real browser paint.
- The kind of bug we caught this session (v2_K StreamlitDuplicateElementKey crashing every tab) would still surface — that's a startup-level failure.

### Coverage we lose

- That the dataframe row-pick + Load button visually wire to `HistoryService.load_run`. Covered well enough by:
  - `test_v2_app_smoke.py::test_history_tab_recognizes_v2_run_as_loadable` (AppTest, calls the service through the same path).
  - `HistoryService` unit tests on synthetic fixtures.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `main.py` honors `?current_run_dir=<path>` on first render | new arch test + AppTest |
| AC2 | `page_with_reference_run_loaded` fixture exists in `e2e_budapest/conftest.py` | grep |
| AC3 | Scenarios B/C/D/E/F use the new fixture; row-pick step removed | grep |
| AC4 | All 6 budapest scenarios green (`pytest -m budapest`) | pytest |
| AC5 | History tab UI unchanged — `run_table.py` and `actions_table.py` untouched | git diff |
| AC6 | Query-param seeder is opt-in (no behaviour change when no query param) | AppTest with no params + smoke regression |

## Out of scope

- Replacing `st.dataframe` with another widget. SIGHTING-091 stays open as a known a11y gap; we'll revisit if a screen-reader-focused spec arises.
- Programmatic tab-switching after seed (deep-link to specific tab) — separate ask.

## Effort estimate

~2 hours: 15-LOC seeder + fixture + 5 test rewrites + run the suite.

## Test coverage strategy per tab (added 2026-06-05)

Unblocking the seed (B passes) exposed that scenarios C–F had **never** run
past the History row-pick, so their assertions were never validated against
the real multi-tab DOM. Three failure classes surfaced; the fix is to test
each tab at the *right layer*, not to force everything through Playwright.

**Guiding principle:** Playwright only where the widget is real DOM (images,
metric strips, Plotly charts, buttons, headers). For canvas `st.dataframe`
widgets it cannot address the cells — use headless AppTest + spec-068
telemetry + service unit tests instead. Streamlit also keeps *every* tab body
in the DOM, so browser selectors MUST be scoped to the active tab (role+label
or visible-scoped), never `.first` across the page.

| Tab | Primary layer | Browser? | Notes |
|---|---|---|---|
| Run | Service test (`run_v2_pipeline`) | A — slow smoke only | AppTest for form; browser A ~10 min, not primary |
| Cluster Analysis | **Playwright (B)** | yes — real DOM | thumbnails + metric strip address fine |
| Face Analysis | **Playwright (D)** | yes — real DOM | target fields by role+label, NOT `.first` |
| Merged Clusters | AppTest + telemetry | header only | table is canvas; assert `n_rows` via telemetry + `MergedClustersService` test |
| Quality | Service test + Playwright | metrics/chart only | needs a run WITH quality data; scope metric selector to visible tab |
| Recluster | Service test (`ReclusterService`) | C — slow smoke only | browser recluster ~5 min, not primary |
| History | Service test + AppTest | no — picker is canvas | browser bypasses picker via the seed (this spec) |

## Resolution plan for the exposed failures

- **Server fragility (root blocker, DONE 2026-06-04):** the session-scoped
  Streamlit server served only the first browser test; the next `page.goto`
  timed out. Made `streamlit_server` **function-scoped** (fresh server per
  scenario) + port-free guard. B and D now pass.
- **D (DONE):** `input[type=number].first` grabbed a hidden-tab input →
  retargeted to `get_by_role("spinbutton", name="Face id")`.
- **E (TODO):** re-point off the canvas grid cells. Assert the Merged Clusters
  data via spec-068 telemetry (`tab.done name=merged_clusters n_rows>=1`)
  and/or `MergedClustersService`, plus a browser check only that the tab
  header + table container render. Drop the `[role='gridcell']` count + the
  canvas row-click detail-panel assertion.
- **F (BLOCKED — needs data decision):** the reference run `6437d335…` is a
  legacy `fc_app` run with **0 filter_decisions** (telemetry:
  `tab.done name=quality n_items=0`). F's "220–240 rejected" premise is
  unmeetable against it. Options: (a) add a v2-produced reference run that
  has quality data and point F at it; (b) redefine F to assert the empty
  state. Tracked as a sighting, not fudged.
- **C (TODO — verify):** retest the recluster scenario now that the server is
  per-test; one ~5-min run to confirm recluster genuinely works vs. was just
  collateral damage from the wedged server.

**AC4 status:** partial. B, D green. E fixable (test-layer). F blocked on a
data decision. C unverified. A unaffected.

## Decision log

- 2026-05-31: Initial draft proposed replacing `st.dataframe` with a DOM-native picker (~1 day, UI regression). Flipped on user feedback to "drive the endpoint instead" — same coverage of the parts that matter, ~2 h, zero UI change.
- 2026-06-05: Added per-tab coverage strategy after unblocking exposed never-validated assertions in C–F. Root blocker was a shared-server fixture, not the seed. Right layer per tab > everything-through-Playwright.
