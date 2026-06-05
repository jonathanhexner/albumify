# spec-066 — Architecture Plan: Gallery + Overview tabs (the last v2 pair)

**Status of this doc**: actionable plan, ready to implement. Refines `spec.md` + `tasks.md`.
**Companion diagram**: [`ARCHITECTURE.html`](ARCHITECTURE.html) (open in a browser).
**Date**: 2026-06-05

> These are the final two tabs. When their e2e (Scenarios G+H) is green, **spec-042 v2 tab parity flips to Implemented** and every legacy tab has a v2 equivalent.

---

## 1. Scope & non-goals

| In scope | Out of scope |
|---|---|
| Gallery tab — scrollable per-cluster thumbnail strips + "Open in Cluster Analysis" | New clustering compute / schema |
| **Thumbnails-per-cluster slider** (§3c-G1) | Async compute (sync + spinner, per SIGHTING-079) |
| **Clickable thumbnails → Face Analysis** (reuse `selected_face_id`, §3c-G2) | Per-stage timing charts (v5 doesn't record them — see `history._summary_from_run_metadata`) |
| **Quality FLAG badges** (blurry / outlier, read-only, §3c-G5) | **Interactive disqualify** → its own spec **[069](../069-gallery-manual-disqualify/spec.md)** (a mutation; breaks read-only) |
| Overview tab — run-level dashboard (metrics strip + 3 charts) | |
| One new Streamlit-free `OverviewService` + typed `DashboardMetrics` | |
| Frontend observability (extend `_telemetry`) | |
| Nearest-outside-cluster strip — **opt-in toggle, gated on Decision D1** (§3c-G4) | |

---

## 2. The framework stack (canonical 4-layer split, spec-042/045)

```
TAB (orchestration, <=80 LOC, NO sql/fs/cfg.get)   app/face_clustering_v2/tabs/*.py
  │  gather session_state → build Service → dispatch to components → emit telemetry
  ▼
COMPONENT (Streamlit render only)                   app/face_clustering_v2/components/*.py
  │  takes a typed view object → st.* calls. No SQL, no compute.
  ▼
SERVICE (typed, Streamlit-FREE, unit-testable)      face_cluster/views/*.py
  │  typed Inputs → typed Result dataclasses. The TESTABLE ENDPOINT.
  ▼
REPOSITORY (SQL / FS)                                sim_bench/db/... , face_cluster/repositories/...
```

Enforced by `tests/architecture/test_v2_layering.py` (auto-scans every tab — no per-tab
registration needed): views must not import `streamlit`; tabs must not import `sqlite3`,
`run_history(_db)`, or `json.loads(...read_text)`.

---

## 3. Data contracts (the testable endpoints)

### 3a. Overview — NEW `face_cluster/views/overview.py`

```python
@dataclass(frozen=True, slots=True)
class AlbumStat:    album: str;   n_runs: int
@dataclass(frozen=True, slots=True)
class ProfileStat:  profile: str; n_runs: int
@dataclass(frozen=True, slots=True)
class RunPoint:                                    # one point on the n_clusters time-series
    run_id: int; started_at: Optional[str]; n_clusters: Optional[int]; album: str

@dataclass(frozen=True, slots=True)
class DashboardMetrics:
    total_runs: int
    total_faces_ever: int
    avg_n_clusters: Optional[float]
    median_n_clusters: Optional[float]
    last_run_at: Optional[str]          # RAW iso string — tab computes "age" for display
    gate_pass_rate: Optional[float]     # fraction of runs with status == "complete"
    per_album: List[AlbumStat]          # desc by n_runs
    per_profile: List[ProfileStat]      # desc by n_runs
    timeseries: List[RunPoint]          # oldest→newest, the recent window

class OverviewService:
    def __init__(self, repo: RunHistoryRepository, runs_dir: Optional[Path] = None): ...
    def compute_dashboard(self, *, limit: int = 50) -> DashboardMetrics: ...
```

**Key design decisions:**
- **No wall-clock inside the service.** Returns `last_run_at` raw; the *tab* turns it into
  "3h ago". Keeps `compute_dashboard()` deterministic → trivially unit-testable. (This is the
  one deviation from spec.md's "last run age" field — moved the clock to the UI layer.)
- **Aggregates from `action_log` alone, not per-run DBs.** `RunRow` already carries
  `n_faces`, `n_clusters`, `display_album`, `status`, `config` — see `history.RUN_COLUMNS`.
  Opening 50 per-run DBs (spec.md decision #2) is unnecessary I/O for these metrics.
  `runs_dir` stays an optional ctor arg for a future deep-stats pass. **Deviation flagged —
  needs your ✓** (§7, Decision D2).
- **Profile** comes from `RunRow.config.get("profile_name")` (tolerant of None → `"(none)"`).
- Endpoint: `repo.find(RunHistoryCriteria(producer="fc_app_v2", limit=limit))` — `producer`
  + `limit` already exist on the criteria (`run_history_repo.py:54,60`).

### 3b. Gallery — composes `ClusterAnalysisService` (the spec's "no new service" lock)

The gap the spec missed: **`ClusterRow` has `size`/`n_exemplars` but NO face ids or crop
paths** (`views/_base.py:33-40`). Two ways to get the 8 exemplar thumbnails per cluster:

| Option | What | Cost | Spec-lock |
|---|---|---|---|
| **A (zero deviation)** | `service.compute_detail(cid).faces[:8]` — `FaceRow` already sorted exemplars-first | full `ClusterView.compute` (distance matrix) ×15 clusters/page | honors lock |
| **B (recommended)** | add ONE thin passthrough `service.exemplar_face_ids(cid, n) -> List[int]` over the repo | cheap row read | minor lock deviation |

**Recommend B** — 15 distance-matrix builds per page render (Option A) is wasteful for a
"glance at people" view. B is a 3-line repo passthrough, no new compute. **Needs your ✓**
(§7, Decision D1). Plan below assumes B; falling back to A changes only `cluster_strip.py`.

### 3c. Gallery interactions (added 2026-06-05 from review Q&A)

All reuse existing endpoints — no new service compute except G4.

| ID | Interaction | Endpoint / mechanism | Notes |
|---|---|---|---|
| **G1** | **Thumbnails-per-cluster slider** (default 8, max "all") | `exemplar_face_ids(cid, n)` already takes `n`; `ClusterView.faces` is the full sorted list | filter-bar `st.slider`; caps the strip width. Heavy if "all" × many clusters — keep default low |
| **G2** | **Click a thumbnail → all fields** | per-thumb `st.button("Open")` → `session_state['selected_face_id']` + `st.rerun()` → **Face Analysis tab** | identical to `face_grid`'s existing per-face Open. All fields render there (`compute_face_detail`). NOT an in-place popup |
| **G5-flag** | **Quality badge** on low-quality faces | read-only: `FaceRow.blur_score` / `area` / `is_outlier`; `ClusterView.outlier_face_ids` | `cluster_strip` renders a `⚠` caption tag. `face_grid` already shows `!` for outliers — same source. **No mutation** |
| **G4** | **Nearest-outside-cluster strip** (opt-in toggle) | `compute_detail(cid).nearest_clusters` → `NearestClusterRow{min_exemplar_dist, merge_candidate}` | needs FULL `compute_detail` per cluster (distance matrices) → **forces Option A cost**; gated on Decision D1 + a `st.toggle` so the cheap default path stays fast |

**Disqualify (G5-action) is explicitly NOT here** — it's a mutation needing a persistence model.
Carved out to **spec-069** (`specs/069-gallery-manual-disqualify/`).

---

## 4. Reuse map — build the minimum

| Need | Reuse (existing) | Build (new) |
|---|---|---|
| Crop path convention `crops/face_{id:04d}_aligned.jpg` | `components/face_grid.py:43` | — (share the constant; see §7 D3) |
| 8-col grid width | `face_grid.GRID_COLS = 8` | — |
| Cluster list + sizes | `ClusterAnalysisService.list_clusters()` | — |
| Cross-tab "Open" → select cluster | `cluster_picker` already defaults to `session_state['selected_cluster']` | strip writes that key + `st.rerun()` |
| Run history rows | `RunHistoryRepository.find(...)` | `OverviewService` aggregation |
| Telemetry | `_telemetry.tab_start/done/skipped` | extend with a disable switch (§5) |
| Charts | Plotly (already a dep; Face Analysis uses it) | `components/dashboard_charts.py` bar+line helpers |

**New files (5):**
1. `face_cluster/views/overview.py` — Service + dataclasses (~90 LOC)
2. `app/face_clustering_v2/tabs/gallery_tab.py` (≤80 LOC)
3. `app/face_clustering_v2/tabs/overview_tab.py` (≤80 LOC)
4. `app/face_clustering_v2/components/cluster_strip.py` (~45 LOC) — one cluster's 8-thumb row + Open button
5. `app/face_clustering_v2/components/dashboard_charts.py` (~50 LOC) — `render_bar(stats, ...)`, `render_timeseries(points)`

**Do NOT reuse `face_grid` for Gallery** — it renders *all* faces and writes
`selected_face_id` (wrong target; Gallery selects a *cluster*). `cluster_strip` is the
correct-semantics sibling.

---

## 5. Frontend observability (disable-able)

Today: `_telemetry.py` logs `tab.start/done/skipped` on a dedicated `fc_app_v2.tabs` logger
(plain ASCII k=v, Windows-safe). Extend, don't replace:

```python
# _telemetry.py additions
_ENABLED = os.getenv("FC_V2_TAB_TELEMETRY", "1") != "0"   # single kill-switch

def component_render(tab: str, component: str, **counts):  # NEW: sub-tab granularity
    if not _ENABLED: return
    logger.info("component.render tab=%s name=%s %s", tab, component, _kv(counts))
```

- **One logger, one switch.** Disable at runtime two ways: env `FC_V2_TAB_TELEMETRY=0`,
  or `logging.getLogger("fc_app_v2.tabs").setLevel(WARNING)`. Default ON.
- New events the new tabs emit (answers "what did the user see?" without screenshots):
  - Gallery: `tab.done name=gallery n_clusters=15 page=1 thumbs_rendered=120`
  - Overview: `tab.done name=overview total_runs=N albums=M profiles=K`
  - `component.render tab=gallery name=cluster_strip cluster_id=1 n_thumbs=8`
- Writes to `logs/<ts>/fc_app_v2.log` (already wired via `setup_logging("fc_app_v2")`).
- Arch test `test_logging_aligned.py` already guards the logger-name convention.

---

## 6. Testable endpoints & test plan

| Layer | File | Cases |
|---|---|---|
| **Service (the endpoint)** | NEW `tests/.../test_overview_service_synthetic.py` | 5: empty-history → zeros; single run; multi-album ranking; profile ranking; median/avg + gate-rate math. Inject a fake `RunHistoryRepository` returning synthetic `RunRow`s → **no DB, no clock, fully deterministic** |
| Service (real) | + 1 `@pytest.mark.slow` opt-in | runs `OverviewService(RunHistoryRepository())` against the user's real action_log; asserts `total_runs >= 1` |
| Gallery | (no new service) covered by arch + e2e | — |
| **Architecture** | NEW `test_gallery_tab.py`, `test_overview_tab.py` (mirror `test_face_analysis_tab.py`) | ≤80 LOC; no sql/fs/cfg.get; imports only services/components/widget_factory |
| **Baseline e2e** | NEW `test_scenario_g_gallery.py`, `test_scenario_h_overview.py` | per spec §"E2E contract"; reuse `page_with_reference_run_loaded` fixture (spec-067 query-param seed) |

E2E uses the established pattern (`test_scenario_d_face_analysis.py`): the **test** clicks the
destination tab — Streamlit has no programmatic tab-switch (see D4).

### Two datasets — use the right one per layer

| Dataset | Access | Use in spec-066 |
|---|---|---|
| **Labeled golden** — `test_data/face_clustering/` (3 people × 3 imgs, `ground_truth.csv`) | `tests.conftest.get_test_data_dir()` | fast deterministic run for **service/integration** tests; lets Gallery assert *semantic* facts (person_1's faces share a cluster), reusing the `test_pipeline_e2e.py` purity/completeness helpers |
| **Reference run** — `D:\Budapest2025_Google` + run `6437d335` (340 faces, 15 clusters, 337 crops) | `e2e_budapest/conftest.py` fixtures | the **Playwright e2e** (Scenarios G + H); skips gracefully if absent |

Both verified present on this machine (2026-06-05): album, reference run DB + 337 crops, and
**40 `fc_app_v2` runs** in `action_log` (incl. `Budapest2025_Google_5`) feeding Overview.

---

## 7. Open decisions (need your ✓ before coding)

| # | Decision | Recommendation |
|---|---|---|
| **D1** | Gallery exemplar source: Option A (compute_detail ×15) vs B (thin `exemplar_face_ids` passthrough) | **B** — cheap, avoids 15 distance-matrix builds/page. Minor deviation from spec's "no new method" lock |
| **D2** | Overview source: action_log only vs action_log + 50 per-run DBs (spec decision #2) | **action_log only** — `RunRow` already carries the funnel counts; 50 DB opens is needless I/O |
| **D3** | Crop-path string `face_{id:04d}_aligned.jpg` is hardcoded in 2 places (face_grid, face_analysis). Add 3rd in cluster_strip, or extract a `crops.crop_path(run_dir, face_id)` helper? | **Extract helper** — kills the STOPGAP-tagged dup (face_grid.py:43 TODO already flags it) |
| **D4** | spec.md Scenario G assertion (4) says Open "switches tab". Streamlit can't. | Reword to "Open selects cluster_id 1; the e2e then clicks the Cluster Analysis tab and asserts the picker shows cluster 1" — matches the working Scenario-D pattern |

---

## 8. Task breakdown (supersedes tasks.md once D1–D4 are decided)

**Phase 1 — Service (~1h)**
- T001 `face_cluster/views/overview.py`: dataclasses (§3a) + `OverviewService.compute_dashboard`
- T002 `test_overview_service_synthetic.py` (5 cases, fake repo)
- T003 `+1` real-fixture `slow` case

**Phase 2 — Components + tabs (~1.5h)**
- T010 `components/dashboard_charts.py` (`render_bar`, `render_timeseries`)
- T011 `components/cluster_strip.py` — thumb row + cluster "Open"→`selected_cluster`; **G2** per-thumb "Open"→`selected_face_id`; **G5-flag** `⚠` badge from `is_outlier`/`blur_score`
- T012 (if D3=extract) `face_cluster/.../crops.py` `crop_path()` + repoint face_grid/face_analysis
- T013 (if D1=B) `Overview/ClusterAnalysisService.exemplar_face_ids` passthrough
- T014 `tabs/gallery_tab.py` (≤80; pagination 10/page, filter bar, **G1** thumbnails-per-cluster slider, optional **G4** nearest-strip toggle)
- T015 `tabs/overview_tab.py` (≤80; 4-metric strip + 3 charts; computes "age" from `last_run_at`)
- T016 `_telemetry.py` disable switch + `component_render`
- T017 wire both into `main.py` (`st.tabs([... "Gallery", "Overview"])`)
- T018 `test_gallery_tab.py`, `test_overview_tab.py` arch tests

**Phase 3 — e2e (~1h)**
- T020 Scenario G + H + README matrix rows (move from Planned→active)

**Phase 4 — Parity close-out (~0.5h)**
- T030 `pytest -m budapest` → 8/8 green
- T031 flip `specs/042-fc-app-v2-tab-parity/spec.md` → Implemented
- T032 `/code-review` → REVIEW.md; CHANGES_LOG; spec-066 → Implemented; commit + push

**Total: ~4h** (matches spec estimate; +0.5h if D3 helper-extract chosen).

---

## 9. Sequencing diagram (data flow)

See [`ARCHITECTURE.html`](ARCHITECTURE.html) for the rendered version. Summary:

```
Gallery:   History.Load → session.current_run_dir
           gallery_tab → ClusterAnalysisService.list_clusters()
                       → [per cluster] exemplar_face_ids() → cluster_strip(crops)
                       → Open → session.selected_cluster → (user) Cluster Analysis tab

Overview:  overview_tab → OverviewService(RunHistoryRepository).compute_dashboard()
                        → DashboardMetrics → metric strip + dashboard_charts (Plotly)
```
