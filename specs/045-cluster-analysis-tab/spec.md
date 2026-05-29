# spec-045 — Cluster Analysis Tab

**Created**: 2026-05-25
**Updated**: 2026-05-29 — Phases 1–8 implemented; REVIEW.md filed; F-2/F-3 folded in.
**Status**: Implemented
**Predecessors**: spec-042 (v2 parity umbrella), spec-046 (SQLAlchemy + Alembic data layer), spec-048 (data-layer cleanup: `_paths`, `ensure_schema`), spec-050 (v2 run picker + per-run UUID dirs), commit `9824d84` (NOISE_LABEL contract).
**Successors**: spec-047 (full-system E2E)

> **Commit-ordering prerequisite**: spec-046 + spec-048 + spec-050 work currently sits uncommitted in the working tree on `unification/spec-040`. Those must land before Phase 1 of this spec starts, because every Repository/Service test in §8 uses the spec-046 `BaseRepository`, the spec-048 `_paths.default_db_path`, and (for the tab phase) the spec-050 `render_run_picker` / `v2_last_run_dir` session key.

---

## Problem

Legacy `app/face_clustering/tabs/cluster_analysis_tab.py` (~383 LOC) reaches directly into the per-run `face_clustering.db` and the filesystem from inside Streamlit callbacks. It mixes UI orchestration, async worker scaffolding, raw SQL, distance-matrix computation, and force-merge mutation in one file.

This tab is **the user's daily workbench** — picking a cluster, drilling into faces, spotting drift, force-merging when the conservative merger refused. v2 must give it the same Tab → Service → Repository → DB layering the History tab pilot established, on top of the SQLAlchemy foundation spec-046 just built.

---

## What we build

| Layer | Module | Purpose |
|---|---|---|
| UI | `app/face_clustering_v2/tabs/cluster_analysis_tab.py` | Declarative orchestrator. ≤200 LOC. No DB, no FS, no SQL. |
| UI components | `app/face_clustering_v2/components/cluster_*.py` | Stateless render functions per region (metrics, exemplars, grid, nearest, force-merge, graph debug). |
| Service | `face_cluster/views/cluster_analysis.py` | `ClusterAnalysisService` — typed compute methods, async work hidden behind `compute_*_async()`. Streamlit-free. |
| Repository | `face_cluster/repositories/cluster_analysis_repo.py` | `ClusterAnalysisRepository(BaseRepository)` — reads the per-run `face_clustering.db` via SQLAlchemy. Schema owned by `RunStore`. |
| ORM models | `face_cluster/repositories/models/{image,scene,face_record,cluster_row,exemplar,cluster_stats}.py` | One model per per-run table. Inherits the spec-046 `Base`. |
| Typed contracts | `ForceMergePreview`, `ForceMergeResult`, `AsyncHandle[T]` | Replace legacy `dict` returns. Live in the service module. |

The Repository **composes**, doesn't own — `RunStore`'s `PRAGMA user_version` + `EXPECTED_ARTIFACTS` remains the schema authority for per-run DBs. Alembic does not manage these DBs (per-run lifecycle, see spec-046 §non-goals).

---

## What we don't build

- New UI invented. Layout matches legacy region-for-region.
- Migration of `RunStore` itself (per-run DB lifecycle is a separate concern; if it earns its keep later, spec-048).
- Async framework changes. Reuse existing `_AsyncState` pattern, wrap in `AsyncHandle[T]`.
- Universal embedding cache (`sim_bench.db`) changes.

---

## Locked decisions

1. **ORM models for the per-run DB** — typed access, even though Alembic doesn't manage migrations (RunStore does). Same SQLAlchemy `Session` pattern as spec-046.
2. **Repository composes RunStore** — the Repository never writes the per-run schema; it queries existing tables and asks RunStore for fresh sessions / artifact paths.
3. **Force-merge mutation writes a new run snapshot** (`merge_snap_{round}/`) — never mutates the parent run dir. Repository exposes `write_merge_snapshot(...)`; filesystem effects are explicit in the contract.
4. **Async hidden behind `compute_*_async()`** — service returns `AsyncHandle[ClusterView]`; tab polls the handle. Worker scaffolding (the 4-branch state machine) lives in `face_cluster/views/_async.py`, not in the tab.
5. **Force-merge preview is typed** (`ForceMergePreview` dataclass) — no untyped dicts crossing the service boundary.
6. **No new UI** — every widget user sees today behaves the same way. Only the code shape changes.

---

## User stories

### US-1 — Pick a cluster (P1)

Dropdown lists every `cluster_id` in the loaded run, sorted ascending, labelled `"Cluster {id} ({size} faces)"`. Selection persists via `st.session_state.selected_cluster`. Switching clusters cancels in-flight compute for the previous selection. No run loaded → friendly empty state pointing at History → Load Run.

### US-2 — Inspect cluster metrics + faces (P1)

- 5-metric strip: Faces / Diameter / Avg intra-dist / Exemplars / Outliers
- Provenance row (when `cluster_stats.origin` set): origin tag + parent cluster ids
- Split-signal banner when bimodal distance distribution detected (gap > 0.12 AND > 3× mean gap)
- Exemplars section: top 5 by default; "Show all N" toggle when count > 5
- All-Faces grid: 8 cols, sorted `(is_exemplar desc, dist_to_exemplar asc)`. Each cell shows crop + role tag ("EX"/"!"/blank) + `face_{id:04d}` + distance
- Per-face "Open Face Analysis" button → sets session state, prompts tab switch
- "Face detail table" expander with full per-face metrics
- All compute via `ClusterAnalysisService.compute_detail_async()`; tab shows "Analysing cluster N…" while in flight

### US-3 — See nearest clusters and jump (P2)

"Nearest Clusters" lists top 10 by `min_exemplar_dist` ascending. Each row: cluster id, size, min exemplar distance, p10 cross distance, threshold, "-- MERGE CANDIDATE" badge when `min_exemplar_dist < threshold`. Up to 4 exemplar thumbnails per row, each with "Open Face Analysis" button. "Go to C{id}" updates session state and invalidates compute workers.

### US-4 — Force-merge with preview (P2)

"Force Merge" expander, collapsed by default.

- Two cluster dropdowns (A, B); B excludes A's selection
- "Preview Merge" computes exemplar-to-exemplar distance vs threshold + 3-gate verdict (exemplar / support / post-diameter) + up to 3 thumbnails per side
- Preview re-renders inline: green if candidate, amber if not. Three PASS/FAIL gate badges with measured values
- "Confirm: Merge C{a} + C{b}" writes manual-merge snapshot, increments `merge_round`, reloads run via Repository, invalidates compute, success toast, rerun
- Errors (missing embeddings, snapshot failure, invalid ids) surface via `st.error` with the underlying message — no silent swallows

### US-5 — Graph diagnostics (P3)

"Graph Debug" section computes asynchronously via `ClusterAnalysisService.compute_debug_async()`.

- 4-metric strip: Edges / Density / Chain score (with help tooltip) / Bridge faces
- Chain score > 1.5 → warning banner with diameter + median distance
- Density < 15% AND n_faces > 4 → "sparse graph" warning
- Bridge faces: face_id badges + up to 8 thumbnails
- Distance heatmap (≥2 faces with embeddings): Plotly imshow, capped at 0.6, expanded by default for clusters ≤ 30 faces
- "Per-face graph connectivity" expander: dataframe (face_id, edge_count, bridge_flag, neighbours)
- "All edges" expander: dataframe sorted by distance asc

---

## Data contracts

### Reused (no changes)

`ClusterRow`, `NearestClusterRow`, `FaceRow`, `EdgeInfo`, `FaceGraphInfo`, `CloseFace`, `ClusterView`, `ClusterDebugView` — all already defined in `face_cluster/views/`. Service returns these as-is.

### New (`face_cluster/views/cluster_analysis.py`)

```python
@dataclass(frozen=True, slots=True)
class ForceMergePreview:
    cluster_a: int
    cluster_b: int
    exemplar_dist: float
    threshold: float
    is_candidate: bool
    passes_exemplar: bool
    passes_support: bool
    passes_diameter: bool
    n_gates_passed: int
    post_diameter: float
    support: int
    cluster_a_size: int
    cluster_b_size: int
    exemplar_face_ids_a: list[int]
    exemplar_face_ids_b: list[int]

@dataclass(frozen=True, slots=True)
class ForceMergeResult:
    snapshot_dir: Path
    merge_round: int
    parent_run_dir: Path
    new_cluster_id: int
    n_merged: int

@dataclass
class AsyncHandle(Generic[T]):
    """Encapsulates the legacy _AsyncState pattern."""
    state: Literal["pending", "running", "done", "failed", "cancelled"]
    result: Optional[T] = None
    error: Optional[Exception] = None
    def cancel(self) -> None: ...
```

### Repository contract

```python
@dataclass(frozen=True, slots=True)
class ClusterAnalysisRepoConfig:
    run_dir: Path   # the run's output directory; Repository reads <run_dir>/face_clustering.db

@dataclass(frozen=True, slots=True)
class ClusterAnalysisCriteria:
    cluster_id: Optional[int] = None
    include_outliers: bool = True

class ClusterAnalysisRepository(BaseRepository):
    # reads
    def list_clusters(self) -> list[ClusterRow]: ...
    def get_cluster(self, cluster_id: int) -> Optional[ClusterRow]: ...
    def list_faces_in_cluster(self, cluster_id: int) -> list[FaceRow]: ...
    def list_exemplars(self, cluster_id: int) -> list[FaceRow]: ...
    def get_cluster_stats(self, cluster_id: int) -> Optional[ClusterStats]: ...
    def find_nearest_clusters(self, cluster_id: int, limit: int = 10) -> list[NearestClusterRow]: ...
    # writes (force-merge)
    def write_merge_snapshot(self, ...) -> ForceMergeResult: ...
```

Constructor accepts a typed :class:`ClusterAnalysisRepoConfig` (see §5.3) — **not** a SQLAlchemy ``Session``. Per spec D3 the per-run DB is not Alembic-managed; the Repository is the **query-shape** (B0b) flavor and inherits ``BaseRepository`` with ``session=None`` purely for the static error-translation helpers. See `docs/architecture/architecture_standards.md` §B0.2.1 for the B0a-vs-B0b distinction.

---

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | Tab file ≤ 200 LOC; no SQL / FS / `cfg.get()` literals | Phase 7 (final review) |
| AC2 | Legacy `cluster_analysis_tab.py` behaviour preserved widget-for-widget | Manual diff + Playwright (Phase 6) |
| AC3 | All compute hidden behind `compute_*_async()`; tab polls a typed `AsyncHandle[T]` | Phase 3 service test |
| AC4 | Service has ≥10 synthetic test cases covering every public method | Phase 3 |
| AC5 | Repository has ≥10 synthetic test cases covering every public method | Phase 2 |
| AC6 | Force-merge writes `merge_snap_{round}/` to disk; never mutates parent run | Phase 4 integration test on real-fixture |
| AC7 | Repository constructed with SQLAlchemy `Session`; tests use `transactional_session` fixture from spec-046 | Phases 2-5 |
| AC8 | Streamlit-free backend — `service.py` + `repo.py` runnable in pytest with no Streamlit import | Phase 3 |

---

## Out-of-scope risks

- Per-run DB schema evolution. If a future version of `RunStore` adds a column, our ORM models must mirror it. RunStore's `PRAGMA user_version` check fails loudly on version mismatch — that's our drift signal.
- Concurrent compute on multiple clusters. The legacy async pattern was single-cluster; we preserve that. Multi-cluster compute is a separate spec.

---

## Definition of Done

- All 8 acceptance criteria green
- v2 Cluster Analysis tab launches; every widget the legacy tab shows is present and works
- `pytest tests/face_clustering/{repositories,views,components}/` green
- New Playwright test for cluster-analysis tab passes
- §B0.2 example extended with the second Repository (proves the foundation is reusable)
- REVIEW.md filed; no high-severity findings
- CHANGES_LOG.md entry

---

## 5 — Data contracts (detailed)

This section is the field-level source of truth for the dataclasses sketched in "Data contracts" above. Phase-1/-2 tasks reference it by number.

### 5.1 — `ForceMergePreview` — see spec body, "Data contracts → New" block. No additional fields.

### 5.2 — `ForceMergeResult` — see spec body. No additional fields.

### 5.3 — `ClusterAnalysisRepoConfig`

```python
@dataclass(frozen=True, slots=True)
class ClusterAnalysisRepoConfig:
    run_dir: Path           # required; <run_dir>/face_clustering.db is read
    read_only: bool = False  # when True, mutation methods raise ValidationError
    log_queries: bool = False  # when True, Repository emits one INFO log per query
```

Constructor validates: `run_dir.is_dir()` and `(run_dir / "face_clustering.db").exists()`. Invalid → raises `ValueError` immediately.

### 5.4 — `ClusterAnalysisCriteria`

```python
@dataclass(frozen=True, slots=True)
class ClusterAnalysisCriteria:
    cluster_id: Optional[int] = None       # filter to one cluster; None = all
    face_ids: Optional[tuple[int, ...]] = None  # additional face_id filter (tuple for hashability)
    iteration: str = "final"               # which clustering iteration to read
    exemplars_only: bool = False           # when True, only is_exemplar=1 rows returned
    include_noise: bool = False            # when True, NOISE_LABEL cluster_id is allowed
```

`include_noise=False` is the default — every Repository read excludes `cluster_id == NOISE_LABEL` unless explicitly requested. Uses `face_cluster.views._base.is_noise` / `NOISE_LABEL` (re-exported from `sim_bench.pipeline.clustering_labels`).

### 5.5 — `Assignment`

```python
@dataclass(frozen=True, slots=True)
class Assignment:
    face_id: int
    cluster_id: int     # NOISE_LABEL when face is noise
    is_exemplar: bool
    iteration: str
```

Lives in `face_cluster/views/_base.py` (shared row type; future tabs reuse it).

---

## 6 — Service contract

### 6.1 — `ClusterAnalysisService`

```python
class ClusterAnalysisService:
    """Owns: a ClusterAnalysisRepository. Provides typed read + compute methods.

    No Streamlit imports. No session_state writes. Async compute is hidden
    behind compute_*_async() returning AsyncHandle[T].
    """

    def __init__(self, repo: ClusterAnalysisRepository) -> None: ...

    # cheap reads (Phase 3)
    def list_clusters(self) -> list[ClusterRow]: ...
    def get_cluster_ids(self) -> list[int]: ...

    # async compute (Phase 4)
    def compute_detail_async(self, cluster_id: int) -> AsyncHandle[ClusterView]: ...
    def compute_debug_async(self, cluster_id: int) -> AsyncHandle[ClusterDebugView]: ...

    # force-merge (Phase 5)
    def preview_force_merge(self, cluster_a: int, cluster_b: int) -> ForceMergePreview: ...
    def apply_force_merge(self, cluster_a: int, cluster_b: int, *, merge_round: int) -> ForceMergeResult: ...
```

Cancellation contract: each `compute_*_async` call cancels any handle still running on the same service instance for the same compute kind. `cluster_picker` change triggers cancellation; force-merge confirm triggers cancellation.

---

## 7 — Tab orchestrator skeleton

### 7.1 — `render_cluster_analysis_tab`

```python
def render_cluster_analysis_tab() -> None:
    """Render the Cluster Analysis tab.

    Layout: run picker → cluster picker → metrics → faces → nearest → force merge → graph debug.
    Reads:  ClusterAnalysisService (queries the per-run face_clustering.db).
    Writes: session_state['selected_cluster'] on dropdown change;
            session_state['current_run_dir'] on force-merge confirm (new snapshot dir).
    """
    st.header("Cluster Analysis")

    # 1. Resolve the run. Reuses spec-050's render_run_picker; falls back to
    #    st.session_state.current_run_dir when set by History → Load Run.
    run_dir = _resolve_current_run_dir()
    if run_dir is None:
        st.info("No run loaded. Open the History tab and click **Load Run**.")
        return

    # 2. Build the service. Lazy: cached on (run_dir, session id).
    service = _get_service(run_dir)

    # 3. Pick a cluster (US-1).
    cluster_id = render_cluster_picker(service.get_cluster_ids())
    if cluster_id is None:
        return

    # 4. Metrics + faces (US-2). Async; handle yields ClusterView.
    detail_handle = service.compute_detail_async(cluster_id)
    render_cluster_metrics(detail_handle)
    render_face_grid(detail_handle)

    # 5. Nearest clusters (US-3). Synchronous from cached cluster_rows.
    render_nearest_clusters(service.list_clusters(), cluster_id)

    # 6. Force merge (US-4). Component owns its own state for the A/B dropdowns;
    #    delegates preview + apply to the service.
    render_force_merge(service, current_cluster=cluster_id)

    # 7. Graph debug (US-5). Async; handle yields ClusterDebugView.
    debug_handle = service.compute_debug_async(cluster_id)
    render_cluster_debug(debug_handle)
```

Constraints: ≤ 80 LOC including helpers; no SQL, no JSON parsing, no `cfg.get(...)` literals, no direct `RunStore` instantiation.

### 7.2 — Run-dir resolution (reconciles with spec-050)

```python
def _resolve_current_run_dir() -> Optional[Path]:
    """Single source of truth for which run this tab is looking at.

    Priority:
      1. st.session_state['current_run_dir'] (set by History → Load Run, T050)
      2. st.session_state['v2_last_run_dir'] (set by spec-050 run-tab on every fresh run)
      3. None (no run available — caller renders empty state)
    """
```

Cluster Analysis does **not** render its own picker — the History tab is the single load/discovery surface in v2. spec-050's `render_run_picker` is reused only by tabs whose primary action is "pick a run" (History, Clusters MVP being deleted).

---

## 8 — Test inventory

Naming: `tests/face_clustering/repositories/test_cluster_analysis_repo_{synthetic,real}.py` and `tests/face_clustering/views/test_cluster_analysis_service_{synthetic,real}.py`. Arch tests in `tests/architecture/test_cluster_analysis_tab.py`. Playwright smoke in `tests/manual/_v2_cluster_analysis_smoke.py`.

Synthetic fixtures use the spec-046 `transactional_session` + a small builder in `tests/face_clustering/views/_seed_cluster_analysis.py` that writes the per-run schema (3 clusters, ~30 faces, exemplars marked) into `tmp_path/face_clustering.db` + `tmp_path/embeddings.npy`. **All tests use `NOISE_LABEL` from `sim_bench.pipeline.clustering_labels` — no bare `-1`.**

### 8.1 — Repository synthetic (12 cases)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_init_validates_run_dir` | Non-existent `run_dir` → `ValueError`; valid → constructs |
| 2 | `test_init_validates_db_exists` | Missing `face_clustering.db` → `ValueError` |
| 3 | `test_get_cluster_rows_returns_typed_list` | Returns `list[ClusterRow]`; length == 3; sorted by `cluster_id` ascending |
| 4 | `test_get_cluster_rows_excludes_noise_by_default` | Synthetic DB has a noise cluster (`NOISE_LABEL`); not returned |
| 5 | `test_get_cluster_ids_matches_rows` | `get_cluster_ids()` == `[r.cluster_id for r in get_cluster_rows()]` |
| 6 | `test_find_assignments_by_cluster` | `Criteria(cluster_id=2)` → only face_ids in cluster 2 |
| 7 | `test_find_assignments_exemplars_only` | `Criteria(exemplars_only=True)` → only `is_exemplar=True` rows |
| 8 | `test_find_assignments_include_noise` | `Criteria(include_noise=True)` → returns rows with `cluster_id == NOISE_LABEL` |
| 9 | `test_get_face_records_filters_by_id` | Pass 5 face_ids → exactly 5 records; unknown id silently dropped |
| 10 | `test_get_run_metadata_shape` | Returns `RunMetadata` with `n_clusters=3`, `n_faces≥30`, `merge_round` int |
| 11 | `test_save_manual_merge_snapshot_writes_snapshot` | Returns `ForceMergeResult`; `snapshot_dir.exists()`; parent dir untouched |
| 12 | `test_save_manual_merge_snapshot_read_only_raises` | `read_only=True` → `ValidationError` raised; nothing written |

### 8.2 — Repository real-fixture (3 cases)

Uses the most-recent v2 Budapest run dir (session fixture `v2_budapest_run_dir` in `tests/conftest.py`). Skips cleanly when no such run exists.

| # | Test | Asserts |
|---|---|---|
| 1 | `test_real_run_loads` | Repository constructs from the fixture path; no error |
| 2 | `test_real_run_has_clusters` | `get_cluster_rows()` returns ≥ 1 row; every row has `face_count > 0` |
| 3 | `test_real_run_metadata_consistent` | `n_clusters == len(get_cluster_rows())` |

### 8.3 — Service synthetic (12 cases)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_list_clusters_passthrough` | Service `list_clusters()` == Repository `get_cluster_rows()` |
| 2 | `test_get_cluster_ids_passthrough` | Same shape as Repository |
| 3 | `test_init_requires_repo` | Constructing with `None` raises |
| 4 | `test_compute_detail_async_returns_cluster_view` | Handle reaches `state == "done"`; `result` is `ClusterView` |
| 5 | `test_compute_detail_async_unknown_cluster_surfaces_error` | Handle reaches `state == "failed"`; `error` is set |
| 6 | `test_compute_debug_async_returns_debug_view` | Handle reaches `state == "done"`; `result` is `ClusterDebugView` |
| 7 | `test_preview_force_merge_typed` | Returns `ForceMergePreview` with all fields populated |
| 8 | `test_preview_force_merge_below_threshold_is_candidate` | Two near clusters → `is_candidate == True` |
| 9 | `test_preview_force_merge_above_threshold_not_candidate` | Two distant clusters → `is_candidate == False` |
| 10 | `test_apply_force_merge_returns_result` | Returns `ForceMergeResult` with `snapshot_dir.exists()` |
| 11 | `test_apply_force_merge_increments_round` | Second apply on same parent → `merge_round` incremented |
| 12 | `test_cancel_in_flight_compute_on_new_call` | `compute_detail_async` × 2 → first handle reaches `state == "cancelled"` |

### 8.4 — Service real-fixture (4 cases)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_real_list_clusters_nonempty` | ≥ 1 cluster |
| 2 | `test_real_compute_detail_completes` | Handle reaches `done` within 10 s |
| 3 | `test_real_compute_debug_completes` | Handle reaches `done` within 10 s |
| 4 | `test_real_preview_force_merge_runs` | Picks two smallest clusters; preview returns typed result (regardless of candidacy) |

### 8.5 — Architecture tests (5 cases)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_tab_has_no_direct_db_or_filesystem_access` | grep tab + components for `sqlite3`, `open(`, `Path(...).read_text`, `RunStore(` → 0 hits |
| 2 | `test_tab_has_no_cfg_get_literals` | grep for `cfg.get(` in v2 tab dir → 0 hits |
| 3 | `test_service_returns_typed_objects` | `inspect.signature(...).return_annotation` ≠ `dict` for every public Service method |
| 4 | `test_repository_takes_typed_config` | `ClusterAnalysisRepository.__init__` parameter type == `ClusterAnalysisRepoConfig` |
| 5 | `test_force_merge_preview_fields_match_writer` | `set(ForceMergePreview fields) ⊇ set(fields apply_force_merge populates in ForceMergeResult)` — drift guard |

### 8.6 — Playwright smoke (manual)

5-step sequence against `streamlit run app/face_clustering_v2/main.py --server.port 8889`:

1. Open History tab; pick a fixture run; click Load Run; expect tab toast.
2. Switch to Cluster Analysis tab; expect cluster picker populated with ≥ 1 cluster.
3. Pick cluster N; within 10 s, expect ≥ 1 face thumbnail and the 5-metric strip visible.
4. Expand "Nearest Clusters"; expect ≥ 1 row.
5. Open Force Merge; pick A=N, B=any-other; click Preview; expect typed preview block (PASS/FAIL gates), no Streamlit exception.

Opt-in (`-m manual`); not part of CI gate.
