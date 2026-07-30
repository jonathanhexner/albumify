# Tasks: FC App v2 Tab Architecture & Parity (042)

Predecessor: spec-041 (FCParams + UI_SPEC + logging + step telemetry)
Target: rebuild the missing v2 tabs against the spec-041 contracts; ship the strangler-fig user-facing migration.

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped (rationale required)

---

## Design notes (the contract every task assumes)

- **D1**: Two-layer split. `face_cluster/views/*.py` is the backend; cannot import `streamlit`. `app/face_clustering_v2/tabs/*.py` is the UI; calls into services, never duplicates their logic. Enforced by `tests/architecture/test_v2_layering.py`.
- **D2**: Every view ships a `*Service` class with typed methods (no dicts) and a clear naming convention: `list_*` / `get_*` are read-only; `update_*` / `delete_*` / `load_*` are mutations with named side effects in the docstring.
- **D3**: Wherever the legacy code has "a list of field names with metadata next to each name," replace with a declarative spec (`ColumnSpec`, `ActionTypeFormat`, or `UI_SPEC`-readonly). One renderer reads from one spec. No per-field literals.
- **D4**: Every public class/function in new v2 code has a docstring describing contract (Args / Returns / Side effects). Functions >30 LOC are split. No `fc1, fc2, fc3` placeholders. Enforced by `tests/architecture/test_v2_module_docstrings.py`.
- **D5**: Three test layers per tab: **service unit (synthetic data, fast, deterministic)** + **service integration (real Budapest fixture, smoke-level)** + **UI smoke (Playwright, opt-in)**. The synthetic-data layer is load-bearing; the real-data layer confirms it.
- **D6**: `tests/face_clustering/views/_seed.py` provides shared row factories so service tests don't repeat DB-seeding boilerplate.
- **D7**: Legacy app at `app/face_clustering/` stays untouched. Retirement is a follow-up after burn-in.
- **D8**: Pilot on History first (Phase 1). The pilot establishes the full pattern; subsequent tabs follow the template.

---

## Phase 0 — Prerequisites

- [ ] **T001** Fix `face_cluster/analysis_views.py::_parse_merge_log` broken import (pre-existing).
- [ ] **T002** Confirm `tests/face_clustering/test_merge_analysis.py` collects after T001.
- [ ] **T003** Add session fixture `v2_budapest_run_dir` to `tests/conftest.py`; skips if `~/.sim_bench/runs/v2_budapest_*` is empty.
- [ ] **T004** Add fixtures `synthetic_action_log_db` and `synthetic_run_dir` to `tests/conftest.py`.
- [ ] **T005** Add `tests/face_clustering/views/_seed.py` with row factories: `_make_action`, `_make_cluster`, `_make_face`, `_make_merge_decision`, `_make_face_score`.
- [ ] **T006** Add `face_cluster/views/__init__.py` and `face_cluster/views/_specs.py` with `ColumnSpec`, `ActionTypeFormat`.
- [ ] **T007** Add `widget_factory.render_field(name, readonly=True, value_override=...)` mode. Renders the same widget disabled, displaying `value_override` (or the field default if None).
- [ ] **T008** Add `tests/face_clustering/test_widget_factory_readonly.py` (≥4 cases: int / float / bool / Optional zero-as-none).
- [ ] **T009** Add `tests/architecture/test_v2_layering.py`: `face_cluster/views/*` does not import `streamlit`; `app/face_clustering_v2/tabs/*` does not access DB / JSON directly.
- [ ] **T010** Add `tests/architecture/test_v2_module_docstrings.py`: every public class and function in `face_cluster/views/` and `app/face_clustering_v2/` has a non-empty docstring.

**Checkpoint**: All Phase 0 tests pass; fixtures resolve; arch tests fail loud if anyone violates the layering.

---

## Phase 1 — Pilot: History tab top-to-bottom

This phase establishes the reference pattern.

### Backend service

- [ ] **T020** Add `face_cluster/views/history.py` with:
  - `@dataclass(frozen=True) HistoryQuery` — filter inputs
  - `@dataclass(frozen=True) RunRow` — table-row shape
  - `@dataclass(frozen=True) RunDetail` — selected-run detail
  - `@dataclass(frozen=True) LoadedRun` — what a Load returns
  - `RUN_COLUMNS: list[ColumnSpec]` — declarative table columns
  - `class HistoryService` — `list_runs(query)`, `list_albums()`, `get_run_detail(id)`, `update_comment(id, text)`, `load_run(id)`
  - Module-level docstring + per-class / per-method docstrings per D4
- [ ] **T021** Add `tests/face_clustering/views/test_history_service_synthetic.py` (≥8 cases):
  - empty DB → empty list
  - filter by album narrows results
  - filter by date range
  - text filter matches album / run_name / comment substring
  - newest-first ordering
  - `get_run_detail` returns full shape with `parent_run_id` populated when parent exists
  - `update_comment` is idempotent and respects 2048-char limit
  - `load_run` returns typed `LoadedRun`; does NOT touch session_state
- [ ] **T022** Add `tests/face_clustering/views/test_history_service_real.py`:
  - against `v2_budapest_run_dir`'s action_log, `list_runs(producer='fc_app_v2')` returns ≥1 row
  - `get_run_detail` on a Budapest run returns a `RunDetail` with config matching the saved profile

### Streamlit components

- [ ] **T030** Add `app/face_clustering_v2/components/run_filter_bar.py` with `render_filter_bar() -> HistoryQuery`. ~30 LOC. Docstring states layout and return.
- [ ] **T031** Add `app/face_clustering_v2/components/run_table.py` with `render_run_table(rows, columns) -> Optional[int]`. Generic over `ColumnSpec`. ~50 LOC.
- [ ] **T032** Add `app/face_clustering_v2/components/run_detail.py` with `render_run_detail(detail: RunDetail) -> None`. Uses `widget_factory.render_field(readonly=True)` for the config view — no `cfg.get('field', '?')` literals.
- [ ] **T033** Add `app/face_clustering_v2/components/load_button.py` with `render_load_button(detail: RunDetail) -> None`. Calls `HistoryService.load_run()`; writes result to `st.session_state` (the only Streamlit-state mutation, isolated to this component).

### Tab

- [ ] **T040** Add `app/face_clustering_v2/tabs/history_tab.py`:
  ```python
  def render_history_tab() -> None:
      """Render the History tab. Layout: filter bar → table → selected detail.
      Reads HistoryService; writes session_state on Load."""
      query = render_filter_bar()
      rows = HistoryService.list_runs(query)
      selected_id = render_run_table(rows, RUN_COLUMNS)
      if selected_id is not None:
          detail = HistoryService.get_run_detail(selected_id)
          render_run_detail(detail)
          render_load_button(detail)
  ```
  Target: ≤ 30 LOC orchestration.
- [ ] **T041** Wire into `main.py`'s `st.tabs([...])`.

### UI smoke

- [ ] **T050** Add `tests/manual/_v2_history_smoke.py`: open History tab, assert ≥1 row visible, no Streamlit exception markdown.

**Checkpoint**: History tab works against real Budapest data. Pattern is concrete enough that subsequent tabs become template-application.

---

## Phase 2 — P1 tabs

### Cluster Analysis (rebuild; replaces v2's existing "Clusters")

- [ ] **T060** Add `face_cluster/views/cluster_analysis.py` with `ClusterRow`, `ClusterDetail`, `ClusterAnalysisService`.
- [ ] **T061** Synthetic-data unit tests in `tests/face_clustering/views/test_cluster_analysis_service_synthetic.py` (≥5 cases).
- [ ] **T062** Real-fixture integration test (33 clusters baseline).
- [ ] **T063** Add `app/face_clustering_v2/components/cluster_card.py` (reuses `face_grid` from Phase 3).
- [ ] **T064** Add `app/face_clustering_v2/tabs/cluster_analysis_tab.py`; remove the old `clusters_tab.py` and its wiring.
- [ ] **T065** UI smoke (`_v2_cluster_analysis_smoke.py`).

### Recluster (needs new runner mode)

- [ ] **T070** Add `RunStore.load_face_records() -> list[FaceRecord]`. Materializes typed records from the v5 DB.
- [ ] **T071** Add `FCAppRunner.recluster(prior_run_dir, output_dir, *, params: FCParams) -> FCAppRunResult`. Reads existing face_records + embeddings from prior run; runs only the unified clustering chain.
- [ ] **T072** `RunExporter` writes `parent_run_id` in `run_metadata` for traceability.
- [ ] **T073** Add `face_cluster/views/recluster.py` with `ReclusterRequest`, `ReclusterResult`, `ReclusterService`.
- [ ] **T074** Synthetic-data unit tests for `ReclusterService` (≥4 cases).
- [ ] **T075** Real-fixture integration test: recluster Budapest with K=3 → success + cluster count differs from 33.
- [ ] **T076** Add **equivalence test** `tests/face_clustering/test_recluster_equivalence.py`: legacy `FaceClusteringPipeline.recluster()` vs `FCAppRunner.recluster()` on the same prior run + FCParams → ≥95% pairwise cluster agreement.
- [ ] **T077** Add `app/face_clustering_v2/tabs/recluster_tab.py` + wire into `main.py`.
- [ ] **T078** UI smoke (`_v2_recluster_smoke.py`).

**Checkpoint**: v2 has 4 tabs working (Run, Cluster Analysis, Recluster, History). User can re-run with new params and inspect output without leaving v2.

---

## Phase 3 — P2 tabs

### Face Analysis

- [ ] **T080** Add `face_cluster/views/face_analysis.py` with `FaceDetail`, `FaceAnalysisService`.
- [ ] **T081** Synthetic unit tests (≥4 cases including missing-landmarks edge case).
- [ ] **T082** Real-fixture integration test.
- [ ] **T083** Add `app/face_clustering_v2/components/face_grid.py` (shared with Cluster Analysis / Gallery).
- [ ] **T084** Add `app/face_clustering_v2/tabs/face_analysis_tab.py` + wire.
- [ ] **T085** UI smoke.

### Merged Clusters

- [ ] **T090** Add `face_cluster/views/merged_clusters.py` with `MergeDecisionRow`, `MergedClustersService`.
- [ ] **T091** Synthetic unit tests (≥3 cases).
- [ ] **T092** Real-fixture integration test (against a `merge_enabled=True` Budapest run).
- [ ] **T093** Add `app/face_clustering_v2/tabs/merged_clusters_tab.py` + wire.
- [ ] **T094** UI smoke.

### Quality

- [ ] **T100** Add `face_cluster/views/quality.py` with `QualityRow`, `QualityService`.
- [ ] **T101** Synthetic unit tests (≥4 cases).
- [ ] **T102** Real-fixture integration test.
- [ ] **T103** Add `app/face_clustering_v2/tabs/quality_tab.py` + wire.
- [ ] **T104** UI smoke.

**Checkpoint**: v2 has 7 tabs. Inspection layer complete.

---

## Phase 4 — P3 tabs

### Gallery

- [ ] **T110** Add `face_cluster/views/gallery.py` (`GalleryService` reuses `ClusterRow` / `FaceRow`).
- [ ] **T111** Synthetic unit tests (≥3 cases).
- [ ] **T112** Add `app/face_clustering_v2/tabs/gallery_tab.py` + wire.
- [ ] **T113** UI smoke.

### Overview

- [ ] **T120** Add `face_cluster/views/overview.py` with `OverviewSummary`, `OverviewService`.
- [ ] **T121** Synthetic unit tests (≥3 cases for aggregation logic).
- [ ] **T122** Real-fixture integration test: aggregator matches the 33-cluster / 340-face baseline.
- [ ] **T123** Add `app/face_clustering_v2/tabs/overview_tab.py` + wire.
- [ ] **T124** UI smoke.

**Checkpoint**: v2 has 9 tabs. Full parity with legacy user-facing surface (Merge ML and Labeling Review excluded as research-only).

---

## Phase 5 — Close-out

- [ ] **T130** Run `/code-review` → `specs/042-fc-app-v2-tab-parity/REVIEW.md`. Resolve high-severity findings.
- [ ] **T131** Update `docs/architecture/classes.html`: add `face_cluster.views.*Service` classes, `widget_factory.render_field(readonly=True)`.
- [ ] **T132** Update `docs/architecture/data_flow.html`: tab → service → DB flow.
- [ ] **T133** Append CHANGES_LOG entry per landed commit.
- [ ] **T134** Flip spec status `Draft` → `Code Review` → `Implemented`.
- [ ] **T135** File follow-up spec for legacy-app retirement (burn-in clock starts at T134).

**Checkpoint**: spec marked Implemented; legacy app retirement planned.

---

## Out of scope (explicit deferrals)

- **Merge ML tab** — research-only.
- **Labeling Review tab** — research-only.
- **Merge Analysis ML scoring sub-panel** — defers with Merge ML.
- **Retiring `app/face_clustering/`** — separate spec, gated on 2-week burn-in after this lands.
- **Migrating `app/shared/merge_controls.py` to FCParams-driven widgets** — separate mini-spec.
- **Legacy tabs consuming the new view services** — optional follow-up; the strangler-fig works either way.
