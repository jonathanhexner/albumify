# spec-050 — v2 App: Per-Run Allocation + History-Driven Loading

**Created**: 2026-05-28
**Status**: Implemented
**Predecessors**: spec-040 (v2 app MVP), spec-046/048 (data layer)
**Linked sightings**: (new — see §Symptoms)

---

## Problem

Three live bugs hit during the first real session with the v2 app:

1. **Clusters tab crashes**: `'RunStore' object has no attribute 'list_clusters'`. `app/face_clustering_v2/tabs/clusters_tab.py` calls `store.list_clusters()` and `store.list_assignments(...)` — neither exists on `RunStore`. The tab was never exercised end-to-end.
2. **Runs overwrite each other**: `run_tab.py` defaults the output dir to a fixed `~/.sim_bench/runs/v2_latest`. Every run clobbers the previous one. No per-run identity, no album metadata captured at the UI layer.
3. **No way to load a specific run**: the Clusters tab only knows about `st.session_state.v2_last_run_dir` (set after the most recent Run). If the user wants to revisit an older run, they have to find and paste the directory by hand — and there's no surface that tells them which dirs correspond to which run.

These are MVP-completeness gaps, not architecture problems. spec-040 shipped the v2 chain but left the UI loop incomplete.

---

## What we build

| Layer | Module | Role |
|---|---|---|
| Allocator | `face_cluster/run_layout.py` (**NEW**) | `allocate_run_dir(base_dir, album_slug) -> (run_dir, run_id)` — creates `<base>/<uuid4-hex>/`, returns the path and a `run_id` string. Pure function over the filesystem; no DB. |
| UI | `app/face_clustering_v2/tabs/run_tab.py` | Adds required **Album name** input. On Run click: allocates a fresh UUID run dir, sets `v2_last_run_dir` in session state **before** kicking off the pipeline (so navigation mid-run doesn't lose it). |
| Pipeline | `app/face_clustering_v2/pipeline.py` | Accepts `album: str` + `run_dir: Path` from the caller (instead of computing them). Uses the caller's `run_id` (the UUID) for action_log instead of a local timestamp. Album persists to `action_log.source_album`. |
| Component | `app/face_clustering_v2/components/run_picker.py` (**NEW**) | Reads the 20 most recent v2 runs via `RunHistoryRepository`, renders a `st.selectbox` with format `{started} — {album} — {n_faces}f/{n_clusters}c ({run_id_short})`. Returns the chosen `output_dir` Path. |
| UI | `app/face_clustering_v2/tabs/clusters_tab.py` | **Rewritten** against the real `RunStore` API (`clusters("latest")`, `faces()`, `crop_path()`). Run picker above the (still-available) free-text path override. |
| Test | `tests/face_clustering/test_v2_run_picker_e2e.py` (**NEW**) | Streamlit AppTest: Run a 3-jpg fixture → switch to Clusters → assert picker has 1 entry, expander renders without error, first cluster has ≥1 face card. |
| Test | `tests/face_cluster/test_run_layout.py` (**NEW**) | Unit: `allocate_run_dir` returns unique paths, creates the dir, run_id is a valid hex. |

---

## What we don't build

- Per-album sub-directories. Disk layout stays flat (`runs/<uuid>/`). The history table is the index.
- Feature parity with the legacy Clusters tab. Read-only view, expander per cluster, thumbnails. Anything richer (split signal, diameter metrics, force-merge) belongs to spec-045 (Cluster Analysis Tab).
- Run deletion / cleanup. Out of scope; track separately if disk fills.
- A new column on `action_log`. Existing columns cover everything (`run_id`, `output_dir`, `source_album`, `n_faces`, `n_clusters`, `started_at`).

---

## Locked decisions

1. **Run dir layout**: `~/.sim_bench/runs/<uuid4-hex>/`. Flat. UUID4, full 32-char hex, no separators.
2. **`run_id` = UUID hex**, not a timestamp. Solves the "two runs in the same second" edge case and makes the dir name == run_id (1-to-1 mapping).
3. **Album name required** on the Run tab. If empty, the Run button is disabled. No silent default — silent defaults are the source-of-truth bug that gave us "untitled run" sprawl in past projects.
4. **Session state set early**: `st.session_state.v2_last_run_dir = str(run_dir)` is written *before* the pipeline starts. Failed runs still leave a recoverable pointer (and a corresponding `failed` row in action_log).
5. **Picker is sourced from `action_log`, not from disk scan**. Single source of truth; rows with `producer='fc_app_v2'` AND `status IN ('complete', 'failed')`, ordered by `started_at DESC`, `LIMIT 20`. Disk-orphan dirs (no history row) are not discoverable via picker — the free-text override handles them.
6. **`run_id_short`** in UI labels = first 8 chars of UUID hex. Full UUID still in `output_dir` and `run_id` columns; the short form is for display only.
7. **No migration of existing `v2_latest` runs**. They keep working via the free-text path field; new runs use the new layout.

---

## Data contracts

### `face_cluster/run_layout.py`

```python
def allocate_run_dir(base_dir: Path, album_slug: str) -> tuple[Path, str]:
    """Allocate a fresh per-run output directory.

    Args:
        base_dir: e.g. ~/.sim_bench/runs. Created if missing.
        album_slug: free-form album name; recorded in action_log.
                    Not used in the path (kept for future album-grouping).

    Returns:
        (run_dir, run_id) where run_dir = base_dir / run_id and
        run_id = uuid4().hex (32 lowercase chars).
    """
```

### `app/face_clustering_v2/components/run_picker.py`

```python
@dataclass(frozen=True, slots=True)
class RunPickerEntry:
    run_id: str
    output_dir: Path
    album: str
    started_at: str
    n_faces: int
    n_clusters: int
    status: str  # 'complete' | 'failed'

def render_run_picker(
    *,
    label: str = "Pick a run",
    key: str = "v2_run_picker",
    limit: int = 20,
) -> Optional[RunPickerEntry]:
    """Render a selectbox of recent v2 runs. Returns the selected entry,
    or None if the dropdown is empty."""
```

### Run tab signature change (no public type)

`run_tab.py` no longer asks for "Output directory" as free text. It asks for **Album name** (required) and an optional "Base runs directory" (defaulted, read-only by default, expanded in an "Advanced" section).

---

## Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| AC1 | Each Run creates a fresh `runs/<uuid>/` dir; two consecutive runs do not collide on disk | Phase 0 unit test; Phase 5 AppTest |
| AC2 | Album name is required; Run button disabled when empty | Phase 1 manual + AppTest |
| AC3 | action_log row for the new run has `run_id == output_dir.name`, `source_album == album input`, `producer == 'fc_app_v2'` | Phase 2 integration test |
| AC4 | After a successful Run, the Clusters tab pre-selects the new run in the picker AND renders ≥1 cluster expander without error | Phase 5 AppTest |
| AC5 | Picking a different run from the dropdown switches the displayed clusters (no rerun-and-lose-state) | Phase 4 manual smoke (AppTest can't easily exercise this) |
| AC6 | If the user navigates to Clusters mid-run, the picker shows the in-flight run as `(running)` and the free-text field is pre-filled | Phase 1 manual smoke |
| AC7 | Free-text override still works for arbitrary paths (legacy runs, ad-hoc inspection) | Phase 4 manual smoke |
| AC8 | `RunStore.list_clusters` AttributeError is gone; clusters_tab uses only documented `RunStore` methods | Phase 4 code review + AppTest |
| AC9 | `pytest tests/face_clustering tests/architecture` — no new failures (the 13 pre-existing ones remain out-of-scope for this spec) | Phase 6 |

---

## Risks

| Risk | Mitigation |
|---|---|
| `RunHistoryRepository().find(...)` is slow on every Clusters-tab rerender | spec-048's `ensure_schema` is now ~1 ms; the `find(producer='fc_app_v2', limit=20)` query hits the `idx_action_log_type` index. Should be sub-10ms. Cache the result for the rerender via `@st.cache_data(ttl=5)` if profiling shows otherwise. |
| Streamlit AppTest can't exercise the full producer chain (slow, needs InsightFace) | Use a fake `run_v2_pipeline` injected via monkeypatch for the AppTest; cover the orchestration/UI plumbing, not the ML. The producer chain is already covered by `test_legacy_vs_v2_equivalence.py` (currently red — SIGHTING-070 — but a separate concern). |
| Picker shows `failed` runs which point at incomplete `face_clustering.db` — Clusters tab will then error on `RunStore.clusters("latest")` | Picker label includes `status`. Clusters tab catches the error and shows "This run failed; pick another" rather than tracebacking. |
| Two browser tabs each clicking Run at the same second | UUID4 collision probability is negligible. |
| Disk fills with orphaned UUID dirs over time | Out of scope. Track as a separate maintenance task. |

---

## Definition of Done

- All 9 acceptance criteria green.
- The 3 reported symptoms reproduced before the fix, gone after.
- One sighting filed (covering all 3 symptoms as a single MVP-completeness gap) and resolved by this spec.
- `CHANGES_LOG.md` entry.
- REVIEW.md filed via `/code-review`.
