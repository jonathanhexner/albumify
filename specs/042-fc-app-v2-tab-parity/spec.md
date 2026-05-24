# spec-042 — FC App v2 Tab Architecture & Parity

Status: **Draft**
Author: Jonathan Hexner
Created: 2026-05-25
Predecessors: [spec-040](../040-unified-pipeline-framework/spec.md), [spec-041](../041-fc-params-container/spec.md)

---

## 1. Objective

Bring the FC App v2 to feature-parity with the original FC App by **re-implementing** the missing tabs against the spec-041 contracts (FCParams, UI_SPEC, widget_factory, RunStore). The original tabs are treated as **UX specification, not source code**.

v2 today has 2 tabs (Run, Clusters). The original FC App has 11 tabs across 3160 LOC, with patterns that drift from the contracts spec-041 introduced — 35+ hand-rolled `cfg.get('field', '?')` literals duplicating FCParams field names, 11 files averaging ~290 LOC each, several mixing 5+ responsibilities, zero unit tests across the entire tab layer.

Goal in one sentence: **a user opening v2 should have the same workflow available as in the original FC App, against runs produced by the unified pipeline — and the code behind it should be testable, documented, and free of the per-field duplication that drifted in the legacy layer**.

Non-goals:
- Replacing the unified pipeline framework (already done by spec-040).
- Adding new functionality not present in the original FC App.
- Porting research-only tabs that have no production users (Merge ML, Labeling Review).
- Modifying `app/face_clustering/` (legacy). It stays in place during a burn-in period.

---

## 2. Why a rebuild, not a port

### 2.1 The legacy tab layer doesn't meet our current bar

| File | LOC | Issues |
|---|---|---|
| `label_tab.py` | 604 | over the 300 threshold |
| `cluster_analysis_tab.py` | 383 | over threshold |
| `history_tab.py` | 365 | over threshold; 16 `cfg.get()` field literals |
| `run_tab.py` | 323 | over threshold |
| `overview_tab.py` | 252 | 18 `cfg.get()` field literals |
| `recluster_tab.py` | 258 | mixes service + UI |
| ... | ... | ... |
| **Total** | **3160 LOC** | **35 `cfg.get('field', '?')` literals** across 3 files, plus mixed Streamlit + DB + business logic throughout |

Symptoms specific to `history_tab.py` (the canonical example):

```python
# Lines 70-94 — 16 lines that duplicate FCParams field names as plain strings
st.text(f"K:                    {cfg.get('K', '?')}")
st.text(f"distance_threshold:   {cfg.get('distance_threshold', '?')}")
st.text(f"blur_min:             {cfg.get('blur_min', '?')}")
# ...
```

```python
# Variable names like fc1, fc2, fc3 — letter-and-number placeholders
fc1, fc2, fc3 = st.columns([2, 2, 3])
with fc1:
    album_filter = st.selectbox(...)
```

```python
# ASCII section comments substituting for function decomposition
# ---- Filter bar --------------------------------------------------------
# ... 200 more lines ...
# ---- Main table --------------------------------------------------------
```

A "copy + rewire" port would propagate all three patterns into v2, immediately undoing spec-041's single-source-of-truth work.

### 2.2 What's salvageable

The read backend (`face_cluster.run_store.RunStore`, `face_cluster.run_history_db`) is sound — typed, tested, used by both apps today. spec-042 keeps it and builds new view-service classes on top.

The legacy tabs' **UX shape** (which expanders, which columns, which order) is the spec for v2 layout — copy the look, rewrite the substance.

---

## 3. Scope

### In scope (this spec)

8 tabs, rebuilt against the new contracts:

| Tab | Priority | Status notes |
|---|---|---|
| **Cluster Analysis** | P1 | Replaces the v2 "Clusters" tab |
| **Recluster** | P1 | Needs new `FCAppRunner.recluster()` runner |
| **History** | P1 | Pilot tab — establishes the full pattern |
| **Face Analysis** | P2 | Per-face popup |
| **Merged Clusters** | P2 | Merge-decisions table viewer |
| **Quality** | P2 | Per-gate verdict visualizer |
| **Gallery** | P3 | Cluster-by-cluster browse |
| **Overview** | P3 | Aggregate metrics dashboard |

Plus the infrastructure pieces:

- New backend layer `face_cluster/views/` (one service per view)
- Reusable Streamlit components `app/face_clustering_v2/components/`
- `widget_factory.render_field(name, readonly=True)` mode
- Architecture tests enforcing the layering
- Documentation contract (binding for all new v2 code)

### Out of scope (deferred)

| Item | Reason |
|---|---|
| Merge ML tab | Research-only |
| Labeling Review tab | Research-only |
| Retiring `app/face_clustering/` | Burn-in clock starts when spec-042 lands; retirement is a follow-up |
| Migrating `app/shared/merge_controls.py` to FCParams | Standalone mini-spec |

---

## 4. Design

### 4.1 Two-layer split

Hard separation between Streamlit and backend:

```
face_cluster/                              (backend — NO streamlit imports)
  views/
    __init__.py
    _specs.py                              # ColumnSpec, ActionTypeFormat, shared declarative types
    history.py                             # HistoryQuery, RunRow, RunDetail, HistoryService
    cluster_analysis.py                    # ClusterRow, ClusterDetail, ClusterAnalysisService
    recluster.py                           # ReclusterRequest, ReclusterResult, ReclusterService
    face_analysis.py                       # FaceDetail, FaceAnalysisService
    merged_clusters.py                     # MergeDecisionRow, MergedClustersService
    quality.py                             # QualityRow, QualityService
    gallery.py                             # GalleryService (reuses ClusterRow/FaceRow)
    overview.py                            # OverviewSummary, OverviewService

app/face_clustering_v2/                    (Streamlit only)
  tabs/
    history_tab.py                         # gather → call HistoryService → render
    cluster_analysis_tab.py
    recluster_tab.py
    ...
  components/
    run_filter_bar.py
    run_table.py
    run_detail.py
    cluster_card.py
    face_grid.py
    load_button.py
```

**Rules** (enforced by `tests/architecture/test_v2_layering.py`):

- `face_cluster/views/*` MUST NOT import `streamlit`. Anywhere.
- `app/face_clustering_v2/*` MAY import from `face_cluster/views/*` but MUST go through service classes — never reach into `_specs.py` or query the DB directly.
- Backend services own state (DB connections, caches); rendering functions are stateless given typed input.

### 4.2 Backend service contract

Every `*Service` class is the API for one view. Contract:

```python
class HistoryService:
    """Owns: connection to ~/.sim_bench/sim_bench.db.

    Provides: read-only queries (list_*, get_*) and named mutations
    (update_*, load_*). All methods take typed inputs and return typed
    outputs — no dicts.
    """

    @classmethod
    def list_runs(cls, query: HistoryQuery) -> list[RunRow]:
        """Return runs matching the filter, newest-first. Read-only."""

    @classmethod
    def list_albums(cls) -> list[str]:
        """Distinct album names ever recorded. Read-only."""

    @classmethod
    def get_run_detail(cls, run_id: int) -> RunDetail:
        """Full detail for one run: row + parent + pipeline_run.json. Read-only."""

    @classmethod
    def update_comment(cls, run_id: int, comment: str) -> None:
        """Mutates: action_log.comment for run_id. Idempotent."""

    @classmethod
    def load_run(cls, run_id: int) -> LoadedRun:
        """Loads cluster data into a typed container. Pure — does NOT touch session_state."""
```

Patterns:

- **Pure read methods** named `list_*` / `get_*` — no side effects, idempotent.
- **Mutation methods** named with action verbs (`update_*`, `delete_*`, `load_*`) — side effects called out in docstring.
- **No `dict` returns** — every API surface is a typed `@dataclass` or Pydantic model.
- **No Streamlit awareness** — services don't touch `st.session_state`. If a mutation needs to populate session state, the tab layer does that after calling the service.

### 4.3 Streamlit tab contract

Every `*_tab.py` is structurally identical:

```python
def render_<tab>_tab() -> None:
    """Render the <Tab> tab.

    Layout: <region 1> → <region 2> → <region 3>.
    Reads:  <ServiceName> (queries the DB).
    Writes: st.session_state[...] when the user clicks <action>.
    """
    st.header("<Tab Name>")

    # 1. Gather inputs from widgets.
    inputs = render_<input_section>()  # → typed input

    # 2. Call the service.
    data = <ServiceName>.<method>(inputs)  # → typed output

    # 3. Render the output.
    render_<output_section>(data)
```

Constraint: **tab files contain only Streamlit calls + service calls.** No SQL, no JSON parsing, no field-name string literals, no formatting logic beyond what the rendering components provide.

### 4.4 Declarative specs

Wherever the legacy code had "a list of field names with metadata next to each name," replace with a declarative spec list + one renderer.

```python
# face_cluster/views/_specs.py

@dataclass(frozen=True)
class ColumnSpec:
    """One column in a tabular display.

    `field`: attribute name on the row object.
    `label`: human-readable column header.
    `fallback_fields`: try these in order if `field` is missing/falsy.
    `formatter`: optional callable; default is str().
    """
    field: str
    label: str
    fallback_fields: tuple[str, ...] = ()
    formatter: Optional[Callable[[Any], str]] = None
```

Used by every tabular tab:

```python
# history.py
RUN_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(field="display_album", label="Album"),
    ColumnSpec(field="run_name", label="Run name", fallback_fields=("run_id", "id")),
    ColumnSpec(field="run_kind", label="Kind", fallback_fields=("action_type",)),
    ColumnSpec(field="output_dir", label="Output", formatter=_short_path),
    ColumnSpec(field="started_at", label="Created", formatter=_iso_to_local),
    ColumnSpec(field="n_faces", label="Faces"),
    ColumnSpec(field="n_clusters", label="Clusters"),
    ColumnSpec(field="status", label="Status"),
    ColumnSpec(field="comment", label="Comment"),
]
```

One `row_to_display(row, columns)` helper iterates over the spec — same pattern as `widget_factory.render_field` reading from `UI_SPEC`.

Similar specs:

- `ActionTypeFormat` — typed dispatch for the dict-of-lambdas at legacy line 349.
- (existing) `UI_SPEC` for config display, read-only mode.

### 4.5 `widget_factory.render_field(name, readonly=True)` — new mode

Today's `widget_factory.render_field(name)` writes a Streamlit input widget. We add a `readonly=True` mode that renders the same field as a non-editable display element (label + value).

Used by History detail view to show "what config did this run use" without duplicating the field list:

```python
# render_run_detail.py
def render_run_config(run_config: dict[str, Any]) -> None:
    """Show a run's saved FCParams config as read-only fields, grouped by UI_SPEC group."""
    for group in ("cluster", "quality", "merge"):
        with st.expander(f"Stage · {group.title()}", expanded=False):
            for field_name in UI_SPEC.fields_in_group(group):
                if field_name in run_config:
                    widget_factory.render_field(field_name, readonly=True, value_override=run_config[field_name])
```

Replaces 30+ lines of `st.text(f"K: {cfg.get('K', '?')}")` literals with 4 lines of iteration. **The field list is FCParams; no duplication.**

### 4.6 Documentation contract

All new v2 code follows these rules:

| Element | Required documentation |
|---|---|
| **Class** | 1–2 line docstring stating *what state it owns and what it's responsible for*. |
| **Public method** | Docstring with: one-line summary, `Args:`/`Returns:` if non-trivial, `Side effects:` if it touches I/O / session_state / DB. |
| **Private helper** (`_foo`) | One-line docstring if the name doesn't make the contract obvious. |
| **`render_*` function** | Docstring stating: what region it produces, what it reads, what it writes back. |
| **Inline comment** | Only when WHY is non-obvious. Never narrate the WHAT. |
| **Variable names** | Self-documenting. `fc1`, `r`, `df` forbidden; `col_album`, `run_row`, `df_actions` are required. |
| **Section ASCII comments** (`# ---- xxx ----`) | Forbidden in new code. They mean the function should have been split. |
| **Functions over 30 lines** | Must be split. If you can't extract a function with a clean name, the structure isn't right yet. |

Enforced by `tests/architecture/test_v2_module_docstrings.py` — asserts every public function/class in `face_cluster/views/` and `app/face_clustering_v2/` has a non-empty docstring. Catches "forgot to document" at PR time without policing prose.

### 4.7 Testing strategy — three layers per tab, plus synthetic-data service tests

Every tab gets three test layers. **Service-level unit tests are the load-bearing layer** — they prove the contract works on known inputs without needing Streamlit or real data.

| Layer | Lives in | Driven by | Speed |
|---|---|---|---|
| **Service unit (synthetic)** | `tests/face_clustering/views/test_<view>_service_synthetic.py` | pytest, in-memory SQLite seeded with known fixture rows | ms |
| **Service integration (real)** | `tests/face_clustering/views/test_<view>_service_real.py` | pytest, `v2_budapest_run_dir` session fixture | sec |
| **UI smoke** | `tests/manual/_v2_<tab>_smoke.py` | Playwright headless against live Streamlit on port 8889 | sec, opt-in |

#### Service unit (synthetic) — the critical layer

For each `*Service`, a unit test that:

1. Creates an empty in-memory SQLite DB (or a real one in `tmp_path`).
2. Seeds it with known rows constructed in Python.
3. Calls the service method.
4. Asserts on the typed return — exact shape, exact values.

Example for History:

```python
def test_history_service_list_runs_filters_by_album(tmp_path):
    db = _seed_history_db(tmp_path, runs=[
        _make_action(id=1, album="Budapest", status="complete"),
        _make_action(id=2, album="Paris", status="complete"),
        _make_action(id=3, album="Budapest", status="failed"),
    ])
    service = HistoryService(db_path=db)
    rows = service.list_runs(HistoryQuery(album="Budapest"))
    assert [r.id for r in rows] == [3, 1]  # newest-first; 2 (Paris) excluded
```

Every service method gets at least one such test. **Synthetic data is preferred** because:

- Tests are fast and deterministic — no dependence on the dev machine having a real Budapest run.
- Edge cases (empty result, ambiguous fallback, malformed payload) are easy to construct.
- Refactors to the SQL or model are caught immediately.

The real-fixture integration test exists as a *smoke* — confirms the service works against actually-shaped data when the Budapest dir is present. It's not the primary correctness signal.

#### Per-tab test matrix

| Tab | Service unit (synthetic) | Service integration (real Budapest) | UI smoke | Extra |
|---|---|---|---|---|
| **History** | `list_runs` filter combinations, `get_run_detail` shape, `update_comment` idempotence, `load_run` returns typed `LoadedRun` (≥6 cases) | List ≥1 row with `producer='fc_app_v2'` | Open tab; assert ≥1 row + no exception | — |
| **Cluster Analysis** | `list_clusters` on synthetic DB with 3 clusters; `get_cluster_detail` returns face thumbnails sorted by exemplar score | Returns 33 clusters; sizes sum ≤ 340 | ≥30 expanders visible | — |
| **Recluster** | `recluster()` on synthetic prior-run dir; `parent_run_id` set | Recluster Budapest with K=3 → success + different cluster count | UI completes a recluster | **Equivalence**: legacy vs v2 ≥95% agreement on same prior run |
| **Face Analysis** | `get_face_detail(face_id)` returns bbox/landmarks/scores on synthetic DB | Returns FaceDetail with non-empty fields | Bbox + landmarks visible | — |
| **Merged Clusters** | `list_merge_decisions()` on synthetic merge_decisions table | Returns rows for actually-merged pairs | Table renders | — |
| **Quality** | `list_face_scores()` on synthetic DB | One row per face | Verdict grid renders | — |
| **Gallery** | (reuses cluster_card/face_grid units) | — | ≥1 cluster + thumbnails | — |
| **Overview** | Aggregator on synthetic DB | Matches 33/340 baseline | Headline metrics visible | — |

### 4.8 Test fixtures

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def v2_budapest_run_dir() -> Path:
    """Most recent v2 Budapest run dir; skips if absent (CI without album)."""

@pytest.fixture
def synthetic_action_log_db(tmp_path) -> Path:
    """Empty action_log SQLite at tmp_path/sim_bench.db with the v5 schema applied.
    Tests insert their own rows via helpers in tests/face_clustering/views/_seed.py."""

@pytest.fixture
def synthetic_run_dir(tmp_path) -> Path:
    """Empty <tmp>/face_clustering.db with v5 schema + empty embeddings.npy.
    Tests populate face / cluster / merge rows via _seed.py helpers."""
```

`tests/face_clustering/views/_seed.py` provides shared row factories (`_make_action`, `_make_cluster`, `_make_face`, `_make_merge_decision`) so service tests don't repeat boilerplate.

### 4.9 `main.py` layout

```python
st.tabs([
    "Run",              # already shipped (spec-040)
    "Cluster Analysis", # rebuild; replaces v2's existing "Clusters" tab
    "Recluster",        # NEW
    "History",          # NEW (pilot)
    "Face Analysis",    # NEW
    "Merged Clusters",  # NEW
    "Quality",          # NEW
    "Gallery",          # NEW
    "Overview",         # NEW
])
```

Order matches the legacy FC App for muscle-memory continuity.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Service classes leak Streamlit imports** | Arch test `test_v2_layering.py` — fails on any `import streamlit` inside `face_cluster/views/`. |
| **Tab files re-implement service logic inline** | Arch test — fails on direct DB / JSON access inside `app/face_clustering_v2/tabs/`. |
| **The "synthetic data" tests drift from real-DB schema** | Both layers run together in CI. If the schema changes, both fail; that's the signal. |
| **Recluster equivalence drift between legacy and v2** | Equivalence test mirroring spec-040 A2 pattern. |
| **Docstring contract turns into prose-policing** | The arch test only checks *presence*, not quality. Reviewer comments handle prose. |
| **Effort estimate slippage** | Phases are sized so each ends with a shippable commit. Pilot (History) absorbs the risk; subsequent tabs are templated. |

---

## 6. Phases

### Phase 0 — Prerequisites (~2 hours)

- Fix `face_cluster/analysis_views.py::_parse_merge_log` broken import (pre-existing).
- Add session fixtures (`v2_budapest_run_dir`, `synthetic_action_log_db`, `synthetic_run_dir`) + `_seed.py` row factories.
- Add `face_cluster/views/_specs.py` with `ColumnSpec` + `ActionTypeFormat`.
- Add `widget_factory.render_field(name, readonly=True, value_override=...)` mode.
- Add arch tests: `test_v2_layering.py`, `test_v2_module_docstrings.py`.

### Phase 1 — Pilot: History tab top-to-bottom (~6 hours)

This phase establishes the full pattern. Every other tab follows from here.

1. `face_cluster/views/history.py` — `HistoryQuery`, `RunRow`, `RunDetail`, `LoadedRun` dataclasses + `HistoryService` (5 methods).
2. Service unit tests against synthetic data (≥6 cases covering filter combos, fallback fields, mutation idempotence).
3. Service integration test against Budapest fixture.
4. `app/face_clustering_v2/components/run_filter_bar.py`, `run_table.py`, `run_detail.py`, `load_button.py`.
5. `app/face_clustering_v2/tabs/history_tab.py` — orchestration only.
6. UI smoke (`_v2_history_smoke.py`).
7. Wire into `main.py`.

**Checkpoint**: History tab works against real Budapest data; pattern documented in spec.md as the reference; arch tests green.

### Phase 2 — Apply pattern to P1 tabs (~6 hours)

8. Cluster Analysis (~3h)
9. Recluster (~3h + `FCAppRunner.recluster()` runner mode + equivalence test)

### Phase 3 — P2 tabs (~5 hours)

10. Face Analysis (~1.5h)
11. Merged Clusters (~1.5h)
12. Quality (~2h)

### Phase 4 — P3 tabs (~3 hours)

13. Gallery (~1.5h)
14. Overview (~1.5h)

### Phase 5 — Close-out (~2 hours)

- `/code-review` → REVIEW.md
- Update `docs/architecture/classes.html` + `data_flow.html`
- CHANGES_LOG per landed commit
- Spec status → Implemented
- File a follow-up sighting / spec for legacy-app retirement (2-week burn-in clock starts).

**Total: ~24 hours** spread across 5 phases. Pilot absorbs the design-uncertainty; the remaining 7 tabs are template applications.

---

## 7. Definition of Done

- [ ] `face_cluster/views/` exists with 8 service modules, each with synthetic-data unit tests AND real-fixture integration tests.
- [ ] `app/face_clustering_v2/tabs/` has 9 tabs (Run + 8 new); each tab file is < 100 LOC of orchestration only.
- [ ] No `cfg.get('field', '?')` patterns in v2 code (arch test enforces).
- [ ] No `streamlit` imports inside `face_cluster/views/` (arch test enforces).
- [ ] Every public class/function in new v2 code has a docstring (arch test enforces).
- [ ] `widget_factory.render_field(readonly=True)` mode implemented and used in History detail view.
- [ ] Recluster equivalence test passes (≥95% agreement with legacy).
- [ ] Architecture HTMLs updated.
- [ ] REVIEW.md produced, high-severity findings closed.
- [ ] Spec status → Implemented.

---

## 8. Open questions

1. **Should the backend services live in `face_cluster/views/` or `sim_bench/views/`?** Recommended: `face_cluster/views/` because they're tied to face-clustering schema. If we ever expose scene-clustering views, those go in `sim_bench/views/` or similar.
2. **Naming: `RunRow` vs `RunSummary` vs `HistoryEntry`?** Recommended: `RunRow` — matches the existing `face_cluster.run_history.RunRow` that legacy code already uses. Keep continuity.
3. **`HistoryService.load_run(id) -> LoadedRun` — should it touch `st.session_state`?** Recommended: NO. The service returns a typed `LoadedRun`; the tab layer writes session_state. Keeps the service Streamlit-free.
4. **Should the legacy app eventually consume the same services?** Recommended: optionally, in a follow-up spec. The strangler-fig story works either way. If the legacy app is retired (per the burn-in plan), the question goes away.
