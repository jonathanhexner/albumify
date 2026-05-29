# spec-059 — RunStore + ClusterAnalysisRepository on SQLAlchemy

**Created**: 2026-05-29
**Status**: Draft
**Predecessors**: spec-057 (RunExporter split), spec-058 (per-run ORM models)
**Successors**: future read-side tabs (Face Analysis, Merged Clusters, Quality, Gallery) inherit this pattern
**Trigger**: spec-045 left `ClusterAnalysisRepository` with 7 column-string literals across 4 raw SQL statements (SMELL-1 in `CODE_REVIEW_SUMMARY.html`). spec-058 provides the ORM models that close the gap.

---

## Problem

After spec-057 (writers split) and spec-058 (ORM models exist), three readers still issue raw SQL against the per-run DB:

| File | LOC | Raw SQL statements |
|---|---|---|
| `face_cluster/run_store.py` | 617 | ~18 (every `_connect()` block) |
| `face_cluster/repositories/cluster_analysis_repo.py` | 268 | 4 |
| `face_cluster/loader.py` (legacy) | ~ | ~10 |

Each statement hardcodes column names that the ORM models now declare typed. Migrating the readers to `select(Cluster.cluster_id, ...)` closes the drift surface and gives IDEs typo-detection for every column access.

This spec migrates the two non-legacy readers (RunStore + ClusterAnalysisRepository); `loader.py` is deprecated and out of scope.

## What we build

### 1. `RunStore` refactor

Every `conn.execute("SELECT ... FROM x")` block in `RunStore` becomes `session.execute(select(X).where(...))`. The public method signatures (`faces()`, `clusters(iteration)`, `metadata()`, `merge_log()`, `image_detail(path)`, etc.) stay byte-identical in return shape.

Side effect: `RunStore` shrinks. Estimate: 617 LOC → ~350-400 LOC (most of the trim is column-by-column row hydration → `Mapped` attribute access).

### 2. `ClusterAnalysisRepository` refactor

4 raw SQL statements → 4 `select(...)` blocks. Removes the 7 hardcoded column literals SMELL-1 flagged. Same public surface (`get_cluster_rows`, `find_assignments`, etc.).

### 3. Shared infrastructure

- `face_cluster/run_store/_session.py` — per-run-DB sessionmaker factory. Different from `face_cluster/repositories/_session.py` (which is for `~/.sim_bench/sim_bench.db`) — different DB, different lifecycle.
- Optional: `BaseRunReader` mixin if RunStore + Repository end up sharing 3+ helpers. Don't add it preemptively; only if duplication shows up.

## What we don't build

- **No public API change.** RunStore methods + ClusterAnalysisRepository methods keep their signatures and return shapes. spec-045 + spec-040 tests pass without modification.
- **No `loader.py` migration.** Legacy reader; spec-040 Phase 7 retires it.
- **No Service-layer changes.** `ClusterAnalysisService` is unaffected — it composes the Repository, doesn't care about its internals.
- **No new Repository methods.** The 7 existing read methods + 1 mutation stay.

## Locked decisions

1. **Per-call sessions.** Same lifecycle as today's per-call `sqlite3.connect()`. No long-lived session in RunStore — that'd break Streamlit's polling.
2. **Read-only by default for RunStore.** RunStore was never a writer; the SQLAlchemy version keeps that contract.
3. **Repository inherits `BaseRepository` with `session=None` for now.** The query-shape (B0b) pattern from spec-045 §B0.2.1 still applies — no Alembic, no schema ownership. A future spec may revisit whether B0b deserves its own base class.
4. **No semantic changes.** Equivalence test: run every existing test in `tests/face_clustering/{repositories,views,run_store}/`. They must all pass without modification.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | Zero raw SQL strings in `run_store.py` and `cluster_analysis_repo.py` (`grep -E 'SELECT |INSERT INTO|UPDATE '` → empty) | grep arch test |
| AC2 | All spec-045 tests pass without modification (15 Repository + 16 Service + 5 arch = 36) | run |
| AC3 | All existing RunStore tests pass without modification | run |
| AC4 | `RunStore` LOC < 450 (down from 617) | `wc -l` |
| AC5 | ClusterAnalysisRepository LOC ≤ 200 (down from 268) | `wc -l` |
| AC6 | spec-057's exporter equivalence test still passes (proves writes still produce the same bytes) | run |
| AC7 | Performance regression test: 1000 `get_cluster_rows()` calls finish within 1.2× the pre-refactor time | new micro-bench in `tests/face_clustering/repositories/test_cluster_analysis_repo_perf.py` |

## Risks

- **Performance.** SQLAlchemy adds overhead per call. AC7 bounds it; if it busts, fall back to `session.execute(text("..."))` for the hot path and document why.
- **Streamlit polling + Session state.** The Cluster Analysis tab polls `compute_*_async` handles. Each poll triggers a new Service call → new Repository call → new SQLAlchemy session. AC2 covers correctness; if latency creeps, cache the engine on `(run_dir, st.session_id)` (already cached at the Service level via spec-045's `_get_service`).
- **Row-hydration semantics.** RunStore today does explicit None handling on nullable columns (`face.pose = None if r["yaw"] is None else (...)`). The ORM version must preserve this.

## Effort estimate

**~5-6 hours.** RunStore refactor is the bulk (3 h). ClusterAnalysisRepository is ~1 h. Shared infra + tests + docs ~1-2 h.

---

## Final binding structure (post-059)

**Binding.** Mirrored in `specs/057/EXECUTIVE_REVIEW_057_059.html` §9 and in specs 057 / 058. Drift = Code Review §1 fail.

**Note:** spec-056 relocated `face_cluster/run_store.py` to `sim_bench/run_db/store.py` and `face_cluster/repositories/cluster_analysis_repo.py` to `sim_bench/db/face_clustering/cluster_analysis_repo.py` before this spec started. Spec-059 flips both to use the ORM models from spec-058 — all at the canonical location, no shims anywhere.

### Package layout (the parts this spec owns)
```
sim_bench/run_db/
├─ store.py                                   # MODIFIED — RunStore on ORM (≤450 LOC; was 546)
└─ _session.py                                # added in spec-058 — used by store.py

sim_bench/db/face_clustering/
└─ cluster_analysis_repo.py                   # MODIFIED — ClusterAnalysisRepository on ORM (≤200 LOC; was 305)
```

### Classes
| Class | Module |
|---|---|
| `RunStore` | `sim_bench/run_db/store.py` |
| `ClusterAnalysisRepository` | `sim_bench/db/face_clustering/cluster_analysis_repo.py` |

No new classes for the session factory — it is a plain function in `sim_bench/run_db/_session.py`.

### Methods on `RunStore` (binding — signatures and return shapes frozen)
```python
__init__(run_dir: Union[str, Path])
_load_and_validate() -> dict                   # PRAGMA user_version + file checks;
                                               # KEEPS raw sqlite3 — runs BEFORE ORM machinery
metadata() -> RunMetadata
faces() -> List[FaceRecord]
merge_log() -> List[MergeDecisionRow]
image_detail(image_path: str) -> ImageDetail  # heaviest; joins Face + FilterDecision +
                                               # ClusterAssignment
filter_decisions() -> List[FilterDecisionRow]
embeddings() -> EmbeddingMatrix                # npy file — not a DB call
crop_path(face_id: int) -> Path                # fs — not a DB call
clusters(iteration: Union[int, str]) -> ClusterResult
iteration_count() -> int
_resolve_iteration(iteration: Union[int, str]) -> int
_session() -> Session                          # REPLACES _connect(); per-call session
```

### Methods on `ClusterAnalysisRepository` (binding)
```python
__init__(config: ClusterAnalysisRepoConfig)
get_cluster_rows(iteration: str = "final") -> List[ClusterRow]
get_cluster_ids(iteration: str = "final") -> List[int]
find_assignments(criteria: ClusterAnalysisCriteria) -> List[Assignment]
get_face_records(face_ids: List[int]) -> List[FaceRecord]
get_run_metadata() -> RunMetadata
get_merge_log() -> List[MergeDecisionRow]
get_cluster_result(iteration: str = "final") -> ClusterResult
save_manual_merge_snapshot(...) -> None        # only mutation
_session() -> Session                          # REPLACES _connect()
_resolve_iteration(iteration: str) -> int
_log(msg: str) -> None
```

### Functions in `sim_bench/run_db/_session.py`
```python
make_run_db_sessionmaker(run_dir: Path) -> sessionmaker
```

### Cross-spec invariants
1. **Public surface frozen.** All 13 `RunStore` methods + all 9 `ClusterAnalysisRepository` methods keep signature and return shape. spec-045 + spec-040 + spec-049 test suites pass without modification.
2. **Two distinct sessionmakers in the codebase.** `face_cluster/repositories/_session.py` (sim_bench.db) vs `sim_bench/run_db/_session.py` (per-run DB). Never cross-import.
3. **`RunStore._load_and_validate()` stays on raw sqlite3.** `PRAGMA user_version` must run before any ORM machinery is invoked. ORM access only starts *after* validation passes.
4. **Per-call sessions only.** No long-lived session held on `RunStore` or `ClusterAnalysisRepository`. Matches today's per-call `sqlite3.connect()` lifecycle — required for Streamlit polling correctness.
5. **Zero raw SQL** in `sim_bench/run_db/store.py` and `sim_bench/db/face_clustering/cluster_analysis_repo.py`. Enforced by arch test (AC1).
6. **Single canonical import path** for both classes. No alternative paths exist (spec-056 deleted `face_cluster.run_store` and `face_cluster.repositories.cluster_analysis_repo`).
