# Code Review — spec-058 (per-run ORM models + drift-guard)

**Date**: 2026-05-30
**Reviewer**: Claude (automated, via `/code-review`)
**Branch**: `unification/spec-040`
**Commits**: `a8b08b7` (implementation), plus diagnostic-only `a54711c`, `459d956` (unrelated SIGHTING-081).
**Checklist**: `docs/guides/CODE_REVIEW_CHECKLIST.md`

---

## Part 1 — How it works

### Module inventory

| File | LOC | Purpose |
|---|---:|---|
| `sim_bench/run_db/models/_base.py` | 26 | `class Base(DeclarativeBase)` + naming convention. Separate from `face_cluster/repositories/_orm_base.py` (different lifecycle, different constraint-name namespace). |
| `sim_bench/run_db/models/face.py` | 43 | `Face` model — 26 mapped columns, mirrors `FACES_DDL`. |
| `sim_bench/run_db/models/face_scores.py` | 22 | `FaceScores` (not in spec's 9-table list — added because `FACE_SCORES_DDL` exists). |
| `sim_bench/run_db/models/cluster.py` | 21 | `Cluster` — composite PK `(cluster_id, iteration)`. |
| `sim_bench/run_db/models/cluster_assignment.py` | 28 | `ClusterAssignment` — FK to `faces`, 2 indexes. |
| `sim_bench/run_db/models/merge_decision.py` | 50 | `MergeDecision` — 28-column row matching spec-030 FR-004. |
| `sim_bench/run_db/models/filter_decision.py` | 26 | `FilterDecision` — spec-032 typed filter log. |
| `sim_bench/run_db/models/image.py` | 34 | `Image` — spec-040 Phase 4 (schema v5). |
| `sim_bench/run_db/models/scene_cluster.py` | 23 | `SceneCluster` — composite PK. |
| `sim_bench/run_db/models/scene_cluster_assignment.py` | 24 | `SceneClusterAssignment` — FK to `images`. |
| `sim_bench/run_db/models/run_metadata.py` | 35 | `RunMetadataRow` — class suffix avoids clash with `RunMetadata` dataclass. |
| `sim_bench/run_db/_schema.py` | 125 | DDL constants now *derived* from `Base.metadata`. `SCHEMA_HISTORY`, `SCHEMA_VERSION`, `EXPECTED_ARTIFACTS` remain hand-maintained. |
| `tests/face_clustering/db/test_orm_matches_schema.py` | 124 | 3-assertion drift-guard. |

Total new code: ~480 LOC (10 model files, average 30 LOC each). Largest file is 50 LOC. All under the 80-LOC step file convention; the 300-LOC module cap doesn't bite.

### Dependency map

```
sim_bench/run_db/models/_base.py
        ▲
        │  imports Base
        │
sim_bench/run_db/models/{face,cluster,...}.py  ─── 10 files, each one mapped class
        ▲
        │  imports Base + every model (via models/__init__)
        │
sim_bench/run_db/_schema.py  ─── derives *_DDL strings via CreateTable/CreateIndex compilers
        ▲
        │  imports SCHEMA_DDL, SCHEMA_VERSION, EXPECTED_ARTIFACTS
        │
sim_bench/run_db/{store,exporter,writers/*,artifact_writers/*}.py
```

Linear, no cycles, no reverse imports. Verified by grep.

### Data flow

1. **Schema bootstrap path (RunExporter)**: at run start, `RunExporter.export()` opens a fresh sqlite3 connection, calls `conn.executescript(SCHEMA_DDL)`. `SCHEMA_DDL` is now built at module-import time from `Base.metadata.sorted_tables` + every table's indexes, compiled via the SQLite dialect. The DB created is byte-identical (at the `sqlite_master` and `PRAGMA table_info` levels) to what the old hand-DDL produced.

2. **Drift-guard test**: opens two tmp DBs, one via `executescript(SCHEMA_DDL)`, one via `Base.metadata.create_all(engine)`. Compares the resulting `sqlite_master` rows and `PRAGMA table_info` columns. Trivially passes after Phase 2 (both sides derive from the same metadata) — the test is now a regression guard against breakage in the derivation chain.

3. **Writers / readers**: unchanged. spec-058 does NOT migrate any caller to ORM-based access — that's spec-059's job. Writers still use raw `INSERT INTO ... VALUES (?, ?, ...)`; `RunStore` still uses `cursor.execute(...)`.

---

## Part 2 — Findings

### §1 Structure

| # | Criterion | Verdict |
|---|---|---|
| 1.1 | No module >300 LOC | **pass** — largest new module is 50 LOC; `_schema.py` shrank from ~290 to 125 |
| 1.2 | No module mixes responsibilities | **pass** — each model = one table; `_schema.py` = derivation only |
| 1.3 | No file holds multiple unrelated classes | **pass** — one `Mapped` class per file |
| 1.4 | Docstring summarizes module in one sentence | **pass** — verified file-by-file |
| 1.5 | No function with >4 parameters | **pass** — `_table_ddl(name)` and `_compile(stmt)` are 1-arg helpers |
| 1.6 | No dead code | **pass** — every model is re-exported via `models/__init__.py`; every helper is invoked |
| 1.7 | Dependency direction respected | **pass** — models → Base → SQLAlchemy; `_schema.py` → models. No reverse imports (grep-verified) |

### §2 Code quality

All criteria **pass**. No try/except in new code; no nested conditionals; explicit `server_default=text("0")` / `text("1")` for the 3 columns with DEFAULTs; no operator-precedence traps; comments explain *why* (the post-058 source-of-truth flip, the naming convention rationale).

### §3 Naming and package structure

| # | Criterion | Verdict |
|---|---|---|
| 3.1 | Module names match contents | **pass** — `face.py` → `Face`, `merge_decision.py` → `MergeDecision`, etc. |
| 3.2 | Naming consistency in sibling group | **pass** — all snake_case files, PascalCase classes, `_base.py` underscore-prefixed because internal |
| 3.3 | Subpackage when ≥3 files share concern | **pass** — 12-file `models/` subpackage with `__init__.py` aggregator |
| 3.4 | `__all__` declared for new public surface | **pass** — `models/__init__.py` declares `__all__`; per-model files expose a single class each (sufficient for declarative ORM modules) |

### §4 Layering and coupling

| # | Criterion | Verdict |
|---|---|---|
| 4.1 | No reverse imports | **pass** |
| 4.2 | No duplicated logic | **pass** — `Base` pattern is intentionally separate from `face_cluster/repositories/_orm_base.py` (locked decision 4: different lifecycles) |
| 4.3 | Single writer per piece of state | **pass** — before 058 the schema had two writers (hand DDL + ORM-if-it-existed). 058 flips ORM to single source; hand DDL strings can no longer be authored (they're derived) |
| 4.4 | Two-way writers documented | **pass** — spec-058 §"Locked decisions" #1 |

### §5 Testability

**Test inventory:**
- **Static / structural**: 3 tests in `test_orm_matches_schema.py` (tables, indexes, columns).
- **Unit**: 0 (declarative ORM models have no behavior to unit-test).
- **Synthetic**: 0.
- **E2E (regression-covered)**: spec-057's `test_exporter_output_matches_golden_hashes` exercises the full RunExporter, which bootstraps from the now-derived DDL. Still green → end-to-end confirmation that schema-as-derived produces byte-identical row hashes.

| # | Criterion | Verdict |
|---|---|---|
| 5.1 | Test inventory recorded | **pass** — see above |
| 5.2 | At least one real E2E | **pass** — spec-057 golden-hash test; spec-053 e2e pipeline (`test_pipeline_100images`) both pass with derived DDL |
| 5.3 | Each test one responsibility | **pass** — 3 drift-guard tests split into one assertion class each |
| 5.4 | Mock usage justified | **pass** — no mocks |
| 5.5 | Failure-mode walk-through | **pass-with-followup (1)** — see below |
| 5.6 | Test placement | **pass** — `tests/face_clustering/db/` for the drift-guard (structural, but not architecture-test-shaped — it executes SQL) |

**Failure-mode walk-through:**

| Bug class | Test that would catch a recurrence |
|---|---|
| Column added to ORM, missing in DDL | `test_columns_match_per_table` |
| Column type wrong (REAL vs INTEGER) | `test_columns_match_per_table` |
| Index defined on one side only | `test_index_set_matches` |
| Table renamed in one place only | `test_table_set_matches` |
| Phase-2 derivation regression (e.g., `CreateTable` API change in future SA version) | spec-057 golden-hash + 797 existing tests cover end-to-end behavior |
| **FK semantics lost via ORM** (e.g., a `ForeignKey` dropped without notice) | **not directly tested** → finding F-1 |
| `server_default` removed from ORM column | `_columns(...)` reads `dflt_value` from `PRAGMA table_info` → caught |

**Finding F-1 (pass-with-followup, low severity)**: the drift-guard reads `PRAGMA table_info` which does *not* include FK relationships. A future model edit that drops `ForeignKey("faces.face_id")` from `FaceScores.face_id` would not fail any test in spec-058's scope. A 4th assertion using `PRAGMA foreign_key_list(table)` would close this gap — see ticket below.

### §6 Boundary contracts

| # | Criterion | Verdict |
|---|---|---|
| 6.1 | Each new boundary has enforcement | **pass** — drift-guard test enforces the ORM↔SQL boundary |
| 6.2 | `extra="forbid"` on new Pydantic | **n/a** — no Pydantic in this spec |
| 6.3 | Pandera `nullable=False` on required | **n/a** — no new Pandera schemas; existing ones in `face_cluster/db/validators.py` unchanged |
| 6.4 | Contract claimed but not invoked | **pass** — drift-guard runs in default pytest collection |
| 6.5 | Config knob → producer check | **n/a** — no config knobs added |

### §7 Documentation deliverables

| # | Artifact | Status |
|---|---|---|
| 7.1 | `spec.md` | **pass** — updated to In Progress |
| 7.2 | `tasks.md` | **pass** — exists; per-task progress tracked in TaskList |
| 7.3 | `REVIEW.md` | **pass** — this document |
| 7.4 | `EXECUTIVE_SUMMARY.html` | **n/a** — optional |
| 7.5 | `docs/architecture/db_schemas.html` | **pass** — "Implementation layer" callout rewritten to reflect ORM source-of-truth; SCHEMA_VERSION 4→5 corrected in two places |
| 7.6 | `docs/architecture/classes.html` | **pass-with-followup (2)** — has a "Domain dataclasses & repositories" section but no entries for the 10 new ORM models. Spec-058 tasks.md T040 calls for an "ORM models (per-run DB)" subsection. See ticket below |
| 7.7 | `docs/architecture/data_flow.html` | **n/a** — no pipeline-step / bridge / exporter / read-side flow changed |
| 7.8 | `docs/architecture/index.html` | **n/a** — no new architecture doc added |
| 7.9 | `CHANGES_LOG.md` | **pass** — prepended entry for spec-058 |
| 7.10 | `LEARNINGS.md` | **n/a** — no new failure class surfaced |
| 7.11 | MEMORY | **n/a** — nothing future-session-applicable |

### §8 Risk register

| # | Item | Status |
|---|---|---|
| 8.1 | Known deferred work has tickets | **pass** — F-1 (FK drift) + F-2 (classes.html) filed below |
| 8.2 | Workarounds named and tested | **pass** — the four `NOT NULL`-on-single-PK changes in `_schema.py` are documented in the commit message + CHANGES_LOG as a no-op alignment toward ORM canonical form |
| 8.3 | Backwards-compat surface | **pass** — existing per-run DBs (created before this commit) remain readable. SQLite doesn't enforce schema-text identity; only `PRAGMA user_version` is checked. No migration needed |
| 8.4 | Hot-path performance | **pass** — DDL derivation runs once at module import (~milliseconds). Hot reads/writes still use raw sqlite3 |

---

## Part 3 — Verdict

**Accept with two minor follow-ups.** No high-severity findings; no §1-§7 fail criteria; spec-058 acceptance criteria AC1–AC6 all satisfied (10 models exist, drift-guard passes, DDL constants still importable, SCHEMA_VERSION=5, no new dependencies). 797 tests pass with zero regressions.

### Follow-up tickets

- **F-1** (TODO): add `test_foreign_keys_match_per_table` to `tests/face_clustering/db/test_orm_matches_schema.py` using `PRAGMA foreign_key_list`. Closes the §5 walk-through gap. Estimated 20 min.
- **F-2** (TODO): add an "ORM models (per-run DB)" subsection to `docs/architecture/classes.html` listing the 10 new model classes + the second `Base` (with a note that it's separate from `face_cluster/repositories/_orm_base.py`). Estimated 30 min.

Neither blocks handoff. Both are housekeeping; recorded in `TODO.md` rather than spec dirs or sightings.
