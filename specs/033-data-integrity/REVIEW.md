# spec-033 — Code Review

**Audience**: future architects (Part 1 = onboarding) and the current architect deciding whether to accept this change (Part 2 = findings).
**Scope**: every module added or changed by spec-033 phases P-A → P-H plus the SIGHTING-061 fix.
**Date**: 2026-05-15.

---

## Part 1 · How this works

### 1.1 What spec-033 is

A boundary-contracts retrofit on the face-clustering path. Two pipelines share one clustering engine: **FC App** (standalone) and **Albumify** (production photo selector). They diverged in quality because the adapter between them (`face_cluster_bridge.py`) silently dropped fields and force-disabled gates. The change set replaces silent failure modes with explicit contracts using Pydantic (object boundaries) and Pandera (DataFrame boundaries).

### 1.2 Module inventory

**Added (12 files, ~700 LOC)**

| Module | LOC | Purpose | Layer |
|---|---:|---|---|
| `face_cluster/types.py` (replaced) | 218 | `FaceRecord` migrated dataclass → Pydantic BaseModel with `extra="forbid"` | face_cluster |
| `face_cluster/db_schemas.py` | 99 | Pandera DataFrameSchemas for faces / face_scores / filter_decisions | face_cluster |
| `face_cluster/image_detail.py` | 74 | Pydantic `ImageDetail` / `FaceDetail` / `FaceFilterDecision` (P-D return shape) | face_cluster |
| `face_cluster/config_diff.py` (extended) | 126 | `effective_config_from_*` + diff CLI; existing `compute()` kept | face_cluster |
| `sim_bench/pipeline/steps/configs/__init__.py` | 36 | `STEP_CONFIG_MODELS` registry | sim_bench |
| `sim_bench/pipeline/steps/configs/_validate.py` | 28 | Shared `validate_step_config(name, dict)` helper | sim_bench |
| `sim_bench/pipeline/steps/configs/filter_quality.py` | 31 | `FilterQualityConfig` BaseModel | sim_bench |
| `sim_bench/pipeline/steps/configs/filter_faces.py` | 47 | `FilterFacesConfig` BaseModel | sim_bench |
| `sim_bench/pipeline/steps/configs/cluster_people.py` | 219 | `ClusterPeopleConfig` BaseModel (~40 fields) | sim_bench |
| `sim_bench/pipeline/steps/configs/extract_face_embeddings.py` | 32 | `ExtractFaceEmbeddingsConfig` | sim_bench |
| `sim_bench/pipeline/steps/configs/insightface_detect_faces.py` | 48 | `InsightFaceDetectFacesConfig` | sim_bench |
| `specs/034-pipeline-context-contract/spec.md` | — | Field-by-field contract for `PipelineContext` + `_RunContext` | docs |

**Changed (8 files)**

| Module | LOC (post) | What changed |
|---|---:|---|
| `sim_bench/pipeline/steps/face_cluster_bridge.py` | 228 | Stopped dropping 5 fields; removed 4 of 5 force-disables; **pinned `blur_min=0`** (SIGHTING-061 workaround) |
| `sim_bench/pipeline/steps/face_cluster_export.py` | 234 | Wired `filters=context.filters`; builds `image_scores` join dict |
| `sim_bench/pipeline/base.py` | +8 | `BaseStep.process()` calls `validate_step_config()` |
| `sim_bench/pipeline/steps/filter_quality.py` | 88 | Validates config via `FilterQualityConfig` at entry |
| `sim_bench/pipeline/steps/filter_faces.py` | 235 | Same pattern |
| `sim_bench/pipeline/steps/cluster_people.py` | +3 | Same pattern |
| `face_cluster/run_exporter.py` | 799 | 4 additive columns; Pandera `.validate()` before INSERT; `image_scores` kwarg |
| `face_cluster/run_store.py` | 617 | New `image_detail(image_path)` single-JOIN reader |

**UI files changed** (P-A label-only): `app/streamlit/components/pipeline_runner.py`, `app/face_clustering/tabs/run_tab.py`, `app/face_clustering/tabs/recluster_tab.py`. Behavior unchanged; help-text only, plus one new slider (`min_bbox_ratio`).

### 1.3 Module dependency map

```
                   ┌──────────────────────────────────────────────┐
                   │  app/streamlit/  +  app/face_clustering/     │ (UI tier)
                   └─────┬────────────────────────────────────────┘
                         │ writes step_configs dict
                         ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  sim_bench/pipeline/                                         │
   │   ┌─────────────────────────────────────────────────────┐    │
   │   │  steps/configs/  (Pydantic models — P-G)            │    │
   │   │     FilterQualityConfig, FilterFacesConfig,         │    │
   │   │     ClusterPeopleConfig, ExtractFaceEmbeddingsConfig│    │
   │   │     InsightFaceDetectFacesConfig                    │    │
   │   │     STEP_CONFIG_MODELS registry, _validate helper   │    │
   │   └────────────┬────────────────────────────────────────┘    │
   │                │  validates dict at BaseStep.process()       │
   │   ┌────────────▼──────────────────────────────────────┐      │
   │   │  steps/  (the step implementations)               │      │
   │   │    filter_quality, filter_faces, cluster_people,  │      │
   │   │    face_cluster_bridge, face_cluster_export       │      │
   │   └───────┬────────────────────────────────────┬──────┘      │
   └───────────┼────────────────────────────────────┼─────────────┘
               │                                    │
        bridge │ converts FaceForClustering         │ export step
               │     → face_cluster.FaceRecord      │ calls RunExporter
               ▼                                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  face_cluster/  (clustering engine + storage I/O)            │
   │     types.py       (Pydantic FaceRecord)                     │
   │     config.py      (dataclass PipelineConfig — pre-existing) │
   │     run_exporter.py (writer, Pandera-validated)              │
   │     run_store.py   (reader, image_detail)                    │
   │     db_schemas.py  (Pandera DataFrameSchemas)                │
   │     image_detail.py (Pydantic ImageDetail / FaceDetail)      │
   │     config_diff.py (effective-config + CLI)                  │
   │     filter_context.py (KNOWN_FILTERS registry — pre-existing)│
   └──────────────────────────────────────────────────────────────┘
```

**Layering rule** (preserved): `sim_bench/` may import from `face_cluster/` (bridge direction). `face_cluster/` must not import from `sim_bench/`. No spec-033 module crosses the line in the wrong direction.

### 1.4 Data flow — one face, end to end

```
UI widget value
    │ (e.g., config_min_iqa=0.2)
    ▼
step_configs[step_name][key]            (PipelineContext)
    │
    ▼
BaseStep.process(context, config)
    │ ── validate_step_config(name, config)   ← P-G: typo'd key raises
    ▼
step body reads cfg.attr / config.get(...)
    │
    ▼ (cluster_people branch)
run_face_cluster_knn(faces, embeddings, config, context)
    │
    ▼
faces_to_face_records(faces, emb, context=context)   ← P-C C-1: recovers
    │                                                  blur/pose/det from
    ▼                                                  context.insightface_faces
FaceRecord (Pydantic, extra="forbid")    ← P-C C-2: typo at construction raises
    │
    ▼
QualityGater.select_core_set(face_records)
    │   ← gates honor config (post-P-C) EXCEPT blur_min, pinned (SIGHTING-061)
    ▼
core_indices, holdout_indices
    │
    ▼ clustering → exemplars → merge → cap → export
    ▼
RunExporter.export(faces=..., filters=..., image_scores=...)
    │ ── FACES_SCHEMA.validate(pd.DataFrame(face_rows))   ← P-H: NULL fails
    ▼
sqlite3 INSERT INTO faces (... 21 columns, +4 from P-C C-3 ...)
    │
    ▼
... later, UI popup or analyst tool ...
    │
    ▼
RunStore(run_dir).image_detail(image_path) -> ImageDetail   ← P-D single JOIN
    │ ── Pydantic validation on return
    ▼
UI renders typed object (no more JSON-manifest scattering)
```

---

## Part 2 · Findings

Each finding has a **severity** (Critical / Major / Minor) and a per-area **Verdict** (Accept / Accept with follow-up / Needs rework).

### 2.1 Structure

| Finding | Where | Severity |
|---|---|:--:|
| `run_exporter.py` is 799 LOC and does five things: schema definition + crop writing + DB writing + run-metadata writing + strict validation | `face_cluster/run_exporter.py` | Major |
| `RunExporter.export(...)` takes **14 keyword arguments** (`faces`, `base_cluster_result`, `merged_cluster_result`, `core_indices`, `merge_log`, `merge_metadata`, `config`, `source_album`, `producer`, `run_id`, `started_at`, `finished_at`, `parent_run_id`, `crop_source_dir`, `filters`, `image_scores`) | `run_exporter.py:217-235` | Major |
| `pipeline.py` is 1064 LOC (pre-existing, not introduced by spec-033) | `face_cluster/pipeline.py` | Major (pre-existing) |
| `ClusterPeopleConfig` is 219 LOC with ~40 fields — covers face_cluster_knn + every legacy method + merge + cap | `configs/cluster_people.py` | Minor (intentional — one model per step) |
| Five separate Pydantic config files for face-clustering steps. Two are tiny (31 + 32 LOC). Could collapse into one `configs.py` if you weight cohesion over filesystem clarity | `sim_bench/pipeline/steps/configs/` | Minor |
| `db_schemas.py` declares 3 schemas; `image_detail.py` declares 3 Pydantic models. Naming inconsistent — one is plural, one is singular, neither has a suffix like `_schemas` / `_models` consistently. `config_diff.py` adds non-diff helpers (`effective_config_from_*`) — module name no longer matches contents | `face_cluster/db_schemas.py`, `image_detail.py`, `config_diff.py` | Minor |

**Verdict — Structure**: Accept with follow-up. The `RunExporter.export` fan-in is real tech debt the refactor inherited and made slightly worse (added 2 kwargs). A future PR should bundle the arguments into a typed `ExportRequest` Pydantic model — same pattern as the rest of the refactor.

### 2.2 Code quality

| Finding | Where | Severity |
|---|---|:--:|
| Hot-path try/except added in `face_cluster_export.py` that swallows all exceptions from v4 dual-write as a WARNING. Pre-existing; not introduced by spec-033 but now the path it guards is load-bearing (filter_decisions, image_scores write). A schema bug would silently log and continue | `face_cluster_export.py:118-138` | Major |
| The bridge `pose` lookup tries two key paths (`if_face.get("pose_scores") or if_scores.get("pose")`) — exploratory; neither key exists today. Dead lookup code | `face_cluster_bridge.py:78-87` | Minor |
| Operator precedence: `if_face.get("pose_scores") or if_scores.get("pose") if isinstance(if_face, dict) else None` — the ternary binds to the second operand only. Almost certainly not what was intended; works only because both `.get()` calls return None today | `face_cluster_bridge.py:80` | Minor (bug-shaped — no observable failure today) |
| `notebook_diagnostic.py` at repo root is debug scaffolding. Untracked-style file in the source tree | `notebook_diagnostic.py` | Minor (pre-existing; not deleted by this PR) |
| `_validate.py` is single-function, 28 LOC, prefix-underscored. Could be a `__init__.py` function; the file split is cosmetic | `configs/_validate.py` | Minor |

**Verdict — Code quality**: Needs rework on the bridge pose lookup (fix the precedence bug or delete the dead path; today it does nothing useful). Other items can be follow-ups.

### 2.3 Layering & duplication

| Finding | Where | Severity |
|---|---|:--:|
| Layering preserved — no `face_cluster/` import of `sim_bench/`. Bridge is the only cross-layer module, by design | — | None |
| `face_cluster/config_diff.py` extended in-place. The new helpers (`effective_config_from_*`, `_load_effective_from_run_dir`, `main`) have a different concern than the pre-existing `compute()` / `ConfigDelta`. Two responsibilities in one file now | `config_diff.py` | Minor |
| `STEP_CONFIG_MODELS` registry is the discovery mechanism for validation. It's a flat dict in `__init__.py`. If a new step is added in another module, it must be imported here or it won't validate. No CI check enforces this | `configs/__init__.py:11-19` | Major |
| `RunExporter._write_filter_decisions` exists (added in spec-032). spec-033 P-C wires `filters=context.filters` into the caller. No double-write today, but two write paths now produce filter_decisions on the Albumify pipeline (`filter_quality` records via `filter_quality.py`, and `face_cluster_export` forwards the FilterContext). Need to verify these don't both insert the same `(item_id, filter_name)` pair | `face_cluster_export.py:151`, `filter_quality.py:79` | Major |

**Verdict — Layering**: Accept with one verification (no duplicate `filter_decisions` rows). The unguarded `STEP_CONFIG_MODELS` registry is a known weak link — add a CI check that every face-clustering step in `sim_bench/pipeline/steps/` has a registered model OR an explicit "no typed config" allowlist entry.

### 2.4 Public surface

- `face_cluster/image_detail.py` — three classes exported; no `__all__` declared. Convention in this repo varies.
- `face_cluster/db_schemas.py` — three module-level `DataFrameSchema` constants; no `__all__`.
- `sim_bench/pipeline/steps/configs/__init__.py` — has `__all__`. Good.

**Verdict — Public surface**: Minor inconsistency. Add `__all__` to the two new modules or document the project convention.

### 2.5 Testability — the most important section

#### Test inventory (new tests added by spec-033)

| Test file | LOC | Kind | What it actually does |
|---|---:|---|---|
| `test_ui_aligns_with_filters.py` | 184 | **Static** | Greps UI source files for widget keys; parses `KNOWN_FILTERS` table |
| `test_pipeline_context_contract.py` | 131 | **Static** | Parses spec-034 markdown tables; compares to dataclass `fields()` |
| `test_typed_step_configs.py` | 92 | **Unit (in-process)** | Constructs each Pydantic model with good + bad input |
| `test_pandera_schemas.py` | 114 | **Unit + static** | Builds DataFrames in memory; also `inspect.getsource()` on `RunExporter._write_faces_and_scores` to confirm `FACES_SCHEMA.validate` is called |
| `test_image_detail.py` | 158 | **Synthetic-data** | Builds a synthetic SQLite DB + npy + crops dir; constructs RunStore; asserts the join |
| `test_config_parity.py` | 110 | **Mixed** | Pure-Python equality on dicts (3 tests); subprocess run of the CLI (1 test); static source inspection (1 test) |
| `test_no_raw_collection_iteration.py` | 219 | **Static** | AST walks for forbidden iteration patterns (pre-existing — spec-032) |

**Mock usage in new tests: 0.** No `unittest.mock`, no `MagicMock`, no `patch`. The synthetic-data tests build real DBs / DataFrames / files.

#### Single-responsibility per test

Spot-check: most pass. Counter-examples:
- `test_pandera_schemas::test_exporter_invokes_faces_schema` and `test_exporter_invokes_face_scores_schema` are two tests that do the same thing on different schema names — fine to keep, both fast.
- `test_image_detail::test_image_detail_returns_populated_for_synthetic_run` does seven assertions across three concepts (image-level scores, face-level cluster assignment, image-level filter decisions). Should be three tests but isn't.
- `test_config_parity::test_cli_runs_with_two_run_dirs` mixes "the CLI exists" with "the CLI returns 0 on parity" — two assertions, defensible.

#### E2E vs static vs synthetic — the breakdown

| Kind | Count | What it can catch | What it can't |
|---|:--:|---|---|
| Static (source inspection / spec parsing) | 14 | Code-shape violations (missing keyword in source, drift between spec markdown and dataclass) | Anything that depends on what the code computes at runtime |
| Unit (in-process model construction) | ~10 | Pydantic / Pandera contract semantics | Whether the contracts are actually invoked in production paths |
| Synthetic-data (build DB → exercise reader) | 2 | Reader / writer round-trips on a fake but realistic input | Whether the upstream pipeline produces the input the reader expects |
| **Real E2E (full pipeline on real data)** | **0** | — | This category is empty |

**Zero new E2E tests.** The architecture suite checks the *shape* of the code (do the right files import the right things, do the right widgets exist, do the right validators exist). It does not exercise the pipeline.

#### What test should have caught SIGHTING-061?

The bug: `face_cluster_bridge.build_fc_config` was changed to honor `cluster_people.blur_min` from yaml, but no upstream step computes blur, so every face had `blur_score=0.0` and got rejected. Pipeline failure: `Step 'identity_refinement' failed: Required context key is empty: people_clusters`.

**No test in the spec-033 suite could have caught this.** The static tests inspect source; the unit tests construct Pydantic objects with hand-supplied values; the synthetic-data tests build DBs with hand-supplied rows. None of them simulate "config knob references upstream field that doesn't get computed."

**The test that would have caught it**: a real E2E test that runs the full Albumify pipeline on a small image fixture and asserts:
- `context.people_clusters` is non-empty after `cluster_people`
- `core_indices` is non-empty after the bridge's quality gate

This test was deliberately deferred (per CLAUDE.md: "Visual smoke test of the Streamlit UIs was not performed in this session"). It's the missing acceptance gate.

A weaker but cheaper test that would also have caught it: a contract test asserting "for every gate the bridge honors, an upstream pipeline step produces the corresponding field on `insightface_faces`." That's a new class of contract — config-to-producer graph — that this refactor introduced the need for but didn't build. It's the architectural gap the SIGHTING-061 learning calls out.

**Verdict — Testability**: Needs rework before merge. At least one E2E acceptance test (Albumify pipeline on a 5-image fixture, asserts non-empty `people_clusters`) should land alongside this PR. Without it, the entire boundary-contracts feature is unproven on real data.

---

## Part 3 · Accept / reject recommendation

| Area | Verdict |
|---|---|
| Concept (Pydantic + Pandera + spec-034 + image_detail + config_diff) | **Accept** — direction is right, contracts are real |
| P-A (UI label honesty) | **Accept** — label-only, no risk |
| P-B (spec-034 + drift test) | **Accept** |
| P-G (typed step configs) | **Accept with follow-up** — registry has no CI guard |
| P-C (Pydantic FaceRecord + bridge plumb + schema additions) | **Needs rework** — bridge pose-lookup precedence bug, SIGHTING-061 root cause unresolved (only worked around) |
| P-H (Pandera schemas) | **Accept** — but write path is guarded by a try/except that swallows errors as warnings (`face_cluster_export.py:118`) |
| P-D (`image_detail`) | **Accept** — fails clean on old runs (raises `RunStoreError`), readable join |
| P-F (config diff + CLI) | **Accept with follow-up** — `config_diff.py` now has two responsibilities |
| Test coverage | **Needs rework** — zero real E2E tests; SIGHTING-061 proves architecture tests aren't enough |
| `RunExporter.export` 14-arg surface | **Accept with follow-up** — pre-existing, this PR made it slightly worse |

### Recommended follow-up tickets

1. **E2E acceptance test** — Albumify pipeline on a 5-image fixture; asserts non-empty `people_clusters` after `cluster_people`. Highest leverage.
2. **Config-to-producer graph contract** — for every `cluster_people.{gate}_min` config knob, assert an upstream step writes the corresponding field. The architectural fix SIGHTING-061 exposed.
3. **Add an `insightface_score_blur` step** — and then revert the SIGHTING-061 workaround in `face_cluster_bridge.build_fc_config`. The pin is temporary by design.
4. **Replace `RunExporter.export(**kwargs)` with `RunExporter.export(request: ExportRequest)`** — uses the same Pydantic pattern as the rest of the refactor.
5. **Verify no duplicate `filter_decisions` rows** on a real Albumify run after `filters=context.filters` is forwarded.
6. **CI check on `STEP_CONFIG_MODELS` registry** — every face-clustering step in `sim_bench/pipeline/steps/` is registered or explicitly allowlisted.
7. **Fix the bridge pose-lookup operator precedence** (`face_cluster_bridge.py:80`) or delete the dead branch.
8. **Decision on `notebook_diagnostic.py`** — delete or move under `scripts/`.

---

## References

- Master plan: `specs/033-data-integrity/MASTER_PLAN.md`
- Context contract: `specs/034-pipeline-context-contract/spec.md`
- Bug fix that motivated this review: `docs/project/SIGHTINGS.md` → SIGHTING-061
- Lesson that motivated this review: `docs/project/LEARNINGS.md` → 2026-05-15 entries
- Test files: `tests/architecture/test_*.py` (8 new), `tests/face_clustering/test_filter_context*.py` (3 new from spec-032)
