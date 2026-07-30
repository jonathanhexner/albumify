# spec-057 — Split RunExporter into per-table writers

**Created**: 2026-05-29
**Status**: Implemented (2026-05-29)
**Predecessors**: spec-046 (SQLAlchemy data layer — established the per-table-file pattern for `action_log`)
**Successors**: spec-058 (per-run ORM models), spec-059 (Cluster Analysis Repository + RunStore on SQLAlchemy)
**Trigger**: spec-045 code review surfaced `face_cluster/run_exporter.py` as 924 LOC / 10 responsibilities / 19-field `export()` — blocks any meaningful SQLAlchemy adoption downstream because per-run schema ownership is currently a monolith.

---

## Problem

`face_cluster/run_exporter.py` violates §1 Structure of `docs/guides/CODE_REVIEW_CHECKLIST.md` on three counts:

1. **924 LOC** — over the 300 LOC bar by 3×, with no single named responsibility.
2. **10 `_write_*` private methods** glued together by a shared `output_dir`. Each one writes a different table (faces / clusters / cluster_assignments / merges / filter_decisions / images / scene_clusters / scene_cluster_assignments / run_metadata) plus three non-DB artifacts (embeddings.npy / pipeline_run.json / crops).
3. **`export()` takes 19 fields.** spec-053 wrapped them in a `RunExportInputs` dataclass — that pins the boundary but doesn't fix the class.

The combination makes the per-run schema un-refactorable: any attempt to introduce ORM models, typed Repositories, or schema-version migrations runs through this 924-LOC class first. Spec-058 + spec-059 are blocked on this split.

## What we build

Decompose `face_cluster/run_exporter.py` into one writer per artifact, all behind a thin facade that preserves today's `export()` / `calc()` signature:

```
face_cluster/run_exporter/
  __init__.py                   # re-exports RunExporter, RunExportInputs, RunExportResult
  exporter.py                   # ~150 LOC — RunExporter facade; orchestrates the writers
  writers/
    _common.py                  # ~50 LOC — shared helpers (connect, transaction wrapping)
    faces_writer.py             # ~150 LOC
    clusters_writer.py          # ~120 LOC (clusters + cluster_assignments)
    merges_writer.py            # ~60 LOC
    filter_decisions_writer.py  # ~50 LOC
    images_writer.py            # ~100 LOC
    scenes_writer.py            # ~100 LOC (scene_clusters + scene_cluster_assignments)
    run_metadata_writer.py      # ~100 LOC
  artifact_writers/
    embeddings_writer.py        # ~50 LOC — embeddings.npy + embedding_face_ids.npy
    pipeline_run_writer.py      # ~60 LOC — pipeline_run.json
    crops_writer.py             # ~60 LOC — crops/ dir copy
```

Each writer module exposes one function: `write_<artifact>(conn_or_dir, inputs)`. Pure I/O; no business logic.

## What we don't build

- **No behavior changes.** Output dirs produced by the new and old exporter must be byte-identical for the same inputs. Drift-guard test: run both on a synthetic input, diff artifacts.
- **No ORM models yet.** Writers still emit raw SQL strings against the schema in `face_cluster/db/schema.py`. ORM models land in spec-058.
- **No `export()` signature change.** The 19-field `RunExportInputs` stays as the public boundary. Internal call sites use it positionally.
- **No `_VALID_PRODUCERS` move.** That constant stays in `run_exporter/__init__.py`.

## Locked decisions

1. **Per-table file granularity, not per-table class.** Functions, not classes. Writers are stateless given a connection + inputs.
2. **One transaction across all DB writers.** The facade opens one connection, BEGINs once, calls every DB writer, COMMITs once. Matches today's behavior in `RunExporter.export()`.
3. **Artifact writers (npy / json / crops) get their own subdir** to keep DB and FS concerns separate.
4. **Public surface unchanged.** Every test that today does `from face_cluster.run_exporter import RunExporter` keeps working. `__init__.py` re-exports.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `face_cluster/run_exporter/` exists; old `run_exporter.py` removed (single-file backward compat shim is OK only as a deprecation marker for one release) | grep |
| AC2 | No file under the new package > 200 LOC | grep + `wc -l` arch test |
| AC3 | Byte-identical output dirs vs the pre-split exporter on the spec-046 golden fixture | new test in `tests/face_clustering/exporter/test_split_equivalence.py` |
| AC4 | All 9 DB writers operate inside one transaction (rollback on any failure) | unit test that monkeypatches the 4th writer to raise, asserts no rows on disk |
| AC5 | spec-053's `RunExportInputs` / `RunExportResult` boundary unchanged | architecture test on the public surface |
| AC6 | Existing test suite (`pytest tests/face_clustering/`) passes without modification | run |

## Risks

- **Atomicity regression** if the per-writer split breaks the all-or-nothing transaction.
- **Performance regression** if naive per-writer connections replace the single shared connection.
- **Hidden coupling** between writers (e.g., scene_cluster_assignments has FK to images) — call order matters; pinning it in the facade is the contract.

## Effort estimate

**~6-8 hours.** Mostly mechanical. The byte-equivalence test is the load-bearing gate.

---

## Final binding structure (post-057)

**Binding.** Mirrored in `EXECUTIVE_REVIEW_057_059.html` §9 and in specs 058 / 059. Drift = Code Review §1 fail.

**Note:** spec-056 relocated `face_cluster/run_exporter.py` to `sim_bench/run_db/exporter.py` (monolith, intact) before this spec started. Spec-057 splits that file into the writer package shown below — all at the canonical location, no shims anywhere.

### Package layout
```
sim_bench/run_db/
├─ __init__.py
├─ exporter.py                                # MODIFIED — class RunExporter facade (≤150 LOC; export() ≤80 LOC)
├─ _inputs.py                                 # NEW — RunExportInputs (spec-053),
│                                             # RunExportResult, RunExporterError
├─ _producers.py                              # NEW — _VALID_PRODUCERS = ("albumify","fc_app",...)
├─ writers/                                   # NEW
│  ├─ __init__.py
│  ├─ _common.py                              # open_run_db(path), transaction CM,
│  │                                          # _strict_validate_merge_log
│  ├─ faces_writer.py                         # write_faces(session, inputs)
│  ├─ clusters_writer.py                      # write_clusters_and_assignments(session, inputs)
│  ├─ merges_writer.py                        # write_merges(session, merge_log)
│  ├─ filter_decisions_writer.py              # write_filter_decisions(session, filters)
│  ├─ images_writer.py                        # write_images(session, inputs)
│  ├─ scenes_writer.py                        # write_scene_clusters(session, inputs)
│  │                                          # write_scene_cluster_assignments(session, inputs)
│  └─ run_metadata_writer.py                  # write_run_metadata(session, inputs)
└─ artifact_writers/                          # NEW
   ├─ embeddings_writer.py · pipeline_run_writer.py · crops_writer.py
```

### Classes
| Class | Module |
|---|---|
| `RunExporter`      | `sim_bench/run_db/exporter.py` |
| `RunExportInputs`  | `sim_bench/run_db/_inputs.py` |
| `RunExportResult`  | `sim_bench/run_db/_inputs.py` |
| `RunExporterError` | `sim_bench/run_db/_inputs.py` |

**Writers are functions, not classes** — locked decision §1.

### Methods on RunExporter (binding)
```python
__init__(output_dir: Path)
calc(inputs: RunExportInputs) -> RunExportResult        # spec-053 entry; thin
export(inputs: RunExportInputs) -> RunExportResult      # ≤80 LOC; opens 1 session,
                                                        # calls every writer in fixed order,
                                                        # commits once
```
All `_write_*` methods are deleted. `_strict_validate_merge_log` moves to `sim_bench/run_db/writers/_common.py`.

### Cross-spec invariants
1. **Single canonical path.** `from sim_bench.run_db.exporter import RunExporter`. No alternative path exists (spec-056 deleted the old `face_cluster.run_exporter` path).
2. **Public surface frozen.** Signatures and return shapes unchanged.
3. **One transaction per `export()`.** Facade opens session, BEGINs, calls writers, COMMITs. Writer modules never open their own session.
4. **Writer call order is the contract.** Pinned in `exporter.py`. Documented because `scene_cluster_assignments` has FK to `images`.
