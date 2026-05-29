# Tasks: Split RunExporter (057)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

## Phase 0 — Capture the golden output (~30 min)
- [ ] **T001** Create `tests/face_clustering/exporter/test_split_equivalence.py` skeleton + a `golden_exporter_input` fixture (small synthetic run input — 3 clusters, ~30 faces, sourced from the spec-045 synthetic builder).
- [ ] **T002** Snapshot today's `RunExporter.export(...)` output: hash every produced file. Commit the hash list as `tests/face_clustering/exporter/_golden_hashes.txt` (load-bearing — every Phase 1-2 task must keep producing the same hashes).

**Gate**: `pytest tests/face_clustering/exporter/test_split_equivalence.py -q` → 1 test passes (current exporter produces the snapshotted hashes).

## Phase 1 — Carve out per-table DB writers (~3-4 h)
- [ ] **T010** Create `face_cluster/run_exporter/` package. Move existing module content into `face_cluster/run_exporter/exporter.py` (rename only; no logic changes).
- [ ] **T011** Extract `_write_faces_and_scores` → `writers/faces_writer.py:write_faces(conn, inputs)`. Run the equivalence test. Loop until it passes.
- [ ] **T012** Repeat T011 for: clusters_writer (clusters + cluster_assignments), merges_writer, filter_decisions_writer, images_writer, scenes_writer, run_metadata_writer.
- [ ] **T013** Extract shared helpers (`_connect`, transaction context manager) into `writers/_common.py`.

**Gate**: equivalence test green after every extraction; never red for more than one commit.

## Phase 2 — Artifact writers (~1 h)
- [ ] **T020** Extract `_write_embeddings_npy` → `artifact_writers/embeddings_writer.py`.
- [ ] **T021** Extract `_write_pipeline_run_json` → `artifact_writers/pipeline_run_writer.py`.
- [ ] **T022** Extract `_write_crops` → `artifact_writers/crops_writer.py`.

**Gate**: equivalence test green.

## Phase 3 — Tighten the facade (~1 h)
- [ ] **T030** `RunExporter.export()` reduces to: open one transaction → call each writer in order → commit. Target ≤ 80 LOC in `exporter.py` for `export()` itself.
- [ ] **T031** `RunExporter.calc()` (spec-053 entry point) stays a thin facade; verify no logic moved into it.
- [ ] **T032** Add arch test in `tests/architecture/test_run_exporter_layering.py`: assert no file under `face_cluster/run_exporter/` > 200 LOC.

**Gate**: `pytest tests/face_clustering/ tests/architecture/ -q` → no regression.

## Phase 4 — Transaction atomicity test (~30 min)
- [ ] **T040** New test: monkeypatch `merges_writer.write_merges` to raise mid-export. Assert the output dir is left empty (no faces.db rows, no embeddings.npy, no crops). Proves the single-transaction contract survived the split.

**Gate**: T040 passes.

## Phase 5 — Cleanup + close-out (~1 h)
- [ ] **T050** Remove backward-compat shim `face_cluster/run_exporter.py` (the package's `__init__.py` is the new import surface).
- [ ] **T051** Update `docs/architecture/classes.html` — replace the single `RunExporter` row with one row per writer module.
- [ ] **T052** Update `docs/architecture/data_flow.html` — replace the single export node with the multi-writer fan-out.
- [ ] **T053** CHANGES_LOG entry.
- [ ] **T054** `/code-review` → `REVIEW.md`. Address any high-severity findings.
- [ ] **T055** Spec status → `Implemented`. Commit + push.

**Final gate**: full suite green; spec-053 + spec-046 + spec-045 baselines unchanged.

## Total estimate
**~6-8 hours.** Phase 1 dominates (one writer extraction per ~30 min, including the equivalence-test cycle).
