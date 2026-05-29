# spec-049 — Test Suite Recovery (13 red tests on `unification/spec-040`)

**Created**: 2026-05-28
**Status**: Draft
**Predecessors**: spec-040 unification, spec-041 FCParams, spec-046/048 data layer
**Successors**: spec-047 (full-system E2E gate — depends on a green suite)
**Linked sightings**: SIGHTING-070, -071, -072, -073, -074

---

## Problem

Running the full test suite on the `unification/spec-040` branch reports:

> **13 failed, 667 passed, 12 skipped, 4 deselected**

The 13 failures were noticed during spec-048's full-suite verification step but were not on anyone's radar before that. They've been red for an unknown number of days/weeks because no one was running the full suite as a gate. **12 of them are real regressions on this branch** (confirmed: same failures with spec-048 changes stashed). One is a test-ordering issue (passes in isolation on every branch state).

This spec catalogues every failure with its surface area, intent, and likely root-cause hypothesis, so the failures can be triaged one by one rather than dismissed as a vague "clustering/merge regression bucket." Each cluster gets a sighting; this PRD is the index.

---

## Failure inventory

### Cluster A — spec-040 legacy-vs-v2 equivalence (8 failures, 1 sighting)

**Sighting**: SIGHTING-070
**Module**: `tests/face_clustering/test_legacy_vs_v2_equivalence.py`
**Common failure**: `Step 'attach_holdout_faces' failed: Validation failed: Required context key is empty: holdout_indices`
**Goal of the test**: Prove the spec-040 v2 clustering chain produces ≥ 0.95 pairwise-agreement with the legacy `run_face_cluster_knn` bridge on a real labeled 9-jpg fixture, across 4 config variants. This is the spec-040 **merge gate** — without it, the unification claim is unverified.

| # | Test method | Parametrization | What it asserts |
|---|---|---|---|
| 1 | `test_legacy_and_v2_agree_on_real_fixture` | `default` | Pairwise cluster-assignment agreement (legacy vs v2) on the 9-jpg fixture ≥ 0.95 with `CANONICAL_PARAMS` |
| 2 | `test_legacy_and_v2_agree_on_real_fixture` | `merge_on` | Same, with `merge_enabled=True` |
| 3 | `test_legacy_and_v2_agree_on_real_fixture` | `tighter_threshold` | Same, with `distance_threshold=0.35` |
| 4 | `test_legacy_and_v2_agree_on_real_fixture` | `larger_K` | Same, with `K=5` |
| 5 | `test_legacy_and_v2_produce_same_cluster_count` | `default` | `abs(n_clusters_legacy - n_clusters_v2) ≤ 1` |
| 6 | `test_legacy_and_v2_produce_same_cluster_count` | `merge_on` | Same with merge on |
| 7 | `test_legacy_and_v2_produce_same_cluster_count` | `tighter_threshold` | Same with tighter threshold |
| 8 | `test_legacy_and_v2_produce_same_cluster_count` | `larger_K` | Same with K=5 |

**All 8 fail at the same point** — the v2 runner crashes inside `attach_holdout_faces` before producing any output to compare. The failure is at v2 runtime, not at the equivalence assertion. So we don't even know what the agreement number would be once v2 runs to completion.

**What's broken**: a producer/consumer mismatch in the v2 chain. `attach_holdout_faces` expects a non-empty `holdout_indices` context key; something upstream (likely `filter_quality_gate`) is producing an empty list (or not producing the key at all). Pandera/context-validation catches it and aborts.

---

### Cluster B — `ClusterPeopleStep` quality gate + bridge (2 failures, 1 sighting)

**Sighting**: SIGHTING-071
**Module**: `tests/face_clustering/test_cluster_faces_knn_method.py`
**Goal of the file**: Cover `ClusterPeopleStep._run_face_cluster_knn` synthetic-data behavior: clustering purity/completeness, quality-gating enforcement, holdout-path correctness, and the static `_faces_to_face_records` bridge that converts `FaceForClustering` objects to `FaceRecord` rows.

| # | Test method | What it asserts | Failure |
|---|---|---|---|
| 9 | `ut_FaceClusterKNNMethod::test_quality_gating_holdout` | When 10 faces have `blur_score=0` and config sets `blur_min=50`, ALL labels must be `-1` (holdout). | Got `[0,0,0,0,0,1,1,1,1,1]` — full clusters; blur gate not honoured. |
| 10 | `ut_FaceClusterKNNMethod::test_faces_to_face_records_bridge` | `ClusterPeopleStep._faces_to_face_records(faces, embeddings)` returns 3 `FaceRecord` instances with populated `face_id`, `image_path`, `embedding`, `embedding_normalized`. | Static method shape changed during spec-040 unification; either signature differs or return type differs. (Exact failure mode not yet captured — needs a verbose run.) |

**What's broken**: Either `ClusterPeopleStep` no longer reads `blur_min` from the passed config dict, or quality gating moved to a different step in spec-040 and this test still hits the legacy code path. Either way, the test surface and the code surface have drifted.

---

### Cluster C — `PipelineConfig` adaptive-threshold cleanup (1 failure, 1 sighting)

**Sighting**: SIGHTING-072
**Module**: `tests/face_clustering/test_merge.py`
**Goal of the file**: `ut_SimplifiedMerger` is the unit suite for the "simplified merger" — the cleaned-up merger after the adaptive-threshold experiment was abandoned. The test in question encodes the claim that the abandoned fields were removed from `PipelineConfig`.

| # | Test method | What it asserts | Failure |
|---|---|---|---|
| 11 | `ut_SimplifiedMerger::test_adaptive_threshold_fields_removed` | `PipelineConfig` does NOT have any of: `merge_use_adaptive_threshold`, `merge_threshold_alpha`, `merge_threshold_beta`, `merge_exemplar_percentile`, `merge_global_percentile`. | `PipelineConfig` still has `merge_threshold_alpha=1.0`, `merge_threshold_beta=0.5`, `merge_exemplar_percentile=90`, `merge_global_percentile=75`, and `use_adaptive_merge_threshold=True`. |

**What's broken**: Test was written ahead of the cleanup (or the cleanup was reverted). Requires a decision: either delete the test (we changed our mind, adaptive threshold lives) or finish the cleanup (delete the 5 fields and any consumers).

---

### Cluster D — v4 merge-stage E2E (1 failure, 1 sighting)

**Sighting**: SIGHTING-073
**Module**: `tests/face_clustering/test_merge_stage.py`
**Goal of the file**: End-to-end proof of spec-030 Phases 1+2 on real images. From the test docstring:

> spec-030 end-to-end on real JPEGs: real images → face detect → embed → cluster → merge → RunExporter (writer) → disk → RunStore (reader). Every piece of in-memory pipeline state must come back bit-identical (or float-close) through the writer/reader pair. This is the test that proves Phases 1+2 work end-to-end on real data; all other tests use synthetic FaceRecord fixtures.

| # | Test method | What it asserts | Failure |
|---|---|---|---|
| 12 | `ut_MergeStageE2E::test_v4_full_round_trip_real_images` | Full round-trip: in-memory `MergeDecisionRow`s and other state equal the rows read back from `RunStore` after a write/read cycle. | Setup runs; specific assertion delta not yet captured. Needs verbose re-run. |

**What's broken**: Schema v5 writes from spec-040 Phase 4 changed the on-disk shape; either the writer is now emitting more fields than `RunStore` reads (silent drop) or fewer (KeyError on read), or a column type changed. Could also be a pre-existing issue from spec-030 that's been red since the schema v5 work landed.

---

### Cluster E — test-ordering interaction (1 failure, 1 sighting)

**Sighting**: SIGHTING-074
**Module**: `tests/face_clustering/test_export.py`
**Goal of the test**: Verify that `export_results` logs a WARNING when a `Face` has `image_path=None` (so silent data loss is caught at export time).

| # | Test method | What it asserts | Failure |
|---|---|---|---|
| 13 | `test_no_null_image_paths_raises_warning` | After calling `export_results` with a `Face(image_path=None)`, `caplog.records` contains a record whose message includes `"null image_path"`. | In isolation: PASSES. Inside the full suite: FAILS. |

**What's broken**: Not a behavioural bug in `export_results` itself. Some earlier test in the run mutates global logging/warnings state (most likely a `caplog.set_level` that doesn't get cleaned up, or a streamlit AppTest fixture). The export warning is presumably still being emitted; `caplog` just can't see it because the logger config has been re-pointed.

---

## Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| AC1 | All 8 spec-040 equivalence cases pass | `pytest tests/face_clustering/test_legacy_vs_v2_equivalence.py` |
| AC2 | `ClusterPeopleStep` quality gating honours `blur_min` | `pytest tests/face_clustering/test_cluster_faces_knn_method.py::ut_FaceClusterKNNMethod::test_quality_gating_holdout` |
| AC3 | `ClusterPeopleStep._faces_to_face_records` returns `FaceRecord`s with documented fields | `pytest tests/face_clustering/test_cluster_faces_knn_method.py::ut_FaceClusterKNNMethod::test_faces_to_face_records_bridge` |
| AC4 | Decision recorded on adaptive-threshold fields (delete test OR delete fields) | `pytest tests/face_clustering/test_merge.py::ut_SimplifiedMerger::test_adaptive_threshold_fields_removed` |
| AC5 | v4 round-trip on real images passes | `pytest tests/face_clustering/test_merge_stage.py::ut_MergeStageE2E::test_v4_full_round_trip_real_images` |
| AC6 | `test_no_null_image_paths_raises_warning` passes inside the full suite | `pytest tests/face_clustering` (full module, no isolation) |
| AC7 | Full suite green: `pytest tests/face_clustering tests/architecture` → 0 failed | Single full run |
| AC8 | A CI/git-hook gate added that runs the full suite and blocks merges on any regression | New `.github/workflows/test.yml` or pre-push hook (see Phase 6) |

---

## Out of scope

- New features. This spec is repair only.
- Re-architecting any of the failing modules — fix the bug, keep the shape. If a fix needs a deeper redesign, file a new spec.
- The 12 skipped tests in the full suite — those skip deliberately. Different problem.

---

## Risk

The bigger risk than any individual fix: **the full suite has been red on the active development branch and nobody noticed.** Whatever process should have caught this didn't. AC8 exists to address that — without it, the next refactor reopens the same gap.

---

## Definition of Done

- All 13 currently-failing tests pass, both in isolation AND inside the full suite.
- One sighting per cluster updated with the root cause and "Resolution" filled in.
- A learnings entry in `docs/project/LEARNINGS.md` covering the meta-failure ("how did 12 acceptance-level tests stay red on the active branch unnoticed").
- AC8's gate in place and verified by introducing a deliberate failure and watching it block.
- REVIEW.md filed (skipping the `/code-review` gate because this spec produces no new code surface — only fixes to existing code; review is on each fix's PR instead).
