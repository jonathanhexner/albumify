# spec-052 — Current 5 red tests on `unification/spec-040`

**Created**: 2026-05-28
**Status**: Draft
**Predecessors**: spec-049 (which catalogued 13 failures; 8 fixed by the spec-040 Phase 6 `BaseStep.validate` change, leaving the 5 below)
**Linked sightings**: SIGHTING-071, -072, -073, -074

---

## Why a new spec instead of editing spec-049

Spec-049 is the historical record of the 13-failure state. Editing it would erase the "we found 13, fixed 8, 5 remain" story. This spec is the current snapshot.

---

## Current state

```
pytest tests/face_clustering tests/architecture
→ 5 failed, 699 passed, 9 skipped, 4 deselected, 45 warnings
```

The 8 spec-040 equivalence failures (spec-049 Cluster A) were resolved by the `BaseStep.validate` relaxation that landed in parallel. The 4 errors in `test_fc_app_v2_e2e.py` were resolved by spec-050's signature-update sweep. What remains:

```
2 real functional bugs (P1)
2 stale tests / half-finished refactor (P2)
1 test-ordering flake (P3)
```

---

## The 5 failures

### #1 · P1 · Quality gate silently bypassed

```
File:    tests/face_clustering/test_cluster_faces_knn_method.py
Class:   ut_FaceClusterKNNMethod
Method:  test_quality_gating_holdout
Failure: AssertionError: Expected all holdout labels, got [0 0 0 0 0 1 1 1 1 1]
```

**What the test does:** Builds 10 faces all with `blur_score = 0`, passes `config["blur_min"] = 50`, calls `ClusterPeopleStep._run_face_cluster_knn(...)`. Every face should be classified as "holdout" (label `-1`) because they're all below the blur threshold.

**What actually happens:** Faces are clustered into two groups — labels `[0,0,0,0,0,1,1,1,1,1]`. The blur threshold is being ignored.

**Why this matters:** If `blur_min` is silently ignored in the unified clustering chain, every production run that relies on it is silently broken — low-quality faces leak into clusters and reduce purity.

**Suspected cause:** Quality gating moved out of `ClusterPeopleStep` during the spec-040 unification (into a separate `filter_quality_gate` step). The test still hits the unified-chain entry point but the entry no longer enforces blur. Either the gating step isn't wired into this code path, or the test needs to call the gate separately.

**Sighting:** SIGHTING-071

---

### #2 · P2 · Test calls a method that no longer exists

```
File:    tests/face_clustering/test_cluster_faces_knn_method.py
Class:   ut_FaceClusterKNNMethod
Method:  test_faces_to_face_records_bridge
Failure: AttributeError: type object 'ClusterPeopleStep' has no
         attribute '_faces_to_face_records'
```

**What the test does:** Calls `ClusterPeopleStep._faces_to_face_records(faces, embeddings)` and asserts the returned `FaceRecord` instances have the right fields.

**What actually happens:** The method was deleted during the spec-040 unification — face-to-record conversion moved into a producer step (`extract_face_embeddings` or similar). The test was not updated.

**Why this matters:** Low — the test asserts a method that intentionally doesn't exist anymore.

**Resolution path:** Delete the test, or rewrite it against the producer step that now owns the conversion.

**Sighting:** SIGHTING-071

---

### #3 · P2 · Half-finished refactor — fields meant to be deleted are still there

```
File:    tests/face_clustering/test_merge.py
Class:   ut_SimplifiedMerger
Method:  test_adaptive_threshold_fields_removed
Failure: AssertionError: assert not True
         (where True = hasattr(PipelineConfig(...), 'merge_threshold_alpha'))
```

**What the test does:** Asserts that `PipelineConfig` does NOT have any of these 5 fields:
- `merge_use_adaptive_threshold`
- `merge_threshold_alpha`
- `merge_threshold_beta`
- `merge_exemplar_percentile`
- `merge_global_percentile`

The test encodes the claim "we abandoned the adaptive-threshold experiment; the fields were removed."

**What actually happens:** `PipelineConfig` still has all 5 fields with default values populated.

**Why this matters:** Low — the adaptive-threshold code may still be live (in which case the test is wrong), or the cleanup was started and dropped (in which case the fields are dead code). A decision is owed.

**Resolution path:** Either delete the test (we changed our minds) or delete the fields + their consumers (we meant to remove them).

**Sighting:** SIGHTING-072

---

### #4 · P1 · Merge writer/reader round-trip drops or duplicates data

```
File:    tests/face_clustering/test_merge_stage.py
Class:   ut_MergeStageE2E
Method:  test_v4_full_round_trip_real_images
Failure: AssertionError: assert 5 == 4
         (count mismatch — specific field not captured in --tb=line output)
```

**What the test does:** End-to-end on real JPEG images:
```
detect → embed → cluster → merge → RunExporter (writer) → disk → RunStore (reader)
```
Every piece of in-memory pipeline state must come back bit-identical (or float-close) through the writer / reader cycle. The test docstring describes this as "the test that proves spec-030 Phases 1+2 work end-to-end on real data."

**What actually happens:** A count assertion fails — 5 of something where the test expects 4. Without a full `--tb=short` run we can't see which field.

**Why this matters:** Medium-high — if the writer is emitting more rows than the reader reads back (or vice versa), every real run is silently losing or duplicating data.

**Suspected cause:** Schema v5 (spec-040 Phase 4) added columns to the on-disk shape. Either the writer is emitting one extra row the reader doesn't pick up, or a count assertion was tuned to an older fixture and the merger now produces one more cluster.

**Sighting:** SIGHTING-073

---

### #5 · P3 · Test passes alone, fails when run after others

```
File:    tests/face_clustering/test_export.py
Method:  test_no_null_image_paths_raises_warning
Failure: Passes in isolation. Fails inside `pytest tests/face_clustering`.
         (Not consistently in the failure set across runs.)
```

**What the test does:** Calls `export_results(...)` with a face whose `image_path = None`. Asserts that `caplog.records` contains a warning-level record whose message includes `"null image_path"`.

**What actually happens:** When run alone, the warning is captured and the assertion passes. When run after the rest of `tests/face_clustering/`, the warning is presumably still emitted but `caplog` doesn't see it — some earlier test mutated the global logging configuration without restoring it.

**Why this matters:** Low — the production code is fine; this is a test-infrastructure issue.

**Suspected cause:** A fixture calls `logging.basicConfig(...)` with `force=True` (which replaces all root handlers) or `caplog.set_level(...)` without restoring afterwards. Candidates include `sim_bench.logging_setup.setup_logging` (which uses `force=True`) and any Streamlit AppTest fixture.

**Sighting:** SIGHTING-074

---

## Module-by-module summary

```
tests/face_clustering/test_cluster_faces_knn_method.py
  ├── ut_FaceClusterKNNMethod::test_quality_gating_holdout         #1 (P1)
  └── ut_FaceClusterKNNMethod::test_faces_to_face_records_bridge   #2 (P2)

tests/face_clustering/test_merge.py
  └── ut_SimplifiedMerger::test_adaptive_threshold_fields_removed  #3 (P2)

tests/face_clustering/test_merge_stage.py
  └── ut_MergeStageE2E::test_v4_full_round_trip_real_images        #4 (P1)

tests/face_clustering/test_export.py
  └── test_no_null_image_paths_raises_warning                      #5 (P3)
```

---

## Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| AC1 | #1 — blur quality gate enforced (or the test ported to the correct entry point) | `pytest tests/face_clustering/test_cluster_faces_knn_method.py::ut_FaceClusterKNNMethod::test_quality_gating_holdout` |
| AC2 | #2 — test deleted or rewritten | Test file no longer references `_faces_to_face_records` OR new test passes |
| AC3 | #3 — adaptive-threshold decision recorded (test deleted OR fields deleted) | Either test passes (cleanup finished) or test removed (cleanup abandoned), with a sighting close-out documenting which path was taken |
| AC4 | #4 — round-trip passes; merger writer/reader produces equal counts | `pytest tests/face_clustering/test_merge_stage.py::ut_MergeStageE2E::test_v4_full_round_trip_real_images` |
| AC5 | #5 — passes in both isolation and full-suite ordering | `pytest tests/face_clustering` |
| AC6 | `pytest tests/face_clustering tests/architecture` → 0 failed | Single full run |
| AC7 | CI or pre-push hook in place that runs the full suite and blocks regressions | New `.github/workflows/test.yml` or `.git/hooks/pre-push` |

---

## Out of scope

- New features. Repair only.
- Refactor of the modules these tests cover beyond what's needed to make assertions accurate.
- The 9 skipped tests in the full suite — those skip deliberately.

---

## Risk note

These 5 went unnoticed because nobody was running the full suite as a gate. AC7 closes that gap; without it, the same kind of drift will accumulate again.

---

## Definition of Done

- All 7 ACs green.
- Each of #1–#5 either fixed or explicitly removed via a small spec (or sighting close-out) under the spec-implementer workflow.
- REVIEW.md filed per fix.
- This spec flipped to Implemented when AC6 is green.
