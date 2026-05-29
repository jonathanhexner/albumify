# Test status — full-suite history

This file is the time-series record of full-pytest-suite runs against the
repo. Maintained so we can answer two questions instantly:

1. **When did a test start failing?** (find the first row where its name appears)
2. **When did the suite last pass cleanly?** (scan for the row with `Failed: 0`)

## When to update

Add a row when you run the full suite **as a checkpoint** — typically at:
- End of a spec (post-`Implemented` flip, alongside REVIEW.md)
- After a triage / sightings round
- Before merging a non-trivial branch into main

Do **not** add a row for every iteration while fixing a single failure —
mid-fix runs are noise.

## How to record a run

1. Run:
   ```
   .venv/Scripts/python -m pytest tests/ --ignore=<known-broken-collection-files> -q
   ```
2. Update the **Current** section below to reflect HEAD.
3. **Prepend** a row to **History** with: date, commit SHA, totals, delta
   vs. previous row, one-line note. List failing tests *only when the set
   changed* from the previous row (compact diff: `+test_foo` / `-test_bar`).
4. Commit alongside whatever change triggered the run.

---

## Current (HEAD: `31b7aea`, 2026-05-29)

| Metric | Value |
|---|---|
| Passed | **1046** |
| Failed | **19** |
| Collection errors | **6** (files) |
| Skipped | 17 |
| Deselected | 11 |

### Currently failing tests (and their sightings)

| Test | Sighting |
|---|---|
| `tests/test_face_pipeline_full.py::TestPipelineVsGroundTruth::test_distance_matrix_correlation` | [081](SIGHTINGS.md) |
| `tests/test_face_pipeline_full.py::TestFullPipeline::test_pipeline_embeddings_match_ground_truth` | [081](SIGHTINGS.md) |
| `tests/test_face_pipeline_full.py::TestFullPipeline::test_pipeline_preserves_identity_structure` | [081](SIGHTINGS.md) |
| `tests/test_ground_truth_fresh.py::TestGroundTruthFresh::test_same_person_distances` | [081](SIGHTINGS.md) |
| `tests/pipeline/test_face_recognition_benchmark.py::TestSimilarityMetrics::test_intra_person_similarity` | [081](SIGHTINGS.md) |
| `tests/clustering/test_hybrid_hdbscan_knn.py::TestInputValidation::test_1d_array_raises` | [082](SIGHTINGS.md) |
| `tests/pipeline/test_scoring_strategy.py::test_person_penalty_strategy` | [083](SIGHTINGS.md) |
| `tests/test_selection_export.py::test_imports` | [084](SIGHTINGS.md) |
| `tests/test_selection_export.py::test_selector_initialization` | [084](SIGHTINGS.md) |
| `tests/test_selection_export.py::test_selector_compute_score` | [084](SIGHTINGS.md) |
| `tests/test_selection_export.py::test_selector_best_selection` | [084](SIGHTINGS.md) |
| `tests/clustering/test_mutual_knn.py::TestHDBSCANPCAClusterer::test_basic_clustering` | [085](SIGHTINGS.md) |
| `tests/clustering/test_mutual_knn.py::TestHDBSCANPCAClusterer::test_factory_loading` | [085](SIGHTINGS.md) |
| `tests/test_photo_analysis.py::test_clip_tagger` | [086](SIGHTINGS.md) |
| `tests/test_photo_analysis.py::test_config_file_exists` | [086](SIGHTINGS.md) |
| `tests/test_ava_training.py::test_dataset` | [087](SIGHTINGS.md) |
| `tests/test_ava_training.py::test_full_training_loop` | [087](SIGHTINGS.md) |

### Currently uncollectible files (sighting [088](SIGHTINGS.md) + [084](SIGHTINGS.md))

| File | Cause |
|---|---|
| `tests/test_full_e2e_flow.py` | `sys.exit(1)` at import (backend not running) — [088](SIGHTINGS.md) |
| `tests/test_quality_assessment.py` | `UnicodeEncodeError` (Hebrew cp1255 + `✗`) — [088](SIGHTINGS.md) |
| `tests/pipeline/test_face_pipeline_e2e.py` | pytest collection error — [088](SIGHTINGS.md) |
| `tests/test_clip_aesthetic.py` | `ModuleNotFoundError: sim_bench.quality_assessment.clip_aesthetic` — [084](SIGHTINGS.md) |
| `tests/quality_assessment/test_learned_clip.py` | same missing module — [084](SIGHTINGS.md) |
| `tests/test_quality_benchmark.py` | `ModuleNotFoundError: sim_bench.quality_assessment.benchmark` — [084](SIGHTINGS.md) |
| `tests/pipeline/test_face_embedding_validation.py` | `ModuleNotFoundError: sim_bench.pipeline.steps.filter_quality_gate` — [084](SIGHTINGS.md) |

---

## History

Newest first.

| Date | Commit | Passed | Failed | Coll.err | Skipped | Δ vs prev | Notes |
|---|---|---:|---:|---:|---:|---|---|
| 2026-05-29 | `31b7aea` | 1046 | 19 | 6 | 17 | (baseline) | First recorded run. Post-spec-057 close-out + triage. 8 sightings filed (081-088). See [TEST_FAILURE_REPORT_20260529.html](TEST_FAILURE_REPORT_20260529.html) for full root-cause breakdown. |

### Change history (when the failing set diffs from the prior row)

| Date | Commit | Tests entering FAIL | Tests leaving FAIL |
|---|---|---|---|
| 2026-05-29 | `31b7aea` | (initial baseline — all 19 + 6 coll.err listed in Current above) | — |
