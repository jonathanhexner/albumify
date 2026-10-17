# spec-113 tasks — Remove the deprecated `cluster_people` step

Status legend: `[ ]` open, `[>]` in progress, `[x]` done.

## T0 — Prerequisite
- [x] T0.1 SIGHTING-132 fix: no step depends on `cluster_people`; regression tests in `tests/pipeline/test_builder.py`. | 2026-10-16
- [x] T0.2 Owner decides O1–O3. | 2026-10-16 as recommended

## R1 — Albumify regression suite (before the removal)
- [x] R1.1 `AlbumSummarizer` / `AlbumComparator` + unit tests (`tests/album/test_regression.py`, 7). | 2026-10-16
- [x] R1.2 Runner `scripts/albumify_regression.py`, datasets.yaml, pytest wrapper + marker. | 2026-10-16
- [x] R1.3 Capture baselines on the post-SIGHTING-132 code: quick, budapest, austria, slovakia. (AC6) | 2026-10-16
- [x] R1.4 Check all four on the post-removal code: 4 PASS. (AC6) | 2026-10-16
- [x] R1.5 HTML comparison report per check (counts with allowed band, people, picks: same / swapped / missing / extra / more from a group, thumbnails); `check --job <id>` re-reports a finished job. | 2026-10-16 — superseded by spec-114 (Compare runs page); report module removed

## T1 — Settings key
- [x] T1.1 `pipeline.yaml`: rename section `cluster_people:` to `people_clustering:`. (AC1)
- [x] T1.2 `pipeline_service`: read `people_clustering` (fallback per O1); test that a changed value reaches the 8 step configs. (AC3)
- [x] T1.3 `pipeline_runner.py`, `face_cluster/config_diff.py`, `fc_app_runner.py`, `run_tab.py`: new key. (AC1)

## T2 — Delete
- [x] T2.1 Delete `steps/cluster_people.py`, `steps/configs/cluster_people.py`, `steps/face_cluster_bridge.py`; unregister. (AC1)
- [x] T2.2 Delete `configs/face_clustering_export.yaml`, `configs/pipeline_custom_arcface.yaml`. (AC1)
- [x] T2.3 Tests: delete old-step-only tests; update the rest. (AC5)
- [x] T2.4 Scripts: archive per O3. (AC1)

## T3 — UI
- [x] T3.1 `explore.py`: removed the face-decision block (no step records per-face decisions since spec-079; it was always empty). Deviation from AC4, see updates.md. (AC4)

## T4 — Optional
- [ ] T4.1 `people_clustering` group expanded by `PipelineBuilder` (O2). Deferred: optional, not needed for AC1-AC6 (see TODO).

## T5 — Close-out
- [x] T5.1 Budapest gate green (AC2); full suite green (AC5); `albumify_regression.py check all` green (AC6).
- [x] T5.2 `docs/architecture/` classes + data_flow HTMLs updated.
- [x] T5.3 `/code-review` -> REVIEW.md; updates.md entry; status Implemented.
