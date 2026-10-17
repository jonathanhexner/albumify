# spec-113 — Remove the deprecated `cluster_people` step

**Status:** Implemented (2026-10-18) — owner approved scope + regression suite; O1–O3 per recommendations. Evidence: updates.md 2026-10-18
**Owner:** Jonathan / Claude
**Priority:** P1 (dead code that already caused one silent quality bug: SIGHTING-132; no test pins Albumify's results)
**Relation:** follows [[spec-079]] (replaced `cluster_people` with the 8-step chain); closes the clean-up left by
SIGHTING-132.

## 1. Problem

spec-079 replaced `ClusterPeopleStep` with an 8-step people-clustering chain ending in `assign_people_clusters`.
The old step stayed in the repo, registered and runnable. Three steps still named it as a dependency, so
`PipelineBuilder` added it to every default run and it overwrote the new chain's people (SIGHTING-132, fixed
2026-10-16 by removing those three dependencies). As long as the step exists, it can come back the same way.
Its name also survives as the settings key the UI uses for people clustering, which hides what actually runs.

## 2. Where it is today

| Area | Item | Use |
|---|---|---|
| Step | `sim_bench/pipeline/steps/cluster_people.py` (`ClusterPeopleStep`) | Registered via `all_steps.py` |
| Step config | `sim_bench/pipeline/steps/configs/cluster_people.py` (`ClusterPeopleConfig`) | Registered in `configs/__init__.py` |
| Helper | `sim_bench/pipeline/steps/face_cluster_bridge.py` | Only `cluster_people` and tests |
| Config | `configs/pipeline.yaml` section `cluster_people:` | Read as the UI's people settings |
| Config | `configs/face_clustering_export.yaml`, `configs/pipeline_custom_arcface.yaml` | Not loaded by any code |
| Service | `sim_bench/api/services/pipeline_service.py:157-193` | Translates the `cluster_people` settings into the 8 steps' configs |
| UI | `app/streamlit/components/pipeline_runner.py` | People settings stored under `cluster_people` |
| UI | `app/streamlit/pages/explore.py:468` | Shows face decisions where `step == "cluster_people"`: always empty now |
| Other | `face_cluster/config_diff.py`, `face_cluster/fc_app_runner.py`, `app/face_clustering/tabs/run_tab.py` | Settings key and comments |
| Tests | ~12 files (legacy-vs-new comparisons, bridge tests) | |
| Scripts | `capture_albumify_baseline`, `diff_fcconfig`, `experiment_clustering_ab`, `localize_gap`, `prove_albumify_unified` | spec-079 investigation tools |

## 3. Goal

1. One people-clustering path in the code: the spec-079 chain. Nothing named `cluster_people` that can run.
2. A regression suite that fails when the Albumify album on a real dataset drifts (R1). Nothing pinned it:
   the Budapest gate tests the face-clustering app, and the Albumify e2e test only checks "people non-empty",
   so SIGHTING-132 went unnoticed since spec-079.

## 4. Scope

1. Delete the step, its config class, `face_cluster_bridge.py`, and the two unused yaml files.
2. Rename the settings key `cluster_people` to `people_clustering` in `pipeline.yaml`, `pipeline_service`, the
   Albumify settings UI, and the face-clustering app.
3. Fix `explore.py` to show the new chain's face decisions (`quality_gate`, `attach_holdout_faces`, ...).
4. Tests: delete those that only test the old step; update the rest to the new key.
5. Scripts: move the spec-079 investigation scripts to `scripts/archive/` (see O3).
6. Optional (O2): a named group `people_clustering` in a pipeline list that the builder expands into the 8 steps.

7. Albumify regression suite (R1), section 7.

Out of scope: changing how the 8-step chain clusters.

## 5. Acceptance criteria

| # | Criterion | Check |
|---|---|---|
| AC1 | No step, config class or yaml section named `cluster_people` | `grep` outside `docs/archive`, `CHANGES_LOG.md`, specs: no code hits |
| AC2 | Default pipeline results unchanged by the removal | Budapest gate (`pytest -m budapest tests/face_clustering/e2e_budapest/`): 15 clusters, 340 faces, same sizes |
| AC3 | A people setting changed in the Albumify UI still reaches the 8-step chain | Test: changed value appears in the step configs `pipeline_service` builds |
| AC4 | Explore page no longer depends on the removed step (amended 2026-10-16: no current step records per-face decisions, so the block was dead and is removed) | Code + page loads without error |
| AC5 | Full test suite green | `pytest tests/` |
| AC6 | Regression suite: baselines for `quick` + 3 real albums captured BEFORE the removal; all pass AFTER it | `scripts/albumify_regression.py check all` |

## 6. Open decisions (owner)

| # | Question | Recommendation |
|---|---|---|
| O1 | Saved profiles and old job configs store people settings under `cluster_people`. | Read the old key when the new one is missing, with a one-line warning; no data migration |
| O2 | Add the `people_clustering` group to pipeline lists? | Yes, but as a separate small task after the removal |
| O3 | Delete or archive the spec-079 investigation scripts? | Archive (`scripts/archive/`); delete the tests that only compare old vs new |

All three decided 2026-10-16 as recommended (O2 stays a follow-up, T4).

## 7. Albumify regression suite (R1)

Runs the product path (`JobService`: stage 768 px → default pipeline → load) on real albums and compares the
album with a saved baseline. "Close enough", not identical: a near-duplicate burst may keep a sibling frame,
a person may gain or lose a face.

| Dataset | Source | Photos |
|---|---|---|
| `quick` | `D:/Slovakia26`, first 48 | 48 (pre-commit check) |
| `budapest` | `D:/Budapest2025_Google` | 122 |
| `austria` | `D:/Austria_24` | 474 |
| `slovakia` | `D:/Slovakia26` | 715 |

| Compared | Pass when |
|---|---|
| Counts: in album, picks, scenes, people, set aside | within 3% / 10% / 10% / 15% / 15% (differences of ≤ 2 always pass) |
| Largest 5 people's face counts | each within 20% |
| Picks | ≥ 85% of baseline picks matched, and ≥ 85% of new picks matched. A pick matches if the other album kept it or a shot from its near-duplicate group |

| Piece | Where |
|---|---|
| Summary + comparison (pure) | `sim_bench/album/regression.py` (`AlbumSummarizer`, `AlbumComparator`, `Tolerances`) |
| Runner (capture / check) | `scripts/albumify_regression.py` |
| Datasets, baselines | `tests/regression/albumify/{datasets.yaml,baselines/*.json}` |
| Pytest entry | `tests/regression/albumify/test_albumify_regression.py`, marker `albumify_regression` (opt-in) |
| Reading results | Albumify app, **Compare runs** page (spec-114). The R1.5 HTML report was removed in spec-114 (decision O3); runs are tagged `regression:<dataset>` in the app's job list |

Re-baselining is a deliberate act (`capture <dataset>`), recorded in updates.md with the reason.
