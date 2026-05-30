# V2 Budapest e2e — functionality matrix

The binding contract for the v2 app. Every new tab / feature MUST add a
scenario to this matrix and a corresponding test file.

Run: `.venv/Scripts/python -m pytest -m budapest tests/face_clustering/e2e_budapest/ -v`

## Reference baseline (verified 2026-05-30)

| Key | Value |
|---|---|
| Source | `D:\Budapest2025_Google` |
| Profile | `profile_4.json` |
| Reference run id | `6437d335de914755bc3edb825c9591c0` |
| Producer (reference) | `fc_app` (legacy) — v2 must reproduce same shape |
| n_faces total | 340 |
| n_faces assigned (in clusters) | 107 |
| n_clusters | **15** |
| Cluster sizes (largest → smallest) | 35, 24, 14, 7, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2 |
| First face_id in cluster 0 | `face_0000` |

Constants live in `conftest.py`. If the reference shape drifts (e.g. a
clustering-algorithm change is intentional), update conftest **and** this
table in the same PR.

## Scenarios

| ID | Test file | What it does (click sequence) | Asserts | What it catches |
|---|---|---|---|---|
| **A** | `test_scenario_a_fresh_run.py` | Run tab → fill Source + Album → pick profile_4 → click Run → wait up to 10 min for "Run complete" message | success message visible; parsed `n_clusters == 15` | clustering-output regression; Run tab UI breakage; pipeline crash; profile loading bug |
| **B** | `test_scenario_b_load_reference.py` | History tab → click row whose run-id contains `6437d335` → "Load into analysis tabs" → Cluster Analysis tab → wait for metric strip | ≥5 metric widgets visible; Faces metric value ∈ known cluster sizes; ≥1 `<img>` thumbnail | SIGHTING-078/079/080/089 + face-grid thumbnail bug + "no exception but blank UI" class |
| **C** | `test_scenario_c_recluster.py` | History tab → click row containing `6437d335` → "Load into analysis tabs" → Recluster tab → leave default params → "Run recluster" → wait | (1) "Recluster complete" message visible; (2) a new run dir exists under `~/.sim_bench/runs/` whose `pipeline_run.json` has `parent_run_id == 6437d335de914755bc3edb825c9591c0`; (3) parsed `n_clusters ∈ [12, 18]`; (4) new run is visible in History tab on rerun | Recluster wiring corrupting input face_records; producer chain invoked accidentally; parent_run_id lineage broken; SIGHTING-079 class regression on Recluster tab |
| **D** | `test_scenario_d_face_analysis.py` | History tab → load `6437d335` → Cluster Analysis tab → wait for face_grid → click first thumbnail's "Open" button → Face Analysis tab opens | (1) Face Analysis tab visible (`h2 "Face Analysis"`); (2) Plotly bbox overlay OR `<img>` crop rendered; (3) >=5 metric widgets present (Blur/Yaw/Pitch/Roll/Area); (4) Face id `number_input` echoes the selected id | face_grid Open button not writing `selected_face_id`; `FaceAnalysisService.compute_face_detail` crash on real face id; SIGHTING-079 regression on Face Analysis tab |

## Adding a scenario for a new tab

When a new tab spec ships, add **one row** above + **one test file** in this dir.
The row's "Asserts" column is the load-bearing part — list concrete checks
beyond `no exception`.

Naming: `test_scenario_<letter>_<short_desc>.py`. Letters in order: A, B, C…

### Planned (each owned by a tab spec)

| ID | Owner spec | Will add | Asserts |
|---|---|---|---|
| E | spec-065 Merged Clusters | Load reference run → Merged Clusters tab | ≥1 row in merge_decisions table; clicking a row shows full 28-column detail panel |
| F | spec-065 Quality | Load reference run → Quality tab | per-gate bar chart has ≥1 bar; total rejected count matches `340 - 107 = 233` (band: 220-240) |
| G | spec-066 Gallery | Load reference run → Gallery tab | ≥1 cluster row visible with ≥1 thumbnail; cluster-1 row shows 8 thumbnails (largest cluster) |
| H | spec-066 Overview | Load reference run → Overview tab | 4-metric strip renders; per-album bar chart has bar for "Budapest2025_Google_5" |

## What we DON'T verify (yet — drift-prone)

- Pixel-perfect screenshots (visual diff)
- Per-face-id cluster membership beyond cluster-0's first face_id
  (face detector occasionally re-numbers on borderline crops)
- Exact stage timings (machine-dependent)

These are deliberate gaps. If a regression slips through them, file a
sighting before tightening — assertion churn is worse than a missed bug
of a specific kind.

## Running

```bash
# Full suite (~10 min if Scenario A pipeline runs from scratch):
.venv/Scripts/python -m pytest -m budapest tests/face_clustering/e2e_budapest/ -v

# Scenario B only (fast — no pipeline run, ~30 s):
.venv/Scripts/python -m pytest -m budapest tests/face_clustering/e2e_budapest/test_scenario_b_load_reference.py -v
```

Failure screenshots land in `_failure_artifacts/<unix_ts>.png`.

Prereq: `.venv/Scripts/playwright install chromium` (one-time).

## CLAUDE.md gate

A v2 commit is not Implemented until `pytest -m budapest` is green. See
CLAUDE.md §"Delivery Quality" → "V2 baseline gate (binding)".
