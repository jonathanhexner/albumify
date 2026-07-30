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
| **E** | `test_scenario_e_merged_clusters.py` | History tab → load `6437d335` → Merged Clusters tab | (1) tab visible (`h2 "Merged Clusters"`); (2) >= 1 row in the merge_decisions table; (3) clicking the first row reveals a detail panel; (4) panel exposes `cluster_a` / `cluster_b` / `actually_merged` / `exemplar_dist` / `support` | Repository → Service → Tab wiring for `merge_decisions`; tab missing detail panel on row select; MergeDecisionRow schema drift |
| **F** | `test_scenario_f_quality.py` | History tab → load `6437d335` → Quality tab | (1) tab visible (`h2 "Quality"`); (2) >= 4 metric widgets in summary strip; (3) per-gate Plotly chart with >= 1 bar; (4) Rejected metric in `EXPECTED_REJECTED_BAND` (220..240, derived from 340 − 107 ± 7) | `QualityService.summary()` aggregation regression; `list_filter_decisions` schema drift; chart short-circuit to `st.info` on empty input |
| **I** | `test_scenario_i_merged_clusters_detail.py` | Seed `current_run_dir` + `selected_merge_pair=a,b` (real pair from the ref run) → Merged Clusters tab → detail panel renders without a canvas row-click | (1) detail subheader `Pair (cluster_a=…`; (2) gate badges + numbers caption in visible body (`exemplar_dist=`, all 5 gate names — `diameter` is not a table column); (3) pair-crop captions `cluster_a =` / `cluster_b =` + >= 1 visible `<img>` | spec-071 gate-badge / pair-crop wiring; the `?selected_merge_pair` seed; SIGHTING-091 canvas-click bypass for the merge detail |
| **J** | `test_scenario_j_face_metrics_click.py` | Load `6437d335` → Face Metrics tab (Grid layout) → click first face's **Open** button | (1) the click alone navigates to Face Analysis (`h2 "Face Analysis"`); (2) Face id spinbutton echoes a numeric id | spec-083 clickable-face regression — the recurring "I can't click on faces" (canvas row-select replaced by a real button); `selected_face_id` wiring |
| **K** | `test_scenario_k_images_click.py` | Load `6437d335` → Images tab (Grid layout) → click first image's **Open** button → Back to images | (1) "People in this photo" detail caption; (2) Plotly boxed-photo overlay rendered; (3) Back button returns to the grid | spec-083 Images master-detail wiring; `selected_image_path` + in-tab Image Analysis; pass-filter box overlay |

## Adding a scenario for a new tab

When a new tab spec ships, add **one row** above + **one test file** in this dir.
The row's "Asserts" column is the load-bearing part — list concrete checks
beyond `no exception`.

Naming: `test_scenario_<letter>_<short_desc>.py`. Letters in order: A, B, C…

### Planned (each owned by a tab spec)

| ID | Owner spec | Will add | Asserts |
|---|---|---|---|
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
