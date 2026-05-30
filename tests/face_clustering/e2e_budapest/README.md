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

## Adding a scenario for a new tab

When a new tab spec ships, add **one row** above + **one test file** in this dir.
The row's "Asserts" column is the load-bearing part — list concrete checks
beyond `no exception`.

Naming: `test_scenario_<letter>_<short_desc>.py`. Letters in order: A, B, C…

### Planned (each owned by a tab spec)

| ID | Owner spec | Will add | Asserts |
|---|---|---|---|
| C | spec-063 Recluster | Load reference run → Recluster tab → pick same params → click Run → wait | new snapshot run dir written; n_clusters ∈ [12, 18] (band around 15); snapshot's `parent_run_id == 6437d335…` |
| D | spec-064 Face Analysis | Load reference run → Cluster Analysis → click thumbnail's "Open" → Face Analysis tab opens | Face Analysis renders face crop + 5 score metrics + bbox/landmarks visible |
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
