# Slovakia26 full roll on the Modal L4 worker (2026-10-09)

**Goal**: produce the Stage A quality-gate input (ROADMAP Stage A, spec-102 updates): the real
`default_pipeline` over the WHOLE 715-photo trip, on the GPU worker, as `album.json`, so the owner can
judge the picks against their own keeper list. Also the first at-scale run of the spec-110 job flow.

**Method**: `scripts/modal_curate_driver.py --src D:/Slovakia26 --n 800 --job-id slovakia_full`
(768 px downsample, EXIF kept, staged to the `albumify-jobs` Volume, `curate_job` on an L4 with
protobuf 4.25 + per-step `device=cuda`, deterministic torch). Full result:
`reports/_modal_job_slovakia_full.json`; originals on `D:\Slovakia26`; displays in the job store.

## Outcome
| | |
|---|---|
| photos sent | 715 (97 MB at 768 px) |
| pipeline time | 341 s = **0.48 s/img**; wall 521 s incl. upload |
| picks | **147** across 96 scene clusters (max 2 per cluster, no global cap yet) |
| near-duplicates dropped | 423 |
| set aside by triage | 56 (screenshots, receipts, signs, posters) |
| people clusters | 121; largest [16, 11, 10, 9, 9] |
| **missing from album.json** | **89** of 715 — removed before scoring (`filter_quality`), invisible (SIGHTING-128) |

Slowest steps (s, whole roll): score_tilt 146, triage_content 55, score_occlusion 29, insightface_detect_faces 28, select_best 24, extract_scene_embedding 10.

## What this says
- The worker is production-shaped: 0.48 s/img all-in; 100 photos ≈ 1 min warm.
- 147 picks is not an album: selection is per scene cluster with no total cap. The Stage B item
  `album_max_picks` (global top-K) is the next engine change; Stage A can still be judged as
  "would you keep each of these" + "which keepers of yours are missing, and what reason did we give".
- 89 photos silently vanish (dropped by the quality filter before scene clustering); the
  show-everything principle requires them in `dropped` with a reason. SIGHTING-128.
- 121 people clusters with the largest at 16 faces on a family trip = identities
  over-split at this scale; spec-111 checkpoint #1 (merge candidates) is the designed fix.

## Stage A — what the owner does next
Open `report.html`, mark each pick keep / not, list keepers of yours that are absent (then find them in
the Dropped section and note the reason given). That verdict + the missing-keeper reasons = Gate 1.
