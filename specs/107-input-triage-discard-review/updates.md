# spec-107 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [REVIEW.md](REVIEW.html) · roadmap stage: **B** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Implemented P0–P2 (2026-10-01). P3 (behavioral tier) and P4 (discard-review UX) open.

## How it was verified
| Increment | Evidence |
|---|---|
| P0 metadata tier (filename / camera EXIF / format / screen dims) | 9 unit tests; full 715-photo Slovakia26 roll: 8/8 screenshots set aside, 707 kept, 0 false positives |
| P1 RAM++ content tier (photographed docs/posters/re-shared screenshots) | 8 unit tests; 75-image real-tag Slovakia subset: caught the 8 screenshots + 3 junk items metadata missed (incl. the poster that had leaked into the album), signs-in-scene downgraded to review, no false positives |
| P2 pipeline steps + `set_aside` contract | 19 triage tests incl. real-roll integration; end-to-end `scripts/verify_spec107_no_junk.py` → **0 junk in the curated album** |
| Builder ordering bug found by the E2E | `tests/pipeline/test_builder.py::test_independent_steps_keep_declared_order`; 191 pipeline+triage tests green |
| Code review | REVIEW.md ACCEPT; architecture + triage suites 149 passed; 2 architecture-guard blockers fixed during review |

## Why it exists
Slovakia26 analysis (2026-10-01, see spec-102 updates): 6/20 picks were junk and scoring rewarded
screenshots. Triage must run before scoring.

## Notable
- The end-to-end verify first FAILED: the pipeline builder was tie-breaking the topological sort
  alphabetically, so scoring ran before triage regardless of pipeline.yaml. Fixed in the builder;
  LEARNINGS entry 2026-10-01.

## Timeline
- 2026-10-01 — P0, P1, P2 built and verified; builder fix; REVIEW.md ACCEPT; architecture HTMLs updated.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [Slovakia26 analysis](../../reports/2026-10-01_slovakia_analysis/report.html) | the proof of need: 6/20 picks junk, Gmail screenshot top score, junk 0.54 vs real 0.41 composite |
| [RAM++ vs CLIP tagging](../../reports/2026-10-01_ram_vs_clip_tagging/report.html) | 30 photos, 3 trips: RAM++ 11.7 tags/img (210 distinct) vs CLIP 2.5 (closed 20) → RAM++ chosen for the content tier |
| [RAM++ for defect detection](../../reports/2026-10-01_ram_defect_detection/report.html) | RAM++ is content-only: occlusion AUC 0.38 vs probe 0.98, no blur tag ever fires → triage uses it for junk content, not quality |
| [CLIP tagging method](../../reports/2026-10-01_clip_tagging_method/report.html) | the poor CLIP tags were the reading method, not the model; CLIP still cannot judge composition |
| Verify log | `reports/_spec107_verify.log`: 11 set aside (8 metadata + 3 content), 57 scored, 0 junk in the 20 picks |
| [REVIEW](REVIEW.html) | ACCEPT; 149 tests |
