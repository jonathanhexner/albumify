# spec-102 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **A (prove the picks)** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — harness + 3-trip runs done; **no human verdict** (Gate 1 still open).

Newest first. Every entry says what was done and how it was verified.

## 2026-10-09 — Stage A input produced: full 715-photo Slovakia26 roll curated on the GPU worker
- Real `default_pipeline` over the whole trip on the Modal L4 (spec-110 job flow): 0.48 s/img, 341 s.
  Album: **147 picks** across 96 scenes (per-scene max 2, no global cap yet), 423 near-dups, 56 set
  aside with reasons, 121 people clusters (over-split at this scale), **89 photos missing** from
  album.json (dropped by `filter_quality` before scene clustering, no record — SIGHTING-128).
- **Judging sheet for the owner:** [report](../../reports/2026-10-09_slovakia_full_gpu/report.html) —
  every pick and every set-aside photo inline, grouped by day, plus 24 near-dup pairs. Verdict per pick
  (keep / not) + the list of the owner's keepers that are absent and the reason given = Gate 1.
- Caveats before reading the verdict as final: the global top-K cap (Stage B) is not built, so 147 is
  "everything that won its scene", not a 20-photo album; the 89 invisible drops must be surfaced first.

## 2026-10-09 — Testing method clarified (recorded by the owner)
- **Decision:** PhotoTriage / CUFED benchmarks are NOT the quality gate for Albumify. PhotoTriage is
  best-of-burst on near-duplicate series, not a real album; CUFED is importance voting. Neither says
  "would these 20 picks make a good trip album." The gate is **real trips** (Slovakia26, Budapest,
  Austria, Germany) judged against the owner's own keeper list.
- **Follow-up completed 2026-10-09:** roadmap Stage A now uses real trips judged against the owner's
  keeper list; PhotoTriage/CUFED removed from the quality gate in line with this recorded decision.
- Note: this decision was not written down earlier; the written trail (finalization plan,
  agentic_layer_design) proposed PhotoTriage/CUFED. Recorded here so it is not re-proposed.

## 2026-10-01 — Slovakia26 real-roll analysis (branch-out to specs 107–109)
- **What:** ran the `faces` pipeline (CPU, 768px, K=20) on a 75-image subset of the real 715-photo
  Slovakia26 trip with screenshots deliberately left in.
- **Finding:** 6 of 20 picks were junk (5 screenshots + 1 photographed poster). A Gmail screenshot
  scored 0.703, the top score in the album. Junk mean composite 0.54 vs 0.41 for real photos: **quality
  scoring rewards screenshots**, so they must be removed before curation, not filtered by score.
- **Also:** the 14 real picks were sound trip moments across 8 scene clusters. Runtime 2.2 s/img with
  the face stack at 75% of it.
- **Branched out:** spec-107 (input triage), spec-108 (semantic layer), spec-109 (GPU). The quality
  verdict itself was deferred until triage is on and a keeper list exists.
- **Verified by:** `reports/2026-10-01_slovakia_analysis/` (report.html with inline samples,
  summary.md, results.json). Repro: `scripts/experiment_slovakia_analysis.py`.

## 2026-07-19 — EXP-2 within-cluster best-frame harness (Budapest)
- **What:** `exp2.py` + blind best-frame picker HTML; Budapest Albumify arm regenerated with
  occlusion/tilt penalties after the SIGHTING-117 memory fix.
- **Verified by:** 7 unit tests (`tests/albumify_vs_vlm/test_exp2.py`). Human picking step
  (T5.2b) **not done**.

## 2026-07-17/18 — Both arms + 3 trips (Budapest, Austria, Germany)
- **What:** Albumify arm (real pipeline on 768px inputs, exactly K=20) vs VLM arm (Claude Opus,
  raw images only, map-shortlist then reduce), blind A/B viewer, metrics, structured annotation (EXP-3).
- **Finding:** pick overlap shrinks with trip size: Budapest 6/20, Austria 3/20, Germany 0/20.
  VLM annotation (album type, moment groups, a caught finger-occlusion) is strong and Albumify has no
  equivalent. Full people-aware pipeline OOMed above ~120 images (SIGHTING-117), so the big trips ran
  the `minimal` pipeline without the person penalty.
- **Verified by:** 3 reports under `reports/2026-07-17_albumify_vs_vlm_budapest/`,
  `2026-07-18_albumify_vs_vlm_austria/`, `2026-07-18_albumify_vs_vlm_germany/`; unit tests
  `tests/albumify_vs_vlm/` (schema, curation, metrics, VLM reduce, viewer/report).
  **Human blind judging (T4.HUMAN) not done.**

## 2026-07-17 — Harness (T1) + decisions locked
- Downsample (EXIF-transposed, HEIC-aware, 768px, content hash), roster with day segmentation,
  versioned prompts. Decisions: Claude-only, solo N=1 pilot, Budapest first.
- **Verified by:** 11 unit tests green.

## Open to close this spec
1. Owner judges the 3 blind viewers (T4.HUMAN) and the EXP-2 picker (T5.2b).
2. Re-run the Albumify arm on Slovakia26 **with triage on** vs the owner's keeper list.
3. `/code-review` (experiment-scoped) + LEARNINGS entry (T7.3, T7.4).

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [Budapest A/B report](../../reports/2026-07-17_albumify_vs_vlm_budapest/report.html) | 122 imgs, K=20 each arm: 6/20 overlap; VLM annotation strong; Albumify ran the `faces` pipeline (OOM, SIGHTING-117) |
| [Austria A/B report](../../reports/2026-07-18_albumify_vs_vlm_austria/report.html) | 474 imgs: 3/20 overlap; Albumify ran `minimal` (no person penalty) |
| [Germany A/B report](../../reports/2026-07-18_albumify_vs_vlm_germany/report.html) | 797 imgs incl. HEIC: 0/20 overlap; redundancy saturates at 0 on big trips |
| [Slovakia26 analysis](../../reports/2026-10-01_slovakia_analysis/report.html) | 75-img subset: 6/20 junk, scoring rewards screenshots, 14 real picks sound, 2.2 s/img |
| [Expert reviews](EXPERT_REVIEW.html) | two independent critiques of the design and how each was addressed |
| [Prompts](PROMPTS.html) | versioned curation / annotation / rater-sheet prompts |
| Index | `reports/EXPERIMENTS.md` |

**Reading the three trips together:** the VLM is a useful reference peer (different, equally valid picks; strong
annotation) but not a truth source, and agreement collapses with trip size. That is why Stage A moved to
the owner's own keeper list on a real roll (Slovakia26) instead of VLM agreement.
