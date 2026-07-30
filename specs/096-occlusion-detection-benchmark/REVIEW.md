# spec-096 — Code Review (against docs/guides/CODE_REVIEW_CHECKLIST.md)

**Date**: 2026-07-09 · **Reviewer**: Claude (session with user adjudication complete)
**Scope**: `sim_bench/occlusion_bench/` (12 modules), `app/occlusion_review/` (3 modules),
`scripts/{refit_after_adjudication,extract_log_features,experiment_*}.py`, dataset artifacts
in `D:\occlusion_dataset\`, `RESULTS.md`.

**Verdict: PASS with follow-ups** (research-bench scope; production integration is spec-097,
which carries the production-grade test obligations).

## 1 · Structure — pass
- All modules < 300 LOC, one responsibility each (`dataset` build/manifest, `grouping`
  near-dup union-find, `eval` CV harness, `saliency`, per-track files `track_{b,c,d}_*`).
- Review app split UI (`main`) / data layer (`data`, Streamlit-free) / explain (`explain`).
- No dead code found; experiment scripts under `scripts/` are named `experiment_*` and
  documented as research one-offs.

## 2 · Code quality — pass-with-followup
- Broad `except Exception` with `logger.warning` on per-image loops (unreadable files must
  not kill a 800-image sweep) — legitimate for research sweeps; production step (spec-097)
  must tighten to per-image skip + count reporting. **Follow-up noted in spec-097 T1.2.**
- No bare `except:`; no silent defaults on load-bearing paths (corrections vocabulary is
  runtime-asserted in `save_correction`).

## 3 · Naming — pass
- Track modules follow one convention; ids carry provenance prefixes (`budapest__…`).

## 4 · Layering & single-writer — pass
- `occlusion_bench` imports nothing from `app/`; review app imports the bench (correct
  direction). `corrections.csv` has exactly one writer (`data.save_correction`,
  append-only, last-write-wins documented).

## 5 · Testability — pass-with-followup (main finding)
- **Test inventory: 0 pytest tests.** The bench was validated by execution: 6 candidate
  tracks ran end-to-end on 832 real images; results human-adjudicated (all 82
  disagreements); refit reproduced numbers deterministically (frozen sha1 split, fixed CV
  seed). That is a real E2E validation of research code, but it is not regression-protected.
- **Follow-ups filed** (spec-097 tasks): unit tests for `grouping` (union-find, the
  leak-prevention mechanism) and `eval._scene_level` (max-over-group), since spec-097's
  reported quality rests on both. The production scorer itself gets full step tests in
  spec-097 T1.4.
- Failure-mode walk-through: near-duplicate leakage (the bug the user caught) is prevented
  by `apply_group_split` and *measured* by its `old_split_leaked_rows` stat (0 on current
  manifest) — a runtime check, not only a test.

## 6 · Boundary contracts — pass-with-followup
- `manifest.csv` / `corrections.csv` schemas are implicit CSV; runtime asserts exist for
  decision vocabulary and npz↔manifest alignment (refit aborts on mismatch). No Pandera
  models (research artifacts, single consumer). Follow-up: if spec-097 reads any of these
  at production time, promote to a typed loader. (Stage 1 does NOT — it reads only the
  serialized model artifact.)

## 7 · Documentation — pass
- spec.md (locked decisions incl. ground-truth policy), tasks.md, RESULTS.md (round-1 +
  FINAL tables), this REVIEW.md, CHANGES_LOG entries per phase, LEARNINGS entries
  (2026-07-05 heuristics ceiling; 2026-07-08 HEIC; verdict entry added at close-out).
- Architecture HTMLs untouched **correctly**: no pipeline step / DB / Pydantic surface
  changed in 096 (that happens in 097 and is on its checklist).

## 8 · Risk register — pass
- D2 CNN never retrained post-adjudication; its 0.87 was a 3-scene artifact — documented
  in RESULTS; not the production pick.
- Track C (Moondream) FAILED on torch 2.3 (`enable_gqa`) — retry requires env upgrade;
  recorded, deferred.
- LoG proposer recall is 23/49 on the full positive set — recorded; it is NOT part of
  Stage 1 (detector only).
- User's Anthropic API key used for Track A should be rotated in the Console (reminder
  standing since 2026-07-07).
- Dataset lives outside the repo (`D:\occlusion_dataset`) by design (personal photos);
  scripts take `OCCLUSION_DATASET` env override.

## Final benchmark verdict (what this spec set out to decide)
Production detector = **CLIP ViT-B/32 probe, global+tile-max concat**:
**0.86 scene PR-AUC [0.79–0.92]** on 832 images / 56 positives, adjudicated labels,
grouped CV. Full ledger in RESULTS.md.
