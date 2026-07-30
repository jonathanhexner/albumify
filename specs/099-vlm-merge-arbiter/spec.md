# spec-099 — VLM Merge Arbiter (adjudicate the `review` tier)

**Created**: 2026-07-12 · **Status**: Draft — awaiting approval · **Priority**: P2
**Depends on**: `face_cluster/merge.py` (`ConservativeMerger`, `group_merge_candidates`)
already emits `auto_approve / review / auto_reject`. Related: `docs/PLAN_ML_CLUSTER_MERGING.md`
(the logistic `MLMerger` this spec produces training data for).

## Goal
A vision-language model (Claude) adjudicates **only the borderline cluster-merge
pairs** — the ones the 4-gate heuristic already labels `review`. The VLM sees a
few exemplar face crops per side, answers *same person? yes/no + confidence +
reason*. Verdicts are (a) applied to the merge, (b) **cached by pair-hash** so
re-runs are free and deterministic, (c) **logged as `(features → label)` rows**
that train the planned `MLMerger`. Arbiter now, teacher for later.

## Why this slot (nothing new to plumb)
```
candidate pairs ─► 4 gates ─► group_merge_candidates()
                                     │
                 ┌───────────────────┼───────────────────┐
            auto_approve          review              auto_reject
              (merge)        ► VLM ARBITER ◄            (drop)
                              THIS SPEC             ~5% of pairs
```
The margin already exists (`merge_margin`, gate C) and the `review` bucket
already exists. This spec fills the empty arbiter slot; it does **not** touch
gate thresholds or the bulk clustering path.

## Scope
**In:**
1. `VLMMergeArbiter` helper (spec-053 shape: `__init__(config)` + `calc(inputs) -> result`).
   - Input: `CandidateGroup`s with `confidence == "review"` + exemplar face
     indices + a crop-loader callback.
   - For each `review` pair: render up to N exemplar crops/side → one VLM call →
     `{merge: bool, confidence: float, reason: str}`.
   - Output: approved pair list (fed into existing `apply_manual_merges`).
2. **Cache** keyed by `sha256(sorted exemplar face embeddings + model_id + prompt_version)`.
   Same faces → same verdict, no re-call. Stored in `universal_cache`
   (feature_type `vlm_merge_verdict`), matching the embedding-cache pattern.
3. **Teacher log**: every verdict appended to
   `results/face_clustering_training/vlm_verdicts/<run_id>.csv` with the full
   12-feature vector from `candidate_pairs` + VLM label + confidence + reason.
   This is drop-in training data for `MLMerger` (PLAN_ML_CLUSTER_MERGING §4.5).
4. **Budget cap** per run (`vlm_arbiter_max_calls`, default e.g. 40). On cap:
   `log()` how many `review` pairs were left un-adjudicated (no silent truncation).
5. Config flags on `PipelineConfig` (all default OFF — opt-in):
   `vlm_arbiter_enabled=False`, `vlm_arbiter_model="claude-opus-4-8"`,
   `vlm_arbiter_max_calls=40`, `vlm_arbiter_exemplars_per_side=4`,
   `vlm_arbiter_min_confidence=0.6` (below → treat as no-merge, keep separate).

**Out (explicitly):**
- No change to gate thresholds, candidate proposal, or `auto_approve`/`auto_reject`.
- Not a runtime dependency for the happy path — with the flag off, behavior is
  byte-identical to today (Budapest E2E must stay `n_clusters == 15`).
- No `MLMerger` training in this spec — we only *produce* its rows.
- No new UI tab; surface verdicts in the existing merge-decisions panel later.

## The VLM contract
```
System: You judge whether two sets of face crops show the SAME person.
        Default to "no" when uncertain — a wrong merge is worse than a miss.
Input:  side A: [crop, crop, crop, crop]   side B: [crop, crop, crop, crop]
Output (StructuredOutput / tool): {"same_person": bool,
                                   "confidence": 0.0-1.0,
                                   "reason": "<=15 words"}
```
Asymmetry is deliberate: escalation exists to *prevent* over-merging, so the
arbiter is biased conservative and `min_confidence` gates the "yes".

## Determinism & safety
- Cache-first: a hashed verdict never re-calls the model → reproducible runs.
- Cap-bounded: `vlm_arbiter_max_calls` is a hard ceiling on cost per run.
- Flag-gated: OFF by default; every existing test path is unaffected.
- Conservative default: uncertain / low-confidence → do **not** merge.

## Acceptance criteria
1. Flag OFF → Budapest E2E unchanged (`n_clusters == 15`, sizes match). *(binding)*
2. Flag ON, cache warm → **zero** VLM calls on a second identical run (assert call count 0).
3. Every adjudicated pair writes exactly one teacher row with all 12 features +
   label; row count == VLM call count.
4. Cap reached → a `log()`/warning names the count of un-adjudicated `review` pairs.
5. `confidence < min_confidence` → pair NOT merged (unit test with a mocked VLM).
6. Unit tests mock the VLM (no network in CI); one integration test behind a
   marker hits the real API on the Budapest `review` set.
7. REVIEW.md produced via `/code-review`; no High findings open.

## Open questions (for approval)
- **Q1 crop source**: aligned crops from spec-091 disk cache, or crop-on-the-fly
  from `image_path + bbox`? (Lean: reuse spec-091 cache — already on disk.)
- **Q2 merge application**: route VLM-approved pairs through the existing
  `apply_manual_merges()` (treat VLM as an auto-labeler of the manual panel), or a
  new path? (Lean: reuse `apply_manual_merges` — same union-find, transitive-safe.)
- **Q3 client**: does this repo already have an Anthropic client/key wired, or is
  adding `anthropic` + key management part of this spec's slice-0?

