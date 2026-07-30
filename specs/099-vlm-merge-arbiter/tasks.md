# spec-099 — Tasks

Status legend: `[ ]` open · `[>]` in progress · `[x]` done

## Slice 0 — client & config (no behavior change)
- [ ] T0.1 Resolve Q3: confirm/add `anthropic` dep + API-key handling (env, not committed).
- [ ] T0.2 Add `vlm_arbiter_*` fields to `PipelineConfig` (all default OFF). Update
      `docs/architecture/` config HTML (doc mandate).
- [ ] T0.3 Verify flag-OFF is a no-op: run Budapest E2E, assert `n_clusters == 15`.

## Slice 1 — arbiter helper (mockable, no network)
- [ ] T1.1 `VLMMergeArbiter` in `face_cluster/vlm_merge.py`: `__init__(config)` +
      typed `ArbiterInputs` / `ArbiterResult` + `calc()` (spec-053 shape).
- [ ] T1.2 VLM call behind a small `VLMClient` interface (so tests inject a fake).
      Structured output `{same_person, confidence, reason}`.
- [ ] T1.3 Conservative gate: `same_person and confidence >= min_confidence` → approve;
      else keep separate.
- [ ] T1.4 Unit tests (mock VLM): low-confidence → no merge (AC-5); yes/high → merge;
      cap reached → un-adjudicated count logged (AC-4).

## Slice 2 — cache (determinism)
- [ ] T2.1 Pair-hash = `sha256(sorted exemplar embeddings + model_id + prompt_version)`.
- [ ] T2.2 Store/read verdicts in `universal_cache` (feature_type `vlm_merge_verdict`).
- [ ] T2.3 Test: warm cache → second identical run makes **0** VLM calls (AC-2).

## Slice 3 — teacher log (feeds MLMerger)
- [ ] T3.1 Append one row per verdict to
      `results/face_clustering_training/vlm_verdicts/<run_id>.csv`:
      12 `candidate_pairs` features + `label` + `confidence` + `reason`.
- [ ] T3.2 Test: row count == VLM call count; columns match
      `merge_training_data.csv` schema (PLAN_ML_CLUSTER_MERGING §4.5) (AC-3).

## Slice 4 — wire into merge (opt-in)
- [ ] T4.1 After `group_merge_candidates()`, route `confidence=="review"` groups
      to `VLMMergeArbiter`; approved pairs → `apply_manual_merges()` (Q2 lean).
- [ ] T4.2 Integration test behind a marker: real API on Budapest `review` set;
      assert deterministic on re-run (cache hit).
- [ ] T4.3 Re-run Budapest E2E flag-ON; record resulting cluster count + which
      `review` pairs the VLM merged (report artifact).

## Close-out
- [ ] T5.1 `reports/<date>_vlm-merge-arbiter/` — sample adjudicated pairs w/ crops +
      VLM reason inline; 2-line entry in `reports/EXPERIMENTS.md`.
- [ ] T5.2 `/code-review` → `REVIEW.md`; resolve High findings.
- [ ] T5.3 `CHANGES_LOG.md` entries; flip spec Status → Implemented.
