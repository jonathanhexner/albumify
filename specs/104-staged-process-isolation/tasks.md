# spec-104 tasks — Staged process isolation

Status legend: `[ ]` todo · `[>]` in progress · `[x]` done · `[!]` blocked

## Phase 0 — cheap in-process win (DONE 2026-07-30)
- [x] T0.1 Add `release()` to `extract_face_embeddings` (`_release_models("_extractor")` + reset config).
- [x] T0.2 Verify with probe: retained delta +366 → +5 MB (freed ~361 MB). CHANGES_LOG + SIGHTING-117.
- [x] T0.3 Audit all step files for model-attr-without-release: only `insightface_score_pose`
      (pure-geometry `FacePoseScorer`, no model) — no action.

## Phase 1 — executor isolation primitive (default OFF, byte-identical when off)
- [ ] T1.1 `IsolatedStage` spec type: `{name, steps: list[str], imports: list[str], exports: list[str]}`.
      Add optional `isolated_stages` to `PipelineSpec` (empty = today's behavior).
- [ ] T1.2 Module-level worker `sim_bench/pipeline/isolation.py::_run_stage(payload)` — rebuilds a
      fresh `PipelineContext` from imports + a local disk-backed cache_handler, runs the stage's
      steps via the existing executor loop (Tier-1 `release()` between steps), returns exports +
      relayed progress. `spawn` context; args picklable.
- [ ] T1.3 Executor: when the next steps match a declared stage, dispatch via `_run_stage` in a
      child process instead of inline; merge returned exports into the parent context.
- [ ] T1.4 Progress relay: child → `Queue` → parent invokes the real `on_progress`.
- [ ] T1.5 Error handling: stage exception → serialized error → failed `StepResult` (respect fail_fast).
- [ ] T1.6 Crash handling: non-zero child exit / no result → failed `StepResult("<stage>", "child exited N — likely OOM")`; **parent survives**.
- [ ] T1.7 Startup guard: assert every declared `exports` key is picklable; fail the spec loudly if not.
- [ ] T1.8 Tests `tests/pipeline/test_isolated_stage.py`:
      - runs a 2-step stage in a child, exports return, parent RSS stays ~flat (psutil assert). [AC1]
      - stage step raises → failed StepResult, parent alive. [AC3]
      - child `os._exit(1)` mid-stage → failed StepResult, parent alive. [AC4]
      - no `isolated_stages` → identical StepResults + context to inline run. [AC5]

## Phase 2 — declare album stages + equivalence gate
- [ ] T2.1 Define image_features / face_clustering / scene_org stages (disjoint model sets;
      crops stay inside image_features — NOT exported).
- [ ] T2.2 Equivalence test: staged vs inline on Budapest anchor → identical clusters (15/340) +
      identical `selected_images`. [AC2]  Byte-compare exports.
- [ ] T2.3 Memory test: staged `faces` on Austria (474) peak RSS < 1.2 GB via the full probe. [AC1]
- [ ] T2.4 Run `tests/face_clustering/e2e_budapest/` — stays green. [AC7]

## Phase 3 — wire real entry points + scale proof
- [ ] T3.1 Albumify arm (`albumify_vs_vlm/albumify_arm.py`) + API `PipelineService`: opt into staged
      mode (flag or image-count threshold — see open Q3).
- [ ] T3.2 Germany (788) `default` pipeline completes without OOM on the dev box; capture peak. [AC6]
- [ ] T3.3 Re-run Austria/Germany Albumify arms on the FULL people-aware pipeline (was blocked by
      the OOM → had fallen back to `minimal`); note in the spec-102 reports.

## Phase 4 — docs, review, close
- [ ] T4.1 Update `docs/architecture/data_flow.html` + `overview.md` (staged execution + boundaries).
- [ ] T4.2 `/code-review` → REVIEW.md; resolve High findings.
- [ ] T4.3 LEARNINGS.md entry (torch CPU memory not OS-returnable on Windows → isolation is the only
      reliable free; measured, corrects SIGHTING-117).
- [ ] T4.4 Flip spec + SIGHTING-117 to Implemented; remove the deferred-half note.

## Decisions needed before Phase 1 (from spec §8)
- [ ] Q1 stage grouping: 3 stages vs finer.
- [ ] Q2 imports/exports: explicit lists vs auto-derived from requires/produces.
- [ ] Q3 staged mode: auto-enable above an image-count threshold vs always-explicit flag.
