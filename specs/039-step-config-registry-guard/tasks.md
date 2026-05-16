# Tasks: STEP_CONFIG_MODELS Registry Guard (039)

## Design Notes

- **D1**: Use the runtime registry (`sim_bench.pipeline.registry`) to discover registered steps. AST scraping is more brittle than running an import.
- **D2**: Keep the allowlist small and explicit. Each entry's reason is reviewed as part of `/code-review` going forward.
- **D3**: The prefix list lives in the test module, not in the configs package — it's a test concern, not production behavior.

## Phase 1: Test

- [ ] T001 Extend `tests/architecture/test_typed_step_configs.py`:
  - Add `FACE_CLUSTERING_STEP_PREFIXES` constant.
  - Add `UNTYPED_STEPS_ALLOWLIST` dict (step_name → reason); seed empty.
  - New test `test_registry_covers_face_clustering_steps`:
    - Import the steps package so all `@register_step` decorators fire.
    - Walk the registry; for each step whose name starts with a prefix, assert it's in `STEP_CONFIG_MODELS` or in the allowlist.
  - New test `test_registry_has_no_stale_entries`: every key in `STEP_CONFIG_MODELS` corresponds to a real registered step.
  - New test `test_allowlist_reasons_non_empty`: every allowlist entry has a non-empty reason.

**Checkpoint**: Tests pass against the current 5 typed steps + 0 allowlisted; failing cases (synthetic) raise the expected messages.

## Phase 2: Hook into review

- [ ] T010 Update `docs/guides/CODE_REVIEW_CHECKLIST.md` §6 — add bullet: "When adding a face-clustering step, `STEP_CONFIG_MODELS` entry exists (gate: `test_registry_covers_face_clustering_steps`)."
- [ ] T011 Update spec-033 REVIEW.md follow-up table entry for FR-033-6 to point at this spec as "Implemented" when it lands.

**Checkpoint**: A reviewer running `/code-review` on a PR adding `filter_new.py` without a Pydantic config gets a clean fail from this test.
