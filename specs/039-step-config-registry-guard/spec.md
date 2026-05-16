# Feature Specification: STEP_CONFIG_MODELS Registry Guard

**Created**: 2026-05-15
**Status**: Draft
**Resolves**: FR-033-6

## Problem Statement

`sim_bench/pipeline/steps/configs/__init__.py` declares a flat `STEP_CONFIG_MODELS` registry mapping step names to their Pydantic config models. `BaseStep.process()` looks up the model by step name to validate the incoming config dict.

If a new face-clustering step is added without an entry in this registry, the step silently takes a `dict` config with no validation — exactly the silent-default behavior the spec-033 refactor was built to prevent.

This spec adds a CI gate that detects un-registered face-clustering steps and fails the build.

## User Stories

### US1 — CI fails when a face-clustering step lacks a typed config (Priority: P1)
As an engineer adding a step under `sim_bench/pipeline/steps/`, I want CI to fail if my step name matches a face-clustering prefix but I forgot to register a Pydantic config model, so that the validation contract can't be silently bypassed.

**Acceptance Criteria**:
- Given a step file `sim_bench/pipeline/steps/filter_NEW.py` with a `register_step`-decorated class, when `STEP_CONFIG_MODELS` has no entry for it, then `tests/architecture/test_typed_step_configs.py::test_registry_covers_face_clustering_steps` fails with a message naming the step.
- Given the same step with an entry in `STEP_CONFIG_MODELS`, the test passes.

### US2 — Explicit opt-out for legitimately untyped steps (Priority: P2)
As an engineer working on a non-face-clustering step that happens to match the prefix, I want a documented way to opt out of validation, so the gate doesn't false-positive on unrelated work.

**Acceptance Criteria**:
- An `UNTYPED_STEPS_ALLOWLIST` set in the test module lists step names that are intentionally not in the registry.
- Each entry has a one-line reason (enforced by a separate test).

## Edge Cases

- A step registered in `STEP_CONFIG_MODELS` but no longer in the codebase → fails (stale entry).
- A step file deleted but the test still expects it → fails (forces cleanup).
- A new step whose name doesn't match the face-clustering prefixes → ignored (out of scope).

## Requirements (brief)

- FR-001: Define `FACE_CLUSTERING_STEP_PREFIXES = ("filter_", "cluster_", "insightface_", "extract_face_", "score_face_", "align_faces", "detect_face")` in the test module.
- FR-002: Discover step files under `sim_bench/pipeline/steps/*.py` that (a) match a prefix and (b) declare a `@register_step` class.
- FR-003: Assert every discovered step appears in `STEP_CONFIG_MODELS` OR in `UNTYPED_STEPS_ALLOWLIST`.
- FR-004: Assert every entry in `STEP_CONFIG_MODELS` corresponds to a real step file (no stale entries).
- FR-005: Assert every `UNTYPED_STEPS_ALLOWLIST` entry has a non-empty reason.
- FR-006: Update `docs/guides/CODE_REVIEW_CHECKLIST.md` §6 with a pointer to this gate.

## Non-Goals

- Not requiring typed configs for non-face-clustering steps (out of spec-033's scope).
- Not auto-generating Pydantic models — registration is manual; the gate just enforces.

## Open Questions

- [NEEDS CLARIFICATION] Should the test discover steps via AST (find `@register_step` decorators) or via the runtime registry? Runtime is simpler; AST is more robust to import-time side effects. Recommendation: runtime, mirroring how `BaseStep.process` looks things up.
