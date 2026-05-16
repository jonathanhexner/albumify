# Feature Specification: Config-to-Producer Graph Contract

**Created**: 2026-05-15
**Status**: Draft
**Resolves**: FR-033-2; the architectural gap SIGHTING-061 exposed

## Problem Statement

Spec-033's contracts catch *shape* mismatches (typo'd config keys, NULL DB columns, unknown Pydantic fields). They do **not** catch *graph* mismatches: a config knob that references an upstream-computed field where no upstream step actually produces that field.

SIGHTING-061 was exactly this: `cluster_people.blur_min: 50.0` in `pipeline.yaml` gated faces on `blur_score`, but no step in the active InsightFace pipeline writes `blur_score`. The bridge respected the config; every face was rejected. The contracts said the SHAPE was fine.

This spec adds the missing contract: for every gate-style config knob, an upstream step must produce its underlying signal.

## User Stories

### US1 — CI fails when a gate has no producer (Priority: P1)
As an architect reviewing a change that adds or modifies a quality gate, I want CI to fail if the gate's underlying signal is not produced by any step in the active pipeline, so that "honor the config" never reactivates a gate against missing data.

**Acceptance Criteria**:
- Given a config knob `cluster_people.blur_min` that gates on `FaceRecord.blur_score`, when no pipeline step in `default_pipeline` writes that field, then the architecture test `test_gate_has_producer.py` fails with a clear message naming the orphan gate.
- Given the same gate with an upstream `insightface_score_blur` step that writes `blur_score`, the test passes.

### US2 — Explicit waiver mechanism for known-pinned gates (Priority: P2)
As an engineer landing a temporary workaround (e.g. SIGHTING-061's pin), I want a documented way to mark a gate as "intentionally not enforced today, here's why," so that the test surfaces the technical debt without blocking unrelated work.

**Acceptance Criteria**:
- A `GATE_WAIVERS` map (gate_name → (reason, target_ticket)) is consulted by the test.
- Waivers without a `target_ticket` (FR id or spec number) fail the test — every waiver names the work that will lift it.
- When the target ticket's spec status is `Implemented`, the waiver is treated as overdue and fails.

## Edge Cases

- A new gate added without a producer mapping → fails with "unknown gate."
- A producer step is renamed → the mapping is by step name, so the test catches the rename mismatch.
- A gate is removed but a waiver remains → fails with "waiver for nonexistent gate."

## Requirements (brief)

- FR-001: Static map `{gate_config_path: required_context_field}` declared in `face_cluster/quality_contracts.py` (new module).
- FR-002: Architecture test `tests/architecture/test_gate_has_producer.py` walks the map and verifies each `required_context_field` is written by at least one step listed in `default_pipeline` from `configs/pipeline.yaml`.
- FR-003: Producer discovery — either scrape step source for `context.<field> =` writes (AST), or declare a `produces_fields` set on each `StepMetadata` (preferred, less brittle).
- FR-004: `GATE_WAIVERS` dict in the same module; tests pass when a waiver is current and fails when overdue.
- FR-005: Initial waiver for `cluster_people.blur_min` → target FR-033-3 (spec-037). When spec-037 lands `Implemented`, this waiver becomes overdue and the test fails — forcing the bridge pin to be lifted.

## Non-Goals

- Not a general dataflow analyzer. Hand-maintained mapping is fine for the ~5 gates that exist.
- Not extended to non-gate config (e.g. `merge_candidate_threshold` doesn't gate on a context field, so it doesn't need a producer).

## Open Questions

- [NEEDS CLARIFICATION] Should `produces_fields` be declared on `StepMetadata` (typed, discoverable) or left as text in the map (cheap, brittle)? Recommendation: typed on StepMetadata; existing `produces` set already exists for context keys, just needs extension to "fields on context-held objects."
