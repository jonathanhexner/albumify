# Feature Specification: ExportRequest Pydantic Bundle

**Created**: 2026-05-15
**Status**: Draft
**Resolves**: FR-033-4

## Problem Statement

`RunExporter.export()` takes 14 keyword arguments. The function fan-in is hard to read, hard to extend (every new caller has to copy-paste a long kwargs block), and inconsistent with the spec-033 refactor's pattern of "Pydantic at every object boundary."

A typed `ExportRequest` bundles the arguments, gets the same `extra="forbid"` validation as the rest of the refactor, and turns the call site into one parameter instead of fourteen.

## User Stories

### US1 — One-parameter export call site (Priority: P1)
As a caller of RunExporter, I want to pass a single typed request object rather than 14 kwargs, so that adding or removing an argument doesn't ripple through every caller.

**Acceptance Criteria**:
- `RunExporter.export(request: ExportRequest)` is the public API.
- All existing callers (`face_cluster/pipeline.py`, `sim_bench/pipeline/steps/face_cluster_export.py`) construct and pass an `ExportRequest`.
- A typo'd field on `ExportRequest` raises `ValidationError` at construction (Pydantic `extra="forbid"`).

### US2 — Backwards-compatible bridge (Priority: P2)
As tests and ad-hoc tools that call `export(faces=..., merge_log=..., ...)` directly, I want a transition window where the kwarg form still works, so the migration doesn't strand callers.

**Acceptance Criteria**:
- For one release: `export()` accepts either an `ExportRequest` positional/keyword OR the legacy kwargs.
- Legacy-kwarg path emits a `DeprecationWarning`.
- After the transition, the kwarg path is removed and a follow-up sighting is filed if any callers still hit it.

## Edge Cases

- Caller passes both an `ExportRequest` and legacy kwargs → `TypeError`.
- `ExportRequest` constructed with `numpy` types (e.g. `np.int64` for face_id) → Pydantic coerces; document the coercion.
- Existing tests that construct `RunExporter().export(faces=...)` continue to work during the transition.

## Requirements (brief)

- FR-001: New `ExportRequest` Pydantic model in `face_cluster/run_exporter.py` (or `face_cluster/export_request.py` if it grows).
- FR-002: Fields mirror the current 14 kwargs of `export()`:
  - `faces`, `base_cluster_result`, `merged_cluster_result`, `core_indices`, `merge_log`, `merge_metadata`, `config`, `source_album`, `producer`, `run_id`, `started_at`, `finished_at`, `parent_run_id`, `crop_source_dir`, `filters`, `image_scores`.
- FR-003: `model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)` for numpy / dataclass interop.
- FR-004: `RunExporter.export` accepts either the request or the legacy kwargs during transition.
- FR-005: All in-repo callers migrated to the request form in the same PR; legacy path scheduled for removal in the next.
- FR-006: Test `tests/face_clustering/test_export_request.py` — round-trip + typo'd field + numpy interop.

## Non-Goals

- Not migrating `RunStore` to a similar request — its read methods take one or two args each, the smell isn't there.
- Not changing the artifact layout or DB schema. Pure refactor.

## Open Questions

- [NEEDS CLARIFICATION] Where does `ExportRequest` live — inline in `run_exporter.py`, or a new `face_cluster/export_request.py`? Recommendation: inline until it grows beyond ~50 lines; promote later.
