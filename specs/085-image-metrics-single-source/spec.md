# spec-085 — One source of truth for the image-metrics API contract

**Created**: 2026-06-25 · **Status**: Implemented · **Priority**: P1
**Predecessors**: spec-084 (Results metrics transparency)
**Source**: user — Results table columns (Body, Frontal, Central, Roll, BodyPose, and spec-084's Reason/Quality/Penalty) render blank. Root cause documented in `docs/architecture/results_image_data_flow.html`.

## Problem
The same per-image record is re-declared as **three independent hand-maintained field
lists** on the server: the producer `_build_image_metrics` (~23 fields), the projection
`_build_image_dict`, and the Pydantic response contract `ImageMetrics` (10 fields). The
narrowest one wins: FastAPI's `response_model=list[ImageMetrics]` **silently drops** every
field `ImageMetrics` doesn't declare. So computed-and-stored fields never reach the UI →
blank columns. Verified: the columns that work are exactly the 10 `ImageMetrics` declares.

## What we build (C-lite)
1. **`ImageMetrics` becomes the single source of truth for the shape.** Add the missing
   fields so it declares the full per-image set (the ~13 currently dropped, incl. spec-084's
   `quality_score`/`person_penalty`/`filter_reason`).
2. **The producer (`_build_image_metrics`) returns an `ImageMetrics` directly**
   (`ImageMetrics(path=..., iqa_score=..., ...).model_dump()`), instead of a free-form dict.
   The schema is the one shape definition; the producer only supplies the *values* (the
   bespoke extraction from context, which is irreducible code).
3. **`_build_image_dict` builds *from* `ImageMetrics`** too
   (`ImageMetrics(**{**metrics, "path": path, "is_selected": ...}).model_dump()`), normalising
   old runs and applying per-view overrides. No hand-copied field list remains anywhere.
4. **Parity test** asserting the producer's emitted field set **==** `ImageMetrics.model_fields`.

Deliberately NOT in C-lite (would be C-full): a metric registry tying each field to its
producing pipeline step, to catch "added a context score but forgot to extract it" and to
show/hide columns by which steps ran. C-lite still allows a forgotten field to default to
`None` silently — that gap is acknowledged, not closed here.

Out of scope: the frontend `ImageInfo` dataclass (already declares every field) and the
client↔server boundary (legitimately separate). Not collapsing those.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | `ImageMetrics` declares every field `_build_image_metrics` emits | parity unit test |
| 2 | `_build_image_dict` output == `ImageMetrics` field set (no hand list) | unit test |
| 3 | `person_detected`/`best_frontal_score`/`filter_reason` survive the API boundary | unit test (round-trip through `ImageMetrics`) |
| 4 | Results table shows the previously-blank columns on real data | manual (needs a run) |
| 5 | spec-084 tests still green; no field regressions | pytest |

## Notes
- This **supersedes spec-084's manual `_build_image_dict` forwarding** (replaced by the SSOT build).
- `ClusterInfo.images: list[ImageMetrics]` inherits the fix automatically (nested).
- `filter_reason`/`quality_score`/`person_penalty` still populate on **new runs only** (spec-084).
- Documentation mandate: update `docs/architecture/results_image_data_flow.html` to mark the break point resolved.
