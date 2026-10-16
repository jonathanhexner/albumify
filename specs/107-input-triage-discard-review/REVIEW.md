# spec-107 — Code Review (Input Triage & Discard Review)

**Reviewed:** 2026-10-01 · **Reviewer:** Claude · **Base:** `main`
**Scope:** P0 (metadata tier) + P1 (content/RAM++ tier) + P2 (pipeline steps + `set_aside` contract)
+ the pipeline-builder ordering fix that P2 surfaced. P3 (behavioral) + P4 (review UX) are not yet
built and are out of scope for this review.
**Verdict:** **ACCEPT** — the 2 architecture-guard blockers (B1, B2) were resolved during review;
architecture + triage suites green (149 passed). P0–P2 Implemented; P3/P4 remain as follow-up specs.

---

## Part 1 — How it works

### Module inventory
| File | LOC | Responsibility |
|---|---|---|
| `sim_bench/triage/metadata.py` | 94 | Tier-1: pure `calc(TriageInputs)->TriageResult` from filename/EXIF/format/dims + `classify_file`. |
| `sim_bench/triage/content.py` | 88 | Tier-2: pure `calc(tags)->TriageResult` (strong/soft junk taxonomy + moment-override) + lazy `RamTagger`. |
| `sim_bench/pipeline/steps/triage_metadata.py` | 39 | Thin step wrapping `metadata.classify_file`. |
| `sim_bench/pipeline/steps/triage_content.py` | 78 | Thin step wrapping `content` (cache template for `ram_tags`). |
| `sim_bench/pipeline/steps/_triage_apply.py` | 38 | Shared step-side helper: record decisions + filter survivors. |
| `sim_bench/pipeline/builder.py` | (edit) | Stable, author-ordered dependency resolution + topo sort. |
| `sim_bench/pipeline/context.py` | (edit) | +`set_aside`, `triage_review`, `ram_tags` fields. |

### Dependency map / layering
`steps/triage_*` (framework) → `sim_bench.triage.*` (framework-agnostic domain, spec-053 compliant)
→ nothing upward. `_triage_apply` is the only framework-coupled helper (touches `PipelineContext`),
correctly placed on the step side. No reverse imports.

### Data flow
`discover_images` → `triage_metadata` (sets aside named screenshots, free) → `triage_content`
(RAM++ on survivors; caches `ram_tags`; sets aside docs/posters) → scoring/face/scene steps run on
survivors only → `select_best`. `set_aside`/`triage_review` ride the context to the result for the UI.
`set_aside` images are removed from `image_paths`; nothing is deleted.

---

## Part 2 — Findings by checklist section

### §1 Structure — **pass**
All modules ≤94 LOC, single responsibility; step files thin (≤80, spec-053). No dead code. No
function >4 params. Domain/step split respected.

### §2 Code quality — **pass**
`try/except` only around image I/O + RAM++ inference, each degrading to a safe default (unreadable →
`review`/empty tags, never raised). No silent defaults on load-bearing fields. Nesting ≤3.

### §3 Naming / package structure — **pass-with-followup**
New `sim_bench/triage/` subpackage (3 files sharing a concern) — correct promotion. **Finding (low):**
no `__all__` in `triage/metadata.py` / `content.py`. Repo is inconsistent on this; follow-up TODO.

### §4 Layering / coupling — **pass**
**Single-writer check:** `set_aside`/`triage_review`/`ram_tags` written only by the triage steps
(via `_triage_apply`). `image_paths` is now written by `discover_images` (producer) and the triage
steps (filter) — a deliberate producer→filter chain, same pattern as `filter_quality`/`filter_portraits`
on `active_images`; documented here as the contract. No duplicated logic (reuses the cache template).

### §5 Testability — **pass**
Test inventory: **19 unit** (`test_metadata_triage` 9, `test_content_triage` 8, `_triage_apply` 2) +
**1 real-data integration** (`needs_data`: 715-roll metadata run) + **2 builder-ordering** +
**1 real-data E2E** (`scripts/verify_spec107_no_junk.py` → PASS: 0 junk in album vs 6/20).
Failure-mode walk-through: the exact bug class the feature prevents (junk in the album) has a named
E2E assertion; the ordering bug that defeated it has `test_independent_steps_keep_declared_order`.
Mocks: none — all tests run against real code / real tags.

### §6 Boundary contracts — **pass-with-followup**
No new Pydantic/Pandera boundary (triage steps have no typed config model — optional, consistent with
many existing steps). `set_aside` is an internal context list, not an external contract. `ram_tags`
cache uses `feature_type` + `model_version` versioning (correct). **Follow-up (low):** when the UI
consumes `set_aside` (P4), promote its entry shape to a typed model with `extra="forbid"`.

### §7 Documentation — **pass-with-followup**
`spec.md` ✓, `tasks.md` ✓, `REVIEW.md` ✓ (this), CHANGES_LOG ✓ (5 entries). **Follow-up:**
`docs/architecture/data_flow.html` should show the two new pre-curation steps + `set_aside` read path;
`classes.html` should note the new context fields. LEARNINGS entry for the builder-ordering bug.

### §8 Risk register — **pass**
Backwards-compat: triage is in `default_pipeline`; other pipelines (`minimal_pipeline`) unaffected (no
hard depends_on pulls RAM++ in). Builder change is behavior-affecting but covered by 191 green pipeline
tests + 2 new ordering tests. Hot-path: triage runs *before* the expensive stack and *reduces* total
work (57 survivors vs 68 scored on the Slovakia subset); RAM++ cost is the one added model, cached.

---

## Part 3 — Verdict & tickets

### Blockers — RESOLVED during review
- **B1 (§7/contract) — FIXED:** added `set_aside`, `triage_review`, `ram_tags` rows to
  `specs/034-pipeline-context-contract/spec.md` §3.1. `test_pipeline_context_contract` green.
- **B2 (§4/guard) — FIXED:** added `triage_metadata.py`, `triage_content.py`, `_triage_apply.py`
  to the raw-iteration `ALLOW_LIST` with reasons (`_triage_apply` is the centralized filter primitive
  this guard exists to encourage). `test_no_raw_collection_iteration` green. Full suite: 149 passed.

### Follow-up tickets (non-blocking → TODO)
- F1 (§3): add `__all__` to `triage/metadata.py` + `content.py`.
- F2 (§7): update `data_flow.html` (two triage steps + `set_aside` read path) + `classes.html`
  (new context fields); add LEARNINGS entry for the builder tie-break bug.
- F3 (§6): typed `extra="forbid"` model for the `set_aside` entry shape when P4 wires the UI.

### Accept criteria met
Core feature proven on real data (0 junk in album), 19+ tests green, thin-step convention held,
no regressions (191 pipeline tests pass). **Accept once B1 + B2 are fixed and the architecture suite
is green.**
