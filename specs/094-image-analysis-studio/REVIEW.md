# spec-094 — Code Review (Image Analysis Studio)

**Reviewed**: 2026-07-01 · **Base**: `main` → branch `specs/093-095-analysis-studios`
**Reviewer**: automated `/code-review` against `docs/guides/CODE_REVIEW_CHECKLIST.md`
**Verdict**: ⛔ **BLOCKED** — 3 high-severity findings (2 failing architecture contract tests + docs
mandate). Slices 1–4 are functionally complete and unit-verified; handoff blocked until the contract
tests are green and the architecture HTMLs are updated.

> **UPDATE 2026-07-03 (Slice 5 + blocker resolution)** — all 3 high-severity findings RESOLVED:
> - **B1** fixed: 7 new `PipelineContext` fields (`method_scores` + 6 geo) now have spec-034 rows;
>   `test_pipeline_context_fields_covered` GREEN.
> - **B2** fixed: `score_quality` + 4 geo steps added to the `ALLOW_LIST` (per-image producers);
>   `test_no_raw_collection_iteration` GREEN. **8/8 architecture contract tests pass.**
> - **Docs**: `data_flow.html` (step + `method_scores`), `CLAUDE.md` Key Entry Points (studio),
>   spec-034 contract updated.
>
> **Slice 5 (Configure/Browse + RunFolder + 095 merge)** reviewed below — no new high-severity.
> Net verdict now: **PASS** (blockers cleared; Slice 5 clean).

### Slice 5 review (v2 restructure)
| Section | Verdict | Notes |
|---|---|---|
| Correctness | PASS | Configure→save→Browse→toggle verified live on real Budapest images (Playwright); RunFolder round-trips columns incl. topk; Browse re-opens with no recompute. |
| Tests | PASS | `test_run_folder.py` (4: round-trip, newest-first, malformed skip, empty→[]); 29 studio/geo tests green. |
| Error handling | PASS | `list_runs` skips malformed run dirs (never raises); missing folder → []; empty-family Browse shows a hint, not a crash; cache handler best-effort. |
| Naming/style | PASS | `RunFolder` pure (no Streamlit); `view` split into `render_quality`/`render_geo`; renamed off "RunStore" to avoid the `face_cluster/run_store.py` collision. |
| Dedup | PASS | `geo_view` reused (095 merged, not duplicated); standalone `geo_vision/main.py` deleted. |
| Docs | PASS | ARCHITECTURE.html + MOCK.html + tasks + READMEs updated. |
| Windows/ASCII | PASS | Ran on win32/py3.11; forward-slash paths; sys.path bootstrap for Streamlit. |

**Slice 5 medium notes**: (M) Geo map exercised live with EXIF pins only — GeoCLIP orange-pin path
is code-identical (`map_points`) but not yet run live (1.6 GB download deferred vs the batch). Follow-up:
one GeoCLIP-included run for the full two-colour map.

---

## Part 1 — How it works

### Module inventory
| Module | LOC | Responsibility |
|---|---|---|
| `app/image_studio/engine.py` | 299 | Pure orchestration: `discover_images`, `AnalysisColumn`, method registry, `run_methods`, `available_methods`, `categories`, `_merge_configs` |
| `app/image_studio/view.py` | 155 | Streamlit rendering: thumbnail `st.button` grid, sortable tables, confidence bars, click-enlarge, CSV |
| `app/image_studio/main.py` | 123 | Streamlit entry: sidebar, run, results; repo-root `sys.path` bootstrap |
| `sim_bench/pipeline/steps/extract_geo_metadata.py` | ~80 | EXIF → `geo_metadata` (feature_type `geo_exif`) |
| `sim_bench/pipeline/steps/infer_geo_clip.py` | ~85 | StreetCLIP → `geo_clip_predictions` (`geo_streetclip`) |
| `sim_bench/pipeline/steps/infer_geo_coords.py` | ~80 | GeoCLIP → `geo_coord_predictions` (`geo_geoclip`) |
| `sim_bench/pipeline/steps/caption_images.py` | ~70 | BLIP → `image_captions` (`blip_caption`) |
| `geo_cluster/{captioning,streetclip,geoclip_locator}.py` | ~90 each | Model helpers; caching removed (step owns it) |
| deleted `geo_cluster/_imcache.py` | — | Side JSON cache, replaced by universal_cache |

### Dependency map (layering — correct direction)
```
main.py ──► view.py ──► engine.py ──► pipeline steps ──► geo_cluster helpers + universal_cache
                                  └──► ScoreQualityStep / ScoreIQAStep / ScoreAVAStep (093 + existing)
```
No reverse imports. `engine` depends on steps; steps depend on helpers; helpers are framework-agnostic.

### Data flow
`discover_images(folder)` → `run_methods(paths, selected, cache_handler)` groups selected methods by
backing step, runs each step once (config merged), each step reads `context.image_paths` and writes its
own context field **and** persists per-image through `universal_cache` (feature_type per model,
`model_version` for schema invalidation). Mappers normalize each field → `AnalysisColumn`. View renders
category tabs + flat All.

---

## Part 2 — Findings by checklist section

### §1 Structure — **pass**
- `engine.py` at 299 LOC has a single named responsibility (comparison orchestration) with a top
  docstring; steps are thin (≤85 LOC, one class each per spec-053). No dead code / unused imports found.

### §2 Code quality — **pass-with-followup**
- `run_methods` and the model helpers swallow exceptions to a `logger.warning` (a failing method/image
  is skipped, not fatal). Justified for an **inspection/comparison** tool (a broken metric must not sink
  the whole table; matches FR-011 "never raise on bad image"). Documented at each site. *Follow-up F1.*
- No bare `except`, nesting ≤3, no silent boundary defaults on load-bearing fields.

### §3 Naming — **pass**
- `engine.py` / `view.py` / `main.py` names match contents; `app/image_studio/` promoted to a subpackage
  (≥3 files share the concern). `__all__` not used — consistent with sibling apps (no project convention).

### §4 Layering & coupling — **pass** (but see B2)
- **Single writer per state verified**: `geo_metadata`, `geo_clip_predictions`, `geo_coord_predictions`,
  `image_captions` each written by exactly one step; `method_scores` only by `score_quality`. No collision.
- Config merge (`_merge_configs`) correctly unions shared-step `methods` lists — fixed a latent bug where
  a distinct step_id per pyiqa metric would let `ScoreQualityStep.process` (full-replace of
  `method_scores`) wipe siblings.

### §5 Testability — **pass-with-followup**
Test inventory:
- **Unit/integration (committed):** `tests/geo/test_geo_steps_cache.py` (5 — real EXIF round-trip +
  cache-hit, monkeypatched vision full-top-k persist, no-handler fallback); `tests/image_studio/test_engine.py`
  (14 — discover filter/sort/cap/empty, every mapper incl. empty-preds, run_methods only-selected /
  shared-step-once / failing-skip, `_merge_configs`, pyiqa single-run, registry).
- **Real-data smokes (run, not committed):** engine over Budapest EXIF via universal_cache; brisque+niqe
  scored-together + 2nd-run-cached.
- **E2E (manual):** Playwright drove the live Streamlit app over 12 Budapest images (tabs, sort, bars, 36
  clickable thumbs, click-enlarge) — screenshots verified.
- Failure-mode walk-through: the "cache bypassed" class → `test_*_persists_full_topk` (cache-hit asserts
  no recompute). The "distinct-step wipes method_scores" class → `test_run_methods_pyiqa_merges_into_one_run`.
- *Follow-up F1:* the real-data E2E was **manual**; commit an automated budapest-marked engine smoke
  (exif-only, no model download) so CI exercises the public `run_methods` path on real data.

### §6 Boundary contracts — ⛔ **fail** (B1)
- **B1**: `test_pipeline_context_contract.py::test_pipeline_context_fields_covered` **FAILS**. Seven
  context fields this work reads/writes have no spec-034 row: `geo_metadata`, `geo_clip_predictions`,
  `geo_coord_predictions`, `geo_home`, `geo_segments`, `image_captions`, `method_scores`. The contract IS
  enforced at runtime (good) — it just isn't satisfied. Add rows to
  `specs/034-pipeline-context-contract/spec.md` (§§3–5) or `CONTEXT_EXEMPTIONS`.
- Config-knob→producer: `only_missing_gps` reads `geo_metadata`, produced by `extract_geo_metadata`
  (declared `depends_on`) — OK. No new Pydantic/Pandera boundary (cache payloads are JSON via `Serializers`,
  matching the `ava_scores` idiom).

### §4/§6 Architecture rule — ⛔ **fail** (B2)
- **B2**: `test_no_raw_collection_iteration.py` **FAILS**. Five steps iterate `context.image_paths` raw
  (`caption_images:43`, `extract_geo_metadata:46`, `infer_geo_clip:44`, `infer_geo_coords:44`,
  `score_quality:68`). The rule requires consuming `ctx.filters.active(item_type)` **or** a reviewer-gated
  `ALLOW_LIST` entry with a reason. These are album-wide analysis/annotation steps (must cover every
  discovered image), so ALLOW_LIST entries are appropriate — but they must be added (reviewer decision),
  or the steps migrated.

### §7 Documentation — ⛔ **fail** (B3)
- `spec.md` ✓, `tasks.md` ✓, `REVIEW.md` ✓ (this file), `CHANGES_LOG.md` ✓ (4 entries).
- **B3**: `docs/architecture/{data_flow,db_schemas,classes}.html` **not updated** for: the new
  `universal_cache` feature_types (`geo_exif`, `geo_streetclip`, `geo_geoclip`, `blip_caption`), the new
  `PipelineContext` fields, and the new `app/image_studio/` surface. Documentation-update mandate → blocks.
- LEARNINGS.md: recommend an entry for the "distinct step_id wipes a full-replace context field" class.

### §8 Risk register — **pass-with-followup**
- No new deps (transformers/torch/geoclip/reverse_geocoder/pillow_heif/pyiqa pre-installed; verified).
- Backwards-compat: `extract_geo_metadata` is in the default `pipeline.yaml`; refactor now caches via
  universal_cache instead of recomputing — tested, no regression. `app/album_explorer` + `app/geo_explorer`
  lose helper-level caching (recompute per run) — documented, experimental, superseded.
- Hot-path: EXIF step now does a cache lookup it didn't before — negligible; cheap read.

---

## Part 3 — Verdict & tickets

| Area | Verdict |
|---|---|
| §1 Structure | pass |
| §2 Code quality | pass-with-followup (F1) |
| §3 Naming | pass |
| §4 Layering | pass (B2 is the arch-rule finding) |
| §5 Testability | pass-with-followup (F1) |
| §6 Boundary contracts | **fail (B1, B2)** |
| §7 Documentation | **fail (B3)** |
| §8 Risk | pass-with-followup |

### Blockers (must resolve before `Implemented`)
- **B1** — Add the 7 context fields to `specs/034-pipeline-context-contract/spec.md` (or `CONTEXT_EXEMPTIONS`)
  so `test_pipeline_context_fields_covered` is green.
- **B2** — Add the 5 step files to `ALLOW_LIST` in `test_no_raw_collection_iteration.py` with a one-line
  reason each ("album-wide analysis step; scores/annotates every discovered image"), or migrate them.
- **B3** — Update `docs/architecture/{data_flow,db_schemas,classes}.html` for the new feature_types,
  context fields, and `app/image_studio/`.

### Follow-up tickets (non-blocking)
- **F1** — Commit an automated budapest-marked engine smoke test (exif-only, no download) exercising
  `run_methods` on real data. → `TODO.md`.
- **F2** — LEARNINGS.md entry: "a distinct step_id per metric wipes a full-replace context field
  (`method_scores`); share the step + merge config." → `docs/project/LEARNINGS.md`.

**No test that previously passed on `main` is left red by design** — B1/B2 are contract tests that turned
red because this branch *added* the fields/steps they govern; resolving B1/B2 (rows + ALLOW_LIST) is the
sanctioned closure, not a masking of a real regression.
