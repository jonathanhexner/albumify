# spec-040 Code Review — Phases 1–6

**Reviewer**: `/code-review` slash command against `docs/guides/CODE_REVIEW_CHECKLIST.md`
**Date**: 2026-05-18
**Branch**: `unification/spec-040` at `72bf376`
**Scope**: all commits since main `740403d`

---

## TL;DR — Verdict

**❌ Block handoff to Implemented.** Phases 1–6 ship scaffolding, not unification. The "v2 path" is unreachable from production code because no producer step writes to `context.face_records`. The equivalence test green-lights the chain on synthetic data only. Two High-severity findings; six Major; several Minor.

The work is **mergeable to main as Phase 0.5 of a longer migration** (additive, no regressions). But it must NOT be marked "spec-040 Implemented" — that requires the items in §A below.

---

## A · High-severity findings (block status=Implemented)

### A1 · No producer writes to `context.face_records` <span style="color:#8a1a1a;">CRITICAL</span>

**Where**: `sim_bench/pipeline/steps/{insightface_detect_faces, extract_face_embeddings, align_faces}.py` all still write to `context.insightface_faces` / `context.face_embeddings`. Grep confirms zero writes to `context.face_records` outside test fixtures.

**Why this is critical**: the entire Phase 3 + 5 chain (`FCAppRunner` → 8 unified clustering steps) reads `context.face_records`. In production, that list is always empty. Result: the v2 path **cannot run on a real album**. The equivalence test passes only because it hand-builds `FaceRecord` objects.

**Fix required before Implemented**: at least one producer step must dual-write (legacy dict + `context.face_records`). Either:
- Modify `insightface_detect_faces.py` to also append `FaceRecord` to `context.face_records`, or
- Add a new step `assemble_face_records` that reads `context.insightface_faces` and populates `context.face_records` — but this contradicts the locked "no bridges / translator steps" constraint in `spec.md`.

The cleanest path is the first: producers dual-write. The "no bridges" rule is satisfied because the *step itself* writes Pydantic; no separate translator class.

**Severity**: Critical. Locked-constraint violation when judged against intent.

---

### A2 · Equivalence test runs on synthetic data only <span style="color:#8a1a1a;">CRITICAL</span> · ✅ RESOLVED 2026-05-19

**Resolution**: `tests/face_clustering/test_legacy_vs_v2_equivalence.py` rewritten to share real producer-chain runs (`detect_persons → insightface_detect_faces → detect_face_orientation → align_faces → extract_face_embeddings`) between the legacy bridge and FCAppRunner. Faces matched by `(image_path, face_index)`. Two layers of coverage:
- **Small fixture** (`test_data/face_clustering/`, 9 jpgs): both `pairwise_agreement ≥ 0.95` and `cluster-count diff ≤ 1` parametrized over 4 configs — `default`, `merge_on`, `tighter_threshold=0.35`, `larger_K=5`. 8/8 pass in 88s. **This also resolves C5** — merge-stage equivalence is no longer deferred.
- **Larger fixture** (`test_data/face_clustering_100/`, 50 jpgs, opt-in via `pytest -m slow`): default config, ~1k-10k pairwise comparisons; passes in 130s.



**Where**: `tests/face_clustering/test_legacy_vs_v2_equivalence.py` builds 15 synthetic faces with hand-crafted embeddings. No real images, no real detector, no real scoring.

**Why this is critical**: the Phase 6 merge gate is "≥95% agreement on a labeled fixture." Synthetic doesn't qualify. The test passes (100% agreement on toy data) but doesn't prove the v2 path matches the legacy path on production-like input.

**Fix required before Implemented**: rewrite the equivalence test against the `test_data/face_clustering_100/` fixture (the same one `test_albumify_e2e.py` uses), or a 5-image subset. Run via the real `insightface_detect_faces` → `align_faces` → ... → clustering chain on both paths. Today this is blocked by A1 (no producer for face_records).

**Severity**: Critical. The merge gate doesn't gate anything.

---

## B · Major findings (block status=Implemented unless waived)

### B1 · Bridge file is not deprecated, not warned-about, still in active use

**Where**: `sim_bench/pipeline/steps/face_cluster_bridge.py` (228 LOC) — unchanged. `cluster_people` step still calls `run_face_cluster_knn` from it on every Albumify run. No `DeprecationWarning`, no log warning, no comment marking it for Phase 7 deletion.

**Recommendation**: at minimum, add a module-level `DeprecationWarning` on import + a note in the docstring saying "scheduled for deletion in spec-040 Phase 7." Without this, a new developer reading `cluster_people.py` doesn't know the bridge is going away.

### B2 · `face_cluster_legacy/` is a silent shim

`face_cluster_legacy/__init__.py` re-exports `FaceClusteringPipeline` etc. without raising `DeprecationWarning`. Old callers using `from face_cluster import FaceClusteringPipeline` and new callers using `from face_cluster_legacy import FaceClusteringPipeline` get identical behavior. The "rename" provides no migration signal.

**Recommendation**: `face_cluster_legacy/__init__.py` should `warnings.warn(DeprecationWarning(...), stacklevel=2)` on import, with a message pointing at spec-040 Phase 7 and `face_cluster.fc_app_runner`.

### B3 · New schema columns and tables are write-but-NULL

**Where**: `face_cluster/db/schema.py` adds `images`, `scene_clusters`, `scene_cluster_assignments` tables + 5 `*_ratio` columns on `faces`.

`RunExporter._write_faces_and_scores` does emit values for the new ratio columns — but always `None` because `getattr(face, "area_ratio", None)` returns `None` (FaceRecord doesn't have those attributes). The three new tables have no `_write_*` method at all. After a fresh run:
- `faces.area_ratio` etc. → all NULL
- `images` table → empty
- `scene_clusters` table → empty
- `scene_cluster_assignments` table → empty

Pandera schemas exist; nothing exercises them.

**Recommendation**: either (a) populate these in Phase 4 (requires producer changes — folds into A1), or (b) ship `RunExporter._write_images` and `_write_scene_clusters_and_assignments` as no-ops with TODO comments so the writer surface is at least complete.

### B4 · `FCAppRunner` isn't wired into any UI

**Where**: `face_cluster/fc_app_runner.py` is added; zero imports of it anywhere except the equivalence test. No Streamlit app uses it. The "new FC App" Phase 5 was supposed to deliver doesn't have a UI surface.

**Recommendation**: either (a) wire FCAppRunner into a new `app/face_clustering_v2/main.py` (real Phase 5 work), or (b) mark Phase 5 as "runner-only" in the spec status table and note Phase 5 UI work is deferred.

### B5 · The "no bridges" rule has a documented exception inside the new steps

**Where**: `sim_bench/pipeline/steps/face_clustering_steps.py:_build_fc_config()` (~40 LOC) translates the step's dict config to `FCConfig` dataclass. The docstring acknowledges this as "ONE residual translation" allowed because clustering algorithms still want `FCConfig`.

**Assessment**: this is documented and conscious, but it means the "no bridges" rule in spec.md isn't literally true today. Every clustering step calls `_build_fc_config(config)` at top — a translator function in disguise.

**Recommendation**: keep as-is for now (rewriting `QualityGater` / `KNNGraphBuilder` etc. is out of scope), but update spec.md's locked constraint to say "no bridges *between PipelineContext and the algorithm layer*; the algorithm layer keeps its `FCConfig` until a future spec replaces it."

### B6 · No Pandera `_write_*` invocations for the 3 new tables

**Where**: Pandera schemas `IMAGES_SCHEMA`, `SCENE_CLUSTERS_SCHEMA`, `SCENE_CLUSTER_ASSIGNMENTS_SCHEMA` are defined but never called from `RunExporter`. The spec-033 P-H pattern (validate before INSERT) is the contract this work was supposed to extend.

**Recommendation**: even no-op `_write_*` methods should call `.validate()` on the (empty) DataFrames so the contract is exercised in CI.

### B7 · Phase 3 step dependencies pull the chain into auto-resolve

**Where**: `face_clustering_steps.py` — each step except the first declares `depends_on=["<prev step name>"]`. `PipelineExecutor.execute(step_names, auto_resolve=True)` then walks the dep graph. If a caller invokes only `assign_people_clusters`, the builder pulls in the whole chain.

**Assessment**: probably fine — the chain is one logical unit. But the order is also enforced by `UNIFIED_CLUSTERING_STEPS` in `fc_app_runner.py` (literal list). Two sources of truth.

**Recommendation**: pick one. Either rely on `depends_on` and have FCAppRunner pass just the last step name, or remove `depends_on` and rely on the explicit list.

---

## C · Minor findings (acceptable; ticket if you want)

### C1 · `face_cluster_legacy/__init__.py` only covers 4 symbols
Existing callers of `face_cluster.FaceRecord`, `face_cluster.QualityGater`, etc. still use the canonical `face_cluster.*` paths. Only `FaceClusteringPipeline`, `PipelineResult`, `PipelineStageError`, `PipelineConfig` are aliased. The phase-1 "rename" is partial. **Acceptable** because those four ARE the FC-App-specific surface; the rest is shared infrastructure that stays in `face_cluster.*` regardless.

### C2 · `_FaceProxy` from Cursor's reverted code lives on in the test for `_collect_faces`
The reverted Cursor code is gone, but `tests/face_clustering/test_unified_clustering_steps.py` doesn't test producer integration (no test invokes `insightface_detect_faces` → expects face_records populated). Reasonable scope cut for the session but worth a ticket.

### C3 · `ApplyDiameterCapStep` is a no-op
The step exists and is registered, but its body just sets `cap_summary = {"note": "diameter cap step is a no-op until integrated", "applied": False}`. The spec-031 cap logic in `face_cluster/cluster_diameter_cap.py` is not invoked. Documented in the docstring but easy to miss.

### C4 · `score_iqa.py` is an empty BaseModel
`ScoreIQAConfig` has zero fields. The architecture test `test_every_field_has_description` passes trivially. Not wrong — `score_iqa` legitimately has no tunables — but a comment in the module would help future readers.

### C5 · No test for legacy-vs-v2 equivalence with `merge_enabled=True` · ✅ RESOLVED 2026-05-19
Subsumed by A2's parametrized sweep: the `merge_on` config (`{**CANONICAL_CONFIG, "merge_enabled": True}`) is one of the 4 sweep variants and asserts ≥0.95 pairwise agreement on the small fixture.

The equivalence test uses default config which has merge_enabled=False. The conservative merge logic is the most complex part of clustering; equivalence isn't proven for runs that exercise it.

---

## D · §-by-§ checklist verdicts

| Checklist § | Verdict | Notes |
|---|---|---|
| §1 Structure | **Pass with follow-up** | `face_clustering_steps.py` is 334 LOC, 8 step classes. Justified by cohesion (one logical chain). |
| §2 Code quality | **Pass** | No new bare `except:`, no deep nesting, no silent defaults beyond the documented `_build_fc_config`. |
| §3 Naming & package structure | **Pass with follow-up** | `face_cluster_legacy/` shim acceptable; `face_clustering_steps.py` plural is slightly inconsistent with peer steps using singular names. |
| §4 Layering & coupling | **Pass** | `sim_bench` → `face_cluster` imports (algorithm layer) are intentional and documented. |
| §5 Testability — load-bearing | **FAIL** | A2 — equivalence test is on synthetic only. No producer-integration tests. Real E2E (`test_albumify_e2e.py`) does NOT exercise the unified chain because A1. |
| §6 Boundary contracts | **FAIL** | B3, B6 — Pandera schemas defined but never called for the new tables. A1 — the Pydantic FaceRecord contract claimed at the producer→context boundary isn't honored because no producer writes there. |
| §7 Documentation deliverables | **Pass** | `spec.md`, `tasks.md`, `CONCRETE_PLAN.md`, `COVERAGE.md`, `ARCHITECTURE.html` all updated. CHANGES_LOG has a comprehensive entry. |
| §8 Risk register | **Pass with follow-up** | Risks were acknowledged in chat ("plow through" mode) and in CHANGES_LOG. But the gating in `spec.md` status table is informal — there's no CI block on "Implemented" status flip. |

---

## E · Follow-up tickets (filed)

| ID | Severity | Action | Where |
|---|---|---|---|
| FR-040-1 | Critical | Producer step(s) write to `context.face_records` (dual-write with legacy dicts) | `insightface_detect_faces.py`, `align_faces.py`, `extract_face_embeddings.py` |
| FR-040-2 | Critical | Real-album equivalence test on `test_data/face_clustering_100/` | `tests/face_clustering/test_legacy_vs_v2_equivalence.py` (extend) |
| FR-040-3 | Major | `DeprecationWarning` on bridge + face_cluster_legacy imports | `face_cluster_bridge.py`, `face_cluster_legacy/__init__.py` |
| FR-040-4 | Major | Write methods + Pandera invocations for `images`, `scene_clusters`, `scene_cluster_assignments` | `face_cluster/run_exporter.py` |
| FR-040-5 | Major | Wire `FCAppRunner` into a real Streamlit app (or split Phase 5 into 5a runner / 5b UI) | new `app/face_clustering_v2/` |
| FR-040-6 | Minor | `ApplyDiameterCapStep` invokes spec-031 logic | `face_clustering_steps.py` |
| FR-040-7 | Minor | Equivalence test with `merge_enabled=True` | `test_legacy_vs_v2_equivalence.py` |

---

## F · Recommended next action

**Do NOT** flip spec-040 to status=Implemented. Land the work as it stands on `unification/spec-040` (already pushed) and treat Phases 1–6 as **infrastructure**, not the unification itself.

The actual unification requires FR-040-1 (producer dual-write) and FR-040-2 (real-album equivalence). Estimate: 2–3 more focused days. After those land green, Phase 7 (delete legacy) can begin its 2-week burn-in.

Today's state is honest: scaffolding complete, contracts defined, the v2 chain provably correct on synthetic data. That's a defensible checkpoint to share with reviewers, just not the finish line.
