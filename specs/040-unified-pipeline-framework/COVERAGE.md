# spec-040 Coverage — open items absorbed or made moot

**Created**: 2026-05-16
**Purpose**: cross-reference between spec-040's scope and the existing open backlog. Items in the "absorbed" / "made moot" columns do not need separate work; items in "orthogonal" stay on the backlog.

## A · Directly absorbed (closes the item)

Spec-040 resolves these as a side-effect of its own scope. They should be marked closed when spec-040 lands.

| Item | Where it lives today | How spec-040 closes it | Phase |
|---|---|---|---|
| **FR-033-2** Config-to-producer graph contract | `specs/036-config-producer-contract/` | Bridge deleted; no gates without producers — structurally impossible. Architecture test still ships but starts with empty waiver list. | Phase 6 |
| **FR-033-3** InsightFace blur step + lift SIGHTING-061 pin | `specs/037-insightface-blur-step/` | Phase 4 lands the blur step + lifts the pin; Phase 6 deletes the bridge that held the pin. | Phase 4 + 6 |
| **FR-033-4** ExportRequest Pydantic bundle | `specs/038-export-request-pydantic/` | RunExporter migration to ExportRequest is required mid-way through Phase 5 (one writer for unified pipeline). | Phase 5 |
| **FR-033-5** Verify no duplicate filter_decisions rows | `SIGHTING-062` | One writer for filter_decisions after Phase 3; PK violation impossible. | Phase 3 |
| **FR-033-6** STEP_CONFIG_MODELS registry guard | `specs/039-step-config-registry-guard/` | Phase 2 makes every face-clustering step Pydantic-typed; CI guard activates with empty allowlist. | Phase 2 |
| **FR-033-7** Bridge pose-lookup operator precedence | `SIGHTING-063` | Bridge file deleted. | Phase 6 |
| **SIGHTING-060** `face.area` unit drift | sighting | Phase 4 adds canonical `*_ratio` columns; one unit across producers. | Phase 4 |
| **SIGHTING-064** `area_ratio` canonical column | sighting | Same as SIGHTING-060 — Phase 4. | Phase 4 |
| **SIGHTING-065** `images` table (image fields off `faces`) | sighting | Phase 4 schema v5 adds dedicated `images` table. | Phase 4 |
| **SIGHTING-066** Scene side has no structured persistence | sighting | Phase 4 schema v5 adds `images` + `scene_clusters` + `scene_cluster_assignments` + `scene_embeddings.npy`. Structural symmetry with faces side. | Phase 4 |

## B · Made moot (no separate work needed; problem disappears)

These items only exist *because* of the dual-framework duplication. After spec-040 the conditions don't exist.

| Item | Where it lives | Why it disappears |
|---|---|---|
| **spec-033 P-F config_diff CLI** | `face_cluster/config_diff.py` | `effective_config_from_albumify` / `effective_config_from_fc_config` split exists only to compare two config shapes. One shape after Phase 2; the CLI simplifies to a generic `compute(parent, child)` (existing function). |
| **Two run-history tables** (`action_log` + `pipeline_runs`) | `face_cluster/run_history_db.py` and `sim_bench/api/database/models.py::PipelineRun` | Two tables exist because two runners exist. After Phase 5, one runner — and the two tables can collapse to one (post-burn-in follow-up, not blocking the merge). |
| **"Persistent Run History in DB"** (2026-04-17 OPEN) | `FEATURE_REQUESTS.md` | Same — one runner, one history table. |
| **"Persist Parameter Defaults in App"** (2026-04-17 OPEN) | `FEATURE_REQUESTS.md` | One config shape → one `config_profiles` table backs both apps directly. |
| **"Face Clustering — Clean Architecture & Full Traceability"** (2026-03-24 OPEN) | `FEATURE_REQUESTS.md` | This IS the unification. Spec-040 supersedes this older ticket. |
| **"Face Clustering — Cohesive Tested Sub-Package"** (2026-04-01 OPEN) | `FEATURE_REQUESTS.md` | Overlapping intent; spec-040 supersedes. |

## C · Partially absorbed (spec-040 helps but doesn't fully close)

| Item | Where it lives | How spec-040 helps | Residual work |
|---|---|---|---|
| **"Pipeline Protective Layer — Step Error Capture + Degenerate Output Alerts"** (2026-04-30 OPEN) | `FEATURE_REQUESTS.md` | Unified framework has one place to wrap step execution and one place for degenerate-output checks. | Still need to actually write the wrapper + checks. Post-spec-040 follow-up. |
| **"Crop-stage observability columns"** (2026-05-11 OPEN) | `FEATURE_REQUESTS.md` | New schema v5 in Phase 4 is the natural moment to add crop-stage columns. | Decide which columns and define producers. Could fold into Phase 4 if scoped. |

## D · Orthogonal (stays on backlog independently of spec-040)

These don't depend on the dual-framework duplication. They live or die on their own merits.

| Item | Why orthogonal |
|---|---|
| spec-031 (max_diameter cap) | Algorithmic feature; lives inside the merge stage. Already implemented; needs UI surfacing. |
| spec-032 (filter_context) | Already implemented. |
| Trip Detection (2026-05-01) | Different feature axis (EXIF GPS + temporal clustering). |
| Merge Label Verification (2026-04-25) | UI feature for ML training; doesn't touch the pipeline framework. |
| ML Model Expansion (2026-04-22 Open) | Lives inside `face_cluster/ml_trainer.py`. |
| Session Operation Pipeline (2026-04-21 Open) | UX feature. |
| Cache Layer (LMDB + Parquet) (2026-02-20 Open) | Storage-tier rework, below the pipeline framework. |
| ML-Based Cluster Merging (2026-04-14 / 2026-02-27) | Algorithm work, not framework. |
| Various "Done" / "Implemented" items | Already shipped. |

## E · Item-level disposition table (single source of truth)

| ID / Title | Disposition | Action when spec-040 lands |
|---|---|---|
| FR-033-1 (specs/035) | **Already done** — landed 2026-05-16 on main | Mark closed; remove from TODO. |
| FR-033-2 (specs/036) | **Absorbed by spec-040 Phase 6** | Close spec; architecture test reuses spec-036's name. |
| FR-033-3 (specs/037) | **Absorbed by spec-040 Phase 4** | Close spec; blur step lands in Phase 4. |
| FR-033-4 (specs/038) | **Absorbed by spec-040 Phase 5** | Close spec; ExportRequest is part of unification. |
| FR-033-5 (SIGHTING-062) | **Absorbed by spec-040 Phase 3** | Close sighting; one writer makes duplicates impossible. |
| FR-033-6 (specs/039) | **Absorbed by spec-040 Phase 2** | Close spec. |
| FR-033-7 (SIGHTING-063) | **Absorbed by spec-040 Phase 6** | Close sighting; bridge deleted. |
| FR-033-8 (TODO) | Trivial, do it inside spec-040 Phase 7 doc cleanup | — |
| SIGHTING-060 | **Absorbed by spec-040 Phase 4** | Close sighting. |
| SIGHTING-064 | **Absorbed by spec-040 Phase 4** | Close sighting. |
| SIGHTING-065 | **Absorbed by spec-040 Phase 4** | Close sighting. |
| 2026-03-24 "Clean Architecture" | **Made moot by spec-040** | Mark superseded in FEATURE_REQUESTS.md. |
| 2026-04-01 "Cohesive Sub-Package" | **Made moot by spec-040** | Mark superseded. |
| 2026-04-17 "Persistent Run History" | **Made moot by spec-040 Phase 5** | Post-merge follow-up: collapse action_log + pipeline_runs. |
| 2026-04-17 "Persist Parameter Defaults" | **Made moot by spec-040 Phase 5** | Post-merge follow-up: use config_profiles for both. |
| 2026-04-30 "Pipeline Protective Layer" | **Partially absorbed** | Could fold into Phase 3 step wrapping. Decide before Phase 3. |
| 2026-05-11 "Crop-stage observability" | **Partially absorbed** | Decide in Phase 4 schema v5 design. |

## F · Net effect on the open backlog

**Before spec-040**: 5 open PRDs (specs/036-039 + spec-033 follow-ups not closed), 3 open sightings (062/063/064/065), several "Open" items in FEATURE_REQUESTS.md from 2026-03 / 2026-04.

**After spec-040 merges to main**: most of those collapse. Specs/036-039 are either closed or replaced by spec-040 phases. Sightings 062/063/064/065 close. Three older FEATURE_REQUESTS entries marked superseded.

**Estimated backlog reduction**: ~10 open items closed or made moot as a side-effect of spec-040 completing.
