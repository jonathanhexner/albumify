# Learnings Log

This file tracks lessons learned from bugs and issues to prevent repeating past mistakes.

---

<!-- Add new entries at the top, newest first -->

### 2026-07-12: spec-098 — a plausible fix that isn't measured is a guess (median blur was not enough)
**What happened**: the "obvious" noise fix (median-blur before Laplacian) only cut noise-inflation 39×→11× on real SIDD noise — the first validation re-run FAILED its own acceptance gates (62% vs ≥95%). The passing formula needed a second, measured idea (subtract the noise's own σ² contribution), found via two quick parameter sweeps against real data.
**Second lesson**: changing a metric's formula silently invalidates every absolute threshold tuned against it — the face blur gate (blur_min=150) over-rejected 4× until re-calibrated (→73.3) from paired old/new scores in the run DBs.
**Prevention**: (1) define acceptance gates BEFORE the fix and re-run the same real-data benchmark after; (2) grep for absolute thresholds on any score whose formula changes; (3) when a baseline check fails, stash-and-rerun on baseline code before attributing — 3 E2E failures + the unreproducible 15-cluster anchor all predated the change.

### 2026-07-09: spec-096 verdict — real positives are the entire game for rare-defect detection
**Root cause of every plateau**: data, not modeling. Prompt engineering (3 ensembles), hand-crafted features (3 rounds), synthetic training (2 attempts) all failed or stalled; meanwhile 5 new REAL positives moved the CLIP probe +0.10 scene PR-AUC (0.76→0.86) and lifted the worst fold 0.61→0.79.
**Lessons**: (1) learned boundary in embedding space ≫ zero-shot text, confirmed at whole-image AND crop level; (2) validate synthetic data by EYEBALLING what the model actually trains on (the "whole-photo positive crops" bug survived until the report's example images exposed it); (3) an LLM validator (Haiku, 0.17 as detector) is still worth its cost as a label-miner — it found the dark-occluder class and the one hidden positive.
**Prevention**: for the next rare-defect detector (subject motion blur, lens flare…), start with a capture session + adjudication loop, not with features or prompts.

### 2026-07-08: HEIC is a recurring blind spot — register pillow_heif at APP ENTRY, not per-module (SIGHTING-114)
**Root cause**: PIL (and thus Streamlit's `st.image`) cannot decode HEIC without `pillow_heif.register_heif_opener()`. Registration was added ad hoc to individual modules (`explain.py`, `saliency.py`, Track D) as each broke — so every NEW image code path (this time the adjudicate page's display) crashed on the 124 `.heic` files, blocking the user mid-adjudication at image 12.
**Lesson**: Second HEIC failure of this class (Track D silently skipped .heic; now the review UI crashed on it). Any path that opens dataset images must assume HEIC.
**Prevention**: register the opener once at app/process entry (done in `app/occlusion_review/main.py`); wrap per-image UI renders in try/except so one unreadable file degrades to a warning instead of blocking a whole worklist.

### 2026-07-05: Finger-occlusion detection needs a trained model — 5 whole-image/heuristic approaches all fail
**Root cause**: A finger over the lens is a LOCALIZED corner defect on an otherwise good photo. Whole-image scorers see the (good) main subject and ignore the corner; hand-crafted rules can't separate a blurry warm finger blob from a warm smooth WALL.
**What was tried (all on the 2 `examples/finger_occlusion/` images vs the 122 Budapest album)**:
1. No-reference IQA (BRISQUE/MANIQA/MUSIQ/HyperIQA/CLIP-IQA/NIQE) — rate the finger photos MIDDLE-to-GOOD (HyperIQA put them in the top 6%). Don't flag it.
2. CLIP-prompt "clear vs finger-over-lens" (`clip_occlusion`, whole-image) — finger P(clear) 0.37/0.32 >= some clean images. Tiling (worst 3x3 tile) only marginal.
3. Classical 4-cue (relative-blur outlier + low-texture + warm/skin + large border-connected blob) — FLAGS the fingers, and correctly rejects cool lakes/sky (warmth cue), but WARM SMOOTH WALLS (beige/white museum walls) score higher than the fingers.
4. + per-cell edge-free cue — measured no help: a flat wall INTERIOR is as edge-free (~0.001-0.003) as a finger.
5. + ring-around-blob sharpness cue (reject blobs surrounded by sharp in-focus frames) — removed the worst wall (27->20 flagged) but 4 warm-wall/night-shot false positives still outrank the fingers (rank 6th/10th of 124).
**Lesson**: Each hand-crafted cue nudges but none cleanly separates finger-over-lens from warm smooth surfaces — the boundary is too subtle for rules. Stop adding cues (diminishing returns).
**Prevention / next step**: Use a LEARNED boundary — cheapest is CLIP image embeddings + a logistic-regression linear probe on ~40-80 labeled finger/clean images (zero-shot CLIP prompts fail, but the embedding still encodes it). `clip_occlusion` is kept in the studio (spec-094) labeled "experimental (does not reliably flag occlusion)".

### 2026-06-29: Multiple OpenCV PyPI variants silently corrupt the cv2 install
**Root cause**: `opencv-python`, `opencv-contrib-python`, `opencv-python-headless` are separate PyPI names but all unpack into the SAME `site-packages/cv2/` dir — last installer wins. Different deps pull different variants (mediapipe→contrib, ultralytics/facexlib→plain, pyiqa/albumentations→headless), so three accumulated. The live build was headless (GUI:NONE) even though pip listed the GUI package as installed — pip metadata != disk truth. Also caused a `WinError 5: cv2.pyd Access denied` when swapping while an app held it loaded.
**Lesson**: No single opencv package satisfies all dependents' declared *names*, so `pip check` will always warn — but functionally they all just `import cv2`, which any one variant provides. Install exactly ONE (the superset `opencv-contrib-python`); treat the residual name-mismatch warnings as cosmetic after verifying the real importers load.
**Prevention**: One opencv pinned in `setup.cfg` (`opencv-contrib-python>=4.8,<4.12`); SIGHTING-111 documents the diagnosis + the "name vs import" rule. When a heavy ML dep is added, `pip check` for opencv duplication.

### 2026-06-25: bbox unit mismatch (pixels vs [0,1]) silently degrades to a placeholder
**Root cause**: spec-079's `FaceRecord.bbox` is in PIXELS; `people_service._bbox_to_xywh` only reshaped it (its docstring lied — said "normalize") while every Streamlit consumer multiplies by image dims, assuming [0,1]. Pixel×dims goes off-canvas → PIL crop collapses → `_load_face_thumbnail` returns None → gray avatar for ALL people. No exception, no log — just a silently wrong image.
**Lesson**: A geometry value crossing a module boundary needs its UNIT in the contract, not just its shape. "xywh" is ambiguous; "xywh normalized [0,1]" is not. Helpers named "normalize" must actually normalize.
**Prevention**: `_normalized_bbox` is now the single normalizer (uses spec-040 `bbox_*_ratio`/`image_*_px`); consumers guard `max(bbox) > 1.5`; `tests/api/test_people_service_bbox.py` asserts [0,1] output. SIGHTING-104.

### 2026-06-06: `st.dataframe` row-select is NOT "click the thumbnail" (3× user frustration)
**Root cause**: v2 used `st.dataframe(on_select="rerun", selection_mode="single-row")` + `ImageColumn` for "clickable thumbnails." Glide-data-grid renders on a canvas and only the ~20px row-select **checkbox column** fires the selection event — clicking the thumbnail or any data cell does nothing. Users reported "I can't click on faces/images" three separate times because the only working target was invisible.
**Lesson**: For click-to-open in this Streamlit app, use a real `st.button("Open")` under each thumbnail (the `face_grid.py` pattern), not dataframe row-select. Bonus: real buttons are Playwright-addressable, so the click path gets actual e2e coverage (scenarios J/K) — the canvas never could.
**Prevention**: spec-083 added `face_pick_grid` / `image_pick_grid`; auto-memory `feedback_clickable_buttons.md`.

### 2026-05-15: Long pointless explanations are unacceptable
**Root cause**: Repeated user feedback ("I didn't understand anything", "this isn't simple", "you're again opting for long unclear explanations") on responses that buried the answer under recap, framing, and symmetric "what was supposed / what actually" templates.
**Lesson**: Direct answer first. One short paragraph or tight bullets. No restating the question. No "let me explain". Caveats only if asked.
**Prevention**: Pasted to auto-memory (`feedback_concise.md`) so future sessions inherit it.

### 2026-05-15: "Plumb the field through" needs an upstream producer
**Root cause**: spec-033 P-C C-1 removed the bridge's hardcoded force-disable of `blur_min` on the assumption that `face_cluster_bridge.faces_to_face_records` could recover `blur_score` from `context.insightface_faces`. It can't — the active InsightFace pipeline has no blur-scoring step. So every FaceRecord's `blur_score` stayed 0.0 and `cluster_people.blur_min=50.0` from `configs/pipeline.yaml` rejected 100% of faces (SIGHTING-061).
**Lesson**: A boundary fix that "stops dropping the field" is only valid if a producer is actually writing the field somewhere upstream. Removing the downstream force-disable BEFORE verifying upstream computation = silent total failure. The pose 3-tuple has the same issue (only a scalar `pose_score` exists; the bridge can't synthesize yaw/pitch/roll), saved only by the fact that the QualityGater is permissive when pose=None.
**Prevention**: When unlocking a gate at a boundary, audit both ends: (a) does an upstream step compute the field? (b) does the bridge read it under the right dict key? If either is missing, the bridge must pin a permissive default with a docstring explaining what's missing. Architecture test asserts the docstring stays in place so the pin can't be "fixed back" without the upstream step landing first.

### 2026-05-09: Strict-write contracts catch real producer-side bugs
**Root cause**: While building the new `RunExporter` for spec-030, the strict-key check (`merge_log[i] keys must equal MergeDecisionRow.field_names()`) refused to accept the on-disk `merge_log.json` from `face_clustering_20260508_000446`. Investigation found 4 rows in the terminal iteration missing `actually_merged` — the early-return path in `merge.py:_select_best_merge` skipped the stamping step when no valid merges existed. SIGHTING-057's fix had stamped `actually_merged` only on the winner-selection branch, leaving the no-winner branch incomplete.
**Lesson**: A strict-write that rejects unknown / missing keys is a contract enforcer, not a nuisance. The first thing it caught was a real bug that would otherwise have stayed hidden behind a `dict.get(key, default)` call. Default values mask producer bugs as long as the consumer happens to be tolerant.
**Prevention**: Keep `_strict_validate_merge_log` strict. Resist any request to add `dict.get` defaults to the writer. The fix is to make the producer correct, not the writer permissive.

### 2026-05-09: Duplication is the disease, schema mismatch is the symptom
**Root cause**: SIGHTING-058 first appeared as "DB schema is missing 5 columns vs the JSON". The instinct was to add the columns or "prefer JSON". Both miss the point. The architecture wrote the same logical data (merge log) to two places (JSON + SQLite) with no declared owner; whichever copy the loader picks would have its own drift over time. Fixing the symptom would have left the structure that produces the symptom.
**Lesson**: When the same fact is written to two stores, one of them is structurally wrong — pick the one that matches the data shape (relational → SQL; bulk numeric → npy; blob → file) and delete the other. Existence-check fallbacks (`if foo.exists(): use foo else bar`) are a tell that no one declared an owner.
**Prevention**: spec-030 proposes a single `RunExporter` (writer) and `RunStore` (reader); Phase 4 deletes legacy artifacts; tests guard against new `if .exists()` chains in `RunStore` and against direct file reads in `app/`.

### 2026-05-02: Do not report a feature as done when you know it's unfinished
**Root cause**: Bounding boxes were discussed 3+ times, planned for 4 locations (popup, People, Explore, Results), but only implemented in 1 (People detail). Each time, reported "done" knowing the other 3 were skipped. The CLAUDE.md rules allowed this because they checked for bugs but not for known-incomplete work.
**Lesson**: If you know a feature needs to be in 4 places and you only did 1, say "25% done" — do not say "done." Either finish the work or be explicit about what's missing.
**Prevention**: Added rule 7 to CLAUDE.md Delivery Quality.

### 2026-05-01: Never return to user with partial fixes — fix everything in one pass and verify end-to-end
**Root cause**: Repeatedly shipped fixes that addressed one symptom but left others broken, requiring 10+ iterations for simple sightings. Example: fixed person naming but didn't check bounding boxes. Fixed step_decisions in DB but not in API schema. Added thumbnails to Explore but not to Scene Clustering.
**Lesson**: When the user reports N issues, fix ALL N in one pass. Then restart the app, run Playwright against the real app, take screenshots, and visually verify every fix before reporting. The user should never be the one discovering that a fix didn't work.
**Prevention**: Added "Delivery Quality" section to CLAUDE.md with a 6-step checklist that must be completed before reporting completion to the user.

### 2026-05-01: Exploration UIs are useless without image thumbnails
**Root cause**: Built Explore tabs as pandas DataFrames with filenames and numbers. No thumbnails, no click-to-inspect, no visual context. Users need to SEE the images to understand pipeline decisions.
**Lesson**: Every UI that references an image must show that image inline. The face_clustering app already does this correctly (face crops in every table, popup on click). Should have reused that pattern.
**Prevention**: Before shipping any image-related UI, verify: (1) can the user SEE the image? (2) can they CLICK for details? If no, the UI is incomplete.

### 2026-05-01: Step decisions not reaching Explore page despite being emitted
**Root cause**: Added `step_decisions` to `get_result()` but `list_results()` (which Explore actually calls) may not include it. The field exists in DB but the API method the UI calls doesn't return it.
**Lesson**: When adding a field to the data pipeline, trace the FULL path from write to read. Verify every API method that could return this data. Write an integration test that runs the pipeline AND reads the result back.

### 2026-04-30: config.get("key", default) is useless when the YAML explicitly sets the key
**Root cause**: Fixed `blur_min` by changing `config.get("blur_min", 0.0)` default from 50 to 0. But `pipeline.yaml` had `blur_min: 50.0`, so `config.get()` found the YAML value and returned 50. The "fix" did nothing — quality gating still rejected 100% of faces for 3 iterations of debugging.
**Lesson**: When a bridge between subsystems must FORCE-DISABLE a parameter (not just default it), hardcode the value directly instead of using `config.get()`. `config.get(key, default)` only uses the default when the key is ABSENT, not when it has an inappropriate value. Write a test that exercises the full pipeline chain (quality gating → graph → clustering → labels) and assert non-trivial output.
**Prevention**: Added `test_multiple_identities_produce_multiple_clusters` that runs the full `_run_face_cluster_knn` with synthetic data and asserts multiple clusters. This test caught the bug immediately when the previous "fix" didn't.

### 2026-04-30: In Streamlit, variable defaults must come BEFORE conditional widget blocks
**Root cause**: `fc_K = 5` was placed AFTER the `fc_K = st.slider(...)` conditional block, overwriting the slider value. The config dict always had `K=5` regardless of user input.
**Lesson**: In Streamlit's top-to-bottom execution model, initialize all variable defaults at the top of the section, then let conditional widget blocks override them. Never set defaults after widget rendering. This class of bug is invisible in the UI (slider shows correct value) but causes wrong behavior (config uses wrong value).
**Prevention**: Added AppTest-based regression test (`test_fc_K_slider_value_reaches_config_dict`) that captures the actual config dict via `_start_pipeline` interception and asserts the value matches the slider.

### 2026-04-30: Bridge functions between subsystems must disable quality gates that depend on unavailable inputs
**Root cause**: `_faces_to_face_records` in `cluster_people.py` correctly defaulted `blur_score=0.0` (unavailable from main pipeline), but `FCConfig` defaulted `blur_min=50.0`, silently rejecting 100% of faces. The step returned all-noise with no exception — just a WARNING log invisible in the UI.
**Lesson**: When bridging two subsystems, explicitly audit which quality thresholds depend on fields the source system doesn't populate, and default those thresholds to "disabled" (0.0, None) in bridge context. Silent degenerate output (0 clusters, all noise) is harder to diagnose than a crash.
**Prevention**: Add post-step sanity checks (degenerate output alerts) to pipeline engine; surface as warnings in UI. File: feature request "Pipeline Protective Layer".

### 2026-04-27: st.dataframe is incompatible with per-row inline images + action buttons
**Root cause**: Replacing per-row `st.columns` rendering with `st.dataframe` eliminates the ability to embed images and interactive buttons inline. The performance problem (too many widgets) and the UX requirement (inline images + per-row buttons) are in direct conflict — switching to `st.dataframe` solved one but broke the other.
**Lesson**: Before replacing a rendering approach for performance reasons, explicitly verify that the replacement supports all required interactive features. `st.dataframe` is appropriate for read-only browsable data; labeling UIs with per-row actions need a different strategy.
**Prevention**: The correct fix is to benchmark removing `st.rerun()` from fragment button handlers first (the cheapest change), before redesigning the rendering model.

### 2026-04-27: p10_cross_dist ~ 0.60 is the empirical merge/reject threshold
**Observation**: Inspecting cluster pairs in the Label Verification tab, p10_cross_dist ≈ 0.60 reliably separates merge candidates from rejects. P10 is robust because it discards the noisiest 90% of cross-pair distances (bad poses, occlusions) and focuses on the closest subset.
**Implication**: p10 < 0.60 alone is a strong single-feature baseline. The ML model's value is in the edge cases — where size, intra-cluster compactness (inter_over_intra), or distribution shape (p10_over_p50) override the simple threshold.
**Action**: Added P10_THRESHOLD = 0.60 to both EDA notebooks with distribution plots and F1 sweep. Added derived features: p10_over_p50, inter_over_intra.

### 2026-04-25: Features should measure pair relationship, not dataset properties
**Root cause**: t_local and t_global measure how spread out clusters are (a property of the dataset/algorithm), not how similar two clusters are to each other. The ML model used them as proxies for "which dataset is this" rather than learning merge-relevant patterns. Similarly, blur, pose, and area features describe face quality, not inter-cluster distance.
**Prevention**: Restrict ML features to those that directly measure the relationship between two clusters: distance metrics (min_exemplar_dist, min_cross_dist, percentiles), merge consequences (post_merge_diameter, diameter_expansion), and evidence breadth (support_fraction).

### 2026-04-25: t_global is constant per run — leaks run identity into ML models
**Root cause**: `t_global` (median of per-cluster P90 intra-exemplar distances) is computed once per run and broadcast to every pair. In multi-run training, the model uses it as a run identifier, not a per-pair merge signal. Similarly, duplicate runs (identical merge decisions under different names) inflate training data without adding information.
**Prevention**: Use ratio features (`dist_over_t_global`) instead of raw `t_global`. Hash merge decisions to deduplicate runs before training. Always validate that features vary per pair, not just per run.

### 2026-04-25: Relative notebook paths break when notebooks move to subdirectories
**Root cause**: `eda_merge_explore.ipynb` used `Path("../results/Germany_18")` — correct when the notebook was in `notebooks/` but wrong after moving to `notebooks/face_clustering/`. All crops appeared "missing" because the path resolved to a non-existent directory.
**Prevention**: Use `PROJECT_ROOT = Path("../../")` relative to the notebook's known depth. Always verify resolved paths with `path.resolve()` and print them for debugging.

### 2026-04-25: Merge labeling must use transitive closure, not direct merge-log lookup
**Root cause**: The merge pipeline iteratively merges clusters — if A+B merge (iter 1) and then A+C merge (iter 2), the merge log records (A,B) and (A,C) but not (B,C). Feature computation uses pre-merge cluster IDs, so pair (B,C) exists in the feature matrix but gets incorrectly labeled as "reject" (false negative).
**Prevention**: Use union-find on merged pairs to build transitive closure before labeling. Any pair whose clusters end up in the same identity group should be labeled positive.

### 2026-04-24: Hardcoded magic numbers in view layers mask config-driven behavior
**Root cause**: `merge_threshold = 0.45` was hardcoded in `cluster_view.py` and `run_overview.py` instead of reading the actual `merge_candidate_threshold` from the run's config. User's config had 0.84 but UI showed 0.450, causing incorrect merge candidate flags.
**Prevention**: View/analysis modules must never hardcode thresholds — always read from `result.summary["config"]`. When adding a configurable threshold, grep for hardcoded values in all consumers. Consider adding a contract test that verifies view outputs change when config values change.

### 2026-04-24: Quality gate fields computed and persisted but never consumed by gating logic
**Root cause**: `det_score` was captured in `FaceRecord`, exported to `faces.csv`, and visible in the UI, but `QualityGater._evaluate_gates()` was never updated to include it as a gate. The field lived in two different modules (types + quality) with no mechanical link forcing them to stay in sync.
**Prevention**: When adding a new per-face quality metric to `FaceRecord`, immediately add a corresponding gate entry to `_evaluate_gates()` — or add a comment explaining why it is intentionally not gated. Never let a quality signal be "computed but not gated" silently.
**Lesson**: "Decisions computed then discarded" (see 2026-04-22 entry) applies to consumers too: a persisted field that is displayed but not acted on is a latent quality regression.

### 2026-04-23: App refactors invalidate test import paths and AppTest APP_PATH simultaneously
**Root cause**: Refactoring `app/face_clustering.py` (monolith) to `app/face_clustering/` (package) changed 3 categories of test dependency: (1) `APP_PATH` constant in AppTest tests, (2) import paths for functions that moved to submodules, (3) `patch()` target strings. All three failed silently until the test suite was run.
**Prevention**: After any app restructuring, grep for old module path in `tests/` before declaring the refactor done. E2E browser tests that require a live server must be excluded from the default pytest run via `addopts = "-m 'not e2e'"` and a registered mark.
**Lesson**: `AppTest` from `streamlit.testing.v1` runs in the same process — `patch()` works — but only if the patch target matches the module's actual import binding (patch where used, not where defined).

### 2026-04-23: Output-path uniqueness must be enforced at the allocation site, not at the call site
**Root cause**: Every action-dispatch in the app independently computed output paths using convention-based naming (e.g. `merge_remerge_<n>`). No central actor checked for existence, leading to silent overwrites (SIGHTING-025).
**Prevention**: `allocate_run_dir` is now the single path-allocation authority; it atomically reserves paths via a DB table with a UNIQUE constraint. All dispatch paths call it instead of computing their own paths.
**Lesson**: "Unique path" logic must be in a single function with an atomic reservation contract — never duplicated across callers.

### 2026-04-22: "Decisions computed then discarded" is a recurring bug pattern
**Root cause**: Spec 012 identified that `QualityGater`, `D10ExemplarSelector`, and `InsightFaceEmbedder` all compute rich per-face data (gate verdicts, d10 values, det_score) that is consumed transiently but never persisted. The same pattern led to SIGHTING-022 and SIGHTING-024. Each new pipeline stage must expose its verdict in output files, not just use it internally.
**Prevention**: Rule: every pipeline stage that makes a decision must write evidence for that decision to the stage's output file before returning. No stage should have side-effect-only decisions.

### 2026-04-22: Apply + Remerge with relaxed config silently absorbs noise into manually-merged clusters
**Root cause**: In `Noa2_5_1/merge_remerge_1`, the user approved 14 pairs that correctly merged 8 base clusters (183 core faces). The subsequent "Apply + Remerge" ran `pipeline.recluster` with much looser config than the base run (`distance_threshold=0.35 → 0.5`, `K=5 → 50`, `merge_exemplar_threshold=0.35 → 0.52`). The loose kNN graph absorbed ~160 previously-noise faces — including face 765, top exemplar of base cluster 9 (a different identity) — into the mega-cluster, which grew 183 → 343. The user had **no visibility** that (a) config changed vs the parent run, or (b) the cluster's membership grew dramatically from non-approved sources.
**Prevention**: (1) Show config **diff** vs parent run in the History tab before any Apply + Remerge. (2) Flag clusters whose size changes >50% between runs. (3) Record per-face "joined_via" lineage (original clustering vs remerge kNN absorption) so the user can distinguish deliberate merges from threshold-driven absorption.
**Also surfaced**: Our 4-gate heuristic was pessimistic on the correct merges (e.g. `1↔10` with `n_gates_passed=1, margin_gap=-0.21` was a legitimate same-person pair). Gate evidence is a weak signal near the decision boundary — ML model (spec 010/011) is the right direction; user annotations on confirmed merges should feed back as positive training pairs.

### 2026-04-22: Run directory naming collides across sessions; no overwrite protection
**Root cause**: Apply + Remerge names its output `merge_remerge_<n>` by fixed convention. When invoked twice from different parent runs within the same album, the second invocation silently overwrites the first. `merge_log.json` went from 41 KB of real decisions to `[]` as a result.
**Prevention**: Central run-naming helper that checks for existence and auto-suffixes (`_2`, `_3`, ...); all session actions must route through it rather than constructing directory names locally. See SIGHTING-025.

### 2026-04-21: ML feedback loops require explicit source tracking, not just label filtering
**Root cause risk**: When ML model pre-fills decisions and those decisions are saved to the training DB unmodified, the model trains on its own predictions — a feedback loop that degrades diversity. A pure label filter (`decisions == "approve"`) does not distinguish human from ML-sourced decisions.
**Prevention**: Track decision source (`"human"` vs `"ml"`) in a parallel `merge_decision_sources` dict. Training DB writes filter on `sources.get(key) == "human"` only. When a user overrides an ML suggestion, the source changes to `"human"` and that pair IS saved — this is the correct signal. Undecided pairs (absent from decisions) are also excluded automatically since they have no source key.

### 2026-04-21: `compute_ml_merge_view` must surface `pair_features` for downstream rendering
**Root cause**: Feature contribution display in gallery cards requires the per-pair `ClusterPairFeatures` objects computed inside `compute_ml_merge_view`. Initially these were discarded after prediction. Adding `pair_features: Optional[Dict] = None` to `MergeAnalysisView` and populating it in the function avoids re-computing features on demand in render code (which would be expensive and duplicate logic).
**Prevention**: Any async compute function that produces intermediate data needed by the rendering layer should include that data in its return value, not require a second compute call. Store expensive results as session state once, read many times.

### 2026-04-20: Snapshot writers must apply user decisions before writing, not just store them as metadata
**Root cause**: `save_manual_merge_snapshot` saved `approved_pairs` in `pipeline_run.json` metadata but wrote `faces.csv` from the unmodified `merged_cluster_result` (ConservativeMerger output). The remerge pipeline loaded this file and ran ConservativeMerger again from the same state, rejecting the same pairs the user had approved.
**Fix**: Apply `approved_pairs` via union-find to produce new cluster assignments BEFORE writing any output files. Metadata storage is for audit/lineage, not for deferred execution.
**Prevention**: Any writer that accepts user decisions must apply those decisions to the data state immediately. "Save for later" patterns that pass decisions through a pipeline stage without applying them first are a recurring bug class. Add a test that reads the writer's output and asserts the decisions are visible in the data (not just in metadata).


### 2026-04-18: Remerge _load_source_remerge must populate cluster_stats before merge stage runs
**Root cause**: `_load_source_remerge` in pipeline.py set `cluster_stats={}` when constructing the `ClusterResult` from a snapshot. The merge stage's `_evaluate_merge_evidence` accesses `cluster_result.cluster_stats[cluster_id_a]` without `.get()`, causing `KeyError(cluster_id)`. The error message was just the cluster ID integer (e.g., "1"), making it cryptic: "Stage 'merge': 1".
**Fix**: Compute `cluster_stats` (diameter, median_dist, etc.) from the distance matrix for each cluster in `_load_source_remerge`. Also made merge.py defensive: `cluster_result.cluster_stats.get(cluster_id_a, {}).get('diameter', 0.0)`.
**Prevention**: Any code path that constructs a `ClusterResult` must populate all fields that downstream stages depend on. `cluster_stats={}` is never safe — downstream code assumes the dict is keyed by cluster_id.

### 2026-04-18: E2E tests that run FaceClusteringPipeline pollute the production DB
**Root cause**: Every `FaceClusteringPipeline().run(...)` call writes to `~/.sim_bench/sim_bench.db` because `run_history_db.get_db_path()` had no test override. 200 stale pytest-temp-dir entries accumulated, appearing in the app's History tab with non-existent output paths ("remerged", "remerged_ex", etc.).
**Fix**: Added `tests/face_clustering/conftest.py` with session-scoped autouse fixture patching `get_db_path` to a temp DB. Deleted 200 stale entries from production DB. `test_pipeline_history_hook.py` per-test monkeypatch correctly overrides the session fixture.
**Prevention**: Any module writing to a shared singleton (DB, cache) must have its path injectable. New pipeline-running tests must verify they write to isolated storage — add conftest pattern to all future test suites that exercise the pipeline.

### 2026-04-17: E2E test assertions must account for quality gate effects on test data
**Root cause**: `test_produces_three_clusters` asserted exactly 3 clusters for 3 persons, but 2 of person_2's 3 images have poses outside production thresholds (roll=35.8°, yaw=-30.3°). Only 1 core face from person_2 survived, which can't form a cluster (min_cluster_size=2). The test was always wrong for this data.
**Fix**: Changed assertions to "one cluster per person with ≥ 2 core faces" — derived from the data rather than assuming all persons have enough quality images.
**Prevention**: E2E test data must include ≥ 2 frontally-facing images per person (yaw<30°, pitch<25°, roll<25°). Add a test fixture validator that checks core-face counts before running clustering assertions.

### 2026-04-17: Pipeline runs never recorded in history DB
**Root cause**: `start_action("pipeline_run")` / `complete_action` were never called in the app's worker done/error blocks; only `merge_apply` and `model_load` were wired up.
**Fix**: Added `start_action` before `_AsyncState.start()` and `complete_action`/`fail_action` in the render-thread done/error blocks. Stored action_id in session_state; cleared to `None` after recording to prevent double-writes.
**Prevention**: When adding a new DB-tracked action type, immediately wire up start+complete+fail in the same PR. Add an integration test that asserts the action appears in `list_actions` after a run.

<!-- Format:
### YYYY-MM-DD: Brief title
**Root cause**: What caused the issue
**Prevention**: How to avoid in future
-->

### 2026-04-17: Stale done-worker silently hijacks History-loaded result
**Root cause**: Recluster tab promotion condition `pipeline_result.output_dir != output_dir_str` fires on every rerun, including after History loads a different run. First attempted fix (clearing `recluster_worker` in `_invalidate_run_caches()`) introduced a regression: Run tab's done-handler also calls `_invalidate_run_caches()` when `pipeline_result is None`, clearing a freshly-started recluster worker.
**Correct fix**: Track `recluster_promoted_dir` separately; only promote a recluster result once per unique output dir. Never add an active worker to `_invalidate_run_caches()` — use per-result tracking instead.
**Prevention**: Promotion logic for any worker must be idempotent (one-shot per result). Use a `promoted_dir` flag in session_state rather than comparing against live widget values.

### 2026-04-11: Questions are not implementation requests
**Root cause**: User asked "do changes require a full rerun?" — a pure question. Claude treated it as an implicit request and added a fallback code change, then reverted it when corrected, compounding the problem.
**Prevention**: CLAUDE.md now has an explicit rule: never write/modify/revert code unless the user specifically asks. Answer questions with answers.

### 2026-04-08: Every index that goes into a CSV must be a real ID, not a positional index
**Root cause**: `export.py` remapped cluster membership from graph-local indices to face-list indices (lines 52-62), but wrote `exemplar_face_ids` raw from `cluster_result.exemplars` without the same remap. Same class of bug as SIGHTING-015 — two index spaces (graph-local 0..n_core-1 vs face_ids) coexist, and any consumer that skips the translation gets wrong data.
**Prevention**: (1) Any value written to a CSV column with "id" in the name must be verified as a real entity ID, not a positional index. (2) Contract test: for every writer, assert that IDs in the output exist in the reference data (e.g., exemplar_face_ids are members of the cluster). (3) The pattern of "remap clusters but forget exemplars" suggests the remap logic should be a single function applied to the entire ClusterResult, not separate per-field code in the export.

### 2026-04-07: Code added between stages must be inside error handling — or it becomes invisible on crash
**Root cause**: SIGHTING-015 fix added a remap block between exemplars and export stages, outside any try/except. It used `np.full()` but `numpy` was never imported. NameError killed the process; `pipeline_run.json` stayed `status: "running"`, `error: null`. No stage recorded the failure. Telemetry showed exemplars "done" and export never started — but no error.
**Prevention**: (1) Never put code between stage try/except blocks — either fold it into an adjacent stage or make it its own stage. (2) Use a single `_execute_stage()` runner so there's no way to forget wrapping. (3) Add a `_finalize()` safety net that marks any still-"running" record as "failed". (4) Declare all stages upfront as "pending" so you can see which never started.

### 2026-04-07: History tab must validate run completeness before allowing load
**Root cause**: Pipeline run Germany_6 crashed after exemplars stage but before export. `pipeline_run.json` existed with `status: "running"`, so History tab listed it. User clicked "Load" and got cryptic `FileNotFoundError: faces.csv not found`. No indication the run was incomplete.
**Prevention**: Always check both run status AND required output files before enabling load actions. Show explicit warning with missing file list and disable the button. Never trust `pipeline_run.json` presence alone as proof of a successful run.

### 2026-04-07: Two code paths producing the same data must use the same index convention
**Root cause**: `pipeline.py` returned `ClusterResult` with graph-local node indices (0..n_core-1). `loader.py` rebuilt it from CSV with face-list indices. Both paths feed the same analysis views, but one was wrong. All live-run cluster displays showed random faces; history loads were correct. Bug was invisible for months because most testing used history loads.
**Prevention**: When two code paths produce the same data structure, add an assertion that validates the invariant (e.g., `assert all(idx < len(faces) for nodes in clusters.values() for idx in nodes)`). The loader should be the reference implementation — test that `pipeline.run()` output matches `load_pipeline_result(output_dir)` in a round-trip test.

### 2026-04-05: Check what a model already produces before building a replacement
**Root cause**: InsightFace buffalo_l has always run `1k3d68` and returned `face.pose`. Instead of reading that field, embedding.py implemented a broken 5-point heuristic. The heuristic was so wrong that the quality gate was disabled entirely. The fix was one line.
**Prevention**: When integrating a model, print all attributes it returns before writing any downstream logic. Never assume a model only produces what the docs highlight.

### 2026-04-04: Persist model outputs at export time — never re-run a model to reload results
**Root cause**: `export.py` saved cluster assignments and crops but not embeddings. `_load_result_from_dir` in the app then had to re-run InsightFaceEmbedder (minutes of work) to reconstruct embeddings for analysis views. This was not an intentional design — it was an omission that was patched with a workaround instead of fixed at the root.
**Prevention**: Any output that a downstream consumer (app, analysis, training) will need must be written to disk by the export stage. If a consumer is re-computing what a producer already computed, that is a sign of a missing export. Rule added: export stage owns all persistent data; consumers load, never recompute.

### 2026-04-04: Session state must never be written from a background thread
**Root cause**: `_run_pipeline` wrote to `st.session_state` from inside a daemon thread. Streamlit's session state is tied to the render context; writes from background threads cause `ScriptRunContext` warnings and may silently fail under concurrent renders.
**Prevention**: Background thread stores result in `worker.result`. Render thread reads `worker.is_done` and applies results to session_state. This is now codified in the CLAUDE.md async pattern.

### 2026-04-04: Reader must be tested against writer — never assume a file format
**Root cause**: `_crop_for_face()` assumed `crop_manifest.json` had format `{id: {crop_path: ...}}`. Actual format written by `crops.py` is `{id: "path_str"}`. No test exercised the read path. The writer and reader were never co-tested: tests verified the writer (crops.py), and tests verified the analysis layer (analysis_views.py), but nothing tested that the app UI could actually load a crop image from the manifest.
**Prevention**: (1) For every file format shared between a writer module and a reader module, add a contract test that writes via the writer and reads via the reader in the same test. (2) Any function that reads a persistent file format is not tested until the test opens the actual file and validates the data type of each field — not just that the file exists. (3) CLAUDE.md rule: "Before writing any code that reads a file written by another module, read that module's writer first."

### 2026-04-03: Test data shape must match production data shape — solo portraits hide multi-face bugs
**Root cause**: `face_id_counter` is a local variable inside `detect_and_embed()`, resetting to 0 per call. Pipeline calls it once per image. Bug was invisible in 9 solo-portrait E2E tests because (a) 1 face/image means face_id=0 always, (b) clustering uses list index not face_id, (c) purity/completeness tests use `image_path` not face_id. 1230 real photos exposed it immediately — only 12 unique IDs across 1103 faces.
**Prevention**: E2E test data must include at least one group photo (multiple faces per image). Always add a `face_id uniqueness` assertion. After any pipeline fix, validate on real album data before declaring done — not just synthetic single-face data.

### 2026-04-01: Optional dependencies that fail silently can reject 100% of data
**Root cause**: `QualityGater` used SixDRepNet for pose estimation. When unavailable, all faces got `pose=None`. The pose check only ran when `pose is not None`, so it was silently skipped — but the bug was that landmark-based heuristic poses (stored earlier in FaceRecord) had inflated pitch/yaw values that failed strict thresholds, causing all 1094 faces to be rejected. No fatal error, just `core=0`.
**Prevention**: (1) When an optional dependency is unavailable, log a clear one-time WARNING stating which filter is being skipped. (2) Add a `require_pose` config flag — default False so the pipeline degrades gracefully. (3) After quality gate, assert `len(core_indices) > 0` immediately and raise with a helpful message listing which thresholds were active. Never let the pipeline silently produce zero core faces.

### 2026-04-01: A "library" without a single entrypoint is not a library — it's a collection of scripts
**Root cause**: face_cluster/ had 10+ modules with good algorithms but no `FaceClusteringPipeline` class. Every consumer (scripts, notebooks, apps) had to re-implement the orchestration sequence, each time slightly differently. Two required stages (`crops.py`, `export.py`) were referenced in the spec but never built. Result: working parts that don't compose.
**Prevention**: For any ML sub-package, define the public API first (`pipeline.py` with a single `run()` method) before implementing stages. The pipeline class is the integration test for the entire sub-package. If it can't be written in <150 lines, the stage APIs aren't clean enough.

### 2026-03-30: Embedding corruption - regeneration script loaded cached data instead of computing fresh
**Root cause**: User reported clustering showing wrong similarities (face 545 similar to 546 instead of 569/573). Investigation revealed stored embeddings in `.npy` file were corrupted (likely face ID offset during original extraction). Critical bug: `regenerate_embeddings_from_crops.py` script had hidden code path that loaded pre-existing embeddings instead of computing fresh from image pixels - ALL 3 regeneration attempts produced IDENTICAL corrupted output (100% match). Only an isolated test in clean directory (no pre-existing .npy files) produced correct embeddings, proving face crops were fine and extraction works correctly.
**Prevention**: (1) **Never trust "regenerate" scripts without verification** - check that output differs from input by comparing embeddings numerically. (2) **Add mandatory validation after extraction**: randomly sample 10 faces, re-compute embeddings fresh, assert cosine similarity > 0.95 between stored and fresh. (3) **Fix regeneration script**: remove ANY code paths that can load cached/pre-existing embeddings - force fresh computation from image pixels only. (4) **Content-based verification**: Store crop image hash alongside embedding, validate they match when loading. (5) **Visual inspection in release process**: Generate HTML report showing face crops + distance matrices for manual spot-checks. (6) **Run validation tests on production data**, not just synthetic test data. See detailed analysis: `docs/EMBEDDING_CORRUPTION_ROOT_CAUSE_ANALYSIS.md`

### 2026-03-23: Never use loop counters for file identifiers
**Root cause**: SIGHTING-006 - `save_face_crops()` used `saved_count` (incremental counter) for crop filenames instead of metadata index. When first 2 faces failed validation, all subsequent faces saved with -2 offset: metadata[2] → face_0000.jpg (should be face_0002.jpg). Created permanent mismatch between array indices and filenames.
**Prevention**: (1) Always use stable identifiers from source data (face_id, metadata index), never incremental counters that skip failures. (2) Add validation immediately after saving to verify filename → data correspondence. (3) Add unit tests that simulate partial failures. (4) Document in code: "Use index from metadata, not saved_count - preserves alignment even when some saves fail."

### 2026-03-06: Transitive closure in kNN graphs creates mixed clusters
**Root cause**: Connected components on kNN graph groups A-B-C even if A≠C. Face B acts as "bridge" between two different people if it's similar enough to both (within threshold). Initial clustering (kNN + connected components) created mixed-identity pre-clusters before any merging happened.
**Prevention**: (1) After connected components, check cluster coherence (diameter, outlier detection), (2) Add post-clustering pruning stage to remove bridges/outliers, (3) Build diagnostics showing kNN neighbors and graph paths to understand WHY incorrect faces clustered together, (4) Consider tighter initial threshold or two-stage clustering (strict → merge).

### 2026-03-06: Don't skip diagnostics when debugging complex ML pipelines
**Root cause**: When merge analysis showed over-merged clusters, initially assumed merge stage was the problem. Built merge diagnostics, only to discover the real issue was earlier (initial clustering). Wasted time optimizing wrong stage.
**Prevention**: For multi-stage ML pipelines, always validate EACH stage's output before proceeding. (1) Check pre-merge clusters first, (2) Then check merge decisions, (3) Build diagnostics for each stage, not just final output. When output is wrong, trace backward through pipeline to find where error originates.

### 2026-02-28: Always compare outputs on identical inputs before declaring a mismatch
**Root cause**: Notebook used most recent embeddings (auto-selected), export script was run manually on older embeddings file. Comparison showed 45% mismatch, concluded "export script is broken". Actually both scripts were correct - just different inputs.
**Prevention**: When validating that script replicates notebook: (1) Run both on EXACT same input file (explicitly specify path, don't rely on "most recent"), (2) Compare row counts and metadata FIRST before comparing content, (3) Document which input file was used in output metadata. Comparison scripts should gracefully handle missing face IDs and different dataset sizes.

### 2026-02-27: Merge analysis - users need cluster-to-cluster distances, not just failure criteria
**Root cause**: User asked "why didn't clusters merge?" I provided `get_merge_decisions_df()` showing which criteria failed (Exemplar, Support, Margin, Diameter). User said "useless". They actually needed: (1) distance matrix BETWEEN CLUSTERS, (2) exemplar distances for all cluster pairs, (3) min/mean/max pairwise distances between cluster pairs. The merge_decisions_df only shows proposed candidates (exemplar_dist < 0.45), not WHY pairs weren't even proposed.
**Prevention**: For "why didn't X happen?" questions, provide the input data first (distances), then decision logic second (criteria). Create `get_cluster_distances()` returning DataFrame with: C1, C2, Exemplar_Dist (for proposal threshold), Min_Dist, Mean_Dist, Max_Dist. This shows "Cluster 4 and 23: exemplar_dist=0.52 > 0.45 → not proposed" directly. Mark this as the PRIMARY method for merge analysis in docstring.

### 2026-02-20: kNN + threshold + connected components prevents over-merging
**Root cause**: HDBSCAN density-based clustering can over-merge when embeddings form one large dense region (no clear density valleys). Parameter tuning (epsilon, min_samples) had no effect because the "wrong" merges happened at high density.
**Prevention**: Add post-clustering split phase: for each large cluster, build kNN graph, prune edges below similarity threshold, find connected components. This ensures every face pair in a cluster is connected through strong similarity paths - weak transitive chains get broken.

### 2026-02-20: Landmark labels are PERSON-relative, not image-relative
**Root cause**: After rotating face image, landmarks were being swapped `[1,0,2,4,3]` to "maintain left/right semantics". But L_eye/R_eye labels refer to the PERSON's left/right eye, not image position. The affine transform correctly maps person's left eye to reference template's left eye position regardless of where it is in the image.
**Prevention**: Never swap landmark indices after rotation. Landmark labels = anatomical identity. Only transform COORDINATES, not LABELS.

### 2026-02-19: Eye-line angle ≠ face orientation
**Root cause**: `compute_roll_angle()` uses `atan2(dy, dx)` on eye positions. This measures eye-line tilt, NOT whether face is upside-down. An upside-down face with level eyes returns roll_angle ≈ 0°.
**Prevention**: Face orientation requires checking spatial relationships: is nose BELOW eyes? is mouth BELOW nose? Implement `detect_face_orientation()` that returns 0°/90°/180°/270°.

### 2026-02-19: Single Responsibility - alignment/cropping/embedding must be separate steps
**Root cause**: `extract_face_embeddings.py` does detection format conversion + cropping + alignment + embedding extraction. Can't test any piece independently. Bug in alignment can't be isolated.
**Prevention**: One step = one responsibility. Create separate `align_faces`, `crop_faces` steps. Each step should be independently testable with unit tests.

### 2026-02-19: Always build debug panels showing all pipeline stages
**Root cause**: Face landmarks didn't match aligned crops because coordinates were from different stages (original image vs transformed output).
**Prevention**: For any ML pipeline, build visualization showing: (1) original input, (2) each intermediate transform, (3) final output. Catches coordinate system mismatches immediately.

### 2026-02-19: Document coordinate systems explicitly
**Root cause**: Confusion between pixel coords, normalized coords, and reference template coords led to landmark-face mismatch.
**Prevention**: Every function that handles coordinates should document: input coord system, output coord system, and any transforms applied. Use type hints like `landmarks_px` vs `landmarks_norm`.

### 2026-02-19: No "backward compatibility" fallbacks in pre-deployment code
**Root cause**: After refactoring face alignment into separate steps, added excessive if/else fallback logic "for backward compatibility" in extract_face_embeddings. This violated single responsibility and added untested code paths.
**Prevention**: In pre-deployment, the pipeline is deterministic. If step A produces output for step B, step B should require that output - no fallbacks. Fallbacks create untested paths and hide integration bugs.

### 2026-02-19: Session context loss leads to incomplete fixes
**Root cause**: After context compaction, specific details about UI bugs (which tabs were broken, what errors were shown) were lost. Made changes to backend code but didn't verify the actual UI issues the user reported.
**Prevention**: Before making fixes, explicitly confirm the specific symptoms with user. After fixing, ask user to test and report results. Don't assume "error handling" fixes unknown bugs.

## 2026-02-27: Store Metadata Paths, Don't Guess Them

**Problem**: Face clustering labeling app couldn't find face_crops directory when export location differed from source location (e.g., `results/training/Budapest_merged` vs `results/Budapest/face_crops`).

**Failed Approaches** (3 iterations):
1. Heuristic path guessing (check parent, check parallel dirs)
2. String matching on "training" in path parts
3. More complex heuristics with name suffix removal

**Why They Failed**: 
- Export directory names are arbitrary (`_merged`, `_test`, `_v2`, etc.)
- Heuristics are fragile and impossible to cover all cases
- No way to guess the original source from modified export names

**Root Cause**: Missing metadata - export didn't store where it came from

**Correct Solution**: Store source paths in metadata
- Added `embeddings_source` and `embeddings_dir` to export_summary.json
- Streamlit app reads from metadata FIRST
- Fallback to heuristics only for legacy exports
- Clear error message with re-export instructions if metadata missing

**Key Learning**: 
- **Always store paths as metadata** rather than reconstructing via string manipulation
- **Fail fast with helpful errors** - tell user how to fix (re-export) vs silent failures
- **Test end-to-end** - trace through actual usage patterns, not just happy paths
- **Heuristics are tech debt** - they work until they don't, then debugging is painful

**Prevention**: When exporting/transforming data that references external files, always save:
1. Source path (where data came from)
2. Timestamp (for cache invalidation)
3. Version (for format compatibility)

**Related**: Similar pattern needed for cache keys, model checkpoints, dataset configs
