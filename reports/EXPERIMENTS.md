# Experiment Index

One entry per experiment, 2 lines each: (1) name + goal, (2) outcome + link.
Convention: `reports/<YYYY-MM-DD>_<slug>/report.html` + `summary.md` (see CLAUDE.md §Experiment reports).
Entries below predate the convention; their reports live at the linked legacy locations.

- **2026-10-09 Slovakia26 full roll (715 photos) on the Modal L4 worker — Stage A input + first at-scale job-flow run** — run the real default_pipeline over the whole trip on the GPU worker and emit album.json for the owner's keeper-list judgement.
  **0.48 s/img (341 s pipeline, 521 s wall incl. upload); 147 picks / 96 scenes (no global cap yet), 423 near-dups, 56 set aside, 121 people clusters (over-split), 89 photos silently missing (quality filter, SIGHTING-128).** Judging sheet with every pick + set-aside inline. [report](2026-10-09_slovakia_full_gpu/report.html) · [summary](2026-10-09_slovakia_full_gpu/summary.md)
- **2026-10-04 spec-109 full-pipeline CPU timing baseline (57 imgs)** — honest per-step timing of the REAL default_pipeline (38 steps incl. occlusion/tilt), not the lean faces variant.
  **9.0 s/img on CPU → non-interactive** (100 imgs=15min). Top costs: RAM++ triage 3.0s + GeoCalib tilt 2.6s = 62%; face stack 2.0s = 23%. All GPU-friendly models. Defer-embedding is NOT free (select_best identity-subclusters need embeddings). GPU mandatory; blocker = host choice (Modal). [summary](2026-10-04_spec109_baseline/summary.md)


- **2026-10-01 Slovakia26 — triage + curation + performance on a real roll (75 of 715)** — run the pipeline on a real messy trip (screenshots left in) to answer: what junk to discard, how curation performs, and what's slow.
  **6/20 album picks were junk** (incl. a Gmail screenshot at 0.70 = top score); junk out-scored real photos (0.54 vs 0.41) -> quality scoring *rewards* screenshots, so triage must precede curation. Metadata caught 8 screenshots free; RAM++ caught 2 more posters. The 14 real picks look sound (8 scene clusters). Runtime ~2.2 s/img (CPU, faces pipeline), **face stack = 75%** (detect 908 + embed 753 ms/img). [report](2026-10-01_slovakia_analysis/report.html) · [summary](2026-10-01_slovakia_analysis/summary.md)

- **2026-10-01 RAM++ for defect detection — occlusion & blur (labeled)** — can a content tagger (RAM++) double as an occlusion/blur detector vs the purpose-built models? Tested on the labeled occlusion benchmark (83 occ + 84 clean) and RealBlur-J pairs.
  **No, decisively.** Occlusion: trained probe AUC 0.979; RAM++ `hand`/`lens flare` 0.384 (below chance); best-of-4,585 steelman tag = `cognac` 0.898 (spurious, still < probe). Blur: no blur/focus/sharp tag exists, none ever fires -> no blur score possible. RAM++ is content-only; occlusion stays on the probe, blur/quality on IQA. [report](2026-10-01_ram_defect_detection/report.html) · [summary](2026-10-01_ram_defect_detection/summary.md)

- **2026-10-01 RAM++ vs fixed-method CLIP — content tagging (30 photos, 3 trips)** — is RAM++'s open-vocabulary tagging worth a ~2.8 GB model on the memory floor vs fixed-method CLIP (closed 20-class scene read)?
  **RAM++ wins decisively: 11.7 tags/img (210 distinct) vs CLIP 2.5 (closed 20).** RAM++ names the actual subject (waffle/syrup/fork, mallard duck, witch hat, pedestrian bridge); CLIP forces wrong buckets (museum->landmark/vehicle) and `a selfie` over-fires. Adopt RAM++ for Tier-1 content tags (post-filter verb noise); keep fixed-CLIP as the near-free fallback. Open cost: RAM++ CPU latency + RSS as an isolated step. [report](2026-10-01_ram_vs_clip_tagging/report.html) · [summary](2026-10-01_ram_vs_clip_tagging/summary.md)

- **2026-10-01 CLIP tagging — scoring method before/after (6 photos, 3 trips)** — is the "poor" CLIP tagging the model (ViT-B/32) or the method (`clip_tagger.py`: raw cosine → top-5 of all 57 prompts)? Same model + prompt text; only the reading changes (per-category softmax + contrastive pairs).
  **Method, not model — for scene/content.** AFTER gives clean scene reads (landmark 99%, group 85%, food 64%) + working people-gating; BEFORE is dominated by always-on tags (`a color photo`) and self-contradicts (sharp+low-quality+high-quality at once). BUT `poorly composed` fires 88–99% everywhere → CLIP can't judge composition; scope CLIP to content tags only and keep quality from IQA/AVA/tilt. [report](2026-10-01_clip_tagging_method/report.html) · [summary](2026-10-01_clip_tagging_method/summary.md)

- **2026-09-21 spec-104 capacity measurement (Germany, 100/300/500 imgs)** — how does peak RSS scale with album size, and what does per-step process isolation cost vs in-process, on the real `faces` pipeline (CPU, cold cache)?
  **Flat scaling: peak ≈ 2.5 GB model floor + 0.4–0.7 MB/img** (isolated 2,845 MB @ N=500; in-process 2,532 MB) → 16 GB fits ~20k imgs/run, runtime (~1.6 s/img) binds first; isolation costs +~300 MB / +10% time; post-release() in-process no longer stacks models. Anomaly: ±1–2 face-count divergence between modes → SIGHTING-122. [report](2026-09-21_spec104_capacity/report.html) · [summary](2026-09-21_spec104_capacity/summary.md)

- **2026-07-21 spec-103 Scene clustering: time+geo fusion vs visual-only (Budapest, Austria, Germany)** — does fusing capture-time (backbone) + GPS (opportunistic) into one per-pair distance improve DINOv2/HDBSCAN scene clustering, and does it degrade gracefully where GPS is sparse? No labels yet — eyeball the change direction/magnitude.
  **Noise bucket (unplaceable orphans) shrank on all 3 trips: 27→15 (−44%), 104→69 (−34%), 191→94 (−51%); 12/52/129 orphans rescued into real scenes.** Time alone does most of the reduction; geo refines. Germany (34% GPS) improves MOST because time carries it — graceful degradation shown empirically. Correctness (vs hand labels) still open. [report](2026-07-21_scene_clustering_fusion/report.html) · [summary](2026-07-21_scene_clustering_fusion/summary.md)

- **2026-07-17/18 spec-102 Albumify vs VLM (3 trips: Budapest, Austria, Germany)** — can Claude Opus 4.8 curate a K=20 trip album as well as the engineered Albumify pipeline, and where does a VLM add value? Same 768px inputs per trip, each picks an ordered K=20.
  **Pick overlap SHRINKS as trips grow: Budapest 6/20 → Austria 3/20 → Germany 0/20** (zero agreement on 797 photos — more images, more equally-valid choices). VLM as diverse or more (redundancy saturates at 0 on big trips). VLM produced structured annotation Albumify has no equivalent for: correct album_type/subtype per trip (city vs road), 30/131/177 editorial moment-groups, caught a finger-occlusion + named a war memorial. **Hardware limits**: full people-aware pipeline OOMs >~120 imgs (SIGHTING-117) → Albumify ran `faces` (Budapest) / `minimal` (Austria, Germany) — no occlusion/tilt, and no person-penalty on the big trips. Human blind A/B verdict PENDING (solo N=1). Reports: [budapest](2026-07-17_albumify_vs_vlm_budapest/report.html) · [austria](2026-07-18_albumify_vs_vlm_austria/report.html) · [germany](2026-07-18_albumify_vs_vlm_germany/report.html)

- **2026-07-12 spec-099 Phase 0: tilt (crooked-photo) detection** — validate classical Hough-based tilt estimation before wiring a penalty; 122 photos × ±2..10° injected rotations.
  **Negative result, penalty not shipped**: 0.32° MAE when confident but only 4.9% coverage, and the confident flags are upright photos with slanted scenery (tunnel, illusion art) — pixels can't separate camera-tilt from world-tilt. [report](2026-07-12_tilt_benchmark/report.html) · [summary](2026-07-12_tilt_benchmark/summary.md)

- **2026-07-11 spec-098 validation: noise-aware quality scoring** — re-run SIDD/RealBlur benchmarks + Budapest face-gate check against pre-registered gates after the noise fix.
  SIDD 0%→**99.4%** pair acc, inflation 39×→0×, RealBlur intact 99.3%; median-blur-only FAILED first (62%) — σ²-subtraction formula found by sweep; face gate re-calibrated 150→73.3. [report](2026-07-11_spec098_validation/report.html) · [summary](2026-07-11_spec098_validation/summary.md)

- **2026-07-10 defect scoring #3: exposure** — pipeline exposure score (clip+entropy, 30% of overall) vs human JOD judgments (PIQA23, 1,500 imgs / 50 scenes).
  Ours SROCC 0.28 — **loses to a one-line mid-gray baseline (0.33)**; MUSIQ 0.57. Root cause: 90% of scores compressed into [0.89, 0.99] (pure-0/255 clip test never fires on smartphone output). [report](2026-07-10_defect_exposure/report.html) · [summary](2026-07-10_defect_exposure/summary.md)

- **2026-07-10 defect scoring #4: rotation** — can the pipeline detect content rotation (EXIF-stripped) vs sky-prior classical vs CLIP zero-shot? 122 photos ×4 orientations.
  Pipeline is exactly chance (AUC 0.500, blind by construction); sky prior 47.5%; CLIP zero-shot **82%** 4-way with no training → viable detector. [report](2026-07-10_defect_rotation/report.html) · [summary](2026-07-10_defect_rotation/summary.md)

- **2026-07-10 defect scoring #2: noise** — pipeline has no noise metric; do existing scores detect real sensor noise (SIDD-Small, 480 GT/noisy pairs)?
  **Inverted, not blind**: noise inflates Laplacian 39×, noisy twin wins 480/480 pairs on our overall score; classical wavelet `estimate_sigma` is ~perfect (AUC 0.993, ~20 ms); MUSIQ weak on patches (61.9%). [report](2026-07-10_defect_noise/report.html) · [summary](2026-07-10_defect_noise/summary.md)

- **2026-07-10 defect scoring #1: blur** — benchmark pipeline Laplacian sharpness vs Tenengrad vs MUSIQ on RealBlur-J (702 sharp/blur pairs).
  Ours wins pair ranking (99.4% vs MUSIQ 98.9%); MUSIQ wins absolute thresholding (AUC 0.893 vs 0.871); failures = dark noisy frames (noise-as-sharpness). [report](2026-07-10_defect_blur/report.html) · [summary](2026-07-10_defect_blur/summary.md)

- **2026-07-10 ResNet18 fine-tune vs CLIP probe** — user request: fine-tuned CNN + on-the-fly augmentation, identical OOF protocol head-to-head.
  CNN loses: scene PR-AUC 0.769 vs 0.899, FP at gate 10 vs 2 (train loss ~0.01 = memorized; 80 positives can't fine-tune a CNN). [report](2026-07-10_resnet_finetune/report.html) · [summary](2026-07-10_resnet_finetune/summary.md)

- **2026-07-10 batch-3 positives + augmentation retrain** — ingest 24 new positives; test heavy augmentation (both classes); report ROC + FP/FN galleries.
  Scene PR-AUC 0.777→0.907±0.006 (+RealBlur best); **augmentation = null result** (CLIP already invariant); 2 FP over gate (Austria burst), 32/80 OOF FN (mostly minor — user-accepted). [report](2026-07-10_augmented_retrain/report.html) · [summary](2026-07-10_augmented_retrain/summary.md)

- **2026-07-10 blur robustness (own albums + RealBlur)** — does real blur fool the occlusion detector; do RealBlur negatives fix it (detection P + explainability)?
  Own albums 0/40 over gate; RealBlur-J FAILS v1 (52/702, 38/234 scenes); +140 RB negatives → 31→0 on holdout at zero PR-AUC/recall cost. [report](2026-07-10_blur_robustness/report.html) · [summary](2026-07-10_blur_robustness/summary.md)

- **2026-07-09 blur-vs-occlusion separation** — verify the occlusion probe learned "occluder" not "blur" (user ship gate, spec-097 T1.1).
  0/60 clean, 0/60 motion, 0/60 defocus over gate vs 56/56 real occlusions. ⚠ variants were score-and-delete — no gallery exists (predates convention); numbers in `D:\occlusion_dataset\results_final.json`, code `scripts/train_occlusion_artifact.py:validate_blur`.

- **2026-07-09 tile-attention explainability** — check the probe's per-tile attention points at the blur region on positives (user ship gate, spec-097 T1.1).
  Hot tile inside LoG box 13/26; detection solid, localization research-grade → Stage 2. Gallery: `D:\occlusion_dataset\research_saliency\tile_explain\index.html`.

- **2026-07-08 crop-verify CLIP probes** — can CLIP (prompts / synth probe / real+diverse probe) verify candidate blurry crops?
  Prompts fail; synth probe INVALID (whole-photo crops bug); real+diverse best but data-starved (23 pos crops). Galleries: `D:\occlusion_dataset\research_saliency\crop_probe\{index,probe_index,real_probe_index}.html`.

- **2026-07-08 blur-box proposer audit** — measure LoG flat-but-noisy box recall/FP on the full positive set.
  23/49 recall, 45 FPs/122 (24 sky) — proposer caps any crop-verify pipeline. Code: `scripts/experiment_blurbox_audit.py`.

- **2026-07-07 SLIC-saliency fusion panels** — object-shaped subject masks from SLIC × (spectral-residual + CLIP-semantic) fusion for Stage 2.
  SR×SLIC traces buildings, CLIPsem×SLIC captures people; each alone fails the other case → fuse. Gallery: `D:\occlusion_dataset\research_saliency\index.html`.

- **2026-07-05..09 spec-096 occlusion model benchmark** — 9 tracks (CLIP probes, ResNet, LoG features, Haiku, zero-shot) on 832 adjudicated images.
  Winner CLIP global+tile-max probe, 0.86 [0.79–0.92] scene PR-AUC. Full results: `specs/096-occlusion-detection-bench/RESULTS.md`.

- **2026-07-13 spec-100 learned tilt (GeoCalib) benchmark** — swap classical Hough for GeoCalib (ECCV'24) on the same injected-rotation harness (122 Budapest photos, ±2–10°).
  Accurate-when-confident (MAE 0.25–0.45°, ~100% recall, fixes the slanted-scenery FPs) but coverage 28% on this people-heavy album (G2≥70% missed, AUC 0.835). Native `roll_uncertainty` abstains honestly (kaleidoscope shots → 10–30° unc). Verdict: gated tie-breaker candidate, not a clean auto-pass. Report: `reports/2026-07-13_geocalib_budapest_tilt/report.html`; gates `reports/2026-07-12_geocalib_tilt/`.

- **2026-07-13 spec-100 GeoCalib tilt tutorial** — step-by-step CV explainer of single-image roll estimation on a high-tilt vs near-zero photo, every intermediate tensor rendered (up-field, latitude, confidence, LM fit, roll geometry, uncertainty, penalty).
  Pedagogical, not a new result. Report: `reports/2026-07-13_geocalib_tutorial/report.html`.

- **2026-07-17 spec-101 auto-straighten before/after** — real subject-aware gate (YOLO persons) on the 10 confident tilted Budapest photos: 7 straightened, 3 declined (portraits below the 70% area floor).
  Median retained area on straightened = 79% → fov penalty 0.086 at fov_weight=0.4 (validates the default). Report: `reports/2026-07-17_auto_straighten/report.html`.
