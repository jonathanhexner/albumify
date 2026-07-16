# Experiment Index

One entry per experiment, 2 lines each: (1) name + goal, (2) outcome + link.
Convention: `reports/<YYYY-MM-DD>_<slug>/report.html` + `summary.md` (see CLAUDE.md §Experiment reports).
Entries below predate the convention; their reports live at the linked legacy locations.

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
