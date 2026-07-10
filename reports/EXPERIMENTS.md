# Experiment Index

One entry per experiment, 2 lines each: (1) name + goal, (2) outcome + link.
Convention: `reports/<YYYY-MM-DD>_<slug>/report.html` + `summary.md` (see CLAUDE.md §Experiment reports).
Entries below predate the convention; their reports live at the linked legacy locations.

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
