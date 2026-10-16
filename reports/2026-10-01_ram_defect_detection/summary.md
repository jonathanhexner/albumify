# RAM++ for defect detection — occlusion & blur (2026-10-01)

**Goal**: test empirically (not by assertion) whether RAM++ — a content tagger — can double as a
defect detector for occlusion and blur, vs the purpose-built models already in the pipeline.

## Occlusion (labeled benchmark: 83 occluded + 84 clean, D:\occlusion_dataset)

| detector | AUC | read |
|---|---|---|
| **trained probe** (`B_clip_oof`) | **0.979** | the dedicated detector — excellent |
| RAM++ `hand` / `lens flare` | **0.384** | **below chance** — its only occlusion-ish tags are anti-correlated (|"hand" fires on hands-as-subject, not fingers-over-lens|) |
| RAM++ best-of-4,585 (steelman) | 0.898 | best single tag = **`cognac`** — a spurious artifact, still < probe |

The steelman is the punchline: the single best-separating tag out of RAM++'s entire 4,585-tag
vocabulary is **"cognac"** (a brandy) — obviously nothing to do with occlusion. It correlates only
because dark/warm occluded frames trip that tag. It (a) still loses to the probe and (b) would never
generalize. That is what "wrong tool" looks like quantitatively: no *meaningful* tag detects
occlusion, and the best-correlated one is overfit noise.

## Blur (RealBlur-J, 20 scene pairs blur vs sharp)

- **No `blur` / `focus` / `sharp` / `out of focus` tag exists** in RAM++'s vocabulary.
- **No blur-related tag ever fired** across the pairs → there is literally no blur score to threshold.
- Tag overlap blurred-vs-sharp = **0.77** (the ~23% shift is lost content detail under heavy blur,
  not blur *awareness*).

## Conclusion

RAM++ is the wrong tool for both defects — now confirmed on labeled data, not asserted. Occlusion
stays on the trained probe (AUC 0.98); blur/quality stay on IQA + noise-aware sharpness (spec-098).
RAM++ earns a place on exactly one axis: **content tagging**. This closes the "is it worth trying
RAM++ for occlusion/quality?" question — no.

Repro: `.venv/Scripts/python scripts/experiment_ram_defect_detection.py` (needs `pillow-heif` for the
HEIC negatives). See [report.html](report.html).
