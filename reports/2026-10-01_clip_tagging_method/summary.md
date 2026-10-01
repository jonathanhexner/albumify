# CLIP tagging — scoring method before/after (2026-10-01)

**Goal**: test whether the "poor" CLIP tagging is the *model* (ViT-B/32) or the *method*
(`clip_tagger.py` takes raw cosine → top-5 across all 57 prompts). Same model, same prompt text;
only the *reading* changes. 6 photos across the 3 trips (budapest/austria/germany).

- **BEFORE** = production `sim_bench/photo_analysis/clip_tagger.py`: raw cosine, top-5 of all 57.
- **AFTER** = per-category **softmax** (prompts compete only with peers) + **contrastive pairs**
  for quality/composition (sharp vs blurry …); CLIP-hopeless composition prompts dropped.

## Findings

1. **Scene / content: the method fix is a clear win.** BEFORE top-5 is dominated by always-on
   tags (`a color photo`, `a daytime scene`) and mixes axes into a mush. AFTER scene softmax is
   clean and plausible: vehicle 79%, group photo 85%, food 64% / person 35%, landmark **99%**.
2. **People gating works.** person-scene mass gates the human read: landmark → "no person" (0%);
   group shot → "multiple faces 84%". BEFORE leaked `a photo with multiple faces` into unrelated
   top-5s.
3. **The self-contradiction disappears.** BEFORE for the food photo returned
   *low-quality + overexposed + poorly composed + sharp + high-quality* simultaneously (raw cosine
   has no notion that these conflict). AFTER forces a single winner per axis.
4. **Quality is only partly rescued; composition is NOT.** Sharpness via contrastive pair looks
   right (sharp 99% on sharp shots, blurry 98% on the soft group shot). But **`poorly composed`
   fires at 88–99% on essentially every image**, and overall-quality hovers at a coin-flip 54–64%.
   This is CLIP's known blind spot for aesthetics/composition — contrastive framing does not fix it.

## Conclusion / recommendation

- **The weakness was largely method, for the job CLIP is good at (scene/content).** Fixing the
  read (softmax-in-group + gating) is cheap, adds no model to the memory floor, and should be a
  control arm against RAM++.
- **Scope CLIP to content/scene tags only.** Drop the quality + composition prompts entirely —
  the pipeline already has dedicated, better signals for those (IQA, AVA, noise, tilt). Using CLIP
  for "composition" just produces `poorly composed` everywhere.
- **RAM++ comparison is still worth it** — not to beat CLIP on quality, but for a far larger open
  tag vocabulary on the content axis (Tier-1 whole-roll tags).

Model: `ViT-B-32 / laion2b_s34b_b79k`, CPU. Full input sets on `D:\albumify_vs_vlm\{trip}\imgs768`.
Repro: `.venv/Scripts/python scripts/experiment_clip_tagging_method.py`. See [report.html](report.html).
