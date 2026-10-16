# RAM++ vs fixed-method CLIP — content tagging (2026-10-01)

**Goal**: decide whether RAM++ (open-vocabulary tagger) is worth adding a ~2.8 GB model to the
pipeline's memory floor, vs the fixed-method CLIP (ViT-B/32 laion2b, per-category softmax over the
20 scene prompts + people gating) from the prior experiment. 30 photos, 10 per trip.

- **RAM++**: `recognize-anything-plus` swin_l/384, checkpoint `models/ram/ram_plus_swin_large_14m.pth`.
- **CLIP**: fixed-method read, closed 20-class scene vocabulary.

## Headline numbers

| | RAM++ | CLIP (fixed) |
|---|---|---|
| avg tags / image | **11.7** | 2.5 |
| distinct tags across 30 imgs | **210** | 20 (closed) |
| vocabulary | open (~4.5k) | fixed 20 scene classes |

## Findings

1. **RAM++ wins decisively on content.** It names the actual subject: `waffle, syrup, fork, plate`
   (breakfast), `mallard duck, mountain lake`, `witch hat, halloween costume`, `cherry blossom,
   glass house` (greenhouse), `pedestrian bridge, fjord`. CLIP can only map these into its nearest
   of 20 buckets.
2. **CLIP's closed vocab forces wrong buckets.** A museum display → CLIP `landmark, vehicle, product
   photo`; a glass vitrine → `product photo, group photo, vehicle`. RAM++: `artifact, museum,
   showcase, glass box` / `crystal, glass vase, showcase`.
3. **CLIP's `a selfie` over-fires** — it appears on ~1/3 of images regardless (people-gating +
   human softmax is too eager). RAM++'s person tags (`person, man, woman, child, girl`) are cleaner.
4. **RAM++ isn't noise-free.** It emits some useless verb/fragment tags (`out, break, bite, push,
   shine, run, beam`) and occasional misses (`moose` for likely livestock). A noun/allow-list
   post-filter would clean this up cheaply.

## Recommendation

- **Adopt RAM++ for Tier-1 content tagging.** The open vocabulary is categorically better and is
  what enables thematic grouping / search / album-type — CLIP's 20 classes can't.
- **Keep fixed-method CLIP as the cheap fallback** (it reuses the already-resident occlusion CLIP,
  zero added floor) when RAM++'s 2.8 GB isn't warranted.
- **Open cost to measure next**: RAM++ swin_large per-image latency on CPU (slow; ~seconds/img) and
  peak RSS as an isolated pipeline step — this is the real trade vs CLIP's near-free reuse.
- **Post-filter** RAM++ tags to a noun allow-list to drop verb/fragment noise.

Note: RAM++ outputs *tags*, not sentence captions. Sentence captions would need a VLM (Qwen3-VL /
MiniCPM-V) — the Tier-2 keeper captioning path.

Install (env-safe): `pip install git+https://github.com/xinyu1205/recognize-anything.git --no-deps`
+ `fairscale --no-deps`; a transformers shim is in the script (RAM's BERT imports moved helpers).
Repro: `.venv/Scripts/python scripts/experiment_ram_vs_clip_tagging.py`. See [report.html](report.html).
