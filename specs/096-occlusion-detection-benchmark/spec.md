# spec-096 — Occlusion Detection Benchmark

**Created**: 2026-07-05 · **Status**: In Progress · **Priority**: P2
**Source**: user — after 5 hand-built approaches failed to reliably flag finger-over-lens
occlusion (see LEARNINGS 2026-07-05), benchmark multiple detector families on a
human-labeled dataset and pick by accuracy-vs-cost.

## Goal
A provenance-tracked occlusion dataset + a benchmark comparing detector candidates on
quality / cost / latency, with special scoring on the hard negatives (warm walls, lakes,
night shots) that defeated every heuristic.

## Ground truth (locked decisions)
- **User labels are ground truth.** `D:\occlusion_examples` (49) + `examples/finger_occlusion`
  (2, Budapest-sourced) = positives. Personal albums (Budapest / Germany_1 / Austria_24) =
  presumed negatives (base-rate noise accepted).
- **Haiku is BOTH a candidate AND a second validator**: its disagreements with the labels are
  reviewed by the user — surfacing hidden occlusions in the albums (label repair), not
  silently overriding.
- **Domain rule**: negatives come ONLY from the same phone-photo domain as positives.
  `D:\DataSets` public sets are excluded (a classifier would learn source, not occlusion).

## Dataset (`D:\occlusion_dataset\`)
```
positives/  occl__<file>.jpg, budapest__<file>.jpg     ← dataset-prefixed ids
negatives/  budapest__… germany1__… austria24__…
manifest.csv  id,label,level,source_dataset,source_path,sha1,split,hard_negative,notes
```
- sha1 dedupe (catches the finger images that exist in both Budapest + examples).
- Deterministic split from sha1 (reproducible; no RNG): ~80/20 train/test, frozen.
- `hard_negative=True` seeded from the classical detector's false positives.
- `level` (0–3 severity) left blank; Haiku proposes, user confirms later.

## Candidates
| # | Approach | Cost profile |
|---|---|---|
| 0 | Classical 4-cue detector (exists) | free, instant — baseline |
| 1 | CLIP global probe | free — known-weak control |
| 2 | SigLIP-2 / CLIP **patch-level** probe (max-pool) | free, fast |
| 3 | Tiny local VLM zero-shot (Moondream / SmolVLM) | free, ~sec/img CPU |
| 4 | Haiku API zero-shot | ~$2.5 / 1k imgs |
| 5 | Small CNN fine-tune + synthetic occlusion augmentation | free at inference |

## Metrics
PR-AUC + precision/recall at tuned threshold; **separate report on hard-negative subset**;
$ per 1k images; sec/image on this CPU. Deliverable = one results table (accuracy-vs-cost
frontier) in `RESULTS.md` + optionally surfaced in the Analysis Studio later.

## Rules
- Frozen test split: no candidate trains on it; probes/CNN use CV on train only.
- **Group-aware split (T1.4, user-caught 2026-07-06)**: albums are full of NEAR-duplicates
  (bursts/retakes) that sha1 dedupe misses. Near-dupes are grouped (dHash hamming ≤ 8 OR
  filename-timestamp within 15 s, per source; union-find) and the split derives from the
  GROUP — a scene can never straddle train/test. Under the naive per-file split, 93 groups /
  377 rows (~45% of the dataset) were leak-prone.
- **Effective sample size = groups, not files**: 51 positive images = **19 distinct scenes**;
  776 negatives = 345 groups. All CV is grouped-by-scene; report scene counts alongside
  image counts everywhere.
- Augmented/synthetic variants: group-split by base image (no leakage).
- Imbalance: keep all negatives; `class_weight`/loss-weighting; never report accuracy.

## Out of scope
VLM service layer (future spec — informed by this benchmark). Studio UI integration.
