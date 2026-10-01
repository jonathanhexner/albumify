# Slovakia26 — triage, curation, performance (2026-10-01)

Ran the pipeline on a 75-image subset (68 scored) of a real 715-photo trip (`D:\Slovakia26`),
screenshots left in deliberately. Faces pipeline, CPU, 768px, K=20. Full report: report.html.

## 1. Discarding irrelevant photos — the headline

- **6 of 20 album picks were junk** (5 screenshots + 1 photographed poster).
- A **Gmail screenshot scored 0.703 — the highest in the whole album.**
- **Junk out-scored real photos: mean composite 0.54 vs 0.41.** Quality scoring *rewards*
  screenshots (flat, sharp, well-exposed, no faces) → you cannot filter them by scoring; they must
  be removed **before** curation. Empirical proof for a triage gate.
- Both triage tiers needed: **metadata** caught all 8 screenshots for free (named `Screenshot_`,
  no camera EXIF); **RAM++ content** caught 2 more real photos (posters) metadata can't see — one
  of which leaked into the album.
- Behavioral (singleton+no-person) was **inconclusive** here: the evenly-spaced subset destroyed
  the temporal bursts, so no scene-cluster-of-one formed. Needs the full contiguous roll.

## 2. Albumify performance

- 14/20 picks are genuine trip moments across **8 scene clusters**, chronological (boats, lake,
  mountains, cathedral at dusk, climbing, family). The engine looks sound — it's being contaminated
  by junk it shouldn't have scored, not making bad judgments on real photos.
- The harder "utilitarian" class is already visible among the real picks: a car+license-plate shot,
  CDs/electronics, a construction worker on a ladder — technically fine, not album moments. The case
  for the behavioral + VLM-intent layer.
- Can't grade curation until triage is on + compared to the user's own keeper list.

## 3. What's slow

- **~2.2 s/image** (faces pipeline, CPU, 768px). Per-step ms/img:
  `insightface_detect_faces 908` + `extract_face_embeddings 753` = **1.66 s = 75% of runtime**;
  DINOv2 scene 216, YOLO persons 129, AVA 97, select_best 33, IQA 31, expression 16.
- The bottleneck is per-image **compute** (not model load), in the face stack.
- At scale: 300 photos ≈ 11 min (faces only), ≈ 21 min with RAM++ triage; the full `default`
  pipeline adds occlusion CLIP + GeoCalib tilt on top. Far beyond interactive on CPU.
- Levers: **GPU** for the InsightFace/ONNX stack (10–50×), feature **caching**, or **defer/skip the
  face stack** when album ranking doesn't need per-face identity.

## 4. Next steps

1. Build the triage gate (metadata → RAM++ → set-aside-for-review) — proven necessary (30% junk leak). **do first**
2. Re-run Stage A quality eval on the cleaned set vs the user's keeper list. **do first**
3. Decide the runtime target (GPU / caching / defer face stack) — sizes the spec-106 serving decision.
4. Test behavioral + VLM-intent on the full contiguous roll (v1.5).
5. Fold triage into finalization Stage B; runtime into Stage C.

Repro: `.venv/Scripts/python scripts/experiment_slovakia_analysis.py` (subset=75). Data: results.json.
