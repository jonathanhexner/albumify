# Sharpness gate vs owner labels (2026-11-06)

**Goal:** does `filter_quality.min_sharpness: 0.2` remove the photos the owner calls blurry?
**Method:** owner labelled 182 photos blind (Fine / Soft but keep / Blurry, drop): 60 the gate removed on the 10-23
Slovakia26 L4 run, 92 that passed (4 score bands x 23), 30 random Budapest. Labels: `sharpness_labels.json`.

**Result:** only 5 of 182 are Blurry. Of the 60 the gate removed, 59 were wrong (37 Fine, 22 Soft); it caught 1 of the 2
blurry Slovakia photos in the sample. Low scores lean Soft (22/60 below 0.2 vs 2/23 above 0.7), so the score tracks
softness and scene content (water, smooth areas; owner note), not "would drop".

| Score band | Fine | Soft | Blurry |
|---|---|---|---|
| < 0.2 (removed) | 37 | 22 | 1 |
| 0.2-0.3 | 17 | 5 | 1 |
| 0.3-0.45 | 20 | 3 | 0 |
| 0.45-0.7 | 18 | 5 | 0 |
| >= 0.7 | 21 | 2 | 0 |
| Budapest (no score) | 12 | 15 | 3 |

**Owner:** blur was not the main problem in the bad photos: people not looking at the camera, and people who do not
belong in the frame (smart cropping) were. **Outcome:** the gate does more harm than good; replacing it is low priority.
Images: `img/` (downscaled, untracked); originals in `D:\Slovakia26`, `D:\Budapest2025_Google`.
