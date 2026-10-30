# spec-115 T2.1 — crop orientation gate (slovakia26), 2026-10-23

- **Goal:** prove PC-made face crops (box from `bundle.json` cut from the PC's own 768 px copy) match the worker's.
- **Method:** 20 faces, 5 per EXIF orientation; ArcFace cosine PC crop vs worker crop, bar >= 0.3 on >= 19/20;
  negative control: rotated photos cut without the EXIF transpose.
- **Result:** 18/20 (FAIL by the pre-set number); control rejected 14/15.
  Rotation handling is right (control ~0, correct crops 0.42-0.82 on rotated photos). Misses: a tilted head (worker
  aligns on landmarks) and an 11x13 px face. **Found SIGHTING-141:** the worker's face crops have red/blue swapped.
- **Report:** `report.html` (crops local only, `img/` untracked).
