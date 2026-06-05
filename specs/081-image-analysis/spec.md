# spec-081 — Image Analysis view (repurpose the Images tab)

**Created**: 2026-06-05 · **Status**: Implemented · **Priority**: P2
**Predecessors**: spec-077 (Images tab), spec-070 (bbox overlay), spec-079 (EXIF/aspect)
**Source**: user feedback #3 — "Images tab is useless; want all face bboxes on the image + face/image metrics + filter status."

## What we build
- `face_cluster/views/image_metrics.py`: `ImageMetricsService.image_detail(path)` →
  `ImageDetail` (passthrough to `RunStore.image_detail` — per-image scores + every face's
  bbox / cluster / gate).
- NEW `app/face_clustering_v2/components/image_analysis.py` — `render_image_analysis(detail)`:
  the source photo (EXIF-upright, aspect-locked) with EVERY face's bbox **colour-coded by
  disposition** (clustered=green / noise=amber / filtered=red) + face-id labels, the
  image-level scores, and a per-face table (disposition / cluster / gate / blur / area / det).
- `app/face_clustering_v2/tabs/images_tab.py`: the per-image table is now selectable; clicking
  a row renders the analysis below.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | `image_detail` returns the image's faces with bbox + disposition | slow test |
| 2 | overlay draws one box per face, coloured by disposition | data + figure build (mirrors validated spec-079 overlay) |
| 3 | per-face table + image scores shown | code |
| 4 | disposition 3-way correct (incl. noise-label) | unit test |
| 5 | Images page renders, 0 exceptions | AppTest |

## Notes
- Static PNG export needs `kaleido` (not installed) — overlay validated via the data
  (9-face image: 1 clustered / 1 noise / 7 filtered) + the identical EXIF/aspect/bbox logic
  proven in spec-079's Face Analysis screenshot.
- Dataframe row-select is canvas-rendered (SIGHTING-091) — not Playwright-addressable, so no
  e2e click; the render path is AppTest + unit covered.
