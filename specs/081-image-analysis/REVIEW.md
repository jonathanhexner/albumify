# REVIEW — spec-081 Image Analysis view

**2026-06-05** · scope: spec-081 diff · ✅ no High findings.

- `image_metrics.py` — `image_detail` passthrough.
- NEW `components/image_analysis.py` — overlay (EXIF-upright, aspect-locked, colour by disposition) + per-face table + image scores.
- `tabs/images_tab.py` — row-select → analysis.
- NEW `tests/.../test_image_analysis.py` (2 + 1 slow).

| § | Finding |
|---|---|
| Correctness | bbox is (x1,y1,x2,y2) in the upright frame (matches spec-079); disposition 3-way incl. noise-label (-1). Off-frame faces (negative x) clip to the axis range — fine. ✅ |
| Reuse | same EXIF/aspect/`scaleanchor` logic as the validated Face Analysis overlay; `RunStore.image_detail` reused (no new query). ✅ |
| Tests | unit (disposition + colours) + slow real-run (faces/bbox) + AppTest 0 exc. Visual: data-validated (kaleido absent for static export; canvas row-select not Playwright-addressable). ✅ |
| Layering | service Streamlit-free; component render-only. ✅ |

**No High → Implemented.**
