# Geo-Vision Studio (spec-095)

Standalone Streamlit app to run geo/vision models (EXIF, StreetCLIP, GeoCLIP,
BLIP) over a photo folder and visually inspect each guess + confidence, with an
EXIF-vs-GeoCLIP map.

```
.venv/Scripts/streamlit run app/geo_vision/main.py
```

## Design
Thin UI on top of the **spec-094 engine** (`app.image_studio.engine.run_methods`,
universal_cache-backed) — no duplicate scoring engine. This package adds only the
geo-specific view layer:

- `geo_view.py` — PURE helpers (no Streamlit): `haversine_km`,
  `geoclip_accuracy` (top-1 vs EXIF GPS), `map_points` (EXIF green / GeoCLIP
  orange), `csv_rows`. Unit-tested in `tests/geo_vision/`.
- `main.py` — sidebar (folder, limit, model checkboxes filtered to the geo
  family + availability), Run → `run_methods`, then summary metrics, the map,
  per-image confidence bars + thumbnails, and CSV export.

## Notes
- Confidence is **relative ranking, not accuracy** (labelled in the UI).
- First run downloads CLIP/BLIP weights (~1.6GB); re-runs hit `universal_cache`.
- Streamlit has no folder picker → text input. CPU inference; keep `limit` small.
