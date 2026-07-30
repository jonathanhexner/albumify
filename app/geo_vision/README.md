# geo_vision — geo view helpers (spec-095, merged into spec-094)

**This is no longer a standalone app.** The Geo-Vision Studio was consolidated into
the **Image Analysis Studio** (`app/image_studio/`) — its map + accuracy live in that
app's **Browse Run → Geo** view.

What remains here is a small **pure helper module** imported by the studio:

- `geo_view.py` — no Streamlit:
  - `haversine_km(...)` — great-circle distance (km)
  - `geoclip_accuracy(columns, threshold_km)` — GeoCLIP top-1 vs EXIF GPS (EXIF-less skipped)
  - `map_points(columns)` — EXIF (green) + GeoCLIP#1 (orange) markers for `st.map`
  - `csv_rows(columns, selected)` — flat per-image export rows

Unit-tested in `tests/geo_vision/test_geo_view.py`.

To run the studio (which includes the geo view):

```
.venv/Scripts/streamlit run app/image_studio/main.py
```
