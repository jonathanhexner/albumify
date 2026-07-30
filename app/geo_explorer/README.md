# Geo Segmentation Explorer (spec-022)

Standalone experiment to *feel out* geo/time segmentation on real photos. It runs
the **real pipeline** (`discover_images → extract_geo_metadata → geo_temporal_segment`)
once on a directory, then serves an interactive Leaflet map + timeline. The same
`geo_cluster/` code powers the live re-segmentation, so anything tuned here ports
straight into Albumify.

## Run (Windows)

```powershell
$env:GEO_EXPLORER_DIR = "D:/Budapest2025_Google"   # any photo folder
.venv/Scripts/python -m uvicorn app.geo_explorer.server:app --port 8077
```

Open http://127.0.0.1:8077

## What you see / can do

- **Map** — one pin per GPS-tagged photo, coloured by its **GPS cluster**. Home
  (auto-detected) is the marker. Click a pin for a thumbnail.
- **Timeline** (bottom) — one tick per timestamped photo, coloured by **time gap**.
- **Sliders** — GPS radius (km), time gap (hours), floor. Drag → live re-segment.
- **Competition table** — every axis scored (separation/coverage/balance/stability/
  parsimony → overall); the winner is highlighted, or **FLAT** if all are below the floor.

## Notes from the Budapest run

- 122 photos · 44 with GPS · 84 with time (Google export stripped the rest).
- Home auto-detected at 47.51, 19.05 (central Budapest) — correct.
- The 44 GPS photos span only ~2 km, so it's a single-place trip: geo can't
  meaningfully split it (1 cluster ≥2 km) and **time wins**. Drop the radius to
  ~1 km to see it split into 2 GPS clusters.

This is exploratory; it does not persist anything. Persistence + UI in Albumify is Slice 2+.
