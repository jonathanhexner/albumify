# Image Analysis Studio (spec-094)

Point it at a photo folder, pick analysis methods across families, and compare
every model's per-image output in a clickable-thumbnail, sortable table grouped
into category tabs.

```
Folder ─► engine.discover_images ─► engine.run_methods(selected) ─► {path: {method: AnalysisColumn}}
                                          │  (each method = a universal_cache-backed pipeline step)
                                          ▼
                              view: category tabs + flat "All"
                              clickable thumbnail grid · sortable · confidence bars · CSV
```

## Run (Windows)

```powershell
.venv/Scripts/streamlit run app/image_studio/main.py
# default port 8501; open the URL it prints
```

## Methods

| Family | Methods | Output |
|---|---|---|
| `geo_location_and_caption` | EXIF, StreetCLIP, GeoCLIP, BLIP | GPS/time · city+conf · lat/lon+conf · caption |
| `image_quality` | Rule-based IQA, Sharpness, AVA | numeric score (sortable) |

- **EXIF** is always available; **StreetCLIP / GeoCLIP / BLIP** need `transformers`/`torch`
  (+ `geoclip`); **AVA** needs the checkpoint at `models/album_app/ava_resnet50.pt`. Unavailable
  methods are greyed out in the sidebar.
- The pyiqa metrics (MANIQA, NIQE, …) join the `image_quality` family when **spec-093** lands.

## Storage / caching

Every method persists through **`universal_cache`** (`~/.sim_bench/sim_bench.db`, shared with
Albumify) via the BaseStep cache hooks — a second run over the same images is a cache hit, no
recompute. First run of the vision/AVA models downloads weights (~1.6 GB) and runs on **CPU**, so
keep the image limit small (default 12).

## Confidence is relative, not accuracy

StreetCLIP/GeoCLIP report a confidence that ranks their own guesses; it is **not** a probability
of being correct. On the Budapest album, high-confidence-but-wrong city guesses are common (see
`specs/022-trip-detection/BUDAPEST_EXPLORATION.html`). The UI labels the bars accordingly.

## Related apps — which to use

Both apps share this one engine (`app.image_studio.engine`) and `universal_cache`; they differ in focus:

| Use | App |
|---|---|
| Compare **all families** (quality + geo + caption) in category tabs, sort by any score | **this** — `app/image_studio` |
| **Geo-only** deep dive: EXIF-vs-GeoCLIP **map** + top-1 accuracy vs EXIF | `app/geo_vision` (spec-095) |

Rule of thumb: reach for `image_studio` for broad multi-method comparison; reach for `geo_vision`
when you specifically want the map and geolocation-accuracy view. Neither duplicates scoring logic —
both call `run_methods`.

## Layout

- `engine.py` — pure orchestration: `discover_images`, `AnalysisColumn`, method registry,
  `run_methods`, `available_methods`, `categories`. No Streamlit.
- `view.py` — rendering: thumbnail grid (real `st.button`, never `st.dataframe` row-select),
  sortable tables, confidence bars, click-to-enlarge, CSV export.
- `main.py` — the Streamlit entry (sidebar + run + results).
