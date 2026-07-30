# Album Pipeline Explorer (spec-022)

Point it at any photo folder, hit **Run**, and it executes the **real pipeline**
(same `execute_spec` engine as Albumify) and shows a per-image table.

```
discover_images → extract_geo_metadata → infer_geo_clip (StreetCLIP)
                → caption_images (BLIP) → geo_temporal_segment
```

## Run (Windows)

```powershell
.venv/Scripts/python -m uvicorn app.album_explorer.server:app --port 8078
```
Open http://127.0.0.1:8078 — type a folder, set a `limit`, click **Run**.

## Table columns

| Column | Source step | Module |
|---|---|---|
| time (EXIF) | `extract_geo_metadata` | `geo_cluster/exif_reader.py` |
| GPS (EXIF) | `extract_geo_metadata` | same |
| StreetCLIP — top 3 cities + scores | `infer_geo_clip` | `geo_cluster/streetclip.py` |
| BLIP caption | `caption_images` | `geo_cluster/captioning.py` |
| (summary: home, geo segments) | `geo_temporal_segment` | `geo_cluster/selector.py` |

## Notes

- **First run downloads models** — StreetCLIP (~600 MB) + BLIP (~1 GB). After
  that, per-image results are disk-cached (`~/.sim_bench/cache_streetclip.json`,
  `cache_blip.json`), so re-runs are instant.
- Inference is **CPU** here (no CUDA) — keep `limit` small (default 12) for snappy runs.
- StreetCLIP is the **GPS fallback**: it guesses the city when EXIF has no GPS.
  Quality varies (Budapest street scenes sometimes score Venice/Nice) — expand
  the candidate list in `geo_cluster/world_cities.py` or refine prompts to improve it.
- **Compat shim**: StreetCLIP/BLIP ship `.bin` only; transformers blocks that on
  torch < 2.6 (CVE-2025-32434). `geo_cluster/_hf_compat.py` narrowly no-ops that
  guard for these trusted repos. Production Albumify should move to torch ≥ 2.6.

These are real, registered pipeline steps — they drop into Albumify's
`default_pipeline` unchanged when ready.
