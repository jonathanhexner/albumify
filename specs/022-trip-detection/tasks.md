# Spec-022 Tasks — First PR (Slices 1 + 2)

**Scope**: EXIF geo/time extraction → adaptive geo-temporal segmentation → reverse-geocode
→ template caption ("Your trip to X") → grouped on the Results page.
**Out of scope this PR**: folium map, CV landmark recognition (Slice 4), local-LLM caption strategy
(stub the flag, template only), album-card/Explore enrichment (Slice 5).

See `PLAN.md` for the full design and the §5 intersection table.

---

## Slice 1 — Geo metadata extraction  ✅ DONE 2026-06-26

- [x] T1.1  `geo_cluster/types.py` — `GeoMetadata` dataclass `(timestamp, lat, lon)` + `has_geo`/`has_time`
- [x] T1.2  `geo_cluster/exif_reader.py` — `GeoMetadataExtractor.calc(ExifInputs) -> GeoMetadataResult`; parse `DateTimeOriginal`, `GPSInfo`; missing → `None`; reject dates < 1990 or > today+1. Pure `extract_from_exif` split out for testing.
- [x] T1.3  `sim_bench/pipeline/steps/extract_geo_metadata.py` — thin `@register_step` step → `context.geo_metadata`; registered in `all_steps.py`
- [x] T1.4  `context.py` — add `geo_metadata: dict[str, GeoMetadata]` field
- [x] T1.5  `configs/pipeline.yaml` — `extract_geo_metadata` after `discover_images` + config block (`min_year`)
- [x] T1.6  `tests/geo/test_geo_metadata.py` — 13 tests: GPS+time, time-only, GPS-only, neither, S/W hemisphere, pre-1990, future, malformed date, malformed GPS, real-file IO, missing file. All assert no raise on missing EXIF (FR-011). **13 passed.**

## Slice 1.5 — Pluggable axis framework + experiment (exploratory)  ✅ DONE 2026-06-26

Architecture-first scaffolding so the open product decisions become late-binding config.
See `PLAN.md §0` and `GEO_APPROACH.html` (multi-axis competition).

- [x] `geo_cluster/types.py` — `Segment`, `Segmentation`, `AxisScore`, `SegmentationOutcome`, `SegmentKind`
- [x] `geo_cluster/_clustering.py` — `cluster_locations` (haversine agglomerative), `split_on_time_gaps`
- [x] `geo_cluster/axes/` — `SegmentationAxis` protocol + `@register_axis` registry; `geo`, `time`, `identity` (stub-until-signal), `semantic` (stub) axes
- [x] `geo_cluster/scoring.py` — `QualityScorer`: separation/coverage/balance/stability **+ parsimony** (anti-fragmentation, found by the experiment)
- [x] `geo_cluster/selector.py` — `SegmentationSelector`: competition + floor → winner or FLAT
- [x] `geo_cluster/home.py` — `HomeAnchor` (dominant-location auto-detect)
- [x] `scripts/experiment_geo_segmentation.py` — synthetic scenarios + `--dir` real album → console + `EXPERIMENT_RESULTS.html`
- [x] `tests/geo/test_segmentation_selector.py` — 6 tests (traveler→geo, kids→time, wedding→FLAT, no-meta→FLAT, identity plug-in, floor forces FLAT)
- [x] **Finding:** naive splitting fragments everyday albums; fixed by (a) GEO = pure-location, (b) parsimony term. Real Budapest run: home auto-detected correctly; single-city trip → time wins.

## Slice 2 — Segmentation, geocode, caption, persist, display

- [ ] T2.1  `geo_cluster/segmenter.py` — `GeoTemporalSegmenter.calc(SegmentInputs) -> SegmentResult`; adaptive geo-first/time-first per spec §Algorithm; hierarchical GPS cluster (geo_radius_km); time-gap split; merge singletons; `has_nothing → "Unsorted"`
- [ ] T2.2  `geo_cluster/geocoder.py` — `ReverseGeocoder.calc(...)` via `reverse_geocoder` (offline); batch + cache centroids
- [ ] T2.3  `geo_cluster/describer.py` — `SegmentDescriber.calc(...)`; `caption_strategy=template` only this PR; `local_llm` raises `NotImplementedError` (flag wired, body deferred)
- [ ] T2.4  `geo_cluster/types.py` — add `GeoSegment` dataclass `(id, label, caption, date_start, date_end, city, country, image_paths, centroid)`
- [ ] T2.5  steps `geo_temporal_segment.py` + `describe_segments.py` (one class each) → `context.geo_segments`; placed after `assign_people_clusters`
- [ ] T2.6  `context.py` — add `geo_segments: list[GeoSegment]`
- [ ] T2.7  `configs/pipeline.yaml` — add both steps + config (`time_gap_hours: 8`, `geo_radius_km: 30`, `min_segment_size: 2`, `caption_strategy: template`)
- [ ] T2.8  **PERSIST** — `pipeline_service.py` ~L286: add `geo_segments=...` to `PipelineResult`; add per-image lat/lon/time into `_build_image_metrics` (one dict feeds blob + spec-086 tables)
- [ ] T2.9  `api/database/models.py` — `PipelineResult.geo_segments` JSON column; `image_repository.py` geo columns
- [ ] T2.10 `api/schemas/result.py` — `GeoSegment` Pydantic model; add to result schema
- [ ] T2.11 `api/services/result_service.py` + `routers/results.py` — `get_segments(job_id)` endpoint
- [ ] T2.12 `app/streamlit/` Results page — group photos by segment; show caption, date range, count; "Unsorted" last
- [ ] T2.13 Test `ut_GeoTemporalSegmenter` — multi-location/multi-time → geo-first; single-location → time-first; empty-metadata album → one "Unsorted", no raise
- [ ] T2.14 Integration test — small fixture album through pipeline; assert `geo_segments` populated and reaches the API response (end-to-end link, the §5 silent-break risk)

## Gate (before flipping spec → Implemented)

- [ ] T3.1  Add deps to `pyproject.toml`: `reverse_geocoder`; `.venv/Scripts/pip install -e .`
- [ ] T3.2  Run with **production-default** config (not relaxed)
- [ ] T3.3  `/code-review` → `REVIEW.md`; resolve High findings
- [ ] T3.4  Update `docs/architecture/{db_schemas,classes,data_flow}.html` (mandate)
- [ ] T3.5  v2 Budapest E2E gate **only if** a v2 tab was touched (Results-page grouping may not be v2 — confirm)
- [ ] T3.6  `CHANGES_LOG.md` entries per change
