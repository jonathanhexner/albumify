# Spec-022 Implementation Plan — Geo-Temporal Enrichment

**Created**: 2026-06-26  **Status**: In Progress (Slice 1 done; axis framework + experiment landing)
**Builds on**: `spec.md` (SPEC READY 2026-05-01) · `GEO_APPROACH.html` (visual design w/ multi-axis competition)

---

## 0. Architecture for flexibility (the pluggable-axis core)

The whole point: the 4 open product decisions (which axes, who names, how "home" is found, which
captioner) become **late-binding config**, not baked-in code. We do that with one small set of
abstractions in `geo_cluster/`. A new way to group an album = one new class + one decorator.

```
                AxisInputs (metadata, home, people_clusters, captions, config)
                     │
   ┌─────────────────┼───────────────────┐         each axis is independent &
   ▼                 ▼                   ▼          registered via @register_axis
GeoAxis           TimeAxis          IdentityAxis ... SemanticAxis
  .propose() ──► Segmentation        (returns None if its signal is absent)
                     │
                     ▼
   QualityScorer.calc(seg, inputs, perturbed) ──► AxisScore
        separation · coverage · balance · stability ─► overall 0–1
                     │
                     ▼
   SegmentationSelector.calc(inputs) ──► SegmentationOutcome
        best axis ≥ floor ? winner : FLAT (no story imposed)
```

| Abstraction | File | Role | Why it gives flexibility |
|---|---|---|---|
| `SegmentationAxis` (Protocol) | `axes/base.py` | `propose(inputs) -> Segmentation \| None` | add an axis without touching the selector |
| `@register_axis` + `get_axes()` | `axes/base.py` | registry | enable/disable per config; discoverable |
| `AxisInputs` | `axes/base.py` | one bag of every signal an axis *might* need | new signals (captions, people) added once |
| `Segmentation` / `Segment` | `types.py` | candidate grouping + its `score_space`/`metric` | axes carry their own distance space for scoring |
| `QualityScorer` | `scoring.py` | axis-agnostic 0–1 score from 4 signals | tune weights/floor in config, not code |
| `SegmentationSelector` | `selector.py` | runs axes, scores, picks winner or FLAT | the "decline to segment" gate lives here |
| `HomeAnchor` | `home.py` | dominant-location anchor | swap auto/manual without touching axes |

**Late-binding of the 4 decisions:**
- *Which axes?* → `enabled` list in config (default: geo, time, identity; semantic off).
- *Who names the kind?* → a separate `SegmentClassifier`/`SegmentDescriber` strategy, chosen by config.
- *Home definition?* → `HomeAnchor` strategy (auto today; manual pin later) — axes just receive `home`.
- *Which captioner?* → only the `SemanticAxis` cares; everything else is unaffected.

Config shape (future `pipeline.yaml` block, mirrored by the experiment's CLI):
```yaml
geo_temporal_segment:
  enabled_axes: [geo, time, identity]      # semantic added in Slice 5
  floor: 0.45                              # below this => FLAT, no segmentation
  weights: {separation: .35, coverage: .25, balance: .20, stability: .20}
  geo_radius_km: 30
  time_gap_hours: 8
```

The experiment in §B exercises exactly these abstractions standalone, before any pipeline wiring.

---

## 1. Guiding decisions (what changed vs the original spec)

| Topic | Original spec | This plan |
|---|---|---|
| Geo as pipeline steps | "new pipeline step" (vague) | **4 discrete steps**, one-class-per-file, ≤80 LOC, domain logic in new `geo_cluster/` package with `calc()` entry (spec-053 convention) |
| Landmark detection | GPS-proximity match vs bundled lat/lon table | **DECIDED 2026-06-26: GPS-proximity table only for now.** CV step `recognize_landmarks` is designed (§4) but deferred to Slice 4 — not in the first PR. |
| Description | template-only | **DECIDED: template default + optional LOCAL LLM behind a flag** (`caption_strategy: template \| local_llm`). Never a network call (NFR-002). |
| Persistence | unspecified | rides the **existing `PipelineResult` blob + spec-086 normalized tables** — one new sub-object, no new write path |

---

## 2. The pipeline steps (where geo logic lives)

Insert into `default_pipeline` in `configs/pipeline.yaml`:

```
discover_images
  -> extract_geo_metadata        # (A) NEW  EXIF GPS+time per image -> context.geo_metadata
score_iqa ... extract_scene_embedding
  -> recognize_landmarks         # (B) NEW, OPTIONAL  CV model -> context.landmark_predictions
cluster_scenes ... assign_people_clusters
  -> geo_temporal_segment        # (C) NEW  segment images by geo+time -> context.geo_segments
  -> describe_segments           # (D) NEW  reverse-geocode + landmark + caption per segment
cluster_by_identity ... select_best
```

**Why these positions**
- **(A) right after discover** — cheap EXIF read, no deps, fail-safe (missing EXIF -> `None`).
- **(B) after `extract_scene_embedding`** — can **reuse the OpenCLIP image embedding** already in `context.scene_embeddings` (zero-shot, no second forward pass) OR run a dedicated model. Uses GPS from (A) to narrow candidates.
- **(C)/(D) after `assign_people_clusters`** — segment captions reference **people count** ("Beautiful family moments…"), so people clusters must exist first. Segmentation itself only needs (A).

Each step is a thin translator: read context -> build `Inputs` dataclass -> `helper.calc(inputs)` -> write `Result` to context. No domain math in the step file.

---

## 3. New domain package `geo_cluster/` (parallel to `face_cluster/`)

```
geo_cluster/
  exif_reader.py        GeoMetadataExtractor.calc(ExifInputs) -> GeoMetadataResult
  segmenter.py          GeoTemporalSegmenter.calc(SegmentInputs) -> SegmentResult   # adaptive geo/time
  geocoder.py           ReverseGeocoder.calc(...) -> city/country  (reverse_geocoder, offline)
  landmarks.py          LandmarkMatcher.calc(...) -> landmark by GPS proximity (bundled table)
  landmark_cv.py        LandmarkRecognizer.calc(...) -> landmark by vision model   # OPTIONAL
  describer.py          SegmentDescriber.calc(...) -> caption  (template strategy; LLM pluggable)
  types.py              GeoMetadata, GeoSegment dataclasses (framework-agnostic)
```

Helpers are notebook-callable directly; pipeline steps must call `calc()` so ordering
constraints (e.g. "geocode before describe") are enforced by construction.

---

## 4. Optional CV landmark recognition — design (your ask)

Three model strategies, ranked by fit. All gated by `recognize_landmarks.enabled` (default **false**).

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **CLIP zero-shot, GPS-narrowed** (recommended) | reverse-geocode GPS -> country/region -> pull candidate landmark names for that region -> score the OpenCLIP image embedding (already computed in `extract_scene_embedding`) against `"a photo of {landmark}"` text prompts | no training, ~0 extra compute (reuses embedding), GPS prunes the candidate set so accuracy is high, works offline after model download | needs a region->landmarks lookup table |
| Dedicated landmark model (GLDv2 / DELG) | run a pretrained Google-Landmarks model | recognizes landmark **without** GPS | heavy (~hundreds MB), another forward pass, weak on non-famous places |
| Scene/category classifier (Places365) | classify "cathedral/beach/mountain" | tiny | gives a *category*, not a named landmark |

**Recommended flow** — GPS and vision **cross-check** each other:

```
        GPS present?
        /          \
      yes           no
       |             |
 region candidates   global top-K candidates
       |             |
       +------> CLIP zero-shot score image ------+
                                                  |
                          confidence >= thresh ?  |
                          yes -> name the landmark, mark source = {gps+cv | cv-only}
                          no  -> fall back to GPS-proximity table (source = gps) or none
```

This makes the CV model *purely additive*: off -> behaves exactly like the original
GPS-only spec; on -> can name a landmark even when GPS is absent, and can correct a
wrong GPS-proximity guess.

---

## 5. Main intersection points (end-to-end data trace)

The spec-053 + "trace data end-to-end" mandate: every link must be wired or the feature is silently broken.

```
LAYER                         FILE / SYMBOL                                  CHANGE
----------------------------- ---------------------------------------------- --------------------------------
1 Config                      configs/pipeline.yaml                          + 4 step names in default_pipeline
                                                                             + 4 config blocks (thresholds)
2 Step registry               sim_bench/pipeline/steps/<one file each>       + 4 thin Step classes (@register_step)
                              extract_geo_metadata.py / recognize_landmarks.py
                              geo_temporal_segment.py / describe_segments.py
3 Domain logic                geo_cluster/*.py                               + new package (calc() helpers)
4 Context                     sim_bench/pipeline/context.py                  + geo_metadata, landmark_predictions,
                                                                               geo_segments fields
5 Persist (blob+tables)       api/services/pipeline_service.py ~L286         + geo_segments=... on PipelineResult
                              api/services/pipeline_service.py _build_image_metrics + per-image lat/lon/time/landmark
6 DB model                    api/database/models.py (PipelineResult)        + geo_segments JSON column
                              api/repositories/image_repository.py           + geo columns (spec-086 normalized)
7 API schema                  api/schemas/result.py                          + GeoSegment, GeoSummary models
8 API read service            api/services/result_service.py get_clusters/   + get_segments(job_id)
                              new get_segments + router in routers/results.py
9 Frontend                    app/streamlit/ Results + Explore pages         + segment grouping, map (folium),
                              app/streamlit/ album card                        landmark badge, caption, date range
10 Architecture docs          docs/architecture/{db_schemas,classes,data_flow}.html  + new column/model/step (mandate)
```

**Highest-risk links** (where features usually go silently broken here):
- **5 -> 7 -> 9**: a new context field that never reaches `PipelineResult` shows nothing in UI. The blob and the spec-086 normalized tables both derive from one dict (`image_metrics`) — add geo there once, not twice.
- **A -> C**: EXIF timezone/`None` handling. `geo_temporal_segment` must treat missing GPS/time as "Unsorted", never raise (FR-011).
- **dual-persistence trap** (memory `dual_persistence_blob_vs_rundb`): Albumify reads the **JSON blob**, not `face_clustering.db`. Wire geo into the blob path or the FC app won't see it.

---

## 6. Dependencies

| Package | For | Size | Notes |
|---|---|---|---|
| `reverse_geocoder` | offline GPS->city/country | ~15 MB | no API key |
| `folium` | map widget | ~1 MB | OSM tiles, network only at display time |
| `open_clip_torch` | CV landmark (if reusing CLIP) | already a scene-embed option | reuse, no new dep if `extract_scene_embedding.model=openclip` |
| region->landmarks table | CV candidate pruning | ~small JSON | curate from Wikidata/GeoNames top-N per country |

---

## 7. Build order (proposed task slices — each independently testable)

1. **Slice 1 — metadata only**: step (A) + `exif_reader.calc` + context field + unit test on a fixture with/without GPS. No UI. Proves EXIF extraction < 2s/1000 imgs (NFR-001).
2. **Slice 2 — segmentation + geocode**: steps (C)(D, template caption, no landmark) + segmenter/geocoder helpers + persist `geo_segments` through layers 5-8 + Results-page grouping (no map yet). First visible "Your trip to X" output.
3. **Slice 3 — map + GPS landmark badge**: folium panel + `landmarks.py` proximity match.
4. **Slice 4 — optional CV landmark**: step (B) + `landmark_cv.py` (CLIP zero-shot, GPS-narrowed), config-gated off by default.
5. **Slice 5 — album card + Explore page enrichment** (US5).

Each slice ends with: production-default config test + the **v2 Budapest E2E gate** if any v2 tab is touched.

---

## 8. Decisions (2026-06-26) & remaining open questions

**Decided:**
1. **CV model** — *Skip for now.* GPS-proximity landmark table only. CV step `recognize_landmarks` (§4) deferred to Slice 4.
2. **Caption** — *Template default + optional local LLM behind a flag.* `describe_segments.caption_strategy: template | local_llm`. No network.
3. **First PR** — *Slices 1 + 2*: metadata extraction + segmentation + reverse-geocode + caption grouping, visible on the Results page. No map, no CV.

**Still open (not blocking the first PR):**
4. **Segment vs album**: keep segments as a *view* over one album, or allow splitting a big album into sub-albums by segment (spec OQ#4, V2)? → defer.
5. **Local LLM choice**: which small local instruct model + where it's hosted? → decide before building the `local_llm` strategy (template path ships first regardless).
