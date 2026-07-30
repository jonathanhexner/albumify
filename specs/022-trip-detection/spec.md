# Feature Specification: Geo-Temporal Album Enrichment

**Created**: 2026-05-01
**Status**: In Progress (Slice 1 done — see PLAN.md / tasks.md)
**Spec**: 022

---

## Problem Statement

Photos in an album are treated as a flat set. The pipeline clusters by visual similarity
and face identity, but has zero awareness of **when** and **where** photos were taken.
A 1000-photo album from a year of travel is presented without any geographic or temporal
structure.

**Current state**: `DateTimeOriginal` is read for cache key generation (`image_cache.py`)
and discarded. GPS is never extracted. No time or location grouping exists.

**Goal**: Extract geo/time metadata per image, add a geo-temporal clustering layer to the
pipeline (above the existing scene/face clustering), enrich album views with location
segments sorted by date, and auto-generate positive-sentiment descriptions per segment.

---

## Core Concept

A new clustering dimension that sits **above** the current scene clustering hierarchy:

```
Album
  |-- Geo-temporal segment ("Dubai, May 2022" / "Austrian Alps, June 2022")
  |     |-- Scene clusters (existing)
  |     |     |-- Photos (sorted by date)
  |     |-- Face clusters (existing)
  |
  |-- Unsorted (photos without metadata)
```

### Adaptive Clustering Strategy

The algorithm picks the best segmentation axis based on the data:

| Data shape | Strategy | Example |
|------------|----------|---------|
| Multiple locations, spread over time | **Geo-first**: cluster by location, then sort by date within each | Year of travel photos |
| Single location, spread over time | **Time-first**: cluster by time gaps | All photos from home, over months |
| Multiple locations, same time window | **Geo-first**: cluster by location | Multi-city conference week |
| Single location, single time window | **No segmentation**: one segment | Weekend birthday party |

The strategy is chosen automatically, but the user can override via config.

### Location Clustering

Neighboring towns should group together — "Tel Aviv" and "Herzliya" are one segment,
not two. Use hierarchical clustering on GPS coordinates with a configurable radius
(default: ~30 km). This avoids fragmentation from minor location changes within a region.

---

## Integration Areas

| # | Area | What changes |
|---|------|-------------|
| 1 | **Metadata Extraction** | New pipeline step: read EXIF GPS + DateTime per image, persist in context |
| 2 | **Geo-Temporal Clustering** | New pipeline step: segment images by location + time, before scene clustering |
| 3 | **Location Enrichment** | Reverse-geocode GPS to city/country, match nearby landmarks |
| 4 | **Segment Description** | Auto-generate short positive-sentiment label per segment |
| 5 | **Album View** | Show segments when browsing album photos (Results page, Explore page) |
| 6 | **Album Card** | Enrich album summary with location/date overview |

---

## User Stories

### US1 — Geo-temporally segmented album browsing (P1)

When I view an album's results, photos are grouped into segments by location and date
instead of shown as a flat gallery. Each segment has a descriptive label, date range,
and photo count.

**Acceptance Criteria**:
- Photos with GPS cluster by geographic proximity (neighboring towns grouped)
- Within a geo cluster, photos are sorted by capture date
- If all photos are from one location, segment by time gaps instead
- Photos without metadata appear in an "Unsorted" group at the end
- Each segment shows: auto-generated description, date range, photo count

### US2 — Segment detail with location context (P1)

Each segment shows location information: city/country name, nearby famous landmark
(if any), and a small map with photo location pins.

**Acceptance Criteria**:
- Reverse-geocoded city/country name shown per segment
- If a famous landmark is within radius of segment centroid, show its name as a badge
- Small map (folium/Leaflet) showing photo pins, auto-zoomed to fit
- People detected within that segment's photos are listed

### US3 — Positive-sentiment description per segment (P1)

Each segment gets a short, warm, auto-generated description based on structured signals
(location, people, time of day, season) — not an LLM call.

**Acceptance Criteria**:
- Description is 1 sentence, ~10-15 words, positive sentiment
- Uses location name, landmark, people count, season/time signals
- Examples:
  - "A magical week exploring Dubai's stunning skyline"
  - "Beautiful family moments in the Austrian Alps"
  - "Lovely evening gathering, July 2022" (no GPS case)

### US4 — Configurable thresholds (P2)

**Acceptance Criteria**:
- `pipeline.yaml` exposes: `time_gap_hours` (default: 8), `geo_radius_km` (default: 30),
  `min_segment_size` (default: 2)
- UI controls in Configure & Run page

### US5 — Album card enrichment (P2)

Album cards/summary show a location overview — e.g. "3 locations: Dubai, Innsbruck, Paris"
and date span.

### US6 — Manual segment editing (P3)

Rename, merge, or split auto-detected segments.

---

## Algorithm

### Step 1: Extract Metadata
- Read EXIF per image: `DateTimeOriginal`, `GPSInfo` (lat/lon), `OffsetTimeOriginal`
- Output: per-image `(timestamp, lat, lon)` tuples. Missing fields → `None`.

### Step 2: Geo-Temporal Segmentation

```
1. Partition images into: has_geo (GPS+time), has_time_only, has_nothing
2. For has_geo images:
   a. Cluster GPS coordinates using hierarchical clustering (geo_radius_km)
   b. Within each geo cluster, sort by timestamp
   c. Split on time gaps > time_gap_hours → sub-segments
3. For has_time_only images:
   a. Sort by timestamp
   b. Split on time gaps > time_gap_hours
4. Merge singletons into nearest adjacent segment
5. Per segment: compute centroid, reverse-geocode, match landmark, generate description
6. has_nothing → "Unsorted" group
```

### Reverse Geocoding
- Offline via `reverse_geocoder` (GeoNames dataset, KD-tree, no API keys, ~15 MB)

### Landmark Matching
- Bundled lightweight dataset (~1000 major world landmarks with lat/lon/name)
- Match if segment centroid is within radius (default: 2 km city, 10 km natural)

### Map Display
- `folium` (Leaflet.js wrapper) embedded in Streamlit via `st.components.v1.html()`
- OpenStreetMap tiles (free, no API key)
- Auto-zoom to fit segment's photo pins

---

## Edge Cases

- **No EXIF at all**: "Unsorted" group. Pipeline must not fail.
- **Timestamps but no GPS**: Time-gap segmentation only. No map, no landmark.
- **GPS but no timestamps**: Geo clustering only, no date sorting within segment.
- **Neighboring towns**: Hierarchical clustering with geo_radius_km handles this.
- **Timezone ambiguity**: Local time is fine — we need relative ordering, not absolute.
- **HEIC files**: Supported via `pillow-heif`.
- **Invalid dates**: Reject < 1990 or > today + 1 day → "Unsorted".
- **10k+ photos**: Offline geocoder is O(log n) per query. Batch and cache.
- **Privacy**: GPS stays local. Only network call is map tile requests at display time.

---

## Requirements

| ID | Requirement |
|----|-------------|
| FR-001 | Pipeline step `extract_metadata`: EXIF GPS + DateTime per image |
| FR-002 | Pipeline step `geo_temporal_cluster`: segments images before scene clustering |
| FR-003 | Hierarchical location clustering with configurable radius |
| FR-004 | Adaptive strategy: geo-first vs time-first based on data shape |
| FR-005 | Offline reverse geocoding (no API keys) |
| FR-006 | Landmark matching against bundled dataset |
| FR-007 | Auto-generated positive-sentiment description per segment |
| FR-008 | Segments shown in album results view with map + landmark |
| FR-009 | Album card enriched with location/date summary |
| FR-010 | Configurable thresholds in pipeline.yaml |
| FR-011 | Graceful degradation when EXIF data is missing |
| NFR-001 | Metadata extraction < 2s for 1000 images |
| NFR-002 | No external API calls during pipeline execution |

---

## New Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `reverse_geocoder` | Offline GPS to city/country | ~15 MB |
| `folium` | Map widget (Leaflet.js, OpenStreetMap tiles) | ~1 MB |

---

## UI Mock

See `specs/022-trip-detection/mock.html` — shows album results view with geo-temporal
segments, landmark badges, map panels, and people list per segment.

---

## Open Questions

1. **Landmark dataset**: Curate our own or find an existing lightweight one?
2. **Description generation**: Template-only for V1, or optionally use LLM?
3. **Pipeline ordering**: `extract_metadata` runs right after `discover_images`.
   `geo_temporal_cluster` runs before `cluster_scenes`. Confirm?
4. **Album subdivision**: Should we support splitting a large album into sub-albums
   by geo segment? (V2 candidate)
