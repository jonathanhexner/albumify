# Implementation Plan: Embed Cache

**Date**: 2026-04-11
**Status**: Draft — awaiting approval

## Problem

A full `pipeline.run()` takes 2-10 minutes depending on album size. ~85-90% of that time is the `embed` stage (InsightFace model load + per-image detection + embedding extraction). When users iterate on clustering/merge parameters, they re-run the entire pipeline even though the embeddings haven't changed.

`recluster()` exists but requires a completed prior run and creates a separate output directory. There is no transparent caching.

## Goal

Cache the expensive stages (discover, embed, quality, crops) so that subsequent runs on the same image directory skip them automatically when valid. Clustering and everything after it always re-runs with current config.

## Design

### Two-layer model

```
Layer 1: EMBED CACHE (expensive, rarely invalidated)
  What:  FaceRecords with embeddings, aligned faces, pose, bbox, area
  Key:   image_dir fingerprint + EMBED_CACHE_VERSION
  When:  Reuse if same images, same code version. Ignore config entirely.
         Embed stage has no config params — model and det_size are hardcoded.

Layer 2: QUALITY GATE (cheap, always re-run)
  What:  core_indices / holdout_indices split
  Why:   Quality config (blur_min, yaw_max, etc.) changes frequently.
         Re-running quality on cached FaceRecords costs <1 second.
         Not worth caching separately.

Clustering / Exemplars / Merge / Export: always re-run (cheap, config-dependent)
```

### Why not cache quality too?

Quality gating depends on 6+ config params that users tweak often. Caching it would require a composite key of all quality params, and invalidation would be frequent. Since it runs in <1 second on cached faces, the complexity isn't justified.

### Cache location

```
{output_dir}/.embed_cache/
    cache_meta.json          # fingerprint, version, timestamps
    faces_cache.pkl          # List[FaceRecord] with embeddings + aligned_face
    embeddings.npy           # n_faces x 512 float32 (redundant but fast to load)
    embedding_face_ids.npy   # face_id per embedding row
    crop_manifest.json       # face_id -> relative crop path
    crops/                   # aligned face JPEGs
```

Alternative: `{image_dir}/.face_cluster_cache/` — keeps cache next to source images. Pro: survives output_dir changes. Con: writes to user's photo directory (may be read-only, surprising).

**Recommendation**: Use `{output_dir}/.embed_cache/`. If output_dir changes between runs, the cache is lost, but this is acceptable — users typically reuse the same output_dir when iterating.

### Cache key: image directory fingerprint

```python
def _compute_image_fingerprint(image_dir: Path, extensions: Set[str]) -> str:
    """Deterministic hash of the image set in a directory."""
    entries = []
    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() in extensions:
            stat = path.stat()
            entries.append(f"{path.relative_to(image_dir)}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()
```

This catches:
- New images added
- Images deleted
- Images modified (mtime changes)
- Filename renames

Does NOT catch:
- Bit-identical copies with fresh mtime (acceptable — rare, causes unnecessary re-embed)
- Images moved to subdirectories (caught — relative path changes)

### Cache version stamp

```python
# face_cluster/embedding.py (or config.py)
EMBED_CACHE_VERSION = "1"
```

Bump manually when:
- Face detection model changes (e.g., buffalo_l -> buffalo_sc)
- Detection size changes (640x640 -> 320x320)
- Alignment logic changes (norm_crop parameters)
- Embedding normalization changes
- FaceRecord fields change in a way that affects downstream stages

Do NOT bump for:
- Logging changes
- Error handling improvements
- Comment/docstring edits
- Quality gating changes (quality always re-runs)
- Clustering/merge changes (always re-run)

### cache_meta.json schema

```json
{
    "embed_cache_version": "1",
    "image_fingerprint": "sha256:abcdef...",
    "image_dir": "/path/to/images",
    "n_images": 247,
    "n_faces": 1023,
    "created_at": "2026-04-11T14:30:00",
    "embed_time_seconds": 142.3
}
```

### Pipeline integration

**Modified `run()` flow:**

```
1. discover         — always run (fast, gets image list)
2. CHECK CACHE      — new step
   a. Compute image fingerprint
   b. Look for {output_dir}/.embed_cache/cache_meta.json
   c. If exists AND fingerprint matches AND version matches:
      - Load faces from faces_cache.pkl
      - Load crop_manifest
      - Log: "Reusing embed cache ({n_faces} faces from {created_at})"
      - SKIP embed + crops
   d. If no valid cache:
      - Run embed + crops normally
      - Write cache artifacts to .embed_cache/
      - Log: "Embed cache written ({n_faces} faces)"
3. quality           — always re-run on (cached or fresh) faces
4. cluster           — always re-run
5. exemplars         — always re-run
6. merge             — always re-run
7. export            — always re-run
```

**`recluster()` is unchanged.** It loads from a completed run, not from the embed cache. The embed cache is internal to `run()`.

**Config flag:**

```python
# face_cluster/config.py
class PipelineConfig:
    embed_cache_enabled: bool = True   # Set False to force re-embed
```

### What gets stored in faces_cache.pkl

A full `List[FaceRecord]` including:
- `face_id`, `image_id`, `image_path`
- `bbox`, `area`, `landmarks`
- `embedding`, `embedding_normalized` (512-dim float32)
- `aligned_face` (112x112x3 uint8 numpy array)
- `pose` (yaw, pitch, roll)
- `blur_score` (pre-computed during embed via Laplacian)

**Why pickle and not CSV?**
- `aligned_face` is a numpy array (112x112x3) — doesn't fit in CSV
- `embedding` is 512-dim float — CSV works but is 10x slower to parse
- Pickle preserves exact types and is fast
- `embeddings.npy` is also saved separately for tools that need raw numpy access

**Why save aligned_face?**
- Quality gating computes blur_score from `aligned_face` (Laplacian on grayscale)
- Without it, quality gating would need to re-extract aligned faces from crops (I/O)
- With it, quality re-runs are pure in-memory compute (<1 second)

**Size estimate** (per 1000 faces):
- aligned_face: 112 * 112 * 3 * 1000 = ~37 MB
- embeddings: 512 * 4 * 1000 = ~2 MB
- metadata: ~1 MB
- crops (JPEG): ~20-40 MB
- **Total: ~60-80 MB per 1000 faces**

### Cache invalidation

| Event | What happens |
|-------|-------------|
| Image added/removed/modified | Fingerprint changes -> full re-embed |
| Quality config changed | Cache still valid -> quality re-runs on cached faces |
| Clustering config changed | Cache still valid -> cluster re-runs |
| `EMBED_CACHE_VERSION` bumped | Version mismatch -> full re-embed |
| `embed_cache_enabled = False` | Cache ignored -> full re-embed |
| `output_dir` changed | No cache found -> full re-embed (new dir) |
| Corrupt cache (shape mismatch) | Validation fails -> full re-embed |

### Cache validation on load

```python
def _validate_cache(meta, faces, image_fingerprint):
    if meta["embed_cache_version"] != EMBED_CACHE_VERSION:
        return False, f"version mismatch: {meta['embed_cache_version']} != {EMBED_CACHE_VERSION}"
    if meta["image_fingerprint"] != image_fingerprint:
        return False, "image fingerprint changed"
    if not faces:
        return False, "empty face list"
    # Spot-check: first face has embedding
    if faces[0].embedding_normalized is None:
        return False, "missing embeddings"
    if faces[0].embedding_normalized.shape != (512,):
        return False, f"bad embedding shape: {faces[0].embedding_normalized.shape}"
    return True, "ok"
```

### Logging

Cache hits and misses must be clearly visible in the run log:

```
[INFO] Computing image fingerprint for 247 images...
[INFO] Fingerprint: sha256:a1b2c3... (247 images, 1.2 GB)
[INFO] Found embed cache: 1023 faces from 2026-04-11T14:30:00
[INFO] Cache valid (version=1, fingerprint match) — skipping embed + crops
[INFO] Loaded 1023 faces from cache in 0.8s (vs ~142s for fresh embed)
[INFO] Re-running quality gate with current config...
[INFO] Quality gate: 891 core, 132 holdout (blur_min=50, yaw_max=45)
```

Or on cache miss:

```
[INFO] Computing image fingerprint for 247 images...
[INFO] Fingerprint: sha256:d4e5f6... (247 images, 1.2 GB)
[INFO] No valid embed cache found (reason: image fingerprint changed)
[INFO] Running full embed stage...
[INFO] Embed complete: 1023 faces in 142.3s
[INFO] Writing embed cache to .embed_cache/ ...
```

## File changes

| File | Change |
|------|--------|
| `face_cluster/config.py` | Add `embed_cache_enabled: bool = True` and `EMBED_CACHE_VERSION = "1"` |
| `face_cluster/pipeline.py` | Add cache check between discover and quality. Add cache write after embed+crops. |
| `face_cluster/cache.py` | **New file.** `compute_image_fingerprint()`, `load_embed_cache()`, `save_embed_cache()`, `validate_cache()` |
| `face_cluster/__init__.py` | No change (cache is internal, not public API) |
| `tests/face_clustering/test_embed_cache.py` | **New file.** Tests for cache hit/miss/invalidation/corruption |

## Tasks

### Phase 1: Cache infrastructure

- [x] T001 Create `face_cluster/cache.py` with `compute_image_fingerprint()`, `save_embed_cache()`, `load_embed_cache()`, `validate_cache()`, `clear_embed_cache()`, `get_cache_info()`
- [x] T002 Add `EMBED_CACHE_VERSION = "1"` constant to `face_cluster/cache.py`
- [x] T003 Add `embed_cache_enabled: bool = True` to `PipelineConfig`
- [x] T004 Write `tests/face_clustering/test_embed_cache.py` — 21 tests, all pass

### Phase 2: Pipeline integration

- [x] T005 Modify `FaceClusteringPipeline.run()` to check cache after discover, skip embed on hit
- [x] T006 Add cache write after embed completes (non-fatal on write failure)
- [x] T007 Quality stage always re-runs on cached faces (blur_score=0.0 in cache → recomputed by QualityGater)
- [ ] T008 Write `test_pipeline_uses_cache_on_second_run` (E2E — requires InsightFace, deferred)
- [ ] T009 Write `test_pipeline_invalidates_cache_on_image_change` (E2E — requires InsightFace, deferred)

### Phase 3: UI integration

- [x] T010 Show cache status in Streamlit Run tab (face count, creation date, time saved)
- [x] T011 Add "Clear embed cache" button in Run tab
- [ ] T012 Add `--no-cache` flag to CLI (no CLI entry points exist yet; defer)

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Pickle version incompatibility across Python upgrades | Low | Cache unusable | Validate on load; fall back to re-embed |
| Stale cache after image metadata edit (no content change, no mtime change) | Very low | Wrong results | Document: cache keys on mtime, not content hash |
| Large cache size for big albums (10k faces = ~600 MB) | Medium | Disk usage | Log cache size; consider configurable max |
| Cache corruption from interrupted write | Low | Invalid cache | Write to temp dir, atomic rename; validate on load |
| User doesn't realize they're seeing cached results | Medium | Confusion | Clear log messages; show cache age in UI |

## Out of scope

- Per-image incremental cache (add 5 images to a 1000-image album without re-embedding all) — significant complexity, defer to v2
- Distributed/shared cache across machines
- Caching quality gate results (too cheap to justify)
- Caching cluster results (too cheap, config-dependent)

## Open questions

1. **Cache location**: `{output_dir}/.embed_cache/` or `{image_dir}/.face_cluster_cache/`? Current recommendation is output_dir. If users want cache to survive output_dir changes, we could add a `cache_dir` config param later.

2. **Atomic writes**: Should we write to a temp dir and rename? Pickle writes are not atomic — an interrupted write produces a corrupt file. The validation-on-load catches this, but atomic writes prevent it entirely.

3. **Blur score**: Currently computed in quality stage from `aligned_face`. Should we pre-compute it during embed and store it in the cache? This would make the quality stage even faster (no Laplacian needed). The downside is that blur_score computation is coupled to embed cache version.
