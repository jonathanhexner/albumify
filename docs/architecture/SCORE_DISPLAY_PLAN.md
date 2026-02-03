# Plan: Surface All Per-Image Scores for Full Reproducibility

## Motivation

The pipeline computes and stores 9 fields per image in `PipelineResult.image_metrics` (JSON column in SQLite). However, only 5 of those fields make it through the result service to the API, and even fewer reach the Streamlit UI. The old album app (`app/album/`) displayed portrait indicators (eyes/smile), sharpness, and a full downloadable metrics table. The new Streamlit app (`app/streamlit/`) shows none of this. Goal: full reproducibility of results with all scores visible and exportable.

Additionally, two bugs were found:
1. The **siamese model is never loaded** due to a config key mismatch.
2. The **clustering step drops most HDBSCAN parameters**, using `min_cluster_size=3` instead of the validated `2`.

---

## Current Data Flow (with loss points marked)

```
PipelineContext          -->  pipeline_service.py  -->  PipelineResult.image_metrics (DB)
  .iqa_scores                   iqa_score                 iqa_score           OK
  .ava_scores                   ava_score                 ava_score           OK
  .sharpness_scores             sharpness                 sharpness           OK
  .scene_cluster_labels         cluster_id                cluster_id          OK
  .faces (len)                  face_count                face_count          OK
  .face_pose_scores             face_pose_scores          face_pose_scores    OK
  .face_eyes_scores             face_eyes_scores          face_eyes_scores    OK
  .face_smile_scores            face_smile_scores         face_smile_scores   OK
  .selected_images              is_selected               is_selected         OK

PipelineResult.image_metrics  -->  result_service.py  -->  API response
  iqa_score                        iqa_score               iqa_score          OK
  ava_score                        ava_score               ava_score          OK
  sharpness                        sharpness               sharpness          OK
  cluster_id                       cluster_id              cluster_id         OK
  face_count                       face_count              face_count         OK
  face_pose_scores                 DROPPED                 ---               LOST
  face_eyes_scores                 DROPPED                 ---               LOST
  face_smile_scores                DROPPED                 ---               LOST
  is_selected                      DROPPED                 ---               LOST

API response  -->  api_client.py  -->  ImageInfo model  -->  gallery.py
  iqa_score        iqa_score          iqa_score             "IQA: 0.72"      OK
  ava_score        ava_score          ava_score             "AVA: 6.3"       OK
  sharpness        DROPPED            ---                   ---              LOST
  face_count       face_count         face_count            "2 faces"        OK
  cluster_id       cluster_id         cluster_id            ---              OK
  is_selected      is_selected        is_selected           "Selected"       OK
  ---              ---                ---                   no portrait info  LOST
```

---

## Bug 1: Siamese Model Never Loaded

**Root cause**: Config key mismatch between `pipeline.yaml` and `select_best.py`.

**`configs/pipeline.yaml` (lines 104-109)** stores siamese config as nested dict:
```yaml
select_best:
  siamese:
    enabled: true
    checkpoint_path: models/album_app/siamese_comparison_model.pt
    duplicate_threshold: 0.95
    tiebreaker_range: 0.05
```

**`sim_bench/pipeline/steps/select_best.py` (line 128)** reads a flat key:
```python
siamese_checkpoint = config.get("siamese_checkpoint")  # Always None!
```

The step never sees the checkpoint path because `config["siamese"]` is a dict, not `config["siamese_checkpoint"]`.

**Fix** (`select_best.py` line 128): Read from the nested structure:
```python
siamese_config = config.get("siamese", {})
siamese_checkpoint = siamese_config.get("checkpoint_path") if siamese_config.get("enabled", False) else None
tiebreaker_threshold = siamese_config.get("tiebreaker_range", config.get("tiebreaker_threshold", 0.05))
duplicate_threshold = siamese_config.get("duplicate_threshold", config.get("duplicate_similarity_threshold", 0.95))
```

**File**: `sim_bench/pipeline/steps/select_best.py` lines 122-131

---

## Bug 2: Clustering Parameters Dropped

**Root cause**: `cluster_scenes.py` only passes `min_cluster_size` to the clustering algorithm. All other parameters (`metric`, `min_samples`, `cluster_selection_epsilon`, `cluster_selection_method`) are dropped.

**Reference config** (produced excellent results):
```yaml
# From configs/run.cluster_budapest_hdbscan.yaml
params:
  metric: cosine
  min_cluster_size: 2
  min_samples: 2
  cluster_selection_epsilon: 0.0
  cluster_selection_method: eom
```

**Current pipeline config** (`configs/pipeline.yaml` lines 75-78):
```yaml
cluster_scenes:
  method: hdbscan
  min_cluster_size: 3    # Different from reference (2)
  min_samples: 2         # Present but NEVER PASSED to algorithm
```

**Current code** (`cluster_scenes.py` lines 45-51) drops everything except `min_cluster_size`:
```python
clustering_config = {
    "algorithm": config.get("algorithm", "hdbscan"),
    "params": {
        "min_cluster_size": config.get("min_cluster_size", 2)   # Only param passed!
    },
    "output": {}
}
```

**Result**: HDBSCAN runs with `min_cluster_size=3` (from pipeline.yaml), `min_samples=None` (HDBSCAN library default = min_cluster_size = 3), instead of the validated `min_cluster_size=2, min_samples=2`.

**Fix** (`cluster_scenes.py` lines 45-51): Pass all parameters through:
```python
clustering_config = {
    "algorithm": config.get("method", config.get("algorithm", "hdbscan")),
    "params": {
        "min_cluster_size": config.get("min_cluster_size", 2),
        "min_samples": config.get("min_samples", 2),
        "metric": config.get("metric", "cosine"),
        "cluster_selection_epsilon": config.get("cluster_selection_epsilon", 0.0),
        "cluster_selection_method": config.get("cluster_selection_method", "eom"),
    },
    "output": {}
}
```

**Also fix** `configs/pipeline.yaml` line 77: Change `min_cluster_size: 3` to `min_cluster_size: 2` to match reference.

---

## Score Display Changes (7 files, in dependency order)

### File 1: `sim_bench/api/services/result_service.py`

**What**: Pass through all stored fields from `image_metrics` instead of cherry-picking 5.

**Current** (repeated 4 times at lines 94-101, 111-118, 133-140, 144-151):
```python
images.append({
    'path': path,
    'iqa_score': metrics.get('iqa_score'),
    'ava_score': metrics.get('ava_score'),
    'sharpness': metrics.get('sharpness'),
    'cluster_id': metrics.get('cluster_id'),
    'face_count': metrics.get('face_count')
})
```

**Change**: Extract a `_build_image_dict` method and add the missing fields:
```python
def _build_image_dict(self, path: str, metrics: dict, is_selected: bool = False) -> dict:
    return {
        'path': path,
        'iqa_score': metrics.get('iqa_score'),
        'ava_score': metrics.get('ava_score'),
        'sharpness': metrics.get('sharpness'),
        'cluster_id': metrics.get('cluster_id'),
        'face_count': metrics.get('face_count', 0),
        'face_pose_scores': metrics.get('face_pose_scores'),
        'face_eyes_scores': metrics.get('face_eyes_scores'),
        'face_smile_scores': metrics.get('face_smile_scores'),
        'is_selected': metrics.get('is_selected', is_selected),
    }
```

Replace all 4 inline dict constructions with calls to `self._build_image_dict(path, metrics)`. For the `selected_only` branch, pass `is_selected=True`.

---

### File 2: `sim_bench/api/schemas/result.py`

**What**: Add missing fields to the `ImageMetrics` Pydantic model (lines 8-15).

**Current**:
```python
class ImageMetrics(BaseModel):
    path: str
    iqa_score: Optional[float] = None
    ava_score: Optional[float] = None
    sharpness: Optional[float] = None
    cluster_id: Optional[int] = None
    face_count: Optional[int] = None
```

**Add these fields**:
```python
    face_pose_scores: Optional[list[float]] = None
    face_eyes_scores: Optional[list[float]] = None
    face_smile_scores: Optional[list[float]] = None
    is_selected: bool = False
```

---

### File 3: `app/streamlit/models.py`

**What**: Add missing fields to the `ImageInfo` dataclass (lines 48-58).

**Current**:
```python
@dataclass
class ImageInfo:
    path: str
    filename: str
    iqa_score: Optional[float] = None
    ava_score: Optional[float] = None
    composite_score: Optional[float] = None
    face_count: int = 0
    cluster_id: Optional[int] = None
    is_selected: bool = False
    thumbnail_url: Optional[str] = None
```

**Add these fields** (before `thumbnail_url`):
```python
    sharpness: Optional[float] = None
    face_pose_scores: Optional[list[float]] = None
    face_eyes_scores: Optional[list[float]] = None
    face_smile_scores: Optional[list[float]] = None
```

---

### File 4: `app/streamlit/api_client.py`

**What**: Parse all new fields in `_parse_image()` (lines 421-432).

**Current**:
```python
def _parse_image(self, data: Dict[str, Any]) -> ImageInfo:
    return ImageInfo(
        path=data.get("path", ""),
        filename=data.get("filename", Path(data.get("path", "")).name),
        iqa_score=data.get("iqa_score"),
        ava_score=data.get("ava_score"),
        composite_score=data.get("composite_score"),
        face_count=data.get("face_count", 0),
        cluster_id=data.get("cluster_id"),
        is_selected=data.get("is_selected", False),
    )
```

**Add these lines** inside the constructor call:
```python
        sharpness=data.get("sharpness"),
        face_pose_scores=data.get("face_pose_scores"),
        face_eyes_scores=data.get("face_eyes_scores"),
        face_smile_scores=data.get("face_smile_scores"),
```

---

### File 5: `app/streamlit/components/gallery.py`

**What**: Show sharpness and portrait indicators in image cards, matching the old app (`app/album/components/gallery.py` lines 62-85).

**Current** (lines 59-76):
```python
if show_scores:
    score_parts = []
    if image.iqa_score is not None:
        score_parts.append(f"IQA: {image.iqa_score:.2f}")
    if image.ava_score is not None:
        score_parts.append(f"AVA: {image.ava_score:.1f}")
    if image.composite_score is not None:
        score_parts.append(f"Score: {image.composite_score:.2f}")
    if score_parts:
        st.caption(" | ".join(score_parts))

if image.face_count and image.face_count > 0:
    faces_label = "face" if image.face_count == 1 else "faces"
    st.caption(f"{image.face_count} {faces_label}")
```

**Change to**:
```python
if show_scores:
    score_parts = []
    if image.iqa_score is not None:
        score_parts.append(f"IQA: {image.iqa_score:.2f}")
    if image.ava_score is not None:
        score_parts.append(f"AVA: {image.ava_score:.1f}")
    if image.sharpness is not None:
        score_parts.append(f"Sharp: {image.sharpness:.2f}")
    if image.composite_score is not None:
        score_parts.append(f"Score: {image.composite_score:.2f}")
    if score_parts:
        st.caption(" | ".join(score_parts))

if image.face_count and image.face_count > 0:
    faces_label = "face" if image.face_count == 1 else "faces"
    parts = [f"{image.face_count} {faces_label}"]
    if image.face_eyes_scores:
        eyes_open = image.face_eyes_scores[0] > 0.5
        parts.append("eyes open" if eyes_open else "eyes closed")
    if image.face_smile_scores:
        smiling = image.face_smile_scores[0] > 0.5
        parts.append("smiling" if smiling else "neutral")
    st.caption(" | ".join(parts))
```

---

### File 6: `app/streamlit/components/metrics.py`

**What**: Add a `render_image_metrics_table()` function that produces a per-image DataFrame with all scores, matching the old app (`app/album/components/metrics.py` lines 10-48).

**Add new function** at end of file:
```python
def render_image_metrics_table(images: List[ImageInfo], selected_paths: set[str] = None) -> None:
    """Render a detailed per-image metrics table with CSV download."""
    import pandas as pd

    if not images:
        st.info("No image metrics available")
        return

    st.subheader("Per-Image Metrics")

    if selected_paths is None:
        selected_paths = set()

    rows = []
    for img in images:
        is_sel = img.is_selected or img.path in selected_paths
        status = "Selected" if is_sel else "Filtered"

        # Best face scores (first face, if any)
        best_pose = img.face_pose_scores[0] if img.face_pose_scores else None
        best_eyes = img.face_eyes_scores[0] if img.face_eyes_scores else None
        best_smile = img.face_smile_scores[0] if img.face_smile_scores else None

        rows.append({
            "Image": Path(img.path).name,
            "Status": status,
            "AVA": f"{img.ava_score:.1f}" if img.ava_score is not None else "N/A",
            "IQA": f"{img.iqa_score:.2f}" if img.iqa_score is not None else "N/A",
            "Sharpness": f"{img.sharpness:.2f}" if img.sharpness is not None else "N/A",
            "Faces": img.face_count or 0,
            "Pose": f"{best_pose:.2f}" if best_pose is not None else "",
            "Eyes": f"{best_eyes:.2f}" if best_eyes is not None else "",
            "Smile": f"{best_smile:.2f}" if best_smile is not None else "",
            "Cluster": img.cluster_id if img.cluster_id is not None else "-",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "image_metrics.csv", "text/csv")
```

Also add required imports at top: `from pathlib import Path` and `from app.streamlit.models import ImageInfo`.

---

### File 7: `app/streamlit/pages/results.py`

**What**: Add a "Metrics Table" tab that calls `render_image_metrics_table`.

**Current** (line 40):
```python
tab1, tab2, tab3 = st.tabs(["Run Pipeline", "View Results", "Export"])
```

**Change to**:
```python
tab1, tab2, tab3, tab4 = st.tabs(["Run Pipeline", "View Results", "Metrics Table", "Export"])
```

**Current** (lines 42-53):
```python
with tab1:
    ...
with tab2:
    _render_results_tab(album)
with tab3:
    _render_export_tab(album)
```

**Change to**:
```python
with tab1:
    ...
with tab2:
    _render_results_tab(album)
with tab3:
    _render_metrics_table_tab(album)
with tab4:
    _render_export_tab(album)
```

**Add new function**:
```python
def _render_metrics_table_tab(album: Album) -> None:
    """Render the per-image metrics table tab."""
    client = get_client()
    results = client.list_results(album.album_id)

    if not results:
        st.info("No pipeline results yet. Run the pipeline first.")
        return

    latest = results[0]
    job_id = latest.get("job_id", latest.get("id", ""))

    all_images = client.get_images(job_id)
    selected = client.get_selected_images(job_id)
    selected_paths = {img.path for img in selected}

    render_image_metrics_table(all_images, selected_paths)
```

**Add import** at top of file:
```python
from app.streamlit.components.metrics import render_image_metrics_table
```

---

## What is NOT changing

- **`pipeline_service.py`**: Already stores all 9 fields correctly (fixed in previous session). No changes needed.
- **Database schema**: `PipelineResult.image_metrics` is a JSON column -- schema-agnostic. No migration needed.
- **API router** (`routers/results.py`): Already passes through whatever the service returns. No changes needed.
- **Pipeline steps**: No changes to scoring logic (except bug fixes above).
- **Clustering algorithm** (`sim_bench/clustering/hdbscan.py`): Already supports all params, just not receiving them.

---

## Summary of All Changes

| # | File | Change | Category |
|---|------|--------|----------|
| 1 | `sim_bench/pipeline/steps/select_best.py:122-131` | Fix siamese config key mismatch | Bug fix |
| 2 | `sim_bench/pipeline/steps/cluster_scenes.py:45-51` | Pass all HDBSCAN params | Bug fix |
| 3 | `configs/pipeline.yaml:77` | Change min_cluster_size from 3 to 2 | Config fix |
| 4 | `sim_bench/api/services/result_service.py:94-151` | Add face scores to image dicts | Score display |
| 5 | `sim_bench/api/schemas/result.py:8-15` | Add face score fields to schema | Score display |
| 6 | `app/streamlit/models.py:48-58` | Add face score fields to ImageInfo | Score display |
| 7 | `app/streamlit/api_client.py:421-432` | Parse face scores in _parse_image | Score display |
| 8 | `app/streamlit/components/gallery.py:59-76` | Show sharpness + portrait indicators | Score display |
| 9 | `app/streamlit/components/metrics.py` (new function) | Add per-image metrics table | Score display |
| 10 | `app/streamlit/pages/results.py:40-53` | Add Metrics Table tab | Score display |

---

## Reference Data for Clustering Debugging

Good reference results exist at:
- `D:\sim-bench\outputs\cluster_runs\budapest_hdbscan\2025-11-07_12-56-46\cluster_stats.json`
  - 15 clusters, 24 noise images, `min_cluster_size=2, min_samples=2, metric=cosine`
- `D:\sim-bench\outputs\cluster_runs\budapest_clustering\2025-11-07_12-36-36\cluster_stats.json`
  - 50 clusters (DBSCAN), `eps=0.3, min_samples=1, metric=cosine`

Both used DINOv2 embeddings cached at `artifacts\feature_cache\dinov2_*.pkl`.

The new pipeline uses the same DINOv2 model via the same `load_method()` factory, same normalization. The clustering quality difference is entirely due to the parameter pass-through bug (min_cluster_size=3 + missing min_samples).

---

## Verification

1. Start API: `python -m sim_bench.api`
2. Start Streamlit: `streamlit run app/streamlit/main.py`
3. Select an album with existing results
4. **View Results tab**: Image cards should now show `Sharp: X.XX` in the score line, and face images should show `eyes open | smiling` or `eyes closed | neutral`
5. **Metrics Table tab**: Should show a DataFrame with columns: Image, Status, AVA, IQA, Sharpness, Faces, Pose, Eyes, Smile, Cluster
6. **Download CSV**: Click "Download CSV" and verify all columns have data for images that had faces
7. **API direct check**: `GET /api/v1/results/{job_id}/images` should return `face_pose_scores`, `face_eyes_scores`, `face_smile_scores`, `is_selected` fields
8. **Re-run pipeline**: After fixes, clustering should produce ~15 clusters (matching reference) instead of fewer large clusters
9. **Siamese check**: In logs, should see `"Loading Siamese model from ..."` and `"Siamese CNN enabled for comparison"` during select_best step
