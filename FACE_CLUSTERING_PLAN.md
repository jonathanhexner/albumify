# Face Clustering Sub-Package — Implementation Plan

**Date**: 2026-04-01
**Triggered by**: SIGHTING-008
**Feature request**: 2026-04-01 entry in docs/FEATURE_REQUESTS.md
**Status**: AWAITING APPROVAL

---

## Goal

Replace the disconnected collection of face-clustering scripts with a single, coherent
`face_cluster` Python sub-package that can run the full cycle — raw images → detected faces →
quality-gated core set → clusters → labeled export — through one clean API, with live progress
reporting for the Streamlit UI, and full A-to-Z test coverage.

---

## What Exists Today (the problem)

| Component | State |
|---|---|
| `face_cluster/embedding.py` | Good — detect + embed works |
| `face_cluster/quality.py` | Broken — silently rejects all faces when SixDRepNet unavailable |
| `face_cluster/knn_graph.py` | Good |
| `face_cluster/clustering.py` | Good |
| `face_cluster/merge.py` | Good |
| `face_cluster/attach.py` | Good |
| `face_cluster/crops.py` | **MISSING** (required by RECOVERY_PLAN.md Stage 3) |
| `face_cluster/export.py` | **MISSING** (required by RECOVERY_PLAN.md Stage 5) |
| `face_cluster/pipeline.py` | **MISSING** — no unified API exists |
| `scripts/export_clustering_data.py` | 36 KB monolith mixing orchestration + algorithm logic — to be archived |
| Streamlit apps | Read-only viewers; no ability to trigger the pipeline or see progress |
| Tests | No A-to-Z tests covering the full detect→export cycle |

---

## Architecture After This Plan

```
face_cluster/
├── types.py          ← (exists, minor update: image_path non-optional)
├── config.py         ← (exists, add: require_pose flag)
├── embedding.py      ← (exists, no change)
├── quality.py        ← (fix: graceful pose-unavailable fallback)
├── knn_graph.py      ← (exists, no change)
├── clustering.py     ← (exists, no change)
├── exemplars.py      ← (exists, no change)
├── merge.py          ← (exists, no change)
├── attach.py         ← (exists, no change)
├── analysis.py       ← (exists, no change)
├── crops.py          ← NEW: save aligned crops + crop_manifest.json
├── export.py         ← NEW: produce faces.csv, clusters.csv, export_summary.json
├── pipeline.py       ← NEW: FaceClusteringPipeline — single public API
└── __init__.py       ← update exports

scripts/
├── run_face_clustering.py   ← thin orchestration script (≤50 lines), calls FaceClusteringPipeline
└── export_clustering_data.py ← ARCHIVE → archive/scripts/

app/
└── face_clustering.py       ← NEW: unified Streamlit app (runner + browser + debug)
    (replaces face_clustering_labeling.py and face_clustering_debug/ runner functionality)

tests/face_clustering/
├── fixtures/                ← synthetic test data (embeddings, FaceRecord lists)
├── test_quality_gating.py
├── test_crops.py
├── test_export.py
├── test_pipeline_e2e.py     ← uses test_data/face_clustering/source_images/ (15 images)
└── test_streamlit_app.py    ← uses streamlit.testing.v1.AppTest
```

---

## Public API — `FaceClusteringPipeline`

```python
from face_cluster import FaceClusteringPipeline, PipelineConfig

config = PipelineConfig(require_pose=False)  # pose filter optional

pipeline = FaceClusteringPipeline(config)

result = pipeline.run(
    image_dir="path/to/album",
    output_dir="path/to/results",
    on_progress=lambda stage, pct, msg: print(f"[{stage}] {pct:.0%} {msg}"),
)
# result.faces          → List[FaceRecord] (all faces, is_core flag set)
# result.cluster_result → ClusterResult
# result.output_dir     → Path (where face_records.json, faces.csv, etc. live)
# result.summary        → dict (n_faces, n_clusters, n_noise, stages_timing)
```

Progress callback signature: `on_progress(stage: str, fraction: float, message: str)`.
Called at start and end of each stage, and every N faces during embedding.
Streamlit integration: wrap in `st.status()` block, call `st.progress(fraction)`.

---

## Work Items (ordered)

### Phase 1 — Fix Quality Gating (prerequisite for everything else)

**Item 1.1** — `face_cluster/config.py`: add `require_pose: bool = False`
- When `False` (default): pose filter is skipped if pose is `None` — face still eligible for core set
- When `True`: face without pose goes to holdout
- This is a non-breaking change (default = current implicit behavior for faces without pose)

**Item 1.2** — `face_cluster/quality.py`: honour `require_pose`
- In `select_core_set()`: if `face.pose is None and not config.require_pose` → skip pose check
- Log a one-time INFO at start of gating: "Pose filter: ENABLED (require_pose=True)" or "Pose filter: SKIP (no pose available)"
- Test: `tests/face_clustering/test_quality_gating.py`
  - `test_no_pose_passes_when_not_required` — face with pose=None passes when require_pose=False
  - `test_no_pose_fails_when_required` — face with pose=None goes to holdout when require_pose=True
  - `test_bad_pose_always_fails` — face with yaw=90° always goes to holdout
  - `test_blur_filter_independent_of_pose` — blur=0 always holdout regardless of pose

### Phase 2 — Missing Pipeline Stages

**Item 2.1** — `face_cluster/crops.py`
Public API: `save_crops(faces: List[FaceRecord], output_dir: Path) -> Dict[int, Path]`
- Saves `face_{face_id:04d}_aligned.jpg` for each face with `aligned_face is not None`
- Writes `crop_manifest.json`: `{face_id: relative_crop_path}`
- Returns `{face_id: absolute_crop_path}` dict
- Test: `tests/face_clustering/test_crops.py`
  - `test_crops_saved_with_correct_filenames`
  - `test_manifest_contains_all_faces`
  - `test_face_id_in_filename_matches_record`

**Item 2.2** — `face_cluster/export.py`
Public API: `export_results(faces, cluster_result, crop_manifest, output_dir, config, source_album) -> Path`
- Writes `faces.csv` (face_id, image_path, crop_path, cluster_id, blur_score, area, is_core, yaw, pitch, roll)
- Writes `clusters.csv` (cluster_id, size, exemplar_face_ids, diameter)
- Writes `export_summary.json` (source_album, run_id, created_at, n_faces, n_clusters, config as dict)
- No heuristic path guessing — all paths stored explicitly
- Test: `tests/face_clustering/test_export.py`
  - `test_faces_csv_has_required_columns`
  - `test_no_null_image_paths`
  - `test_summary_json_has_source_album`
  - `test_cluster_ids_match_between_csv_and_result`

### Phase 3 — `face_cluster/pipeline.py`

**Item 3.1** — `FaceClusteringPipeline` class
```
run(image_dir, output_dir, on_progress=None) → PipelineResult
```
Stages (each emits on_progress calls):
1. Discover images (glob jpg/jpeg/heic/png)
2. Detect & embed (InsightFaceEmbedder) — progress per image
3. Quality gate (QualityGater)
4. Save crops (crops.py)
5. Build kNN graph + cluster
6. Select exemplars
7. Optional merge
8. Optional holdout attach
9. Export (export.py)

Each stage catches exceptions, logs to file, and raises `PipelineStageError(stage, cause)` — never silently continues.

`PipelineResult` dataclass:
```python
@dataclass
class PipelineResult:
    faces: List[FaceRecord]
    cluster_result: ClusterResult
    output_dir: Path
    summary: dict  # n_faces, n_clusters, n_noise, n_core, stages_timing
```

**Item 3.2** — `face_cluster/__init__.py`: export `FaceClusteringPipeline`, `PipelineResult`

### Phase 4 — Thin Script

**Item 4.1** — `scripts/run_face_clustering.py` (update or create, ≤60 lines)
```bash
python scripts/run_face_clustering.py --images <dir> --output <dir> [--config configs/face_clustering.yaml]
```
- Reads config from YAML (optional), falls back to defaults
- Calls `FaceClusteringPipeline.run()` with a console progress callback
- Prints summary on completion
- **Archive** `scripts/export_clustering_data.py` → `archive/scripts/`

### Phase 5 — Unified Streamlit App

**Item 5.1** — `app/face_clustering.py` — 3 tabs:

**Tab 1 — Run Pipeline**
- Input: album directory path (text input) + output directory (text input)
- Config sliders: `distance_threshold`, `blur_min`, `require_pose`, `max_faces_per_image_core`
- Run button → `st.status("Running face clustering...")` → calls `FaceClusteringPipeline.run()`
  with `on_progress` that drives `st.progress()` + per-stage status messages
- On completion: summary metrics (n_faces, n_core, n_clusters, n_noise) + link to results dir

**Tab 2 — Browse Clusters (Labeling)**
- Load from results dir (auto-populated from Tab 1 on completion)
- Grid view per cluster with face crops
- Identity assignment per cluster → saves `corrected_labels.csv`
- (Supersedes `app/face_clustering_labeling.py`)

**Tab 3 — Debug**
- Cluster distance heatmap
- Per-cluster face detail (blur, pose, bbox)
- kNN graph stats
- (Subsumes key views from `app/face_clustering_debug/`)

### Phase 6 — Tests

**Item 6.1** — `tests/face_clustering/test_pipeline_e2e.py`
Uses `test_data/face_clustering/source_images/` (15 real images, HEIC + JPG).
```python
def test_e2e_produces_clusters():
    result = FaceClusteringPipeline(PipelineConfig(require_pose=False)).run(
        image_dir="test_data/face_clustering/source_images",
        output_dir=tmp_path / "output",
    )
    assert result.cluster_result.n_clusters >= 2
    assert (tmp_path / "output" / "faces.csv").exists()
    assert (tmp_path / "output" / "clusters.csv").exists()
    assert (tmp_path / "output" / "export_summary.json").exists()
    # All face records have non-null image_path
    assert all(f.image_path is not None for f in result.faces)
```

**Item 6.2a** — `tests/face_clustering/test_streamlit_app.py` — unit/smoke layer
Uses `streamlit.testing.v1.AppTest` (headless, no browser). Tests widget logic and Python exceptions only.
```python
from streamlit.testing.v1 import AppTest

def test_app_loads_without_error():
    at = AppTest.from_file("app/face_clustering.py")
    at.run(timeout=10)
    assert not at.exception

def test_run_tab_shows_run_button():
    at = AppTest.from_file("app/face_clustering.py")
    at.run()
    assert any("Run" in b.label for b in at.button)

def test_browse_tab_loads_empty_state():
    at = AppTest.from_file("app/face_clustering.py")
    at.run()
    assert not at.exception
```

**Item 6.2b** — `tests/face_clustering/test_streamlit_e2e.py` — browser E2E layer
Uses **Playwright** (real Chromium browser). Runs against `D:\Google_Germany`.
Setup: `pip install playwright && playwright install chromium`

```python
# Requires: pytest-playwright, streamlit running on localhost:8501
# Run with: pytest tests/face_clustering/test_streamlit_e2e.py --headed

def test_full_pipeline_on_real_data(page):
    page.goto("http://localhost:8501")
    # Set image directory
    page.get_by_label("Image directory").fill(r"D:\Google_Germany")
    page.get_by_role("button", name="Run Pipeline").click()
    # Wait for pipeline to complete (up to 5 minutes for large album)
    page.wait_for_selector("text=Clustering complete", timeout=300_000)
    # Verify no error banner
    assert not page.query_selector(".stAlert[data-baseweb='notification'][kind='error']")
    # Verify summary shows clusters
    assert page.inner_text("text=Clusters").strip() != "0"

def test_browse_tab_renders_faces(page):
    page.goto("http://localhost:8501")
    page.get_by_role("tab", name="Browse Clusters").click()
    # Load existing results from D:\Google_Germany\clustering_output
    page.get_by_label("Results directory").fill(r"D:\Google_Germany\clustering_output")
    page.get_by_role("button", name="Load").click()
    page.wait_for_selector("img")  # at least one face image rendered
    assert len(page.query_selector_all("img")) > 0

def test_debug_tab_loads(page):
    page.goto("http://localhost:8501")
    page.get_by_role("tab", name="Debug").click()
    page.wait_for_load_state("networkidle")
    assert not page.query_selector(".stException")
```

Playwright tests are marked `@pytest.mark.e2e` and excluded from default `pytest` run.
Run explicitly: `pytest tests/face_clustering/test_streamlit_e2e.py -m e2e`
Requires Streamlit server running: `streamlit run app/face_clustering.py &`

---

## What This Does NOT Change

- `face_cluster/knn_graph.py`, `clustering.py`, `exemplars.py`, `merge.py`, `attach.py` — untouched
- `sim_bench/pipeline/` — untouched (separate production pipeline)
- `app/face_clustering_debug/` — kept as-is (read-only debug viewer); new Tab 3 covers the key views
- Ground truth tests (SIGHTING-007) — separate effort

---

## Definition of Done

- [ ] `python scripts/run_face_clustering.py --images test_data/face_clustering/source_images --output /tmp/test_out` exits 0 and produces `faces.csv`, `clusters.csv`, `export_summary.json`
- [ ] `python -m pytest tests/face_clustering/ -v` all green
- [ ] `streamlit run app/face_clustering.py` launches and shows all 3 tabs without error
- [ ] `python scripts/export_clustering_data.py` archived (moved to `archive/scripts/`)
- [ ] No face has `image_path=None` in any output file

---

## Risks

| Risk | Mitigation |
|---|---|
| InsightFace detection slow on 15-image E2E test in CI | Cache detection results in `tests/face_clustering/fixtures/` after first run |
| `streamlit.testing.v1` not in current venv | Check version: `python -c "import streamlit; print(streamlit.__version__)"` before writing tests |
| Unified Streamlit app breaks existing labeling workflow | Keep `app/face_clustering_labeling.py` until new Tab 2 is verified by user |
