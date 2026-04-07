# Face Clustering Debug App - Architecture Document

**Date:** 2026-02-17
**Status:** Draft - Pending Approval
**Prerequisites:** [REQUIREMENTS.md](REQUIREMENTS.md) approved

---

## 1. Architecture Overview

### 1.1 Design Principles

| Principle | Application |
|-----------|-------------|
| **Single Responsibility** | Each module does one thing |
| **Dependency Inversion** | Pages depend on abstractions (interfaces), not implementations |
| **Open/Closed** | New data sources can be added without modifying existing code |
| **Don't Repeat Yourself** | Reusable components, no algorithm reimplementation |

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Overview │ │  Merge   │ │  Attach  │ │ Distance │ ... pages  │
│  │   Page   │ │Decisions │ │Decisions │ │  Lookup  │            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │
│       │            │            │            │                   │
│  ┌────┴────────────┴────────────┴────────────┴────┐             │
│  │              COMPONENTS (reusable widgets)      │             │
│  │  FaceGrid │ DistanceMatrix │ DecisionCard │ ... │             │
│  └─────────────────────┬───────────────────────────┘             │
└────────────────────────┼─────────────────────────────────────────┘
                         │
┌────────────────────────┼─────────────────────────────────────────┐
│                   SERVICES LAYER                                  │
│  ┌─────────────────────┴───────────────────────┐                 │
│  │           DataLoaderProtocol                 │ ◄── Interface  │
│  └─────────────────────┬───────────────────────┘                 │
│           ┌────────────┴────────────┐                            │
│  ┌────────┴────────┐    ┌───────────┴───────┐                    │
│  │   FileLoader    │    │    DBLoader       │ ◄── Implementations│
│  │ (JSON/NPY files)│    │ (SQLite database) │                    │
│  └─────────────────┘    └───────────────────┘                    │
│                                                                   │
│  ┌─────────────────────────────────────────────┐                 │
│  │          ClusteringRunner                    │                 │
│  │  (wraps sim_bench.clustering.base)           │                 │
│  └─────────────────────────────────────────────┘                 │
└───────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────────┐
│                  EXISTING sim_bench MODULES                        │
│  sim_bench.clustering.base.load_clustering_method()                │
│  sim_bench.clustering.hybrid_hdbscan_knn.HybridHDBSCANKNN         │
│  sim_bench.clustering.hybrid_closest_face.HybridHDBSCANClosestFace│
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. Folder Structure

```
app/face_clustering_debug/
├── __init__.py
├── main.py                      # Entry point, navigation, page routing
│
├── pages/                       # One module per page/feature
│   ├── __init__.py
│   ├── overview.py              # Cluster overview with thumbnails
│   ├── merge_decisions.py       # Why clusters merged/didn't merge
│   ├── attach_decisions.py      # Why noise points attached/stayed noise
│   ├── distance_lookup.py       # Query distance between any two faces
│   ├── parameter_tuning.py      # Adjust params, re-run clustering
│   └── algorithm_comparison.py  # Side-by-side HDBSCAN vs Hybrid
│
├── components/                  # Reusable UI widgets
│   ├── __init__.py
│   ├── face_grid.py             # Grid of face thumbnails (clickable)
│   ├── face_detail.py           # Single face with landmarks overlay
│   ├── distance_heatmap.py      # Distance matrix visualization
│   ├── threshold_display.py     # Show threshold with Q1/Q3/IQR
│   ├── decision_card.py         # Single merge/attach decision
│   └── param_sliders.py         # Parameter input widgets
│
├── services/                    # Business logic, data access
│   ├── __init__.py
│   ├── protocols.py             # Abstract interfaces (Protocol classes)
│   ├── file_loader.py           # Load from benchmark JSON/NPY files
│   ├── db_loader.py             # Load from SQLite database
│   └── clustering_runner.py     # Execute clustering via sim_bench
│
└── models/                      # Data structures
    ├── __init__.py
    └── schemas.py               # Pydantic models / dataclasses
```

---

## 3. Layer Specifications

### 3.1 Models Layer (`models/`)

**Purpose:** Define data structures passed between layers.

```python
# models/schemas.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class FaceInfo:
    """Single face metadata."""
    index: int
    image_path: str
    bbox: tuple  # (x, y, w, h)
    confidence: float
    crop_path: Optional[str] = None
    landmarks: Optional[List[Tuple[float, float]]] = None  # 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
    pose_angles: Optional[Tuple[float, float, float]] = None  # pitch, yaw, roll

@dataclass
class ClusterInfo:
    """Single cluster with its faces and stats."""
    cluster_id: int
    face_indices: List[int]
    threshold: float
    exemplar_indices: List[int]
    # d3 stats
    q1: float
    q3: float
    iqr: float
    raw_threshold: float

@dataclass
class MergeDecision:
    """Record of a merge decision."""
    cluster_a: int
    cluster_b: int
    threshold_used: float
    pairs_within_threshold: int
    exemplars_a_involved: int
    exemplars_b_involved: int
    min_distance: float
    merged: bool
    reason: str  # 'merged', 'not_enough_pairs', etc.

@dataclass
class AttachDecision:
    """Record of an attachment decision."""
    face_index: int
    attached_to: Optional[int]
    candidates: List[Dict]  # cluster_id, distance, qualified

@dataclass
class ClusteringResult:
    """Complete clustering result."""
    labels: np.ndarray
    embeddings: np.ndarray
    faces: List[FaceInfo]
    clusters: List[ClusterInfo]
    merge_decisions: List[MergeDecision]
    attach_decisions: List[AttachDecision]
    algorithm: str
    params: Dict
```

### 3.2 Services Layer (`services/`)

#### 3.2.1 Protocol Definition

```python
# services/protocols.py
from typing import Protocol, List, Dict, Optional
import numpy as np
from models.schemas import FaceInfo, ClusteringResult

class DataLoaderProtocol(Protocol):
    """Interface for loading clustering data."""

    def load_embeddings(self) -> np.ndarray:
        """Load face embeddings matrix [N, 512]."""
        ...

    def load_faces(self) -> List[FaceInfo]:
        """Load face metadata."""
        ...

    def load_clustering_result(self, method: str) -> Optional[ClusteringResult]:
        """Load pre-computed clustering result for given method."""
        ...

    def get_available_methods(self) -> List[str]:
        """List available clustering methods in this data source."""
        ...

    def get_face_crop(self, face_index: int) -> Optional[bytes]:
        """Get face crop image as bytes."""
        ...
```

#### 3.2.2 File Loader Implementation

```python
# services/file_loader.py
class FileLoader:
    """Load data from benchmark JSON/NPY files."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: Optional[List[Dict]] = None

    def load_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            npy_files = list(self.results_dir.glob("embeddings_*.npy"))
            if not npy_files:
                raise FileNotFoundError("No embeddings file found")
            self._embeddings = np.load(npy_files[0])
        return self._embeddings

    def load_faces(self) -> List[FaceInfo]:
        # Parse from benchmark JSON metadata
        ...

    def load_clustering_result(self, method: str) -> Optional[ClusteringResult]:
        # Parse from benchmark JSON results
        ...

    def get_face_crop(self, face_index: int) -> Optional[bytes]:
        crop_path = self.results_dir / "face_crops" / f"face_{face_index:04d}.jpg"
        if crop_path.exists():
            return crop_path.read_bytes()
        return None
```

#### 3.2.3 Database Loader Implementation

```python
# services/db_loader.py
class DBLoader:
    """Load data from sim_bench SQLite database."""

    def __init__(self, album_id: int, pipeline_run_id: Optional[int] = None):
        self.album_id = album_id
        self.pipeline_run_id = pipeline_run_id
        self._db_path = Path.home() / ".sim_bench" / "sim_bench.db"

    def load_embeddings(self) -> np.ndarray:
        # Query universal_cache for face_embedding entries
        ...

    def load_faces(self) -> List[FaceInfo]:
        # Query universal_cache for insightface_faces entries
        ...

    def load_clustering_result(self, method: str) -> Optional[ClusteringResult]:
        # Query people table + pipeline_results
        ...
```

#### 3.2.4 Clustering Runner

```python
# services/clustering_runner.py
from sim_bench.clustering.base import load_clustering_method

class ClusteringRunner:
    """Execute clustering using sim_bench algorithms."""

    @staticmethod
    def get_available_algorithms() -> List[str]:
        return [
            'hdbscan',
            'hybrid_hdbscan_knn',
            'hybrid_closest_face',
        ]

    @staticmethod
    def get_algorithm_params(algorithm: str) -> Dict[str, Dict]:
        """Return parameter definitions for UI sliders."""
        params = {
            'hybrid_hdbscan_knn': {
                'min_cluster_size': {'type': 'int', 'min': 2, 'max': 10, 'default': 2},
                'knn_k': {'type': 'int', 'min': 1, 'max': 10, 'default': 3},
                'iqr_multiplier': {'type': 'float', 'min': 0.5, 'max': 5.0, 'default': 2.0},
                'threshold_floor': {'type': 'float', 'min': 0.1, 'max': 1.0, 'default': 0.5},
                'threshold_ceiling': {'type': 'float', 'min': 0.5, 'max': 1.5, 'default': 0.9},
                'max_exemplars': {'type': 'int', 'min': 3, 'max': 20, 'default': 10},
                'merge_min_pairs': {'type': 'int', 'min': 1, 'max': 10, 'default': 3},
                'merge_min_distinct': {'type': 'int', 'min': 1, 'max': 5, 'default': 2},
                'attach_min_exemplars': {'type': 'int', 'min': 1, 'max': 5, 'default': 2},
            },
            # ... other algorithms
        }
        return params.get(algorithm, {})

    @staticmethod
    def run(
        algorithm: str,
        params: Dict,
        embeddings: np.ndarray,
        collect_debug_data: bool = True
    ) -> ClusteringResult:
        """Run clustering and return structured result."""
        config = {
            'algorithm': algorithm,
            'params': params,
        }
        clusterer = load_clustering_method(config)
        labels, stats = clusterer.cluster(embeddings, collect_debug_data=collect_debug_data)

        # Convert to ClusteringResult schema
        return ClusteringResult(
            labels=labels,
            embeddings=embeddings,
            clusters=_parse_clusters(stats),
            merge_decisions=_parse_merge_decisions(stats.get('debug', {})),
            attach_decisions=_parse_attach_decisions(stats.get('debug', {})),
            algorithm=algorithm,
            params=params,
        )
```

### 3.3 Components Layer (`components/`)

**Purpose:** Reusable UI widgets with no business logic.

```python
# components/face_grid.py
import streamlit as st
from typing import List, Optional, Callable
from models.schemas import FaceInfo

def render_face_grid(
    faces: List[FaceInfo],
    get_crop_fn: Callable[[int], Optional[bytes]],
    columns: int = 6,
    highlight_indices: Optional[List[int]] = None,
    selectable: bool = True,
) -> Optional[int]:
    """Render a grid of face thumbnails.

    Args:
        faces: List of FaceInfo objects to display
        get_crop_fn: Function to get crop bytes by index
        columns: Number of columns in grid
        highlight_indices: Faces to highlight (e.g., exemplars)
        selectable: If True, faces are clickable and return selected index

    Returns:
        Index of selected face, or None if nothing selected
    """
    selected = None
    cols = st.columns(columns)

    for i, face in enumerate(faces):
        with cols[i % columns]:
            crop_bytes = get_crop_fn(face.index)
            is_exemplar = highlight_indices and face.index in highlight_indices
            border = "2px solid gold" if is_exemplar else "none"

            if crop_bytes:
                if selectable:
                    if st.button(f"#{face.index}", key=f"face_{face.index}"):
                        selected = face.index
                st.image(crop_bytes, caption=f"#{face.index} {'⭐' if is_exemplar else ''}")
            else:
                st.warning(f"#{face.index} (no crop)")

    return selected
```

```python
# components/face_detail.py
import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
from models.schemas import FaceInfo

def render_face_detail(
    face: FaceInfo,
    crop_bytes: bytes,
    show_landmarks: bool = True,
    show_pose: bool = True,
) -> None:
    """Render enlarged face with optional landmark overlay.

    Args:
        face: Face metadata including landmarks
        crop_bytes: JPEG bytes of face crop
        show_landmarks: Draw 5-point landmarks on face
        show_pose: Show pitch/yaw/roll angles
    """
    img = Image.open(BytesIO(crop_bytes))

    if show_landmarks and face.landmarks:
        draw = ImageDraw.Draw(img)
        colors = ['red', 'red', 'green', 'blue', 'blue']  # eyes, nose, mouth
        labels = ['L-Eye', 'R-Eye', 'Nose', 'L-Mouth', 'R-Mouth']
        for (x, y), color, label in zip(face.landmarks, colors, labels):
            r = 3
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

    st.image(img, caption=f"Face #{face.index}", use_container_width=True)

    if show_pose and face.pose_angles:
        pitch, yaw, roll = face.pose_angles
        st.caption(f"Pose: pitch={pitch:.1f}° yaw={yaw:.1f}° roll={roll:.1f}°")

    st.caption(f"Confidence: {face.confidence:.2f}")
```

```python
# components/param_sliders.py
import streamlit as st
from typing import Dict, Any

def render_param_sliders(
    algorithm: str,
    param_definitions: Dict[str, Dict],
    current_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render sliders for algorithm parameters.

    Returns:
        Dictionary of parameter values from sliders
    """
    values = {}
    for param_name, defn in param_definitions.items():
        default = current_values.get(param_name, defn['default']) if current_values else defn['default']

        if defn['type'] == 'int':
            values[param_name] = st.slider(
                param_name,
                min_value=defn['min'],
                max_value=defn['max'],
                value=default,
            )
        elif defn['type'] == 'float':
            values[param_name] = st.slider(
                param_name,
                min_value=float(defn['min']),
                max_value=float(defn['max']),
                value=float(default),
                step=0.05,
            )
    return values
```

### 3.4 Pages Layer (`pages/`)

**Purpose:** Compose components and call services.

```python
# pages/overview.py
import streamlit as st
from services.protocols import DataLoaderProtocol
from components.face_grid import render_face_grid
from components.face_detail import render_face_detail
from components.threshold_display import render_threshold_info

def render_overview_page(loader: DataLoaderProtocol, method: str) -> None:
    """Render cluster overview page."""
    st.header(f"Cluster Overview: {method}")

    result = loader.load_clustering_result(method)
    if result is None:
        st.error(f"No results found for {method}")
        return

    faces = loader.load_faces()
    faces_by_idx = {f.index: f for f in faces}

    col1, col2 = st.columns([3, 1])

    with col1:
        st.metric("Clusters", len(result.clusters))
        st.metric("Noise Points", sum(1 for l in result.labels if l == -1))

        for cluster in result.clusters:
            with st.expander(f"Cluster {cluster.cluster_id} ({len(cluster.face_indices)} faces)"):
                render_threshold_info(cluster)
                cluster_faces = [faces_by_idx[i] for i in cluster.face_indices]
                selected = render_face_grid(
                    faces=cluster_faces,
                    get_crop_fn=loader.get_face_crop,
                    highlight_indices=cluster.exemplar_indices,
                )
                if selected is not None:
                    st.session_state['selected_face'] = selected

    # Detail panel (right side)
    with col2:
        st.subheader("Face Detail")
        selected_idx = st.session_state.get('selected_face')
        if selected_idx is not None and selected_idx in faces_by_idx:
            face = faces_by_idx[selected_idx]
            crop_bytes = loader.get_face_crop(selected_idx)
            if crop_bytes:
                render_face_detail(face, crop_bytes, show_landmarks=True)
        else:
            st.info("Click a face to see details with landmarks")
```

```python
# pages/parameter_tuning.py
import streamlit as st
from services.clustering_runner import ClusteringRunner
from services.protocols import DataLoaderProtocol
from components.param_sliders import render_param_sliders
from components.face_grid import render_face_grid

def render_parameter_tuning_page(loader: DataLoaderProtocol) -> None:
    """Render parameter tuning page."""
    st.header("Parameter Tuning")

    # Algorithm selection
    algorithm = st.selectbox(
        "Algorithm",
        ClusteringRunner.get_available_algorithms()
    )

    # Parameter sliders
    param_defs = ClusteringRunner.get_algorithm_params(algorithm)
    params = render_param_sliders(algorithm, param_defs)

    # Run button
    if st.button("Run Clustering"):
        embeddings = loader.load_embeddings()
        with st.spinner("Running clustering..."):
            result = ClusteringRunner.run(
                algorithm=algorithm,
                params=params,
                embeddings=embeddings,
                collect_debug_data=True,
            )

        st.success(f"Found {len(result.clusters)} clusters")
        # Display results...
```

### 3.5 Main Entry Point

```python
# main.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Face Clustering Debug", layout="wide")

def main():
    st.title("Face Clustering Debug")

    # Sidebar: Data source selection
    with st.sidebar:
        source_type = st.radio("Data Source", ["Benchmark Files", "Database"])

        if source_type == "Benchmark Files":
            results_dir = st.text_input(
                "Results Directory",
                value="results/face_clustering_benchmark"
            )
            from services.file_loader import FileLoader
            loader = FileLoader(Path(results_dir))
        else:
            album_id = st.number_input("Album ID", min_value=1, value=1)
            from services.db_loader import DBLoader
            loader = DBLoader(album_id)

    # Page navigation
    page = st.sidebar.selectbox("Page", [
        "Overview",
        "Merge Decisions",
        "Attach Decisions",
        "Distance Lookup",
        "Parameter Tuning",
        "Algorithm Comparison",
    ])

    # Route to page
    if page == "Overview":
        method = st.sidebar.selectbox("Method", loader.get_available_methods())
        from pages.overview import render_overview_page
        render_overview_page(loader, method)
    elif page == "Merge Decisions":
        from pages.merge_decisions import render_merge_decisions_page
        render_merge_decisions_page(loader)
    # ... etc

if __name__ == "__main__":
    main()
```

---

## 4. Data Flow Diagrams

### 4.1 Loading Existing Results

```
User selects "Benchmark Files" + results directory
                    │
                    ▼
            ┌──────────────┐
            │  FileLoader  │
            └──────┬───────┘
                   │ load_clustering_result("hybrid_knn")
                   ▼
    ┌──────────────────────────────┐
    │  Parse benchmark_*.json      │
    │  Parse embeddings_*.npy      │
    │  Load face_crops/*.jpg       │
    └──────────────┬───────────────┘
                   │
                   ▼
            ClusteringResult
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Pages render using          │
    │  components + result data    │
    └──────────────────────────────┘
```

### 4.2 Re-running Clustering with New Parameters

```
User adjusts sliders + clicks "Run Clustering"
                    │
                    ▼
            ┌──────────────────┐
            │  param_sliders   │ → Dict of params
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ ClusteringRunner │
            └────────┬─────────┘
                     │ run(algorithm, params, embeddings)
                     ▼
    ┌────────────────────────────────────┐
    │ sim_bench.clustering.base          │
    │   .load_clustering_method(config)  │
    │   .cluster(embeddings, debug=True) │
    └────────────────┬───────────────────┘
                     │
                     ▼
            ClusteringResult
                     │
                     ▼
            Display in UI
```

---

## 5. Interface Contracts

### 5.1 DataLoaderProtocol

All data loaders must implement:

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `load_embeddings()` | - | `np.ndarray [N, 512]` | Face embedding matrix |
| `load_faces()` | - | `List[FaceInfo]` | Face metadata |
| `load_clustering_result(method)` | `str` | `Optional[ClusteringResult]` | Pre-computed result |
| `get_available_methods()` | - | `List[str]` | Available methods |
| `get_face_crop(index)` | `int` | `Optional[bytes]` | JPEG bytes for face |

### 5.2 ClusteringRunner

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `get_available_algorithms()` | - | `List[str]` | Algorithm names |
| `get_algorithm_params(alg)` | `str` | `Dict[str, Dict]` | Param definitions for UI |
| `run(alg, params, embeddings)` | `str, Dict, np.ndarray` | `ClusteringResult` | Execute clustering |

---

## 6. File Size Budget

| File | Max Lines | Responsibility |
|------|-----------|----------------|
| `main.py` | 80 | Entry point, routing |
| `pages/*.py` | 150 | Page composition |
| `components/face_grid.py` | 60 | Thumbnail grid |
| `components/face_detail.py` | 50 | Enlarged face with landmarks |
| `components/*.py` (others) | 80 | Single widget |
| `services/protocols.py` | 50 | Interface definitions |
| `services/file_loader.py` | 150 | File parsing |
| `services/db_loader.py` | 150 | DB queries |
| `services/clustering_runner.py` | 100 | Clustering wrapper |
| `models/schemas.py` | 120 | Data structures |

**Total budget:** ~1050 lines across ~16 files (vs current 1663 lines in 2 files)

---

## 7. Dependencies

### 7.1 External Dependencies

- `streamlit` - UI framework
- `numpy` - Embeddings
- `Pillow` - Image handling
- `scipy` - Distance calculations (in components only if needed)

### 7.2 Internal Dependencies

```
pages/*
  └── imports from: components/*, services/*

components/*
  └── imports from: models/* (only for type hints)

services/*
  └── imports from: models/*, sim_bench.clustering.*

models/*
  └── imports from: (standard library only)
```

### 7.3 Forbidden Dependencies

- Pages MUST NOT import from `sim_bench.clustering.*` directly
- Components MUST NOT import from `services/*`
- No circular imports

---

## 8. Migration Steps

1. Create folder structure
2. Implement `models/schemas.py`
3. Implement `services/protocols.py`
4. Implement `services/file_loader.py` (extract from current code)
5. Implement `services/clustering_runner.py`
6. Implement components (extract from current code)
7. Implement pages (extract from current code)
8. Wire up `main.py`
9. Test all functionality
10. Delete old files

---

## Approval

- [ ] Architecture approved by user
- [ ] Ready to proceed to detailed design / task breakdown

