# Album Organization Module

Photo album organization pipeline with layered architecture.

## Architecture

```
sim_bench/album/
├── domain/              # Data models (no dependencies)
│   ├── models.py        # WorkflowResult, ClusterInfo
│   └── types.py         # WorkflowStage, type aliases
├── services/            # Business logic (depends on domain only)
│   ├── album_service.py # Main orchestration service
│   └── selection_service.py # Best image selection
├── stages.py            # Pure pipeline stage functions
├── preprocessor.py      # Thumbnail preprocessing
├── telemetry.py         # Performance tracking
└── export/              # Export functionality
```

## Usage

```python
from sim_bench.album import AlbumService, WorkflowResult
from sim_bench.config import get_global_config

# Create service with config
config = get_global_config().to_dict()
service = AlbumService(config)

# Run album organization
result = service.organize_album(
    source_directory=Path("/path/to/photos"),
    output_directory=Path("/path/to/output")
)

# Access results
print(f"Selected {len(result.selected_images)} from {len(result.clusters)} clusters")
```

## Domain Models

### WorkflowResult

Main output of the organization pipeline:

```python
@dataclass
class WorkflowResult:
    source_directory: Path
    total_images: int
    filtered_images: int
    clusters: Dict[int, List[str]]  # cluster_id -> image paths
    selected_images: List[str]
    metrics: Dict[str, ImageMetrics]
    # ... additional fields
```

### WorkflowStage

Enum for progress tracking:

```python
class WorkflowStage(Enum):
    DISCOVER = auto()
    PREPROCESS = auto()
    ANALYZE = auto()
    FILTER_QUALITY = auto()
    FILTER_PORTRAIT = auto()
    EXTRACT_FEATURES = auto()
    CLUSTER = auto()
    SELECT = auto()
    EXPORT = auto()
    COMPLETE = auto()
```

## Services

### AlbumService

Main orchestration service:

- `organize_album(source, output, callback)` - Run full pipeline
- `get_config()` - Get current configuration

### SelectionService

Best image selection:

- `select_best(clusters, metrics, hub)` - Select from clusters
- `compute_score(metric)` - Compute weighted score
- `select_diverse(images, metrics)` - Diversity-aware selection

## Configuration

All services are configured via dict:

```python
config = {
    'album': {
        'quality': {'min_iqa_score': 0.3, 'min_ava_score': 4.0},
        'portrait': {'require_eyes_open': True},
        'clustering': {'method': 'hdbscan', 'min_cluster_size': 3},
        'selection': {'ava_weight': 0.5, 'iqa_weight': 0.2, 'portrait_weight': 0.3},
        'preprocessing': {'enabled': True, 'num_workers': 4},
        'export': {'organize_by_cluster': True}
    }
}
```

## FastAPI Integration

The service layer is designed for easy API wrapping:

```python
from fastapi import FastAPI
from sim_bench.album import AlbumService

app = FastAPI()

@app.post("/albums/{album_id}/organize")
async def organize_album(album_id: str, request: OrganizeRequest):
    service = AlbumService(request.config)
    result = service.organize_album(request.source, request.output)
    return result_to_response(result)
```
