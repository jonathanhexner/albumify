# Photo Organization System Architecture

## Overview

The Photo Organization System extends sim-bench with intelligent photo management capabilities, enabling users to analyze, organize, and export photo collections using AI-powered vision-language models.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Photo Organization System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Analysis   │  │Organization  │  │   Layout &   │      │
│  │   Pipeline   │→ │   Engine     │→ │   Export     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↑                  ↑                  ↑              │
│         │                  │                  │              │
│  ┌──────┴──────────────────┴──────────────────┴──────┐      │
│  │          Shared Infrastructure                     │      │
│  │  - Vision-Language Models (CLIP, BLIP)            │      │
│  │  - Image Processing (Thumbnails, Enhancement)     │      │
│  │  - Quality Assessment                              │      │
│  │  - Global Configuration                            │      │
│  └────────────────────────────────────────────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Image Processing (`sim_bench/image_processing/`)

**Purpose**: Multi-resolution image pyramid and preprocessing

```python
sim_bench/image_processing/
├── base.py              # ImageProcessor abstract base class
├── thumbnail.py         # ThumbnailGenerator (multi-resolution caching)
├── enhancement.py       # ImageEnhancer (auto-enhancement, placeholder)
├── cropping.py          # SmartCropper (CLIP-guided cropping)
└── __init__.py          # Factory: create_image_processor()
```

**Key Features**:
- Multi-resolution pyramid (128px, 512px, 1024px, 2048px)
- Disk-based caching with content hashing
- Lazy generation (only create sizes actually needed)
- Thread-safe batch processing

**API Example**:
```python
from sim_bench.image_processing import ThumbnailGenerator

generator = ThumbnailGenerator(cache_dir=".cache/thumbnails")
thumbnails = generator.generate_batch(
    image_paths=["photo1.jpg", "photo2.jpg"],
    sizes=['tiny', 'small'],
    num_workers=4
)
# Returns: {'photo1.jpg': {'tiny': 'path/to/tiny.jpg', 'small': '...'}}
```

### 2. Photo Analysis (`sim_bench/photo_analysis/`)

**Purpose**: High-level photo understanding using vision-language models

```python
sim_bench/photo_analysis/
├── base.py              # PhotoAnalyzer abstract base class
├── clip_tagger.py       # CLIPTagger (55-prompt zero-shot tagging)
├── batch_processor.py   # BatchProcessor (parallel pipeline)
├── routing.py           # RoutingEngine (decide which models to apply)
├── metadata_store.py    # MetadataStore (JSON/SQLite storage)
└── __init__.py          # Factory: create_photo_analyzer()
```

**Key Features**:
- 55 zero-shot CLIP prompts (scene, quality, composition, technical)
- Routing logic (which specialized models to trigger)
- Importance scoring (for layout decisions)
- Metadata persistence

**API Example**:
```python
from sim_bench.photo_analysis import CLIPTagger

tagger = CLIPTagger(model_name='ViT-B-32')
metadata = tagger.analyze_image("photo.jpg")
# Returns:
# {
#   'clip_tags': {'outdoor': 0.9, 'landscape': 0.85, ...},
#   'importance_score': 0.82,
#   'routing': {'needs_face_detection': True, 'needs_landmark': False},
#   'composition': {'type': 'rule_of_thirds', 'visual_weight': 0.7}
# }
```

### 3. Specialized Models (`sim_bench/specialized_models/`)

**Purpose**: Domain-specific deep learning models (triggered by routing)

```python
sim_bench/specialized_models/
├── base.py              # SpecializedModel abstract base class
├── faces.py             # FaceModel (detection, recognition, clustering)
├── landmarks.py         # LandmarkModel (place recognition)
├── objects.py           # ObjectDetector (YOLO/DETR, optional)
└── __init__.py          # Factory: create_specialized_model()
```

**Key Features**:
- Lazy loading (only import when needed)
- Multiple backend support (DeepFace, InsightFace for faces)
- Embedding-based clustering
- Seamless integration with photo_analysis routing

**API Example**:
```python
from sim_bench.specialized_models import FaceModel

face_model = FaceModel(backend='deepface')
results = face_model.process_batch(
    image_paths=["photo1.jpg", "photo2.jpg"],
    routing_hints=metadata['routing']  # Only process if routing suggests faces
)
# Returns:
# {
#   'photo1.jpg': {
#     'faces': [{'bbox': [x,y,w,h], 'embedding': [...], 'cluster_id': 0}],
#     'face_count': 1
#   }
# }
```

### 4. Photo Organization (`sim_bench/photo_organization/`)

**Purpose**: Smart hierarchical clustering and album creation

```python
sim_bench/photo_organization/
├── base.py                      # Organizer abstract base class
├── hierarchical_clustering.py   # HierarchicalOrganizer (multi-level)
├── event_organizer.py           # EventOrganizer (time + location + scene)
├── people_organizer.py          # PeopleOrganizer (face clustering)
├── theme_organizer.py           # ThemeOrganizer (CLIP embedding clustering)
├── quality_filter.py            # QualityFilter (ranking, discard suggestions)
└── __init__.py                  # Factory: create_organizer()
```

**Key Features**:
- Multi-level clustering (coarse-to-fine)
- Multiple strategies (event, people, theme, quality)
- Flexible feature spaces (CLIP, faces, landmarks)
- AI agent for album naming

**API Example**:
```python
from sim_bench.photo_organization import HierarchicalOrganizer

organizer = HierarchicalOrganizer(levels=[
    {'strategy': 'event', 'features': 'clip_embeddings', 'min_size': 20},
    {'strategy': 'people', 'features': 'face_embeddings', 'min_size': 5}
])

hierarchy = organizer.organize(
    metadata=analysis_results,
    specialized_results={'faces': face_results}
)
# Returns hierarchical album structure
```

### 5. Layout Engine (`sim_bench/layout/`)

**Purpose**: Arrange photos into visually appealing layouts

```python
sim_bench/layout/
├── base.py              # LayoutEngine abstract base class
├── templates/
│   ├── grid.py          # GridLayout (simple equal grid)
│   ├── magazine.py      # MagazineLayout (varied sizes, asymmetric)
│   ├── hero.py          # HeroLayout (one large + supporting)
│   └── story.py         # StoryLayout (chronological narrative)
├── optimizer.py         # LayoutOptimizer (aesthetic scoring, optional)
├── renderer.py          # LayoutRenderer (render to images/HTML)
└── __init__.py          # Factory: create_layout_engine()
```

**Key Features**:
- Template-based layouts (grid, magazine, hero, story)
- Importance-weighted placement
- CLIP-based composition analysis
- Visual balance optimization (optional)

**API Example**:
```python
from sim_bench.layout import create_layout_engine

layout_engine = create_layout_engine('magazine')
layout = layout_engine.generate(
    album=organized_photos,
    page_size=(1200, 1600),
    use_importance_scores=True
)
# Returns layout specification (coordinates, sizes)
```

### 6. Export System (`sim_bench/export/`)

**Purpose**: Export organized photos to various formats

```python
sim_bench/export/
├── base.py              # Exporter abstract base class
├── json_exporter.py     # JSONExporter (metadata JSON)
├── html_exporter.py     # HTMLExporter (web gallery)
├── jalbum_exporter.py   # JAlbumExporter (XML config, optional)
├── scribus_exporter.py  # ScribusExporter (SLA layout, optional)
└── __init__.py          # Factory: create_exporter()
```

**API Example**:
```python
from sim_bench.export import HTMLExporter

exporter = HTMLExporter(template='modern')
exporter.export(
    hierarchy=organized_photos,
    layouts=layout_specs,
    output_dir="output/gallery/"
)
```

### 7. AI Agent (`sim_bench/ai_agent/`)

**Purpose**: High-level orchestration and decision-making

```python
sim_bench/ai_agent/
├── base.py              # Agent abstract base class
├── rule_based_agent.py  # RuleBasedAgent (pattern matching)
├── llm_agent.py         # LLMAgent (GPT-4/Claude, optional)
├── orchestrator.py      # Orchestrator (workflow coordinator)
└── __init__.py          # Factory: create_agent()
```

**Key Features**:
- Workflow orchestration (end-to-end pipeline)
- Decision-making (which models, strategies, layouts)
- Album naming and description generation
- User preference learning (future)

**API Example**:
```python
from sim_bench.ai_agent import Orchestrator

orchestrator = Orchestrator.from_config("configs/workflow.yaml")
results = orchestrator.run(
    input_dir="photos/",
    output_dir="output/",
    preferences={'organize_by': 'event', 'layout': 'magazine'}
)
```

## Processing Pipeline

### Stage 1: Thumbnail Generation
```python
ThumbnailGenerator → Generate multi-resolution pyramid
├── Input: Original photos
├── Output: Cached thumbnails (128px, 512px, 1024px, 2048px)
└── Storage: .cache/thumbnails/
```

### Stage 2: CLIP Analysis (Fast Routing)
```python
CLIPTagger → Analyze on tiny (128px) thumbnails
├── Input: Tiny thumbnails
├── Processing: 55 zero-shot prompts
├── Output: Tags, importance scores, routing decisions
└── Speed: ~100 images/minute on CPU
```

### Stage 3: Specialized Models (Selective)
```python
RoutingEngine → Apply specialized models based on routing
├── If needs_faces: FaceModel on large (2048px) thumbnails
├── If needs_landmarks: LandmarkModel on large thumbnails
├── If needs_objects: ObjectDetector on large thumbnails
└── Output: Specialized embeddings and metadata
```

### Stage 4: Quality Assessment
```python
QualityAssessor → Assess on medium (1024px) thumbnails
├── Input: Medium thumbnails
├── Processing: Sharpness, composition, aesthetic scores
├── Output: Quality scores per image
└── Integration: Combine with CLIP aesthetic
```

### Stage 5: Hierarchical Organization
```python
HierarchicalOrganizer → Multi-level clustering
├── Level 1: Event clustering (CLIP + time + GPS)
├── Level 2: People clustering (face embeddings)
├── Level 3: Quality tiers (quality scores)
└── Output: Hierarchical album structure
```

### Stage 6: Layout Generation
```python
LayoutEngine → Create page layouts
├── Input: Organized albums + importance scores
├── Processing: Template selection, photo placement
├── Output: Layout specifications (coordinates, sizes)
└── Preview: Rendered on small (512px) thumbnails
```

### Stage 7: Export
```python
Exporter → Generate final outputs
├── Input: Full resolution photos + layouts
├── Processing: Resize, arrange, add metadata
├── Output: HTML gallery, jAlbum XML, or Scribus SLA
└── Quality: Full resolution for print/web
```

## Design Patterns

### Factory Pattern
Each module provides a factory function for creating instances:

```python
# sim_bench/photo_analysis/__init__.py
def create_photo_analyzer(analyzer_type: str, **config):
    """
    Factory function for photo analyzers.

    Args:
        analyzer_type: 'clip', 'blip', etc.
        **config: Analyzer-specific configuration

    Returns:
        PhotoAnalyzer instance
    """
    if analyzer_type == 'clip':
        return CLIPTagger(**config)
    elif analyzer_type == 'blip':
        return BLIPTagger(**config)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
```

### Strategy Pattern
Different strategies for organization:

```python
# Strategies are selected at runtime based on config
organizer = HierarchicalOrganizer(levels=[
    {'strategy': 'event', ...},    # EventStrategy
    {'strategy': 'people', ...},   # PeopleStrategy
    {'strategy': 'theme', ...}     # ThemeStrategy
])
```

### Singleton Pattern (Global Config)
Global configuration is loaded once and shared:

```python
# sim_bench/config.py
class GlobalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Load from configs/global_config.yaml
        pass

# Usage
from sim_bench.config import get_global_config
config = get_global_config()  # Always returns same instance
```

## Configuration Hierarchy

Configuration is resolved in this order (highest to lowest priority):

1. **CLI Arguments** (highest priority)
   ```bash
   python run_photo_analysis.py --device cuda --output-dir custom/
   ```

2. **Experiment Config Files**
   ```yaml
   # configs/photo_analysis_experiment.yaml
   device: cpu  # Overrides global if specified
   ```

3. **Global Config File**
   ```yaml
   # configs/global_config.yaml
   device: cpu
   output_dir: outputs/
   ```

4. **Hardcoded Defaults** (lowest priority)
   ```python
   device = config.get('device', 'cpu')  # Default to 'cpu'
   ```

## Multi-Resolution Strategy

### Why Multi-Resolution?

| Task | Resolution | Reason |
|------|-----------|--------|
| CLIP Tagging | 128px (tiny) | Fast, CLIP works well on small images |
| UI Preview | 512px (small) | Fast loading, good enough for browsing |
| Quality Assessment | 1024px (medium) | Balance between detail and speed |
| Face Detection | 2048px (large) | Needs detail for small/distant faces |
| Final Export | Original (full) | Preserve quality for print/archival |

### Processing Time Estimates

For 1000 images on CPU:

| Stage | Resolution | Time |
|-------|-----------|------|
| Thumbnail Generation | All sizes | ~5 minutes (one-time) |
| CLIP Analysis | 128px | ~10 minutes |
| Face Detection | 2048px | ~30 minutes (only if routed) |
| Quality Assessment | 1024px | ~15 minutes |
| Layout Preview | 512px | ~2 minutes |
| **Total** | - | **~30-60 minutes** |

## Error Handling

### Graceful Degradation
- If specialized model fails → continue without it
- If routing suggests faces but FaceModel unavailable → log warning, continue
- If thumbnail generation fails for one image → skip it, continue batch

### Logging Strategy
```python
import logging

logger = logging.getLogger(__name__)

# In critical paths
logger.info("Processing batch of 100 images")
logger.warning("Face detection unavailable, skipping")
logger.error("Failed to load image: photo.jpg", exc_info=True)

# In debug mode
logger.debug(f"CLIP similarity scores: {scores}")
```

## Testing Strategy

Each module includes:
1. **Unit tests**: Test individual components
2. **Integration tests**: Test module interactions
3. **Example scripts**: Demonstrate usage

Example:
```
tests/
├── test_thumbnail_generator.py     # Unit test
├── test_clip_tagger.py             # Unit test
├── test_photo_analysis_pipeline.py # Integration test
└── test_end_to_end.py              # Full workflow test
```

## Future Extensions

### Phase 2 Enhancements
- Google Photos integration (OAuth)
- Video support (frame extraction + analysis)
- Multi-user collaboration
- Cloud storage backends (S3, GCS)

### Phase 3 Advanced Features
- Image enhancement (auto-correct, HDR fusion)
- Smart cropping suggestions
- Duplicate detection
- Auto-captioning (BLIP-2, LLaVA)

## API Stability

| Module | Status | Stability |
|--------|--------|-----------|
| `image_processing` | Implementing | Beta |
| `photo_analysis` | Implementing | Beta |
| `specialized_models` | Planned | Alpha |
| `photo_organization` | Planned | Alpha |
| `layout` | Planned | Alpha |
| `export` | Planned | Alpha |
| `ai_agent` | Planned | Alpha |

## Dependencies

### Core Dependencies (Required)
- `torch` - PyTorch for deep learning
- `open-clip-torch` - OpenCLIP for vision-language
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `PyYAML` - Configuration files

### Optional Dependencies
- `deepface` or `insightface` - Face recognition
- `opencv-python` - Advanced image processing
- `streamlit` - Web UI
- `openai` or `anthropic` - LLM agent

## Performance Considerations

### Memory Management
- Process images in batches (default: 32)
- Clear embeddings cache periodically
- Use thumbnails for all non-export operations

### Parallelization
- Thumbnail generation: Multi-threaded
- CLIP encoding: Batched GPU operations
- Specialized models: Sequential (memory constraint)

### Caching Strategy
- Thumbnails: Disk cache (persistent)
- CLIP embeddings: Memory cache (session)
- Quality scores: Memory cache (session)
- Metadata: Disk cache (JSON/SQLite)

## Conclusion

This architecture provides a modular, extensible foundation for AI-powered photo organization. Each component is independently testable and can be used standalone or as part of the full pipeline.

**Next Steps**: Implement Phase 1 modules (image_processing, photo_analysis) following this architecture.
