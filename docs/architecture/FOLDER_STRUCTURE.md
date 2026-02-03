# Proposed Folder Structure

## Overview

```
sim_bench/
│
├── ─────────────────────────────────────────────────────────────
│   EXISTING CODE (unchanged, still works independently)
├── ─────────────────────────────────────────────────────────────
│
├── album/                      # Existing album services
│   ├── domain/
│   ├── services/
│   ├── stages.py
│   └── ...
│
├── face_pipeline/              # Existing face processing
│   ├── crop_service.py
│   ├── pose_estimator.py
│   ├── quality_scorer.py
│   └── pipeline.py
│
├── portrait_analysis/          # Existing portrait analysis
│   ├── analyzer.py
│   ├── eye_state.py
│   └── smile_detection.py
│
├── feature_extraction/         # Existing feature extractors
│   ├── base.py
│   ├── dinov2.py
│   ├── resnet50.py
│   └── ...
│
├── clustering/                 # Existing clustering
│   ├── base.py
│   ├── hdbscan.py
│   └── ...
│
├── model_hub/                  # Existing model hub
│   ├── hub.py
│   └── types.py
│
├── ─────────────────────────────────────────────────────────────
│   NEW CODE (pipeline architecture)
├── ─────────────────────────────────────────────────────────────
│
├── pipeline/                   # Pipeline engine
│   ├── base.py                 # PipelineStep protocol, StepMetadata
│   ├── context.py              # PipelineContext dataclass
│   ├── registry.py             # StepRegistry
│   ├── builder.py              # PipelineBuilder (dependency resolution)
│   ├── executor.py             # PipelineExecutor
│   ├── config.py               # PipelineConfig
│   │
│   └── steps/                  # Step implementations (wrap existing)
│       ├── discover.py
│       ├── score_iqa.py
│       ├── score_ava.py
│       ├── detect_faces.py
│       ├── score_face_pose.py
│       ├── score_face_eyes.py
│       ├── score_face_smile.py
│       ├── filter_quality.py
│       ├── filter_portrait.py
│       ├── filter_pose.py
│       ├── extract_scene_embedding.py
│       ├── extract_face_embedding.py
│       ├── cluster_scenes.py
│       ├── cluster_faces.py
│       ├── cluster_people.py
│       ├── select_best.py
│       └── select_best_per_person.py
│
├── api/                        # FastAPI backend
│   ├── main.py                 # FastAPI app entry point
│   ├── dependencies.py         # Dependency injection
│   │
│   ├── routers/                # API endpoints
│   │   ├── albums.py
│   │   ├── pipeline.py
│   │   ├── steps.py
│   │   ├── results.py
│   │   ├── people.py
│   │   └── websocket.py
│   │
│   ├── schemas/                # Pydantic request/response models
│   │   ├── album.py
│   │   ├── pipeline.py
│   │   ├── step.py
│   │   ├── result.py
│   │   └── people.py
│   │
│   ├── services/               # Business logic (calls pipeline)
│   │   ├── album_service.py
│   │   ├── pipeline_service.py
│   │   ├── result_service.py
│   │   └── people_service.py
│   │
│   └── database/               # SQLAlchemy models & session
│       ├── models.py
│       └── session.py
│
├── ─────────────────────────────────────────────────────────────
│   FRONTEND (separate from backend)
├── ─────────────────────────────────────────────────────────────

app/
├── nicegui/                    # NiceGUI frontend
│   ├── main.py                 # Entry point
│   ├── api_client.py           # HTTP/WebSocket client
│   │
│   ├── pages/                  # Page components
│   │   ├── home.py
│   │   ├── configure.py
│   │   ├── progress.py
│   │   ├── results.py
│   │   └── people.py
│   │
│   ├── components/             # Reusable UI components
│   │   ├── upload.py
│   │   ├── pipeline_builder.py
│   │   ├── step_config.py
│   │   ├── progress_bar.py
│   │   ├── image_gallery.py
│   │   ├── cluster_view.py
│   │   └── face_grid.py
│   │
│   └── state/                  # State management
│       └── app_state.py
│
├── ─────────────────────────────────────────────────────────────
│   OTHER
├── ─────────────────────────────────────────────────────────────

configs/
├── pipelines/                  # Pipeline presets
│   ├── default.yaml
│   ├── face_aware.yaml
│   ├── people.yaml
│   └── quick.yaml
│
├── steps/                      # Step default configs
│   ├── score_iqa.yaml
│   ├── score_face_pose.yaml
│   └── ...
│
└── ...                         # Existing configs (unchanged)

docs/
├── architecture/               # Architecture documentation
│   ├── PIPELINE_ARCHITECTURE_PLAN.md
│   ├── CLASS_DIAGRAM.md
│   ├── FACE_PIPELINE_PLAN.md
│   └── FOLDER_STRUCTURE.md
│
└── ...                         # Existing docs

tests/                          # Tests
├── pipeline/                   # Pipeline engine tests
│   ├── test_context.py
│   ├── test_registry.py
│   ├── test_builder.py
│   └── test_executor.py
│
├── steps/                      # Individual step tests
│   ├── test_discover.py
│   ├── test_score_iqa.py
│   └── ...
│
├── api/                        # API tests
│   ├── test_albums.py
│   ├── test_pipeline.py
│   └── ...
│
└── integration/                # End-to-end tests
    └── test_full_pipeline.py
```

## Key Principles

1. **Existing code untouched** - All current functionality continues to work
2. **Steps wrap, don't modify** - Steps import and call existing code
3. **Clear boundaries** - pipeline/, api/, app/nicegui/ are separate concerns
4. **Testable** - Each layer can be tested independently
5. **Gradual migration** - Old Streamlit app still works during transition
