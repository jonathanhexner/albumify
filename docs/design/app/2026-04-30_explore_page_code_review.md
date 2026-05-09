# Explore Page Code Review — Path to SOLID

## Current State (Problems)

### Problem 1: Hardcoded Constants in Display Layer

```python
# explore.py lines 433-440
if img.iqa_score is not None and img.iqa_score < 0.2:    # WHERE DOES 0.2 COME FROM?
    return "IQA below threshold"
if img.sharpness is not None and img.sharpness < 0.1:    # WHERE DOES 0.1 COME FROM?
    return "Sharpness below threshold"
if img.composite_score is not None and img.composite_score < 0.4:  # WHERE DOES 0.4 COME FROM?
    return "Composite score too low"
```

**Why it's wrong**: These thresholds exist in `pipeline.yaml` and are configurable by the user. The display layer is guessing them with hardcoded values instead of reading the actual config that was used during the pipeline run.

**What should happen**: The pipeline should emit the thresholds it used AND the decision per image. The UI just displays what the pipeline decided.

---

### Problem 2: Logic in the Display Layer

```python
def _infer_selection_reason(img, all_images) -> str:
    """Infer why an image was selected or rejected."""
    # This function REIMPLEMENTS pipeline logic:
    # - Compares composite scores within clusters
    # - Checks thresholds
    # - Determines rank
```

**Why it's wrong**: The pipeline (`select_best.py`, `filter_quality.py`) already makes these decisions. The UI shouldn't re-derive them — it should READ them from pipeline output. If the pipeline logic changes, this function silently becomes wrong.

**What should happen**: Each pipeline step should write a **decision record** per image:

```python
# Written by filter_quality step:
{
    "image": "IMG_0045.jpg",
    "step": "filter_quality",
    "decision": "rejected",
    "reason": "IQA below threshold",
    "details": {"iqa_score": 0.08, "threshold": 0.20}
}

# Written by select_best step:
{
    "image": "IMG_0142.jpg",
    "step": "select_best",
    "decision": "selected",
    "reason": "Best in cluster 3",
    "details": {"composite_score": 0.87, "cluster_id": 3, "rank": 1, "cluster_size": 5}
}
```

The UI just renders these records — zero logic, zero thresholds.

---

### Problem 3: 400-line Monolith with Duplicated Patterns

`explore.py` has 6 tab renderers that all follow the same pattern:
1. Check if data exists → show info message
2. Show metrics row
3. Build rows list from images
4. Create DataFrame
5. Show dataframe

This pattern is repeated 6 times with slight variations. Each tab function is 40-60 lines of nearly identical boilerplate.

**What should happen**: Each tab should be its own file (like the face clustering app: `tabs/overview_tab.py`, `tabs/merge_analysis_tab.py`). Shared patterns should be extracted into reusable components.

---

## Target Architecture (SOLID)

### Single Responsibility: One file per tab

```
app/streamlit/pages/explore/
    __init__.py          # render_explore_page() — just tab orchestration
    quality_tab.py       # Image quality filtering analysis
    detection_tab.py     # Person detection results
    face_scoring_tab.py  # Face detection & scoring
    scene_tab.py         # Scene clustering
    face_cluster_tab.py  # Face clustering + deep-link
    selection_tab.py     # Selection decisions
```

Each tab file is <100 lines and imports from shared components.

### Open/Closed: Decision records from pipeline

**Pipeline side** (backend changes):

```
PipelineContext gains:
    image_decisions: Dict[str, ImageDecision]  # path -> decision record

@dataclass
class ImageDecision:
    image_path: str
    filter_decision: Optional[StepDecision]   # from filter_quality
    selection_decision: Optional[StepDecision] # from select_best

@dataclass
class StepDecision:
    step: str           # "filter_quality", "select_best"
    decision: str       # "passed", "rejected", "selected"
    reason: str         # "IQA below threshold", "Best in cluster"
    config_used: Dict   # {"min_iqa_score": 0.2, ...} — ACTUAL config, not hardcoded
    details: Dict       # step-specific details
```

**API side**: New field on result response:
```python
# GET /api/v1/results/{job_id}
{
    ...
    "image_decisions": [
        {"image": "IMG_0045.jpg", "filter_decision": {"decision": "rejected", "reason": "IQA 0.08 < threshold 0.20"}, ...},
        {"image": "IMG_0142.jpg", "selection_decision": {"decision": "selected", "reason": "Best in cluster 3 (score 0.87)"}, ...},
    ]
}
```

**UI side**: Tab just renders the records:
```python
def _render_selection_tab(decisions: List[ImageDecision]):
    rows = [
        {"Image": d.image_path, "Status": d.selection_decision.decision,
         "Reason": d.selection_decision.reason}
        for d in decisions if d.selection_decision
    ]
    st.dataframe(pd.DataFrame(rows))
```

Zero logic. Zero thresholds. Just display what the pipeline said.

### Liskov Substitution: Tab interface

Each tab implements the same interface:
```python
class ExploreTab(Protocol):
    title: str
    def render(self, images: List[ImageInfo], result: dict, album: Album) -> None: ...
```

Tabs are registered, not hardcoded:
```python
TABS = [QualityTab(), DetectionTab(), FaceScoringTab(), ...]
for tab in TABS:
    with st.tabs([t.title for t in TABS])[i]:
        tab.render(images, result, album)
```

### Interface Segregation: Shared components

Extract repeated patterns:
```python
# components/metrics_summary.py
def render_metrics_row(metrics: Dict[str, int]) -> None:
    """Render a row of metric cards."""

# components/image_table.py
def render_image_dataframe(rows: List[Dict], progress_columns: List[str]) -> None:
    """Render a sortable dataframe with optional progress bar columns."""
```

### Dependency Inversion: Data comes from API, not computed in UI

Current (wrong):
```
UI → reads raw scores → reimplements pipeline logic → displays result
```

Target (right):
```
Pipeline → computes decisions → stores in DB
API → returns decisions
UI → displays decisions
```

---

## Implementation Plan

### Step 1: Add decision records to pipeline (backend)
- `filter_quality.py`: Write `filter_reason` per image to context
- `select_best.py`: Write `selection_reason` per image to context
- Store in `PipelineResult.image_decisions` (JSON column)
- Return via API

### Step 2: Split explore.py into tab files
- Create `app/streamlit/pages/explore/` package
- Move each `_render_*_tab` to its own file
- Extract shared patterns to components

### Step 3: Replace _infer_selection_reason with pipeline data
- Tab reads `image_decisions` from API response
- No hardcoded thresholds
- No reimplemented logic

### Step 4: Extract shared rendering patterns
- `render_metrics_row()`
- `render_image_dataframe()` with configurable columns

---

## Priority

1. **Step 1 is the most important** — without pipeline decision records, the UI will always be wrong or guessing
2. Step 2 is structural cleanup — makes the code maintainable
3. Step 3 is the payoff — clean display of accurate data
4. Step 4 is polish — reduces duplication

**Estimated effort**: Step 1 is 2-3 hours (backend + API). Steps 2-4 are 1-2 hours (frontend refactor).
