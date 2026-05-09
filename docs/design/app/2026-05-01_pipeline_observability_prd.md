# PRD: Pipeline Observability — Per-Step Decision Records

**Date**: 2026-05-01
**Status**: Draft
**Originated from**: `docs/design/app/2026-04-30_explore_page_code_review.md`
**Addresses sightings**: 043, 044, 045, 046

---

## Problem

The Explore page in the main album app reimplements pipeline logic with hardcoded thresholds to guess why images were selected or rejected. This is fragile, inaccurate, and violates separation of concerns.

**Root cause**: Pipeline steps compute decisions but don't emit structured records explaining them. The data dies in memory after execution. The UI has no choice but to reverse-engineer.

## Objective

Every pipeline step emits a **decision record** for every item it processes (image or face). The UI renders these records. Zero logic in the display layer.

## User Stories

1. As a user viewing the Explore page, I want to see exactly why each image was filtered, so I can adjust thresholds.
2. As a user viewing the Selection tab, I want to see the composite score breakdown (quality + penalty components) per image, so I can understand the ranking.
3. As a user viewing Face Detection, I want to see why each face passed or failed filtering, with the actual thresholds used.
4. As a user clicking on an image, I want a popup showing all decisions made about it across every pipeline step.

## Non-Goals

- Changing how pipeline steps make decisions (only how they report them)
- Real-time observability during execution (this is post-run analysis)
- Undo/replay of pipeline decisions

---

## Data Model

### StepDecision

One record per item (image or face) per step.

```python
@dataclass
class StepDecision:
    item_id: str          # image path or face key
    item_type: str        # "image" or "face"
    step: str             # pipeline step name (matches step registry)
    decision: str         # "passed", "rejected", "selected", "detected", "not_detected"
    reason: str           # human-readable, e.g. "IQA 0.08 < threshold 0.20"
    config_used: Dict     # actual thresholds/params used for this decision
    metrics: Dict         # measured values, e.g. {"iqa_score": 0.08, "sharpness": 0.32}
```

### Where decisions are produced

| Step | item_type | decision values | reason examples | config_used keys | metrics keys |
|------|-----------|----------------|-----------------|-----------------|-------------|
| **filter_quality** | image | passed, rejected | "IQA 0.08 < threshold 0.20" | min_iqa_score, min_sharpness | iqa_score, sharpness |
| **detect_persons** | image | detected, not_detected | "Person detected (conf 0.91)" | confidence_threshold | confidence, body_facing_score |
| **insightface_detect_faces** | image | detected (N faces) | "3 faces detected" | detection_threshold, min_face_size | face_count, face_confidences |
| **filter_faces** | face | passed, rejected | "BBox ratio 0.01 < threshold 0.02" | min_confidence, min_bbox_ratio, min_relative_size | confidence, bbox_ratio, relative_size |
| **score_face_frontal** | face | clusterable, not_clusterable | "Frontal score 0.22 < threshold 0.40" | min_frontal_score | frontal_score, eye_bbox_ratio, asymmetry |
| **cluster_scenes** | image | cluster_N, noise | "Assigned to cluster 3 (7 images)" | algorithm, min_cluster_size | cluster_id, cluster_size |
| **cluster_people** | face | cluster_N, noise | "Assigned to person 2 (Jonathan)" | K, distance_threshold | cluster_id, n_faces |
| **select_best** | image | selected, rejected | "Best in cluster 3 (score 0.87)" | max_per_cluster, min_score_threshold, dissimilarity_threshold | composite_score, quality_score, penalty, rank, cluster_id |
| **select_best** (penalty) | image | (part of above) | "Penalty: eyes_closed -0.15, face_turned -0.10" | (none — fixed penalty table) | penalty_total, penalty_components |
| **select_best** (duplicate) | image | duplicate, unique | "Duplicate of IMG_0097 (sim 0.92)" | dissimilarity_threshold | similarity, compared_to |

### Composite score breakdown (select_best detail)

The `metrics` dict for select_best should decompose the composite score:

```python
{
    "composite_score": 0.87,
    "quality_score": 0.87,
    "quality_strategy": "weighted_average",
    "quality_components": {"iqa": 0.82, "iqa_weight": 0.3, "ava": 0.89, "ava_weight": 0.7},
    "penalty_total": 0.00,
    "penalty_components": {},  # or {"eyes_closed": -0.15, "face_turned": -0.10}
    "cluster_id": 3,
    "rank_in_cluster": 1,
    "cluster_size": 5,
    "dissimilarity_check": null  # or {"compared_to": "IMG_0097.jpg", "similarity": 0.92, "is_duplicate": true}
}
```

---

## Storage

### PipelineContext (in-memory during run)

```python
# New field on PipelineContext
step_decisions: List[StepDecision] = field(default_factory=list)
```

Each step appends decisions via:
```python
context.step_decisions.append(StepDecision(
    item_id=image_path,
    item_type="image",
    step=self.metadata.name,
    decision="rejected",
    reason=f"IQA {iqa:.2f} < threshold {min_iqa:.2f}",
    config_used={"min_iqa_score": min_iqa, "min_sharpness": min_sharpness},
    metrics={"iqa_score": iqa, "sharpness": sharpness},
))
```

### Database (PipelineResult model)

New JSON column:
```python
class PipelineResult(Base):
    ...
    step_decisions = Column(JSON, nullable=True)  # List[StepDecision] serialized
```

Migration: idempotent `ALTER TABLE pipeline_results ADD COLUMN step_decisions JSON`.

### API Response

```python
# GET /api/v1/results/{job_id}
{
    ...
    "step_decisions": [
        {"item_id": "IMG_0045.jpg", "item_type": "image", "step": "filter_quality", "decision": "rejected", "reason": "IQA 0.08 < threshold 0.20", ...},
        {"item_id": "IMG_0142.jpg", "item_type": "image", "step": "select_best", "decision": "selected", "reason": "Best in cluster 3 (score 0.87)", ...},
    ]
}
```

---

## Pipeline Step Changes

### filter_quality.py

Current: computes `quality_passed` set, logs aggregate count.

Change: for each image, emit a StepDecision with pass/fail + which threshold was violated.

```python
for img_path in context.image_paths:
    iqa = context.iqa_scores.get(str(img_path), 0)
    sharp = context.sharpness_scores.get(str(img_path), 0)

    passed = iqa >= min_iqa and sharp >= min_sharpness
    if passed:
        reason = f"Passed (IQA {iqa:.2f} >= {min_iqa:.2f}, sharpness {sharp:.2f} >= {min_sharpness:.2f})"
    elif iqa < min_iqa and sharp < min_sharpness:
        reason = f"IQA {iqa:.2f} < {min_iqa:.2f} AND sharpness {sharp:.2f} < {min_sharpness:.2f}"
    elif iqa < min_iqa:
        reason = f"IQA {iqa:.2f} < threshold {min_iqa:.2f}"
    else:
        reason = f"Sharpness {sharp:.2f} < threshold {min_sharpness:.2f}"

    context.step_decisions.append(StepDecision(
        item_id=str(img_path), item_type="image", step="filter_quality",
        decision="passed" if passed else "rejected", reason=reason,
        config_used={"min_iqa_score": min_iqa, "min_sharpness": min_sharpness},
        metrics={"iqa_score": iqa, "sharpness": sharp},
    ))
```

### select_best.py

Current: computes composite_scores, selects images, logs Siamese comparisons.

Change: for each image in each cluster, emit a StepDecision with score breakdown and selection reason.

```python
for img_path in cluster_images:
    score = composite_scores[img_path]
    quality = quality_scores[img_path]
    penalty = penalties[img_path]

    if img_path in selected:
        decision = "selected"
        reason = f"Rank {rank} in cluster {cluster_id} (score {score:.2f})"
    elif score < min_threshold:
        decision = "rejected"
        reason = f"Score {score:.2f} < threshold {min_threshold:.2f}"
    elif is_duplicate:
        decision = "rejected"
        reason = f"Duplicate of {dup_target} (similarity {sim:.2f})"
    else:
        decision = "rejected"
        reason = f"Outranked in cluster {cluster_id} (rank {rank}/{cluster_size})"

    context.step_decisions.append(StepDecision(
        item_id=img_path, item_type="image", step="select_best",
        decision=decision, reason=reason,
        config_used={"max_per_cluster": max_per, "min_score_threshold": min_threshold, "dissimilarity_threshold": dup_thresh},
        metrics={"composite_score": score, "quality_score": quality, "penalty_total": penalty,
                 "penalty_components": penalty_breakdown, "cluster_id": cluster_id, "rank": rank},
    ))
```

### Other steps

Same pattern. Each step appends decisions for items it processes. Detection steps emit "detected"/"not_detected". Clustering steps emit cluster assignments.

---

## Frontend Changes

### explore.py refactor

Split into `app/streamlit/pages/explore/` package:

```
explore/
  __init__.py          # Tab orchestration only
  quality_tab.py       # Reads step_decisions where step="filter_quality"
  detection_tab.py     # Reads step_decisions where step="detect_persons"
  face_scoring_tab.py  # Reads step_decisions where step in ("filter_faces", "score_face_frontal")
  scene_tab.py         # Reads step_decisions where step="cluster_scenes"
  face_cluster_tab.py  # Reads step_decisions where step="cluster_people" + deep-link
  selection_tab.py     # Reads step_decisions where step="select_best"
```

Each tab is <80 lines:
```python
def render(decisions: List[StepDecision], images: List[ImageInfo]):
    my_decisions = [d for d in decisions if d.step == "filter_quality"]
    passed = [d for d in my_decisions if d.decision == "passed"]
    rejected = [d for d in my_decisions if d.decision == "rejected"]

    col1, col2 = st.columns(2)
    col1.metric("Passed", len(passed))
    col2.metric("Rejected", len(rejected))

    rows = [{"Image": d.item_id, "Decision": d.decision, "Reason": d.reason,
             **d.metrics} for d in my_decisions]
    st.dataframe(pd.DataFrame(rows))
```

Zero hardcoded thresholds. Zero reimplemented logic. Just display what the pipeline said.

### Image detail popup

Reads all decisions for the clicked image:
```python
my_decisions = [d for d in all_decisions if d.item_id == image_path]
for d in my_decisions:
    st.write(f"**{d.step}**: {d.decision} — {d.reason}")
```

---

## Implementation Order

| Phase | Scope | Effort | Dependencies |
|-------|-------|--------|--------------|
| 1. Data model | Add `StepDecision` dataclass, `step_decisions` to context, DB column, API field | 1h | None |
| 2. filter_quality | Emit decisions in filter_quality.py | 30min | Phase 1 |
| 3. select_best | Emit decisions with score breakdown in select_best.py | 1h | Phase 1 |
| 4. Other steps | detect_persons, insightface_detect_faces, cluster_scenes, cluster_people | 1h | Phase 1 |
| 5. Frontend refactor | Split explore.py into tab package, consume decisions | 1h | Phases 2-4 |
| 6. Image popup | Show per-image decisions from all steps | 30min | Phase 5 |

**Total: ~5 hours**

## Acceptance Criteria

1. Every pipeline step emits StepDecision records for every item processed
2. Decisions are stored in PipelineResult DB and returned via API
3. Explore page tabs display decisions with zero hardcoded thresholds
4. Image detail popup shows all decisions for clicked image
5. `reason` field contains the actual threshold values used (from config), not constants
6. `metrics` field contains the measured values that were compared against thresholds
7. Existing tests pass (no behavior change in pipeline logic)
8. New test: run pipeline on test data, verify step_decisions are populated with correct reasons

## Risks

- **Performance**: 428 images x 8 steps = ~3400 decision records. As JSON in DB, this is <100KB — negligible.
- **Migration**: New nullable JSON column — backward compatible. Old runs have `step_decisions = null`.
- **API payload size**: Decisions are small dicts. Can add `include_decisions=true` query param if needed.
