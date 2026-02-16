# Common Interface Solution

## Problem Solved

Instead of having different attribute names for different backends, we now use a **common interface** that both MediaPipe and InsightFace write to.

## The Solution

### Before (Backend-Specific Attributes)

```python
# MediaPipe writes to:
context.face_eyes_scores
context.face_pose_scores  
context.face_smile_scores

# InsightFace wrote to:
context.insightface_eyes_scores    ❌ Different names!
context.insightface_pose_scores    ❌ Different names!
context.insightface_expression_scores  ❌ Different names!

# Problem: select_best couldn't find InsightFace scores
```

### After (Common Interface)

```python
# Both pipelines write to the SAME attributes:
context.face_eyes_scores   ✅ Common interface
context.face_pose_scores   ✅ Common interface
context.face_smile_scores  ✅ Common interface

# Result: select_best works with both backends automatically!
```

## Changes Made

### 1. InsightFace Score Eyes (`insightface_score_eyes.py`)

```python
# Changed from:
context.insightface_eyes_scores = {...}

# To:
context.face_eyes_scores = {...}
```

### 2. InsightFace Score Expression (`insightface_score_expression.py`)

```python
# Changed from:
context.insightface_expression_scores = {...}

# To:
context.face_smile_scores = {...}
```

### 3. InsightFace Score Pose (`insightface_score_pose.py`)

```python
# Changed from:
context.insightface_pose_scores = {...}

# To:
context.face_pose_scores = {...}
```

### 4. Scoring Strategy (`scoring/strategy.py`)

Updated `FacePenaltyStrategy` to use standard attribute names:

```python
# Changed from:
eyes_score = context.insightface_eyes_scores.get(face_key, 0.5)
smile_score = context.insightface_expression_scores.get(face_key, 0.5)
pose_score = context.insightface_pose_scores.get(face_key, 0.5)

# To:
eyes_score = context.face_eyes_scores.get(face_key, 0.5)
smile_score = context.face_smile_scores.get(face_key, 0.5)
pose_score = context.face_pose_scores.get(face_key, 0.5)
```

## Benefits

### 1. ✅ No Changes to `select_best` Needed

The existing `_score_face_cluster()` method works perfectly with both backends:

```python
def _score_face_cluster(self, context, image_paths, weights):
    for path in image_paths:
        # These work for BOTH MediaPipe and InsightFace now!
        eyes_scores = self._get_face_scores_for_image(context.face_eyes_scores, path)
        pose_scores = self._get_face_scores_for_image(context.face_pose_scores, path)
        smile_scores = self._get_face_scores_for_image(context.face_smile_scores, path)
        # ... rest of scoring ...
```

### 2. ✅ Clean Abstraction

`select_best` doesn't need to know which backend was used. It just reads from the standard interface.

### 3. ✅ Follows Dependency Inversion Principle

Both backends depend on the abstraction (common attribute names), not on concrete implementations.

### 4. ✅ Easy to Add More Backends

Future backends (e.g., DeepFace, FaceNet) just need to write to the same attributes:

```python
# Future: DeepFace backend
context.face_eyes_scores = {...}  # Just use standard names
context.face_pose_scores = {...}
context.face_smile_scores = {...}
```

### 5. ✅ Simpler Architecture

No need for:
- Backend detection logic
- Strategy pattern for score getters
- Config passing through method chains
- Conditional attribute access

## How It Works Now

### MediaPipe Pipeline

```
score_face_eyes
  ↓ Sets: context.face_eyes_scores

score_face_pose
  ↓ Sets: context.face_pose_scores

score_face_smile
  ↓ Sets: context.face_smile_scores

select_best
  ↓ Reads: context.face_eyes_scores ✓
  ↓ Reads: context.face_pose_scores ✓
  ↓ Reads: context.face_smile_scores ✓
```

### InsightFace Pipeline

```
insightface_score_eyes
  ↓ Sets: context.face_eyes_scores (same name!)

insightface_score_pose
  ↓ Sets: context.face_pose_scores (same name!)

insightface_score_expression
  ↓ Sets: context.face_smile_scores (same name!)

select_best
  ↓ Reads: context.face_eyes_scores ✓
  ↓ Reads: context.face_pose_scores ✓
  ↓ Reads: context.face_smile_scores ✓
```

## Testing

### Test 1: MediaPipe Pipeline

```bash
python -m sim_bench.pipeline.run --pipeline default_pipeline
```

**Expected**:
- MediaPipe steps execute
- `context.face_eyes_scores` populated by `score_face_eyes`
- `select_best` uses MediaPipe scores
- Selection works correctly

### Test 2: InsightFace Pipeline

```bash
python -m sim_bench.pipeline.run --pipeline insightface_pipeline
```

**Expected**:
- InsightFace steps execute
- `context.face_eyes_scores` populated by `insightface_score_eyes`
- `select_best` uses InsightFace scores
- Selection works correctly

### Test 3: Verify Scores Are Used

Add logging to verify scores are actually being used:

```python
# In select_best._score_face_cluster()
logger.info(f"Eyes scores for {path}: {eyes_scores}")
logger.info(f"Pose scores for {path}: {pose_scores}")
logger.info(f"Smile scores for {path}: {smile_scores}")
```

Run both pipelines and verify:
- MediaPipe: Scores from MediaPipe models
- InsightFace: Scores from InsightFace models (currently 0.5 neutral, but extensible)

## Design Pattern: Interface Segregation

This follows the **Interface Segregation Principle** from SOLID:

```
┌─────────────────────────────────┐
│   Common Interface (Abstract)   │
│                                 │
│  - face_eyes_scores             │
│  - face_pose_scores             │
│  - face_smile_scores            │
└─────────────────────────────────┘
           ▲         ▲
           │         │
    ┌──────┘         └──────┐
    │                       │
┌───────────┐         ┌──────────────┐
│ MediaPipe │         │ InsightFace  │
│  Backend  │         │   Backend    │
└───────────┘         └──────────────┘
```

Both backends implement the same interface, so consumers (like `select_best`) don't need to know which backend is being used.

## Comparison with Alternative Approaches

### ❌ Backend Detection

```python
# Would require:
if hasattr(context, 'insightface_eyes_scores'):
    eyes_scores = context.insightface_eyes_scores
else:
    eyes_scores = context.face_eyes_scores
```

**Problems**:
- Violates project guidelines (if/else)
- Tight coupling to backend names
- Harder to add new backends

### ❌ Strategy Pattern for Score Access

```python
# Would require:
score_getter = FaceScoreGetter(backend)
eyes_scores = score_getter.get_eyes(context, path)
```

**Problems**:
- More complex
- Extra abstraction layer
- Still needs backend detection somewhere

### ✅ Common Interface (Our Solution)

```python
# Simple and clean:
eyes_scores = context.face_eyes_scores
```

**Benefits**:
- No conditionals
- No backend detection
- No extra abstractions
- Just works!

## Future Enhancements

### 1. Add Type Hints

```python
from typing import Dict

class PipelineContext:
    face_eyes_scores: Dict[str, float]
    face_pose_scores: Dict[str, float]
    face_smile_scores: Dict[str, float]
```

### 2. Document the Interface

Add to `PipelineContext` docstring:

```python
"""
Standard Face Scoring Interface:
- face_eyes_scores: Eye openness scores (0-1)
- face_pose_scores: Face orientation scores (0-1, 1=frontal)
- face_smile_scores: Smile/expression scores (0-1, 1=smiling)

All face scoring backends must populate these attributes.
"""
```

### 3. Validation

Add validation to ensure backends follow the interface:

```python
def validate_face_scores(context):
    """Ensure face scoring interface is properly implemented."""
    required = ['face_eyes_scores', 'face_pose_scores', 'face_smile_scores']
    for attr in required:
        assert hasattr(context, attr), f"Missing required attribute: {attr}"
```

## Conclusion

By using a **common interface**, we achieved:

✅ **Simplicity** - No complex backend detection or strategy patterns
✅ **Maintainability** - Easy to understand and modify
✅ **Extensibility** - Easy to add new backends
✅ **Compatibility** - Works with existing code without changes
✅ **Clean Architecture** - Follows SOLID principles

This is a textbook example of **"Program to an interface, not an implementation"** from the Gang of Four design patterns!
