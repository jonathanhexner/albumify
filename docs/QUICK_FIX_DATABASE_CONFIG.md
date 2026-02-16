# Quick Fix: Database Config Issue

## 🚨 The Problem

**You're getting this error**:
```
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
File "d:\sim-bench\sim_bench\pipeline\steps\score_face_eyes.py", line 147
```

**Why**: The OLD MediaPipe step (`score_face_eyes`) is running instead of the NEW InsightFace step (`insightface_score_eyes`)!

## 🔍 Root Cause

The API caches config from `pipeline.yaml` in the database when it first starts. Your database still has the OLD MediaPipe pipeline config, even though you updated `pipeline.yaml` to use InsightFace.

```
Database (OLD):          pipeline.yaml (NEW):
- detect_faces           - detect_persons
- score_face_eyes   ❌    - insightface_score_eyes ✅
- score_face_smile       - insightface_score_expression
- score_face_pose        - insightface_score_pose
```

## ✅ Solution

### Option 1: Run the Reset Script (Easiest)

```bash
python scripts/reset_db_config.py
```

This will:
1. Show you the old vs new pipeline
2. Update the database to match `pipeline.yaml`
3. Tell you to restart the API

Then restart:
```bash
python -m uvicorn sim_bench.api.main:app --reload --port 8000
```

### Option 2: Delete Database (Nuclear)

```bash
# Stop the API (Ctrl+C)

# Delete database
del sim_bench.db  # Windows
rm sim_bench.db   # Linux/Mac

# Restart API - it will recreate with new config
python -m uvicorn sim_bench.api.main:app --reload --port 8000
```

## 🎯 What Changed

I also fixed the metadata in the InsightFace steps:

```python
# Before (wrong):
produces={"insightface_eyes_scores"}

# After (correct):
produces={"face_eyes_scores"}  # Common interface
```

This ensures the metadata matches what the steps actually write to context.

## 🔄 Always Do This

Whenever you update `configs/pipeline.yaml`:

```bash
# 1. Update the file
# 2. Reset database config
python scripts/reset_db_config.py

# 3. Restart API
```

Otherwise the database will keep using the old config!

## ✅ Verify It Worked

After restarting, check the logs. You should see:

**✅ Good (InsightFace pipeline)**:
```
2026-02-07 21:16:49 - sim_bench.pipeline.steps.detect_persons - INFO - Detecting persons...
2026-02-07 21:16:49 - sim_bench.pipeline.insightface_pipeline.face_analyzer - INFO - Loaded InsightFace model: buffalo_l
2026-02-07 21:18:01 - sim_bench.pipeline.steps.insightface_score_eyes - INFO - Scoring eyes...
```

**❌ Bad (MediaPipe still running)**:
```
2026-02-07 21:18:07 - sim_bench.pipeline.steps.score_face_eyes - INFO - MediaPipe face mesh loaded...
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
```

## 📚 More Info

See `docs/TROUBLESHOOTING.md` for detailed troubleshooting guide.
