# Troubleshooting Guide

## Common Issues and Solutions

### ❌ MediaPipe Protobuf Error

**Error**:
```
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
```

**Cause**: Old MediaPipe pipeline steps are being executed instead of InsightFace steps. This happens because the database config profile still has the old configuration.

**Solution**: Reset the database config profile.

#### Option 1: Run Reset Script (Recommended)

```bash
python scripts/reset_db_config.py
```

This script will:
1. Show you the old vs new pipeline configuration
2. Update the database to match `configs/pipeline.yaml`
3. Prompt you to restart the API

#### Option 2: Delete Database (Nuclear)

```bash
# Stop the API
# Delete database
del sim_bench.db  # Windows
rm sim_bench.db   # Linux/Mac

# Restart API - it will recreate with new config
python -m uvicorn sim_bench.api.main:app --reload --port 8000
```

#### Option 3: API Call (If Reset Endpoint Exists)

```bash
curl -X POST http://localhost:8000/api/v1/config/profiles/default/reset
```

### Why This Happens

1. The API caches config from `pipeline.yaml` in the database
2. When you update `pipeline.yaml`, the database still has the old config
3. The API uses the database config, not the file
4. You need to reset the database config to pick up changes

### Verification

After resetting, check the logs when starting the pipeline. You should see:

```
✅ Good (InsightFace):
2026-02-07 21:16:49 - sim_bench.pipeline.steps.detect_persons - INFO - Detecting persons...
2026-02-07 21:16:49 - sim_bench.pipeline.insightface_pipeline.face_analyzer - INFO - Loaded InsightFace model: buffalo_l
2026-02-07 21:18:01 - sim_bench.pipeline.steps.insightface_detect_faces - INFO - Detected 311 faces across 122 images

❌ Bad (MediaPipe):
2026-02-07 21:18:07 - sim_bench.pipeline.steps.score_face_eyes - INFO - MediaPipe face mesh loaded...
AttributeError: 'SymbolDatabase' object has no attribute 'GetPrototype'
```

---

## Database Config vs File Config

### How Config Loading Works

```
1. API Startup
   ↓
2. Load configs/pipeline.yaml
   ↓
3. Store in database (if not exists)
   ↓
4. All pipeline runs use DATABASE config
```

### When to Reset Config

Reset the database config whenever you:
- ✅ Update `configs/pipeline.yaml`
- ✅ Change pipeline steps
- ✅ Modify step configurations
- ✅ Switch between MediaPipe and InsightFace

### Config Priority

1. **Runtime overrides** (passed in API request) - Highest priority
2. **Database profile config** - Medium priority
3. **pipeline.yaml** - Lowest priority (only used at DB creation)

---

## Other Common Issues

### Issue: Pipeline Uses Wrong Steps

**Symptoms**:
- Old MediaPipe steps executing
- InsightFace models not loading
- Unexpected errors

**Solution**: 
1. Check database config: `python scripts/reset_db_config.py`
2. Verify `DEFAULT_PIPELINE` in `sim_bench/api/services/pipeline_service.py`
3. Restart API after config changes

### Issue: Protobuf Version Conflicts

**Symptoms**:
```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

**Cause**: MediaPipe requires protobuf < 4.0, but newer versions are installed.

**Solution**: 
- Don't fix protobuf! Fix the pipeline config (see above)
- The new pipeline uses InsightFace, which works with newer protobuf

### Issue: Step Not Found

**Symptoms**:
```
KeyError: 'insightface_score_eyes'
```

**Cause**: Step not registered in the registry.

**Solution**:
1. Check step file has `@register_step` decorator
2. Check `sim_bench/pipeline/steps/all_steps.py` imports the step
3. Restart API to reload registry

---

## Prevention

### Keep Config in Sync

When updating `configs/pipeline.yaml`:

```bash
# 1. Update the file
vim configs/pipeline.yaml

# 2. Reset database config
python scripts/reset_db_config.py

# 3. Restart API
# Ctrl+C in API terminal
python -m uvicorn sim_bench.api.main:app --reload --port 8000
```

### Use Version Control

```bash
# Check what changed
git diff configs/pipeline.yaml

# If database needs reset
python scripts/reset_db_config.py
```

---

## Quick Diagnostic Script

Save this as `scripts/check_config.py`:

```python
"""Check current pipeline configuration."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sim_bench.api.database.models import ConfigProfile
from sim_bench.api.services.config_service import get_default_config

db_path = project_root / "sim_bench.db"
engine = create_engine(f'sqlite:///{db_path}')
Session = sessionmaker(bind=engine)
session = Session()

# Database config
profile = session.query(ConfigProfile).filter(
    ConfigProfile.name == "default"
).first()

print("DATABASE CONFIG:")
if profile:
    db_pipeline = profile.config.get('default_pipeline', [])
    for i, step in enumerate(db_pipeline, 1):
        print(f"  {i}. {step}")
else:
    print("  No default profile found")

print("\nFILE CONFIG (pipeline.yaml):")
file_config = get_default_config()
file_pipeline = file_config.get('default_pipeline', [])
for i, step in enumerate(file_pipeline, 1):
    print(f"  {i}. {step}")

print("\nMATCH:", "✅ YES" if db_pipeline == file_pipeline else "❌ NO (reset needed!)")
session.close()
```

Run with:
```bash
python scripts/check_config.py
```

---

## Still Having Issues?

1. **Check logs** in `logs/` directory
2. **Verify steps are registered**: `GET /api/v1/steps`
3. **Check pipeline execution**: Watch WebSocket messages or check database
4. **Ask for help**: Include error logs and config

---

## Related Documentation

- `docs/architecture/ACTIVE_VS_LEGACY_SUMMARY.md` - Which components are active
- `docs/architecture/PIPELINE_ARCHITECTURE_CURRENT_STATE.md` - Detailed architecture
- `configs/pipeline.yaml` - Configuration file
- `sim_bench/api/services/pipeline_service.py` - Default pipeline definition
