# Dead Code Audit Report

**Date**: 2025-12-11
**Status**: Comprehensive audit complete

## Executive Summary

Found multiple categories of dead/outdated code:
1. **VS Code launch configurations** referencing deleted `train_multifeature_ranker.py`
2. **34 markdown documentation files** in project root (potential clutter)
3. **Duplicate/test scripts** that may no longer be needed
4. **run_hyperparameter_search.py** still references old training script

---

## 1. Critical: VS Code Launch Configuration Issues

### File: `.vscode/launch.json`

**Problem**: Multiple launch configurations (lines 483-927) reference `train_multifeature_ranker.py` which has been **deleted/deprecated**.

#### Outdated Configurations:

1. **Line 483-499**: "Train Multi-Feature Ranker - Full (Recommended) ⭐"
   ```json
   "program": "${workspaceFolder}/train_multifeature_ranker.py"
   ```

2. **Line 501-517**: "Train Multi-Feature Ranker - Quick Test (Small MLP)"
3. **Line 519-535**: "Train Multi-Feature Ranker - Deep MLP"
4. **Line 537-554**: "Train Multi-Feature Ranker - Quick Experiment (10%) ⚡"
5. **Line 556-572**: "Train Multi-Feature Ranker - Full Dataset ⭐"
6. **Line 574-591**: "Train Multi-Feature Ranker - No Series Sampling (OLD)"
7. **Line 593-611**: "Train Multi-Feature Ranker - CLIP Only"
8. **Line 801-827**: "Train VGG16 Paper Replication - Single Run"
9. **Line 829-860**: "Paper Exact - Quick Test (5 epochs) ⚡"
10. **Line 862-892**: "Paper Exact - Full 30 Epochs (End-to-End CNN + Paper Preprocessing) 🎯"
11. **Line 894-924**: "Paper Exact - Frozen CNN Baseline (30 epochs) 📊"

### Recommended Action:

**Replace all references to `train_multifeature_ranker.py` with new training scripts:**

- For **frozen features** → `sim_bench/training/train_frozen.py`
- For **end-to-end** → `sim_bench/training/train_siamese_e2e.py`

**Example replacement:**
```json
{
    "name": "Train Frozen Features - Multi-feature ⭐",
    "type": "python",
    "request": "launch",
    "module": "sim_bench.training.train_frozen",
    "args": [
        "--config", "configs/frozen/multifeature.yaml"
    ],
    "cwd": "${workspaceFolder}",
    "console": "integratedTerminal",
    "justMyCode": true
}
```

---

## 2. run_hyperparameter_search.py References Old Script

### File: `run_hyperparameter_search.py`

**Issue**: This script spawns `train_multifeature_ranker.py` as a subprocess.

### Action Required:
- Update to use `sim_bench.training.train_frozen` or deprecate if no longer used
- Check if hyperparameter search is still actively used

---

## 3. Documentation Clutter (34 Markdown Files)

### Current State:
```
AGENT_ARCHITECTURE.md
AI_AGENT_IMPLEMENTATION_SUMMARY.md
ALL_METHODS_UPDATED.md
APP_ARCHITECTURE.md
CLEANUP_SUMMARY.md
CLIP_AESTHETIC_EXPERIMENT.md
CLIP_AESTHETIC_INTEGRATED.md
ENV_SETUP.md
FINAL_SUMMARY.md
HYPERPARAM_SEARCH_SUMMARY.md
HYPERPARAMETER_SEARCH_QUICKSTART.md
IMPLEMENTATION_COMPLETE.md
IMPLEMENTATION_SUMMARY.md
LEARNED_CLIP_PROMPTS_SUMMARY.md
METHODS_FIXED.md
MIGRATION_GUIDE.md
PAIRWISE_BENCHMARK_COMPLETE.md
PAIRWISE_BENCHMARK_IMPLEMENTATION_PLAN.md
PHOTO_ORGANIZATION_IMPLEMENTATION.md
PHOTOTRIAGE_CONTRASTIVE_SUMMARY.md
PROJECT_SUMMARY.md
README.md
REFACTORING_COMPLETE.md
REFACTORING_PLAN.md
REFACTORING_PROGRESS.md
REFACTORING_SUMMARY.md
REPRODUCE_RESULTS.md
SIAMESE_NETWORK_UPDATE.md
SYNTHETIC_DEGRADATION_SUMMARY.md
TESTING_REPORT.md
TRAINING_QUICKSTART.md
TRAINING_REFACTORING_COMPLETE.md
TRAINING_SCRIPTS_REFACTORING.md
VISION_LANGUAGE_MODULE_SUMMARY.md
```

### Analysis:

#### Keep (Essential Documentation):
- ✅ `README.md` - Main project documentation
- ✅ `ENV_SETUP.md` - Environment setup guide
- ✅ `PROJECT_SUMMARY.md` - Project overview
- ✅ `MIGRATION_GUIDE.md` - Migration from old to new scripts

#### Consolidate (Redundant Implementation Summaries):
These are all implementation summaries that should be consolidated:
- 🔄 `IMPLEMENTATION_COMPLETE.md`
- 🔄 `IMPLEMENTATION_SUMMARY.md`
- 🔄 `REFACTORING_COMPLETE.md`
- 🔄 `REFACTORING_SUMMARY.md`
- 🔄 `TRAINING_SCRIPTS_REFACTORING.md`
- 🔄 `TRAINING_REFACTORING_COMPLETE.md`
- 🔄 `FINAL_SUMMARY.md`

**Suggested consolidation**: Create single `CHANGELOG.md` or `DEVELOPMENT_HISTORY.md`

#### Archive (Historical/Completed):
Move to `docs/archive/`:
- 📦 `REFACTORING_PLAN.md` - Completed plan
- 📦 `REFACTORING_PROGRESS.md` - Completed progress tracking
- 📦 `PAIRWISE_BENCHMARK_IMPLEMENTATION_PLAN.md` - Completed plan
- 📦 `TESTING_REPORT.md` - One-time report
- 📦 `CLEANUP_SUMMARY.md` - One-time cleanup
- 📦 `ALL_METHODS_UPDATED.md` - One-time update
- 📦 `METHODS_FIXED.md` - One-time fix

#### Module-Specific (Move to docs/):
- 📁 `AGENT_ARCHITECTURE.md` → `docs/agent/`
- 📁 `AI_AGENT_IMPLEMENTATION_SUMMARY.md` → `docs/agent/`
- 📁 `APP_ARCHITECTURE.md` → `docs/app/`
- 📁 `CLIP_AESTHETIC_EXPERIMENT.md` → `docs/quality_assessment/`
- 📁 `CLIP_AESTHETIC_INTEGRATED.md` → `docs/quality_assessment/`
- 📁 `LEARNED_CLIP_PROMPTS_SUMMARY.md` → `docs/vision_language/`
- 📁 `VISION_LANGUAGE_MODULE_SUMMARY.md` → `docs/vision_language/`
- 📁 `PHOTO_ORGANIZATION_IMPLEMENTATION.md` → `docs/clustering/`
- 📁 `SIAMESE_NETWORK_UPDATE.md` → `docs/quality_assessment/`
- 📁 `SYNTHETIC_DEGRADATION_SUMMARY.md` → `docs/quality_assessment/`
- 📁 `PHOTOTRIAGE_CONTRASTIVE_SUMMARY.md` → `docs/quality_assessment/`

#### Quickstart/Guides (Move to docs/guides/):
- 📖 `HYPERPARAMETER_SEARCH_QUICKSTART.md` → `docs/guides/`
- 📖 `TRAINING_QUICKSTART.md` → `docs/guides/`
- 📖 `REPRODUCE_RESULTS.md` → `docs/guides/`
- 📖 `PAIRWISE_BENCHMARK_COMPLETE.md` → `docs/guides/`
- 📖 `HYPERPARAM_SEARCH_SUMMARY.md` → `docs/guides/`

---

## 4. Potentially Unused Scripts (Root Directory)

### Testing/Development Scripts:
- ❓ `test_simple_training.py` - May be obsolete after training refactor
- ❓ `test_training_pipeline.py` - May be obsolete after training refactor
- ❓ `test_agent_with_images.py` - Test script, should be in `tests/`
- ❓ `verify_learned_clip_integration.py` - Verification script, may be one-time use

### App Variants:
- ❓ `app.py` - Which app is canonical?
- ❓ `app_agent.py` - v1?
- ❓ `app_agent_v2.py` - v2?

**Action**: Determine which app version is current, deprecate others

### Duplicate Agent CLI:
- ❓ `run_agent_cli.py` - Is this duplicate of `sim_bench.agent` module?

### Analysis Scripts:
- ❓ `analyze_phototriage_results.py` - Still used? Move to `notebooks/` or `scripts/`?
- ❓ `visualize_quality_benchmark.py` - Still used? Move to `scripts/`?

---

## 5. Cleanup Recommendations by Priority

### 🔴 CRITICAL (Do Immediately):

1. **Fix `.vscode/launch.json`**
   - Update 11 outdated launch configurations
   - Replace `train_multifeature_ranker.py` references with new scripts

2. **Update `run_hyperparameter_search.py`**
   - Update to use new training scripts or deprecate

### 🟡 HIGH PRIORITY (Do Soon):

3. **Consolidate Root Documentation**
   - Create `docs/archive/` folder
   - Move 15-20 historical/completed docs to archive
   - Create single `CHANGELOG.md` for historical summaries

4. **Organize Module Documentation**
   - Move module-specific docs to `docs/[module]/`
   - Create clear doc structure

### 🟢 MEDIUM PRIORITY (When Time Permits):

5. **Clean Up Test Scripts**
   - Move test scripts to `tests/` directory
   - Remove obsolete test files

6. **Clarify App Variants**
   - Determine canonical app version
   - Deprecate/remove old versions

7. **Reorganize Analysis Scripts**
   - Move to `scripts/analysis/` or `notebooks/`

---

## 6. Suggested Directory Structure

```
sim-bench/
├── README.md                    # Main documentation (keep)
├── ENV_SETUP.md                 # Setup guide (keep)
├── PROJECT_SUMMARY.md           # Project overview (keep)
├── CHANGELOG.md                 # NEW: Consolidated development history
├── docs/
│   ├── guides/                  # User guides & quickstarts
│   │   ├── TRAINING_QUICKSTART.md
│   │   ├── HYPERPARAMETER_SEARCH_QUICKSTART.md
│   │   └── REPRODUCE_RESULTS.md
│   ├── agent/                   # Agent documentation
│   │   ├── AGENT_ARCHITECTURE.md
│   │   └── AI_AGENT_IMPLEMENTATION_SUMMARY.md
│   ├── quality_assessment/      # Quality assessment docs (already exists)
│   ├── vision_language/         # NEW: Vision-language docs
│   ├── clustering/              # NEW: Clustering docs
│   └── archive/                 # NEW: Historical/completed docs
│       ├── REFACTORING_PLAN.md
│       ├── IMPLEMENTATION_COMPLETE.md
│       └── ...
├── scripts/
│   ├── analysis/                # NEW: Analysis scripts
│   │   ├── analyze_phototriage_results.py
│   │   └── visualize_quality_benchmark.py
│   └── phototriage/             # Already exists
└── tests/                       # All test files here
    ├── test_training_pipeline.py
    └── ...
```

---

## 7. Dead Code Summary Table

| Item | Type | Status | Action |
|------|------|--------|--------|
| `.vscode/launch.json` configs | Config | 🔴 BROKEN | Update to new scripts |
| `train_multifeature_ranker.py` refs | References | 🔴 BROKEN | Already deleted, update refs |
| `run_hyperparameter_search.py` | Script | 🟡 OUTDATED | Update or deprecate |
| 34 markdown files in root | Docs | 🟡 CLUTTERED | Organize into `docs/` |
| `test_*.py` in root | Tests | 🟢 MISPLACED | Move to `tests/` |
| `app*.py` variants | Apps | ❓ UNCLEAR | Clarify which is current |
| Analysis scripts in root | Scripts | 🟢 MISPLACED | Move to `scripts/` |

---

## 8. Immediate Action Items

### Step 1: Fix Broken References (Critical)
```bash
# Update .vscode/launch.json to reference:
# - sim_bench/training/train_frozen.py
# - sim_bench/training/train_siamese_e2e.py
```

### Step 2: Update or Deprecate run_hyperparameter_search.py
```bash
# Check if still used, then either:
# - Update to use new training scripts
# - Add deprecation notice
# - Delete if no longer needed
```

### Step 3: Organize Documentation
```bash
# Create directories
mkdir -p docs/archive docs/guides docs/agent docs/vision_language docs/clustering

# Move files (examples)
mv REFACTORING_*.md docs/archive/
mv HYPERPARAMETER_SEARCH_QUICKSTART.md docs/guides/
mv AGENT_ARCHITECTURE.md docs/agent/
```

### Step 4: Clean Up Root
```bash
# Move test files
mv test_*.py tests/

# Move analysis scripts
mkdir -p scripts/analysis
mv analyze_phototriage_results.py scripts/analysis/
mv visualize_quality_benchmark.py scripts/analysis/
```

---

## 9. Files Confirmed Clean

✅ `sim_bench/training/` - No legacy files, already cleaned up
✅ Training scripts refactored and working
✅ `sim_bench/training/README.md` - Up to date

---

## 10. Parameter Mismatch Fixed ✅

### File: `sim_bench/training/train_siamese_e2e.py`

**Issue**: Incorrect parameter names when creating `MultiFeatureConfig`:
- Used `use_cnn` instead of `use_cnn_features`
- Used `mlp_dropout` instead of `dropout`
- Used non-existent `mlp_embedding_dim`

**Fixed**:
- ✅ Changed `use_cnn=True` → `use_cnn_features=True`
- ✅ Changed `mlp_dropout=...` → `dropout=...`
- ✅ Removed `mlp_embedding_dim` parameter
- ✅ Updated `configs/siamese_e2e/resnet50.yaml` to remove unused parameter

**Status**: Script now compiles without errors

---

## Conclusion

**Total Issues Found**:
- 🔴 Critical: 2 (launch.json, run_hyperparameter_search.py)
- 🟡 High: 2 (doc organization, module docs)
- 🟢 Medium: 3 (test scripts, app variants, analysis scripts)
- ✅ Fixed: 1 (parameter mismatch in train_siamese_e2e.py)

**Estimated Cleanup Time**:
- Critical fixes: 30 minutes
- Full cleanup: 2-3 hours

**Priority**: Start with critical fixes to prevent confusion when using IDE launch configurations.
