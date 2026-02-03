# Method Configuration Fixed - Now Matches Pairwise Benchmark Exactly

## Issue Identified

The synthetic degradation testing was using **incorrect method names** that didn't match the actual registry.

### Problem
- ❌ Used `rule_based_sharpness` as a method type (doesn't exist in registry)
- ❌ Created separate methods for each rule-based variant
- ❌ Didn't match pairwise benchmark configuration format

### Root Cause
Only **ONE** rule-based method registered: `'rule_based'`

Different weights are passed as **configuration**, not as separate method types!

## Registry Check

Actual registered methods (verified):
```python
['clip_aesthetic',           # Legacy CLIP method
 'clip_aesthetic_overall',   # ✓ Correct
 'clip_color',               # ✓ Correct
 'clip_composition',         # ✓ Correct
 'clip_cropping',            # ✓ Correct
 'clip_exposure',            # ✓ Correct
 'clip_sharpness',           # ✓ Correct
 'clip_subject_placement',   # ✓ Correct
 'cnn',                      # CNN method
 'musiq',                    # MUSIQ method
 'nima',                     # NIMA method
 'rule_based',               # ✓ ONE rule-based method (weights in config)
 'transformer',              # Transformer method
 'vit']                      # ✓ Correct
```

## Solution

Now using **exact same format** as [pairwise_benchmark.phototriage_final.yaml](configs/pairwise_benchmark.phototriage_final.yaml):

### Correct Configuration Format

```python
{
    'name': 'RuleBased_Sharpness',      # Display name for results
    'type': 'rule_based',                # Registered method type
    'weights': {                         # Method-specific config
        'sharpness': 1.0,
        'exposure': 0.0,
        'colorfulness': 0.0,
        'contrast': 0.0
    }
}
```

**Key insight**:
- `type` = what's registered in `QualityMethodRegistry`
- `name` = display name for results/output
- Everything else = method-specific configuration

## Updated Files

### 1. [notebooks/synthetic_degradation_analysis.ipynb](notebooks/synthetic_degradation_analysis.ipynb)

**METHODS_TO_TEST** now uses **config dicts** (not strings):

```python
METHODS_TO_TEST = [
    # Rule-based with different weights
    {'name': 'RuleBased_Sharpness', 'type': 'rule_based', 'weights': {...}},
    {'name': 'RuleBased_Exposure', 'type': 'rule_based', 'weights': {...}},
    # ...

    # CLIP methods
    {'name': 'CLIP_Sharpness', 'type': 'clip_sharpness', 'model_name': 'ViT-B-32', ...},
    # ...

    # ViT
    {'name': 'ViT_Base', 'type': 'vit', 'model_name': 'google/vit-base-patch16-224', ...},
]
```

**Assessment cell** now uses config dicts directly:

```python
for method_config in METHODS_TO_TEST:
    method_name = method_config['name']
    assessor = create_quality_assessor(method_config)
    # ...
```

### 2. [examples/test_degradations.py](examples/test_degradations.py)

**create_method_configs()** now returns exact pairwise benchmark format:

```python
all_methods_configs = [
    {'name': 'RuleBased_Sharpness', 'type': 'rule_based', 'weights': {...}},
    {'name': 'CLIP_Sharpness', 'type': 'clip_sharpness', ...},
    # ... all 15 methods
]
```

## Verification

### Method Names Match Pairwise Benchmark

| Degradation Test | Pairwise Benchmark | Registry Type |
|------------------|-------------------|---------------|
| `RuleBased_Sharpness` | `RuleBased_Sharpness` | `rule_based` |
| `RuleBased_Exposure` | `RuleBased_Exposure` | `rule_based` |
| `CLIP_Sharpness` | `CLIP_Sharpness` | `clip_sharpness` |
| `CLIP_Exposure` | `CLIP_Exposure` | `clip_exposure` |
| `CLIP_Color` | `CLIP_Color` | `clip_color` |
| `ViT_Base` | `ViT_Base` | `vit` |

✅ **100% match** - No discrepancies!

### Configuration Format Matches

**Pairwise Benchmark Config**:
```yaml
- name: "CLIP_Sharpness"
  type: "clip_sharpness"
  model_name: "ViT-B-32"
  pretrained: "laion2b_s34b_b79k"
  device: "cpu"
  enable_cache: true
```

**Degradation Test Config** (now):
```python
{'name': 'CLIP_Sharpness',
 'type': 'clip_sharpness',
 'model_name': 'ViT-B-32',
 'pretrained': 'laion2b_s34b_b79k',
 'device': 'cpu',
 'enable_cache': True}
```

✅ **Exact same structure!**

## All 15 Methods (Correct Names)

### Rule-Based (7)
1. `RuleBased_Sharpness`
2. `RuleBased_Exposure`
3. `RuleBased_Contrast`
4. `RuleBased_Colorfulness`
5. `RuleBased_Balanced`
6. `RuleBased_SharpnessFocused`
7. `RuleBased_ExposureFocused`

### CLIP Attributes (7)
8. `CLIP_AestheticOverall`
9. `CLIP_Composition`
10. `CLIP_SubjectPlacement`
11. `CLIP_Cropping`
12. `CLIP_Sharpness`
13. `CLIP_Exposure`
14. `CLIP_Color`

### Deep Learning (1)
15. `ViT_Base`

## Usage Examples

### CLI - Test All Methods
```bash
python examples/test_degradations.py --input path/to/images/ --methods all
```

### CLI - Test Specific Methods (Use Exact Names)
```bash
python examples/test_degradations.py \
  --input path/to/images/ \
  --methods RuleBased_Sharpness,CLIP_Sharpness,ViT_Base
```

### Notebook
Just run cells - `METHODS_TO_TEST` is pre-configured with all 15 methods.

## What Changed

### Before (WRONG ❌)
```python
METHODS_TO_TEST = [
    'rule_based_sharpness',  # ❌ Not registered!
    'clip_sharpness',        # ✓ OK (but inconsistent naming)
]

# Then had to parse strings and create configs manually
if name == 'rule_based_sharpness':
    config = {'type': 'rule_based', ...}  # Confusing!
```

### After (CORRECT ✅)
```python
METHODS_TO_TEST = [
    {'name': 'RuleBased_Sharpness', 'type': 'rule_based', 'weights': {...}},
    {'name': 'CLIP_Sharpness', 'type': 'clip_sharpness', ...},
]

# Pass config dict directly to create_quality_assessor()
assessor = create_quality_assessor(method_config)  # Clean!
```

## Benefits

1. ✅ **No discrepancies** - Uses exact same method names/configs as pairwise benchmark
2. ✅ **Same registry** - Calls same `QualityMethodRegistry.create()`
3. ✅ **Consistent results** - Same methods = same scores
4. ✅ **Easy comparison** - Can directly compare degradation test results with pairwise results
5. ✅ **No confusion** - Method names match everywhere (code, configs, outputs, docs)

## Guaranteed Consistency

The configuration format is now **identical** between:
- Pairwise benchmark YAML configs
- Degradation test Python configs
- Both use `create_quality_assessor(config)`
- Both call `QualityMethodRegistry.create(type, config)`

**Result**: Testing the same methods with the same code!
