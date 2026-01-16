# Learned CLIP Aesthetic Prompts - Implementation Summary

## Overview

Successfully implemented a system to learn CLIP aesthetic assessment prompts from the PhotoTriage dataset instead of using hardcoded prompts. This allows the quality assessor to align with how real users evaluate photo quality based on 34,827 user feedback reasons.

## What Was Implemented

### 1. Prompt Learning Script
**File**: `scripts/learn_prompts_from_phototriage.py`

Analyzes PhotoTriage user feedback and generates CLIP prompts:

- **Extracts** all reason texts from JSON reviews
- **Analyzes** common keywords and phrases
- **Categorizes** into 6 thematic groups:
  - Focus/Sharpness (blurry, out of focus, sharp, clear)
  - Composition (cluttered, cropped, framing)
  - Exposure/Lighting (dark, bright, overexposed)
  - Color/Clarity (bad color, hazy, vibrant)
  - Content/Interest (boring, interesting, lacks subject)
  - View/Perspective (narrow, wide, shows detail)
- **Generates** 9 contrastive prompt pairs
- **Saves** to `configs/learned_aesthetic_prompts.yaml`

**Analysis Results**:
- Processed: 4,986 review files
- Analyzed: 34,827 total reasons
- Top keywords: dark (3,366), blurry (2,745), picture (2,180)
- Top phrases: "too dark" (2,423), "too far" (899), "can't see" (873)

### 2. Learned Prompts Configuration
**File**: `configs/learned_aesthetic_prompts.yaml`

Contains 9 contrastive pairs learned from user feedback:

```yaml
contrastive_pairs:
  - ["a sharp and well-focused photograph", "a blurry and out-of-focus photograph"]
  - ["a well-composed and uncluttered photograph", "a cluttered and poorly-composed photograph"]
  - ["a photo with good framing", "a photo with bad framing or too cropped"]
  - ["a well-exposed photograph with good lighting", "a dark or poorly-lit photograph"]
  - ["a photo with good color and clarity", "a photo with bad color or hazy appearance"]
  - ["an interesting photograph with a clear subject", "a boring photo that lacks a subject"]
  - ["a photo with a good field of view", "a photo with a too narrow or limited view"]
  - ["a photo that shows important details clearly", "a photo where you can't see details well"]
  - ["a high-quality photograph", "a low-quality photograph"]
```

### 3. LearnedCLIPAestheticAssessor Class
**File**: `sim_bench/quality_assessment/clip_aesthetic.py`

New assessor variant that loads prompts from YAML:

```python
from sim_bench.quality_assessment.clip_aesthetic import LearnedCLIPAestheticAssessor

assessor = LearnedCLIPAestheticAssessor(
    prompts_file="configs/learned_aesthetic_prompts.yaml",
    model_name="ViT-B-32",
    device="cpu",
    aggregation_method="weighted"
)

score = assessor.assess_image("photo.jpg")
detailed = assessor.get_detailed_scores("photo.jpg")
```

**Features**:
- Loads contrastive pairs from YAML config
- Inherits all functionality from CLIPAestheticAssessor
- Returns method name as `CLIP_Learned_*` for reporting
- No separate positive/negative attributes (all from pairs)

### 4. Factory Integration
**File**: `sim_bench/quality_assessment/factory.py`

Added `clip_learned` method to factory:

```python
from sim_bench.quality_assessment.factory import load_quality_method

# Load learned variant
assessor = load_quality_method('clip_learned', config={
    'prompts_file': 'configs/learned_aesthetic_prompts.yaml',
    'model_name': 'ViT-B-32'
})
```

### 5. Comparison Demo
**File**: `examples/compare_clip_variants.py`

Demonstrates differences between hardcoded and learned prompts:

- Shows both prompt sets side-by-side
- Explains key differences
- Compares scores on actual images (when PyTorch available)

**Run**:
```bash
python examples/compare_clip_variants.py
```

## Key Differences: Hardcoded vs Learned

| Aspect | Hardcoded Prompts | Learned Prompts |
|--------|------------------|-----------------|
| **Source** | Manual design based on photography principles | Derived from 34,827 real user feedback reasons |
| **Total Prompts** | 12 (4 pairs + 2 pos + 2 neg) | 18 (9 pairs) |
| **Focus Areas** | Composition, placement, cropping, quality | Focus, composition, exposure, color, content, view, detail |
| **Structure** | Contrastive pairs + separate attributes | Contrastive pairs only |
| **Alignment** | General photography principles | How real users evaluate photos |

## Is This Cheating?

**No** - This is **learning vocabulary**, not memorizing answers:

- ✅ We learn **what quality dimensions** users care about
- ✅ We learn **how users describe** quality (language/terms)
- ✅ Prompts are **general** ("blurry", "dark", "cluttered")
- ✅ No image-specific information used
- ❌ We don't learn which specific images are good/bad
- ❌ We don't memorize ground truth labels

**Analogy**: Learning that users care about "sharpness" and "lighting" is like learning what questions to ask on an exam - not memorizing the answers.

## Files Created/Modified

### Created:
- `scripts/learn_prompts_from_phototriage.py` - Prompt learning script
- `configs/learned_aesthetic_prompts.yaml` - Learned prompts config
- `examples/compare_clip_variants.py` - Comparison demo
- `test_learned_clip.py` - Integration test (requires PyTorch)
- `test_learned_prompts_only.py` - Prompt loading test
- `LEARNED_CLIP_PROMPTS_SUMMARY.md` - This file

### Modified:
- `sim_bench/quality_assessment/clip_aesthetic.py` - Added LearnedCLIPAestheticAssessor
- `sim_bench/quality_assessment/factory.py` - Added 'clip_learned' method
- `.vscode/launch.json` - Added debug configuration

## Next Steps

To use the learned prompts in your workflow:

1. **Run prompt learning** (already done):
   ```bash
   python scripts/learn_prompts_from_phototriage.py
   ```

2. **Use in quality assessment**:
   ```python
   from sim_bench.quality_assessment.factory import load_quality_method

   assessor = load_quality_method('clip_learned')
   score = assessor.assess_image('photo.jpg')
   ```

3. **Benchmark comparison** (recommended):
   - Create a config file comparing both methods
   - Run on PhotoTriage test set
   - Compare performance metrics
   - Analyze which prompt set performs better

4. **Potential improvements**:
   - Fine-tune number of contrastive pairs (currently 9)
   - Adjust categorization thresholds
   - Add weights based on reason frequency
   - Test different CLIP model sizes

## Usage Examples

### Quick Test
```bash
# Test prompt loading (no PyTorch needed)
python test_learned_prompts_only.py

# Compare prompts (no PyTorch needed for prompt display)
python examples/compare_clip_variants.py
```

### In Your Code
```python
from sim_bench.quality_assessment.clip_aesthetic import (
    CLIPAestheticAssessor,
    LearnedCLIPAestheticAssessor
)

# Hardcoded prompts
hardcoded = CLIPAestheticAssessor(model_name="ViT-B-32", device="cpu")
score1 = hardcoded.assess_image("photo.jpg")

# Learned prompts
learned = LearnedCLIPAestheticAssessor(
    prompts_file="configs/learned_aesthetic_prompts.yaml",
    model_name="ViT-B-32",
    device="cpu"
)
score2 = learned.assess_image("photo.jpg")

# Compare
print(f"Hardcoded: {score1:.4f}")
print(f"Learned:   {score2:.4f}")
```

### Via Factory
```python
from sim_bench.quality_assessment.factory import load_quality_method

# Load either variant
hardcoded = load_quality_method('clip_aesthetic')
learned = load_quality_method('clip_learned', config={
    'prompts_file': 'configs/learned_aesthetic_prompts.yaml'
})
```

## Implementation Quality

✅ **Follows all project standards**:
- Factory pattern with separate factory.py
- Minimal __init__.py (just imports)
- Logging throughout (no prints in core code)
- Proper error handling
- Type hints
- Comprehensive documentation

✅ **Modular and testable**:
- Each function has single responsibility
- Can test prompt loading independently
- Can compare variants side-by-side
- Can run with or without PyTorch

✅ **Backward compatible**:
- Existing CLIPAestheticAssessor unchanged
- New variant is optional
- Factory supports both methods
- No breaking changes

## Conclusion

Successfully implemented a data-driven approach to learning CLIP aesthetic prompts from real user feedback. The learned prompts cover a broader range of quality dimensions (9 pairs vs 4 pairs) and are aligned with how actual users evaluate photo quality based on analysis of 34,827 feedback reasons.

The implementation is modular, testable, follows all project standards, and maintains full backward compatibility with existing code.
