# CLIP Aesthetic Assessment Experiment

## Quick Start

This experiment tests whether CLIP text-prompt similarity can assess image aesthetic quality.

### 1. Run the Test Script

```bash
# Basic test on sample images
python test_clip_aesthetic.py
```

This will:
- ✅ Load OpenCLIP model (ViT-B-32)
- ✅ Assess sample images using aesthetic prompts
- ✅ Compare with rule-based methods (sharpness, contrast)
- ✅ Show correlations and rankings
- ✅ Test different aggregation methods

### 2. Quick Demo (Single Image)

```bash
# Run the built-in demo
python -m sim_bench.quality_assessment.clip_aesthetic
```

### 3. Integrate into Benchmark

If results are promising, add to your quality benchmark config:

```yaml
# configs/quality_benchmark.clip_test.yaml
dataset:
  name: "phototriage"
  root: "D:/PhotoTriage"
  # ... (your existing config)

methods:
  # Existing methods
  - name: "sharpness"
    class: "SharpnessAssessor"

  - name: "nima"
    class: "NIMAAssessor"
    params:
      model_name: "mobilenet"

  # NEW: CLIP aesthetic
  - name: "clip_aesthetic"
    class: "CLIPAestheticAssessor"
    params:
      model_name: "ViT-B-32"
      pretrained: "laion2b_s34b_b79k"
      device: "cuda"  # or "cpu"
      aggregation_method: "weighted"  # or "contrastive_only", "mean"

output:
  dir: "outputs/quality_benchmarks/clip_test"
```

Then run:

```bash
python run_quality_benchmark.py configs/quality_benchmark.clip_test.yaml
```

## Expected Results

### Test 1: Sample Images

```
RESULTS SUMMARY
================================================================================
            image  clip_aesthetic  sharpness  contrast
    ukbench00001         0.1234      0.872      0.654
    ukbench00002         0.0987      0.765      0.543
    ...

CORRELATIONS
================================================================================
                clip_aesthetic  sharpness  contrast
clip_aesthetic            1.00       0.35      0.28
sharpness                 0.35       1.00      0.45
contrast                  0.28       0.45      1.00
```

**What to look for**:
- ✅ Positive correlation with sharpness/contrast (0.2-0.5)
- ✅ Well-composed images score higher
- ✅ Blurry/dark images score lower

### Test 2: PhotoTriage Burst

```
BEST IMAGE SELECTION
================================================================================
Best by CLIP Aesthetic: IMG_1234.jpg
  CLIP Score: 0.1567
  Composite Score: 0.8234

Best by Composite Quality: IMG_1234.jpg
  CLIP Score: 0.1567
  Composite Score: 0.8234

Methods Agree: YES
```

**What to look for**:
- Agreement with composite quality method
- Detailed scores explain *why* image was selected

### Test 3: Aggregation Methods

```
AGGREGATION METHOD COMPARISON
================================================================================
            image  contrastive_only  weighted    mean
    ukbench00001              0.12      0.15    0.13
    ukbench00002              0.08      0.11    0.09
    ...

CORRELATIONS BETWEEN AGGREGATION METHODS
================================================================================
                  contrastive_only  weighted  mean
contrastive_only             1.000     0.923  0.887
weighted                     0.923     1.000  0.967
mean                         0.887     0.967  1.000
```

**What to look for**:
- High correlation (0.8-1.0) between methods
- Suggests prompts are capturing consistent signal
- Choose method based on interpretability/performance

## Understanding the Scores

### Score Ranges

**Contrastive pairs**: [-2, +2]
- Positive: Image matches positive prompt more
- Negative: Image matches negative prompt more
- Near zero: Ambiguous

**Overall score**: [-1, +1]
- Higher = Better aesthetic quality
- Typical range: [-0.2, +0.3]
- Absolute values less important than relative ranking

### Detailed Score Interpretation

Example detailed scores:

```python
{
  'contrast_a well-composed photograph': 0.15,  # High = good composition
  'pos_a well-composed photograph': 0.28,       # Raw positive similarity
  'neg_a poorly composed photograph': 0.13,     # Raw negative similarity

  'pos_a photo with leading lines': 0.22,       # Present
  'pos_a photo with symmetry': 0.18,            # Somewhat present
  'pos_rule of thirds composition': 0.25,       # Strong signal

  'neg_a cluttered composition': 0.12,          # Low = not cluttered (good!)
  'neg_a flat composition': 0.15,               # Low = has depth (good!)

  'overall': 0.1456                              # Aggregated final score
}
```

## Troubleshooting

### Issue: "open_clip module not found"

```bash
pip install open-clip-torch
```

### Issue: "Model download fails"

Check internet connection. Models are downloaded from HuggingFace:
- ViT-B-32 (laion2b_s34b_b79k): ~350MB
- First run only, then cached

### Issue: "CUDA out of memory"

Use CPU instead:

```python
assessor = CLIPAestheticAssessor(device='cpu')
```

Or use smaller model:

```python
assessor = CLIPAestheticAssessor(
    model_name='ViT-B-32',  # Smaller than ViT-L-14
    device='cuda'
)
```

### Issue: "Scores all very similar"

This could indicate:
1. **Prompts not discriminative**: Images are genuinely similar
2. **Prompt ambiguity**: Prompts capture content, not quality
3. **Need more diverse test set**: Test on more varied images

**Solutions**:
- Try different prompts (edit `CONTRASTIVE_PAIRS` in `clip_aesthetic.py`)
- Test on more diverse images
- Check detailed scores to see which prompts activate

## Customizing Prompts

Edit `sim_bench/quality_assessment/clip_aesthetic.py`:

```python
class CLIPAestheticAssessor(QualityAssessor):
    # Add your own prompts here!
    CONTRASTIVE_PAIRS = [
        ("a well-composed photograph", "a poorly composed photograph"),
        ("excellent framing", "poor framing"),
        # Add more...
    ]

    POSITIVE_ATTRIBUTES = [
        "professional photography",
        "aesthetically pleasing",
        # Add more...
    ]

    NEGATIVE_ATTRIBUTES = [
        "amateur snapshot",
        "badly lit",
        # Add more...
    ]
```

## Next Steps

### If Results are Good (correlation >0.4)

1. ✅ Add to benchmark suite
2. ✅ Test on full PhotoTriage dataset
3. ✅ Compare with NIMA/MUSIQ
4. ✅ Consider ensemble methods
5. ✅ Write paper/blog post!

### If Results are Mixed (correlation 0.2-0.4)

1. 🔧 Iterate on prompt engineering
2. 🔧 Test different CLIP models (ViT-L-14, larger)
3. 🔧 Try learned aggregation weights
4. 🔧 Fine-tune on AVA dataset
5. 📊 Use for specific aesthetic dimensions only

### If Results are Poor (correlation <0.2)

1. 📝 Document negative result (still valuable!)
2. 🔬 Analyze failure cases
3. 🤔 Consider fundamental limitations
4. 🎓 Compare with literature (CLIP-IQA, LIQE)
5. 💡 Pivot to other approaches

## Further Reading

### Papers

- **CLIP-IQA** (2022): CLIP for image quality assessment
- **LIQE** (2023): Language-guided quality estimation
- **LAION-Aesthetics** (2022): CLIP-based aesthetic scoring

### Documentation

- [`docs/quality_assessment/clip_aesthetic_analysis.md`](docs/quality_assessment/clip_aesthetic_analysis.md) - Full analysis
- [`sim_bench/quality_assessment/clip_aesthetic.py`](sim_bench/quality_assessment/clip_aesthetic.py) - Implementation
- [`sim_bench/quality_assessment/README.md`](sim_bench/quality_assessment/README.md) - Module overview

## Questions?

This is an experimental feature. Results will vary based on:
- Dataset composition (portraits, landscapes, etc.)
- Prompt choice and phrasing
- CLIP model variant
- What "quality" means in your context

**Recommended**: Start with the test script, examine correlations, iterate!

---

**Author**: Claude Code
**Date**: 2025-11-14
**Status**: Experimental
