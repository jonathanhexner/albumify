# sim-bench Project Summary

## What I Just Completed

Successfully implemented **learned CLIP aesthetic prompts** for quality assessment, including:

1. ✅ **Prompt learning script** - Analyzes 34,827 PhotoTriage user feedback reasons
2. ✅ **LearnedCLIPAestheticAssessor** - Loads prompts from YAML
3. ✅ **Factory integration** - `clip_learned` method available
4. ✅ **Detailed scores in benchmark** - Evaluator now captures per-prompt scores
5. ✅ **Regression training script** - Learn optimal prompt aggregation from data

## Project Overview

**sim-bench** is a comprehensive image analysis benchmarking framework with three main capabilities:

### 1. Image Similarity / Retrieval
Find visually similar images using various methods:
- **Classical**: Chi-Square, EMD (Wasserstein), SIFT BoVW
- **Deep Learning**: ResNet50, DINOv2, OpenCLIP
- **Datasets**: UKBench (10,200 images), INRIA Holidays (1,491 images), PhotoTriage (12,988 images)
- **Metrics**: mAP, Recall@k, Precision@k, N-S Score

**Best performer**: DINOv2 (mAP@10: 0.885 on Holidays, 0.958 on UKBench)

### 2. Clustering
Automatically group similar images:
- **Methods**: KMeans, DBSCAN, HDBSCAN
- **Features**: Color histograms or deep features (DINOv2, CLIP)
- **Output**: HTML gallery visualization
- **Datasets**: Budapest (310 images), PhotoTriage (12,988 images)

### 3. Quality Assessment ⭐ (Current Focus)
Select best photo from series/burst:
- **Rule-Based**: Sharpness, exposure, contrast, colorfulness, noise
- **CNN**: NIMA (MobileNetV2, ResNet50)
- **Transformer**: ViT-Base
- **CLIP Aesthetic** (NEW): Text-prompt based quality assessment
  - **Hardcoded prompts**: 4 pairs + 2 pos + 2 neg attributes
  - **Learned prompts** (NEW): 9 pairs learned from PhotoTriage user feedback

**Best performer**: Sharpness-only (64.95% Top-1 accuracy on PhotoTriage)

## Current Quality Assessment Results

From latest benchmark (`benchmark_2025-11-16_00-30-54`):

| Method | Top-1 Acc | Top-2 Acc | MRR | Time (ms) |
|--------|-----------|-----------|-----|-----------|
| **sharpness_only** | **64.95%** | 77.47% | 0.581 | 55.3 |
| contrast_only | 48.37% | 83.17% | 0.660 | 46.7 |
| exposure_only | 42.99% | 85.12% | 0.685 | 47.0 |
| **nima_mobilenet** | 42.43% | 84.57% | 0.682 | 34.6 |
| colorfulness_only | 41.56% | 84.95% | 0.675 | 46.8 |
| **clip_aesthetic** | 41.42% | 83.83% | 0.675 | 76.2 |
| nima_resnet50 | 41.40% | 83.79% | 0.675 | 57.3 |
| vit_base | 41.20% | 84.81% | 0.676 | 159.7 |

**Key findings**:
- Sharpness dominates quality assessment (65% accuracy)
- Deep learning methods cluster around 41-42% accuracy
- CLIP aesthetic comparable to NIMA and ViT
- Speed: NIMA MobileNet fastest (34ms), ViT slowest (160ms)

## Architecture & Design Patterns

### Factory Pattern
All major components use factory pattern for extensibility:
- **Methods**: `load_method()` in `sim_bench/methods/base.py`
- **Datasets**: `load_dataset()` in `sim_bench/datasets/base.py`
- **Metrics**: `MetricFactory` in `sim_bench/metrics/factory.py`
- **Quality Assessors**: `load_quality_method()` in `sim_bench/quality_assessment/factory.py`

### Module Organization
```
sim-bench/
├── sim_bench/
│   ├── methods/              # Image retrieval methods
│   ├── datasets/             # Dataset loaders
│   ├── metrics/              # Evaluation metrics
│   ├── quality_assessment/   # Photo quality methods ⭐
│   │   ├── base.py           # Abstract base class
│   │   ├── factory.py        # Factory (EMPTY __init__.py)
│   │   ├── rule_based.py     # Sharpness, exposure, etc.
│   │   ├── cnn_methods.py    # NIMA
│   │   ├── transformer_methods.py  # ViT, MUSIQ
│   │   ├── clip_aesthetic.py # CLIP-based (hardcoded + learned)
│   │   ├── evaluator.py      # Benchmark evaluator
│   │   ├── benchmark.py      # Benchmark runner
│   │   └── analysis.py       # Results analysis
│   ├── clustering/           # Clustering methods
│   ├── vision_language/      # CLIP, BLIP, etc.
│   ├── image_processing/     # Thumbnails, preprocessing
│   └── photo_analysis/       # CLIP tagging, metadata
├── configs/                  # YAML configuration files
│   ├── quality_benchmark.*.yaml
│   ├── learned_aesthetic_prompts.yaml ⭐ (NEW)
│   └── global_config.yaml
├── scripts/
│   ├── learn_prompts_from_phototriage.py ⭐ (NEW)
│   └── train_clip_aggregation_model.py   ⭐ (NEW)
├── examples/
│   ├── clip_aesthetic_detailed_demo.py
│   └── compare_clip_variants.py ⭐ (NEW)
├── notebooks/
│   └── quality_assessment_analysis.ipynb
└── docs/
    ├── quality_assessment/
    ├── clustering/
    └── image_similarity/
```

### Key Design Principles
1. **Empty `__init__.py`** files (only docstring + imports, NO logic)
2. **Separate factory.py** files for factories
3. **Logging over prints** (except user-facing demos)
4. **Minimal if statements** - Use design patterns (Factory, Strategy, Singleton)
5. **Global config** for paths, devices, etc.

## Configuration Files

### Quality Benchmarks
- `quality_benchmark.deep_learning.yaml` - CNN + Transformer + CLIP methods
- `quality_benchmark.comprehensive.yaml` - All methods
- `quality_benchmark.phototriage.yaml` - PhotoTriage-specific
- `quality_benchmark.rule_based_only.yaml` - Fast rule-based methods

### Datasets
- `dataset.phototriage.yaml` - 12,988 images, 4,986 groups
- `dataset.budapest.yaml` - 310 images for clustering
- `dataset.ukbench.yaml` - Image retrieval
- `dataset.holidays.yaml` - Image retrieval

### Global
- `global_config.yaml` - Output folders, device, logging
- `learned_aesthetic_prompts.yaml` ⭐ (NEW) - 9 learned contrastive pairs

## Recent Work: Learned CLIP Prompts

### Problem
CLIP aesthetic assessor used hardcoded prompts:
- Only 4 contrastive pairs (composition, placement, cropping, quality)
- Didn't align with how real users evaluate photos
- Limited coverage of quality dimensions

### Solution
Learn prompts from PhotoTriage user feedback:
1. **Extract** 34,827 user reasons from JSON reviews
2. **Analyze** keywords ("blurry": 2,745, "too dark": 2,423)
3. **Categorize** into 6 themes (focus, composition, exposure, color, content, view)
4. **Generate** 9 contrastive pairs reflecting user language
5. **Save** to `configs/learned_aesthetic_prompts.yaml`

### Results
**Learned prompts** (9 pairs = 18 prompts):
- Focus/Sharpness: "sharp and well-focused" vs "blurry and out-of-focus"
- Composition: "well-composed and uncluttered" vs "cluttered and poorly-composed"
- Framing: "good framing" vs "bad framing or too cropped"
- Exposure/Lighting: "well-exposed with good lighting" vs "dark or poorly-lit"
- Color/Clarity: "good color and clarity" vs "bad color or hazy appearance"
- Content/Interest: "interesting with clear subject" vs "boring that lacks subject"
- View/Perspective: "good field of view" vs "too narrow or limited view"
- Detail Visibility: "shows important details clearly" vs "can't see details well"
- Overall Quality: "high-quality photograph" vs "low-quality photograph"

**Hardcoded prompts** (4 pairs + 2 pos + 2 neg = 12 prompts):
- Focused on composition, placement, cropping
- General photography principles

### New Capabilities

#### 1. LearnedCLIPAestheticAssessor
```python
from sim_bench.quality_assessment.clip_aesthetic import LearnedCLIPAestheticAssessor

assessor = LearnedCLIPAestheticAssessor(
    prompts_file="configs/learned_aesthetic_prompts.yaml",
    model_name="ViT-B-32",
    device="cpu"
)
score = assessor.assess_image("photo.jpg")
```

#### 2. Detailed Scores in Benchmark
Benchmark now automatically captures per-prompt scores:
```json
{
  "detailed_scores": [
    {
      "contrast_0_a sharp and well-focused photograph": 0.1234,
      "pos_0_a sharp and well-focused photograph": 0.6789,
      "neg_0_a blurry and out-of-focus photograph": 0.5555,
      ...
    }
  ]
}
```

#### 3. Regression Model Training
Learn optimal aggregation from data instead of hardcoded weights:
```bash
python scripts/train_clip_aggregation_model.py
```

This trains Ridge/Lasso/ElasticNet/RandomForest/GradientBoosting models to learn:
- Which aesthetic dimensions are most predictive
- Optimal weights for combining prompt scores
- Better aggregation than simple mean/weighted average

## Next Steps

### Immediate Tasks
1. **Run benchmark with learned prompts**:
   ```bash
   # Add to configs/quality_benchmark.deep_learning.yaml:
   - name: clip_learned
     type: clip_learned
     config:
       prompts_file: configs/learned_aesthetic_prompts.yaml
       model_name: ViT-B-32
       device: cpu

   # Run benchmark
   python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml
   ```

2. **Train regression model**:
   ```bash
   # After running benchmark with detailed scores
   python scripts/train_clip_aggregation_model.py
   ```

3. **Compare hardcoded vs learned vs regression**:
   - Accuracy comparison
   - Per-prompt score analysis
   - Feature importance from regression model

### Research Questions
1. **Does learning from user feedback improve accuracy?**
   - Hypothesis: Learned prompts should outperform hardcoded
   - Test: Compare `clip_aesthetic` vs `clip_learned` on PhotoTriage

2. **Can we learn better aggregation?**
   - Hypothesis: Regression model learns optimal weights
   - Test: Compare `weighted` vs regression-based aggregation

3. **Which quality dimensions matter most?**
   - Analyze regression coefficients
   - Compare to sharpness-only (64.95% accuracy)
   - Understand why sharpness dominates

4. **Cross-dataset generalization?**
   - Train on PhotoTriage, test on other datasets
   - Do learned prompts transfer?

### Future Enhancements
1. **Add more CLIP models**:
   - ViT-L-14 (larger, more accurate)
   - ViT-H-14 (huge, best quality)

2. **Ensemble methods**:
   - Combine sharpness + CLIP learned
   - Weighted voting across methods

3. **Active learning**:
   - Iteratively refine prompts
   - Bootstrap from initial learned prompts

4. **Multi-modal approach**:
   - CLIP + traditional metrics
   - Learn which to trust for each image

## How to Use

### Run Quality Benchmark
```bash
# Full benchmark (all methods)
python run_quality_benchmark.py configs/quality_benchmark.comprehensive.yaml

# Deep learning only
python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml

# Quick test (subset)
python run_quality_benchmark.py configs/quality_benchmark.quick_test.yaml
```

### Analyze Results
```bash
# Open Jupyter notebook
jupyter notebook notebooks/quality_assessment_analysis.ipynb

# Or use analysis scripts
python -m sim_bench.analysis.comparison_viz
```

### Compare CLIP Variants
```bash
# Show prompt differences
python examples/compare_clip_variants.py

# Get detailed scores for images
python examples/clip_aesthetic_detailed_demo.py
```

### Learn Prompts (Already Done)
```bash
# Re-run if you want to modify categorization
python scripts/learn_prompts_from_phototriage.py
```

## Key Files to Know

### Configuration
- `configs/quality_benchmark.deep_learning.yaml` - Main benchmark config
- `configs/learned_aesthetic_prompts.yaml` - Learned prompts
- `configs/global_config.yaml` - Global settings

### Core Code
- `sim_bench/quality_assessment/clip_aesthetic.py` - CLIP assessors (both variants)
- `sim_bench/quality_assessment/factory.py` - Factory for loading methods
- `sim_bench/quality_assessment/evaluator.py` - Benchmark evaluation logic
- `sim_bench/quality_assessment/benchmark.py` - Benchmark runner

### Scripts
- `scripts/learn_prompts_from_phototriage.py` - Prompt learning
- `scripts/train_clip_aggregation_model.py` - Regression training
- `run_quality_benchmark.py` - Main benchmark entry point

### Analysis
- `notebooks/quality_assessment_analysis.ipynb` - Interactive analysis
- `sim_bench/quality_assessment/analysis.py` - Analysis functions

### Documentation
- `LEARNED_CLIP_PROMPTS_SUMMARY.md` - Complete documentation of learned prompts
- `README_QUALITY_BENCHMARK.md` - Quality benchmark guide
- `docs/quality_assessment/` - Quality assessment docs

## Dependencies

### Core (required)
- numpy, scipy, scikit-learn
- opencv-contrib-python, Pillow
- pandas, PyYAML, tqdm

### Deep Learning (for CLIP, NIMA, ViT)
- torch, torchvision
- open-clip-torch
- transformers

### Analysis (for notebooks)
- jupyter, matplotlib, seaborn
- nbconvert (for HTML export)

## Performance Notes

### Feature Caching
All methods cache features to `artifacts/<method>/` for 10-300x speedup on repeated runs.

### Speed Comparison
- **Rule-based**: 45-55ms/image (CPU)
- **NIMA MobileNet**: 35ms/image (CPU)
- **NIMA ResNet50**: 57ms/image (CPU)
- **CLIP Aesthetic**: 76ms/image (CPU)
- **ViT-Base**: 160ms/image (CPU)

### Memory
- Full PhotoTriage (12,988 images): ~2-4GB RAM
- DINOv2 features: ~15MB per 1000 images
- CLIP features: ~10MB per 1000 images

## Git Status

Modified files from recent work:
- `sim_bench/quality_assessment/evaluator.py` - Added detailed scores capture
- `sim_bench/quality_assessment/clip_aesthetic.py` - Added LearnedCLIPAestheticAssessor
- `sim_bench/quality_assessment/factory.py` - Added 'clip_learned' method
- `.vscode/launch.json` - Added debug configs

New files:
- `configs/learned_aesthetic_prompts.yaml`
- `scripts/learn_prompts_from_phototriage.py`
- `scripts/train_clip_aggregation_model.py`
- `examples/compare_clip_variants.py`
- `tests/quality_assessment/test_learned_prompts_only.py`
- `LEARNED_CLIP_PROMPTS_SUMMARY.md`

## Contact & Support

For questions or issues:
- Check documentation in `docs/quality_assessment/`
- Review examples in `examples/`
- Open issues on GitHub (if applicable)

---

**Status**: Ready for experimentation with learned CLIP prompts and regression-based aggregation.

**Last Updated**: 2025-11-16 (After implementing learned CLIP prompts)
