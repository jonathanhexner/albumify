# Quality Assessment Module (Minimal)

**Status**: Minimal maintenance mode  
**Legacy code**: Archived in `sim_bench/legacy/quality_assessment/`  
**New system**: `sim_bench/image_quality_models/`

## Current Contents

This module now contains only the essential components used by the unified image quality models system:

- `base.py` - `QualityAssessor` base class
- `rule_based.py` - Rule-based IQA (sharpness, exposure, colorfulness, contrast)
- `__init__.py` - Minimal exports

## For New Development

**Use the unified interface instead:**

```python
from sim_bench.image_quality_models import create_model

# Create a rule-based IQA model
model = create_model({
    'type': 'rule_based_iqa',
    'name': 'IQA',
    'device': 'cpu'
})

# Score an image
score = model.score_image(Path('image.jpg'))

# Compare two images
result = model.compare_images(Path('img1.jpg'), Path('img2.jpg'))
```

## Available Models

The new system supports:
- **Siamese E2E**: Trained on PhotoTriage (pairwise ranking)
- **AVA ResNet**: Aesthetic score prediction (1-10 scale)
- **Rule-Based IQA**: Hand-crafted features (from this module)

See `sim_bench/image_quality_models/README.md` for complete documentation.

## Running Benchmarks

```bash
# New unified benchmark
python scripts/image_quality_utilities/test_model_degradations.py \
    --config configs/image_quality_benchmarks/degradation_test.yaml

# Results
ls outputs/benchmark_ava_vs_siamese/
```

## Legacy System

The old benchmark system (NIMA, MUSIQ, CLIP, pairwise evaluators) has been archived to:
- `sim_bench/legacy/quality_assessment/`
- `archive/scripts/` (old runner scripts)

See `sim_bench/legacy/quality_assessment/README.md` for migration guidance.

## Architecture

```
quality_assessment/          (minimal, kept for IQA)
├── __init__.py             (exports QualityAssessor, RuleBasedQuality)
├── base.py                 (QualityAssessor base class)
└── rule_based.py           (rule-based IQA implementation)

image_quality_models/        (new unified system)
├── base_model.py           (BaseQualityModel interface)
├── siamese_model_wrapper.py
├── ava_model_wrapper.py
├── iqa_model_wrapper.py    (uses rule_based.py)
└── model_factory.py

legacy/quality_assessment/   (archived)
└── [old benchmark code]
```

## Why This Change?

The old system had grown complex with multiple evaluation paradigms. The new unified system:
- Single interface for all models
- Consistent pairwise comparison API
- Config-driven model creation
- Degradation-based testing
- Easier to extend and maintain

## Questions?

See:
- `sim_bench/image_quality_models/README.md` - New system docs
- `sim_bench/legacy/quality_assessment/README.md` - Archive info
- `outputs/benchmark_ava_vs_siamese/KEY_TAKEAWAYS.md` - Benchmark results
