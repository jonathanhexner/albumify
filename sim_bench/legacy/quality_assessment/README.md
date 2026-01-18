# Archived Quality Assessment System

**Archived:** January 19, 2026  
**Reason:** Replaced by unified image quality models interface

## What Was Archived

This directory contains the legacy quality assessment benchmark system including:
- **NIMA** (CNN-based methods) - `cnn_methods.py`
- **MUSIQ** (Transformer methods) - `musiq.py`, `transformer_methods.py`
- **CLIP Aesthetic** scorers - `clip_aesthetic.py`, `clip_attribute_methods.py`
- **Pairwise benchmark framework** - `pairwise_benchmark.py`, `pairwise_evaluator.py`
- **PhotoTriage trained classifiers** - `trained_models/` directory
- **Old benchmark infrastructure** - `benchmark.py`, `evaluator.py`, `registry.py`
- **Analysis tools** - `analysis/` directory
- **Visualization** - `visualization.py`
- **Bradley-Terry model** - `bradley_terry.py`

## Current System

The new unified system is located in:
- **Main interface**: `sim_bench/image_quality_models/` - Unified `BaseQualityModel` interface
  - `siamese_model_wrapper.py` - Siamese E2E (trained on PhotoTriage)
  - `ava_model_wrapper.py` - AVA ResNet (aesthetic scores)
  - `iqa_model_wrapper.py` - Rule-based IQA wrappers
  - `model_factory.py` - Factory for creating models from config
- **New benchmark**: `scripts/image_quality_utilities/test_model_degradations.py`
- **Unified results**: `outputs/benchmark_ava_vs_siamese/`

## Key Differences

### Old System (Archived)
- Separate evaluation framework for each method type
- Different APIs for different models (NIMA, MUSIQ, CLIP)
- Focused on ranking within series
- Complex registry system

### New System (Current)
- Unified `BaseQualityModel` interface
- All models implement `score_image()` and `compare_images()`
- Config-driven model creation via factory
- Degradation-based benchmarking
- Simpler, more focused architecture

## Migration Guide

If you need functionality from this archive:

1. **Port to new interface**: Implement `BaseQualityModel`
   ```python
   from sim_bench.image_quality_models.base_model import BaseQualityModel
   
   class MyModel(BaseQualityModel):
       def score_image(self, image_path: Path) -> float:
           # Your implementation
           pass
   ```

2. **Register in factory**: Add to `MODEL_REGISTRY` in `model_factory.py`
   ```python
   MODEL_REGISTRY = {
       'my_model': MyModel,
       # ...
   }
   ```

3. **Use in benchmark**: Add to config YAML
   ```yaml
   models:
     - type: my_model
       checkpoint: path/to/weights
       name: My Model
   ```

## Why Archived?

The new unified system provides:
- **Cleaner architecture**: Single interface for all models
- **Better testing**: Consistent degradation-based evaluation
- **Easier extension**: Factory pattern for new models
- **Production focus**: Tested on Siamese, AVA, and IQA baselines

The old system had grown complex with different evaluation paradigms for different model types. The new system consolidates around pairwise comparison with a unified interface.

## Original Documentation

See `README_OLD.md` in this directory for the original documentation.

## Need Help?

For questions or to restore functionality, contact the project maintainer or refer to:
- `sim_bench/image_quality_models/README.md` - New system documentation
- `outputs/benchmark_ava_vs_siamese/KEY_TAKEAWAYS.md` - Benchmark results
- `MILESTONES.md` - Project milestones
