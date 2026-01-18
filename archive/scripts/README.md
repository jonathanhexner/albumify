# Archived Scripts

**Archived:** January 19, 2026

This directory contains scripts from the legacy quality assessment benchmark system.

## Archived Files

- `run_quality_benchmark.py` - Old benchmark runner (replaced by `scripts/image_quality_utilities/test_model_degradations.py`)
- `visualize_quality_benchmark.py` - Old visualization tools

## Current System

Use the new unified benchmark system:

```bash
# Run benchmark
python scripts/image_quality_utilities/test_model_degradations.py \
    --config configs/image_quality_benchmarks/degradation_test.yaml

# View results
cat outputs/benchmark_ava_vs_siamese/KEY_TAKEAWAYS.md

# Open notebook for analysis
jupyter notebook notebooks/siamese_e2e/explore_model_behavior.ipynb
```

## Why Archived?

These scripts were part of the old quality assessment framework that used different evaluation paradigms for different model types (NIMA, MUSIQ, CLIP). The new system provides a unified interface with consistent degradation-based testing.

See `sim_bench/legacy/quality_assessment/README.md` for more information.
