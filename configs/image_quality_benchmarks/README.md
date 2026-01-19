# Image Quality Benchmarks

Configuration files for benchmarking image quality assessment models.

## Quick Start

Run the degradation benchmark:

```bash
python scripts/image_quality_utilities/test_model_degradations.py \
    --config configs/image_quality_benchmarks/degradation_test.yaml
```

## Configuration Structure

### Image Sources

- `phototriage`: Use PhotoTriage validation images (for comparing with Siamese model)
- `ava`: Use AVA validation images (from trained AVA model predictions)

### Model Types

Supported models:
- `siamese`: Siamese E2E pairwise ranking model
- `ava`: AVA aesthetic scoring model (1-10 scale)
- `rule_based_iqa`: Combined hand-crafted features (sharpness, exposure, colorfulness, contrast)
- `sharpness_iqa`: Sharpness only
- `exposure_iqa`: Exposure only
- `colorfulness_iqa`: Colorfulness only
- `contrast_iqa`: Contrast only

### Degradation Types

- **Blur**: Gaussian blur with varying sigma
- **JPEG**: Compression artifacts at different quality levels
- **Exposure**: Over/under exposure
- **Crop Edge**: Aggressive crop from one side
- **Crop Corner**: Shifted crop to corner
- **Crop Aspect**: Tall/wide aspect distortion
- **Crop Center**: Remove center subject area

## Output Files

Benchmark produces:
- `unified_results.csv`: All model predictions with degradation metadata
- `summary.json`: Accuracy statistics by model and degradation type
- `degradations_metadata.csv`: Degradation specifications
- `degraded_images/`: Generated degraded images

## Example Results

The benchmark compares how different models detect quality degradations:

- **Siamese**: Strong on pairwise ranking, trained on human preferences
- **AVA**: Measures aesthetic appeal, may be less sensitive to technical artifacts
- **IQA**: Direct technical quality metrics (blur, exposure, etc.)

Use this to understand model strengths/weaknesses across degradation types.
