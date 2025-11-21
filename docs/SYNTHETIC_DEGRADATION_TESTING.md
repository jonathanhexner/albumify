# Synthetic Image Degradation Testing

A system for validating quality assessment methods using controlled synthetic degradations.

## Overview

This module applies known degradations to test images and evaluates how quality assessment methods respond. It's designed as a **sanity check** to ensure methods detect obvious quality changes before using them on real data.

## Components

### 1. Degradation Module
**Location**: [`sim_bench/image_processing/degradation.py`](../sim_bench/image_processing/degradation.py)

**Class**: `ImageDegradationProcessor`

**Degradation Types**:
- **Gaussian Blur**: Tests sharpness detection (sigma: 0.5 to 8.0)
- **Exposure Adjustment**: Tests exposure/histogram quality (-3 to +3 stops)
- **JPEG Compression**: Tests artifact detection (quality: 10 to 95)

**Usage**:
```python
from sim_bench.image_processing import create_degradation_processor

processor = create_degradation_processor(output_dir="outputs/degraded")

# Apply single degradation
blurred = processor.apply_gaussian_blur("image.jpg", sigma=2.0)
darkened = processor.apply_exposure_adjustment("image.jpg", stops=-2)
compressed = processor.apply_jpeg_compression("image.jpg", quality=20)

# Apply complete suite
degraded_variants = processor.apply_degradation_suite(
    image_path="image.jpg",
    blur_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
    exposure_stops=[-3, -2, -1, 1, 2, 3],
    jpeg_qualities=[95, 80, 60, 40, 20, 10]
)
```

### 2. Command-Line Testing Script
**Location**: [`examples/test_degradations.py`](../examples/test_degradations.py)

Generates degraded variants and runs quality assessments.

**Usage**:
```bash
# Test with single image
python examples/test_degradations.py \
  --input path/to/image.jpg \
  --methods rule_based,clip_sharpness,clip_aesthetic_overall \
  --output-dir outputs/test_results

# Test with directory of images
python examples/test_degradations.py \
  --input path/to/images/ \
  --methods rule_based,clip_sharpness \
  --device cuda

# Customize degradation levels
python examples/test_degradations.py \
  --input image.jpg \
  --blur-sigmas 1.0,2.0,4.0 \
  --exposure-stops -2,-1,1,2 \
  --jpeg-qualities 80,60,40,20
```

**Arguments**:
- `--input`: Image path or directory (required)
- `--methods`: Comma-separated method list (default: rule_based,clip_sharpness,clip_aesthetic_overall)
- `--output-dir`: Output directory (default: auto-timestamped)
- `--degradations`: Types to test (blur, exposure, jpeg)
- `--blur-sigmas`: Blur levels (default: 0.5,1.0,2.0,4.0,8.0)
- `--exposure-stops`: Exposure adjustments (default: -3,-2,-1,1,2,3)
- `--jpeg-qualities`: JPEG quality levels (default: 95,80,60,40,20,10)
- `--device`: Device for deep learning methods (cpu or cuda)

**Output**:
```
outputs/degradation_test_{timestamp}/
├── original/
│   └── image.jpg
├── blur_sigma_2.0/
│   └── image.jpg
├── exposure_minus_2/
│   └── image.jpg
├── jpeg_q_20/
│   └── image.jpg
├── results.csv          # All quality scores
└── metadata.json        # Test configuration
```

### 3. Simple Test Script
**Location**: [`examples/test_degradations_simple.py`](../examples/test_degradations_simple.py)

Quick validation test with synthetic or provided images.

**Usage**:
```bash
# Create synthetic test image
python examples/test_degradations_simple.py

# Use your own image
python examples/test_degradations_simple.py path/to/image.jpg
```

**Output Example**:
```
Quality Scores:
--------------------------------------------------
Original                  : 0.6473
Blur (mild, sigma=2.0)    : 0.4863
Blur (severe, sigma=8.0)  : 0.4865
Exposure (-2 stops)       : 0.3407
JPEG (quality=20)         : 0.5446
```

### 4. Analysis Notebook
**Location**: [`notebooks/synthetic_degradation_analysis.ipynb`](../notebooks/synthetic_degradation_analysis.ipynb)

Interactive Jupyter notebook for comprehensive analysis.

**Sections**:
1. **Setup & Configuration**: Select images, methods, degradation levels
2. **Generate Degradations**: Apply suite and visualize examples
3. **Quality Assessment**: Run all methods on all variants
4. **Analysis & Visualization**:
   - Dose-response curves (score vs degradation level)
   - Method sensitivity comparison
   - Correlation analysis (do methods agree?)
   - Monotonicity check (scores decrease as degradation increases?)
5. **Sanity Check Summary**: Pass/fail assessment of each method

**Key Visualizations**:
- **Dose-Response Curves**: Show how scores change with degradation severity
- **Sensitivity Bar Charts**: Compare method responsiveness
- **Correlation Heatmap**: Identify method agreement/disagreement
- **Image Grids**: Visual comparison with scores overlaid

**Run Notebook**:
```bash
jupyter notebook notebooks/synthetic_degradation_analysis.ipynb
```

## Validation Criteria

### Expected Behavior (PASS)
- ✓ **Monotonic decrease**: Scores decline as degradation increases
- ✓ **Sufficient sensitivity**: Score delta > 0.1 for severe degradation
- ✓ **Ranking correctness**: Original > mild degradation > severe degradation

### Failure Modes (FAIL)
- ✗ **Insensitive**: No score change despite severe degradation
- ✗ **Non-monotonic**: Scores increase with degradation
- ✗ **Saturated**: Scores hit floor/ceiling too early

### Interpretation Example

```
Method: rule_based
  Blur sensitivity: 0.163 (Original: 0.647 → Worst: 0.484)
  Exposure sensitivity: 0.307 (Original: 0.647 → Worst: 0.340)
  JPEG sensitivity: 0.102 (Original: 0.647 → Worst: 0.545)

  Monotonic: 3/3 degradation types ✓
  Status: PASS - Reliable method
```

## Recommended Test Images

Use **3-5 diverse images** covering different photographic scenarios:

1. **Portrait**: Tests face/subject quality
2. **Landscape**: Tests composition/color
3. **High-detail macro**: Maximizes blur detectability
4. **Low-light scene**: Tests exposure handling
5. **Colorful image**: Tests color preservation

### Industry Standard Test Images

Common test images used in image processing research:
- **Kodak PhotoCD**: 24 high-quality images (kodim01-kodim24)
- **Lena**: Classic test image (512×512)
- **Baboon**: High-frequency detail test
- **Peppers**: Color fidelity test
- **Barbara**: Texture and fine detail test

These can be downloaded from:
- Kodak: http://r0k.us/graphics/kodak/
- USC-SIPI: http://sipi.usc.edu/database/

## Example Workflow

### Quick Sanity Check
```bash
# 1. Run simple test
python examples/test_degradations_simple.py

# 2. Check output
# Expected: Degraded images should have lower scores than original
```

### Comprehensive Analysis
```bash
# 1. Generate degradations and assess quality
python examples/test_degradations.py \
  --input path/to/test_images/ \
  --methods rule_based,clip_sharpness,clip_exposure,clip_aesthetic_overall \
  --output-dir outputs/comprehensive_test

# 2. Analyze results in notebook
jupyter notebook notebooks/synthetic_degradation_analysis.ipynb

# 3. Review sanity check summary:
#    - Which methods passed?
#    - Which degradations are detected?
#    - Are methods correlated?
```

## Design Constraints

The implementation follows strict coding guidelines:
- **No try/except blocks**: Errors propagate cleanly
- **Minimal if/else**: Use guard clauses and early returns
- **Modules < 300 lines**: Split functionality if needed
- **OpenCV only**: All image operations use cv2

## Next Steps After Validation

Once methods pass sanity checks:
1. **Interpret findings**: Which methods are reliable?
2. **Filter methods**: Exclude failed methods from benchmarks
3. **Tune parameters**: Adjust rule-based weights if needed
4. **Real-world validation**: Compare with PhotoTriage pairwise results
5. **Document baseline**: Use as reference for method selection

## Key Insights

### Why Synthetic Degradations?

Real-world datasets have confounding factors:
- Multiple quality dimensions vary simultaneously
- Ground truth may be noisy or subjective
- Hard to isolate what methods actually detect

Synthetic degradations provide:
- **Known ground truth**: We control what changed
- **Isolated variables**: Test one quality dimension at a time
- **Graduated severity**: Observe dose-response relationships
- **Reproducibility**: Same test conditions every time

### Limitations

Synthetic degradations are **simplified** versions of real-world quality issues:
- Blur is uniform (real blur may be motion-based, depth-based, etc.)
- Exposure is global (real photos have local exposure issues)
- JPEG artifacts are predictable (real compression varies by content)

**Therefore**: Passing synthetic tests is **necessary but not sufficient**. Always validate on real data afterward.

## Troubleshooting

### Issue: All scores are similar
**Cause**: Method insensitive to tested degradations
**Solution**: Try different method or more extreme degradations

### Issue: Scores increase with degradation
**Cause**: Method broken or detecting wrong features
**Solution**: Investigate method implementation or configuration

### Issue: JPEG degradation not detected
**Cause**: JPEG artifacts may be subtle at moderate quality levels
**Solution**: Test lower quality levels (10-40) or use different assessor

### Issue: Exposure adjustment has no effect
**Cause**: Image already at histogram extremes or method doesn't check exposure
**Solution**: Use well-exposed test images or different method

## Citation

If using this system in research, please cite the sim-bench repository and note:

```
Synthetic degradation testing methodology based on common practices in
image quality assessment research, including established degradations
from JPEG compression studies and blur kernel analysis.
```

## References

- Bradley & Terry (1952): Pairwise comparison model (used in quality benchmarking)
- Kodak PhotoCD: Standard test image dataset
- JPEG compression quality: ITU-T Recommendation T.81
- Gaussian blur: Standard image processing degradation
