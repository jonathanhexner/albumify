# PhotoTriage Training Set Mean Color

## What is this?

The `training_mean_color.json` file contains the mean RGB pixel color computed across all 11,716 unique images in the PhotoTriage training set.

This value is used for **aspect-ratio preserving preprocessing** as described in the PhotoTriage paper:

> "we resize each image so that its larger dimension is the required size, while maintaining the original aspect ratio and padding with the mean pixel color in the training set."

## File Contents

```json
{
  "mean_rgb_normalized": [0.460, 0.450, 0.430],  // For use in configs
  "mean_rgb_255": [117.3, 114.7, 109.6],         // For reference
  "num_images": 11716,
  "description": "Mean RGB color computed from PhotoTriage training set..."
}
```

- **mean_rgb_normalized**: RGB values in [0,1] range - use these in your training config
- **mean_rgb_255**: RGB values in [0,255] range - for reference only
- **num_images**: Number of training images used in computation

## How to Use

When training with paper-accurate preprocessing, pass these values via command line:

```bash
python train_multifeature_ranker.py \
    --use_paper_preprocessing true \
    --padding_mean_color 0.460 0.450 0.430 \
    --cnn_freeze_mode none \
    # ... other args
```

Or in a config file:

```python
config = MultiFeatureConfig(
    use_paper_preprocessing=True,
    padding_mean_color=[0.460, 0.450, 0.430],
    cnn_freeze_mode="none",
    # ...
)
```

## How to Regenerate

If this file doesn't exist or you want to recompute it:

```bash
python scripts/phototriage/compute_mean_color.py
```

This script:
1. Loads all unique image paths from `data/phototriage/pairs_train.jsonl`
2. Computes the mean RGB color across all pixels in all training images
3. Saves the result to `data/phototriage/training_mean_color.json`

**Note**: This computation takes 5-10 minutes as it processes ~12K images.

## Comparison with ImageNet Mean

- **ImageNet mean**: [0.485, 0.456, 0.406] (standard for pretrained models)
- **PhotoTriage mean**: [0.460, 0.450, 0.430] (computed from training set)

The PhotoTriage training set is slightly darker than ImageNet on average.

## Why Does This Matter?

Using the correct mean color for padding ensures:
1. **Paper replication**: Matches the exact preprocessing described in the paper
2. **Better performance**: The padding color matches the training set distribution
3. **No distortion**: Images maintain aspect ratio (not cropped/stretched)

When `use_paper_preprocessing=False` (default), the standard ImageNet preprocessing is used:
- Resize to 256px (shorter side)
- Center crop to 224x224
- This **crops** the image rather than padding it

## Reference

Chang, H., Yu, F., Wang, J., Ashley, D., & Finkelstein, A. (2016).
Automatic Triage for a Photo Series. ACM Transactions on Graphics, 35(6).
