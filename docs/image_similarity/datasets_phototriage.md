# PhotoTriage Dataset

## Overview

PhotoTriage is a dataset of photo bursts - when you take multiple photos in quick succession. This dataset contains 4,986 photo bursts from real users, with a total of 12,988 photos. Each burst has a human-labeled best photo.

## Dataset Statistics

- **Location**: D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs
- **Total images**: 12,988
- **Total bursts**: 4,986
- **Average burst size**: 2.6 photos (range: 1-8)
- **Structure**: All files in single directory with naming pattern GGGGGG-II.JPG
- **Ground truth**: YES - each burst has one photo marked as best by humans

## Understanding Photo Bursts

When you take photos by holding the camera button, you get a burst of similar photos:
- Same scene
- Same moment in time
- Slightly different poses, expressions, or angles

Problem: Which photo is the BEST?

PhotoTriage provides human judgments so you can train algorithms to automatically pick the best photo.

## File Naming Pattern

All files follow: GGGGGG-II.JPG

Where:
- GGGGGG = Burst ID (000001, 000002, etc.)
- II = Photo number in burst (01, 02, 03, etc.)

Examples:
```
000001-01.JPG  (Burst 1, Photo 1)
000001-02.JPG  (Burst 1, Photo 2)
000001-03.JPG  (Burst 1, Photo 3)
000002-01.JPG  (Burst 2, Photo 1)
000002-02.JPG  (Burst 2, Photo 2)
```

## Two Ways to Use PhotoTriage

### Use Case 1: Group Similar Photos (Image Similarity)

Test if your algorithm can identify which photos belong to the same burst.

Config: `configs/run.phototriage_example.yaml`

```yaml
experiment:
  name: phototriage_similarity

dataset: phototriage
method: sift_bovw

sampling:
  max_groups: 100

metrics:
  - recall@1
  - recall@4
  - map

output_dir: outputs/phototriage_runs/
```

### Use Case 2: Pick Best Photo (Quality Assessment)

Test if your algorithm can identify which photo in a burst is the best.

Config: `configs/quality_benchmark.phototriage.yaml`

```yaml
datasets:
  - name: phototriage
    config: configs/dataset.phototriage.yaml
    sampling:
      strategy: random
      num_series: 100
      seed: 42

methods:
  - name: sharpness_only
    type: rule_based
    config:
      weights:
        sharpness: 1.0
```

## Usage Examples

### Test Image Similarity (Grouping)

```bash
# Run similarity search to test photo grouping
python -m sim_bench.cli --run-config configs/run.phototriage_example.yaml
```

This tests if the algorithm can correctly identify that 000001-01.JPG and 000001-02.JPG belong together.

### Test Quality Assessment (Pick Best)

```bash
# Run quality assessment to test best photo selection
python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml
```

This tests if the algorithm can correctly identify which photo in each burst is the best one.

## Our Benchmark Results (Quality Assessment)

We tested 8 different methods to automatically pick the best photo:

**Best Method: Sharpness Only**
- Top-1 Accuracy: 64.95%
- What it does: Picks the sharpest (most in-focus) photo
- Result: Correctly picks best photo 65% of the time

**Second Best: Contrast Only**
- Top-1 Accuracy: 48.37%
- What it does: Picks photo with best contrast
- Result: Good but not as good as sharpness

**Composite Methods**
- Top-1 Accuracy: 42%
- What they do: Combine sharpness, exposure, color, contrast
- Result: Surprisingly WORSE than just using sharpness alone

### Key Finding

People strongly prefer SHARP photos. When choosing the best photo from a burst, being in-focus matters way more than brightness, color, or other factors.

This makes sense: A blurry photo is bad even with perfect lighting. A sharp photo is usually acceptable even if lighting is not perfect.

## Real World Example

Imagine you took 5 photos of your dog:

1. 000123-01.JPG - Dog is blinking
2. 000123-02.JPG - Dog is looking away
3. 000123-03.JPG - PERFECT! Dog looking at camera, sharp, good lighting (BEST)
4. 000123-04.JPG - Photo is blurry
5. 000123-05.JPG - Dog mid-movement

PhotoTriage has 4,986 scenarios like this where humans picked which photo is best.

## Why PhotoTriage is Useful

1. LARGE SCALE: 12,988 images is a lot of real-world data
2. REAL USERS: Actual photo bursts from real people
3. GROUND TRUTH: Humans labeled the best photo
4. DUAL PURPOSE: Can test both grouping AND quality picking
5. VARIABLE SIZES: Bursts range from 1-8 photos (average 2.6)

## Comparison with Other Datasets

**UKBench:**
- 10,200 images in groups of exactly 4
- Photos of objects from different angles
- Good for testing object recognition

**Holidays:**
- 1,491 vacation/holiday photos
- Groups of 2-20+ images
- Good for testing scene recognition

**PhotoTriage:**
- 12,988 images in groups of 1-8 (average 2.6)
- Real photo bursts from users
- Good for photo selection and burst grouping

## Summary in Simple Terms

PhotoTriage = When you take 5 selfies and need to pick which one to post

- You have multiple similar photos
- One is marked as best by humans
- You can test algorithms that either:
  1. Group similar photos together
  2. Pick the best photo automatically





