# Understanding Quality Assessment Evaluation

A comprehensive guide to how PhotoTriage evaluation works, with concrete examples.

## The Basic Setup

### What is a Photo Series?

When you take multiple photos in quick succession (burst mode), you get a series of similar images. PhotoTriage contains 4,986 such series.

Example Series:
```
Series 000002: 8 photos of the same moment
- 000002-01.JPG
- 000002-02.JPG
- 000002-03.JPG
- 000002-04.JPG
- 000002-05.JPG
- 000002-06.JPG
- 000002-07.JPG
- 000002-08.JPG
```

### The Ground Truth

For EACH series, humans have labeled which photo is the BEST:
- Series might have 2 images, 3 images, or up to 8 images
- ONE image in each series is marked as "best" by humans
- This is your ground truth

### The Task

Your algorithm must:
1. Look at ALL images in a series
2. Assign a quality score to EACH image
3. Rank them from best to worst
4. We check: Did you rank the human-labeled "best" image at position 1?

This is NOT pairwise comparison - you rank ALL images in the series simultaneously.

## Concrete Example: Series 000002

Let's walk through a complete example with 8 images.

### Step 1: The Series

```
Series ID: 000002
Number of images: 8
Human-labeled best: 000002-04.JPG (let's assume for this example)
```

### Step 2: Your Algorithm Scores Each Image

Let's say your algorithm uses sharpness to score quality:

```
Image            | Sharpness Score | Quality Score
-----------------|-----------------|---------------
000002-01.JPG    | 145.2          | 0.725
000002-02.JPG    | 98.3           | 0.492
000002-03.JPG    | 187.6          | 0.938  <- Highest score (your pick)
000002-04.JPG    | 156.1          | 0.781  <- Human pick
000002-05.JPG    | 112.4          | 0.562
000002-06.JPG    | 134.8          | 0.674
000002-07.JPG    | 89.2           | 0.446
000002-08.JPG    | 101.5          | 0.508
```

### Step 3: Rank Images by Score

Your algorithm produces this ranking:

```
Rank | Image         | Score | Is Human Pick?
-----|---------------|-------|---------------
  1  | 000002-03.JPG | 0.938 | NO
  2  | 000002-04.JPG | 0.781 | YES  <- Human pick at rank 2
  3  | 000002-01.JPG | 0.725 | NO
  4  | 000002-06.JPG | 0.674 | NO
  5  | 000002-05.JPG | 0.562 | NO
  6  | 000002-08.JPG | 0.508 | NO
  7  | 000002-02.JPG | 0.492 | NO
  8  | 000002-07.JPG | 0.446 | NO
```

### Step 4: Evaluate Performance

For this ONE series, we calculate:

Top-1 Accuracy: Is best image at rank 1?
- NO (it's at rank 2)
- Score: 0

Top-2 Accuracy: Is best image in top 2?
- YES (it's at rank 2)
- Score: 1

Top-3 Accuracy: Is best image in top 3?
- YES (it's at rank 2)
- Score: 1

Mean Reciprocal Rank (MRR):
- Formula: 1 / rank_of_best_image
- Calculation: 1 / 2 = 0.5

Mean Rank:
- Just the rank: 2

## Understanding the Metrics

### Top-1 Accuracy (Most Important)

What it measures: Percentage of series where your 1 pick matches the human's pick

Formula: (Number of series where best image is ranked 1) / (Total number of series)

Example over 5 series:
```
Series | Human Best | Your Rank | Top-1 Hit?
-------|------------|-----------|------------
000001 | 01.JPG     | Rank 1    | YES
000002 | 04.JPG     | Rank 2    | NO
000003 | 02.JPG     | Rank 1    | YES
000004 | 03.JPG     | Rank 3    | NO
000005 | 01.JPG     | Rank 1    | YES

Top-1 Accuracy: 3/5 = 0.60 = 60%
```

Interpretation:
- 60% = Your algorithm correctly picks the best photo 60% of the time
- 100% = Perfect (always pick human's choice)
- Random guessing with 3 images = 33.3%

### Top-2 Accuracy

What it measures: Percentage of series where the human's pick is in your top 2

Example over same 5 series:
```
Series | Human Best | Your Rank | Top-2 Hit?
-------|------------|-----------|------------
000001 | 01.JPG     | Rank 1    | YES
000002 | 04.JPG     | Rank 2    | YES
000003 | 02.JPG     | Rank 1    | YES
000004 | 03.JPG     | Rank 3    | NO
000005 | 01.JPG     | Rank 1    | YES

Top-2 Accuracy: 4/5 = 0.80 = 80%
```

Why it matters: Shows how "close" you are when you don't get 1 exactly right

### Top-3 Accuracy

Same as Top-2 but with top 3 positions.

### Mean Reciprocal Rank (MRR)

What it measures: Average of 1/rank across all series

Formula for one series: 1 / rank_of_best_image

Example over 5 series:
```
Series | Human Best | Your Rank | Reciprocal Rank
-------|------------|-----------|----------------
000001 | 01.JPG     | Rank 1    | 1/1 = 1.000
000002 | 04.JPG     | Rank 2    | 1/2 = 0.500
000003 | 02.JPG     | Rank 1    | 1/1 = 1.000
000004 | 03.JPG     | Rank 3    | 1/3 = 0.333
000005 | 01.JPG     | Rank 1    | 1/1 = 1.000

MRR: (1.000 + 0.500 + 1.000 + 0.333 + 1.000) / 5 = 0.767
```

Interpretation:
- 1.0 = Perfect (always rank best image 1)
- 0.5 = On average, best image is ranked 2
- 0.333 = On average, best image is ranked 3

### Mean Rank

Simply the average position of the best image.

Example:
```
Series | Rank of Best Image
-------|-------------------
000001 | 1
000002 | 2
000003 | 1
000004 | 3
000005 | 1

Mean Rank: (1 + 2 + 1 + 3 + 1) / 5 = 1.6
```

Lower is better (1.0 = perfect).

## How Dataset Size Affects Metrics

### Series with Different Sizes

PhotoTriage has variable series sizes:
- Some series: 2 images
- Some series: 4 images
- Some series: 8 images (maximum)
- Average: 2.6 images

### Impact on Difficulty

Series with 2 images:
- Random guessing: 50% chance
- Easier to get right

Series with 8 images:
- Random guessing: 12.5% chance
- Much harder to get right

### Why This is Fair

All series are weighted equally regardless of size:
- Each series contributes 1 to the total count
- Series with 2 images: counts as 1
- Series with 8 images: counts as 1

This means your 64.95% accuracy is an average across easy (2-image) and hard (8-image) series.

## Our Benchmark Results Explained

### What We Tested

We tested 8 different quality assessment methods on 100 randomly sampled series from PhotoTriage.

### Sharpness-Only Method (Winner: 64.95%)

What it does:
```python
for each image in series:
    sharpness_score = calculate_laplacian_variance(image)

rank images by sharpness_score (highest = best)
```

Results:
- Top-1 Accuracy: 64.95%
- Top-2 Accuracy: 77.57%
- MRR: 0.789

Interpretation:
- In 64.95 out of 100 series, the sharpest photo was the one humans picked
- In 77.57 out of 100 series, the human's pick was in the top 2 sharpest
- On average, the human's pick was at rank 1.27 (very close to 1)

### Contrast-Only Method (48.37%)

What it does:
```python
for each image in series:
    contrast_score = calculate_rms_contrast(image)

rank images by contrast_score (highest = best)
```

Results:
- Top-1 Accuracy: 48.37%
- Top-2 Accuracy: 83.23%
- MRR: 0.702

Interpretation:
- Only 48% of the time the highest contrast image was the best
- BUT 83% of the time the best image had top-2 contrast
- This suggests: contrast helps, but it's not the PRIMARY factor

### Composite Method (42.05%)

What it does:
```python
for each image in series:
    sharpness = calculate_sharpness(image)
    exposure = calculate_exposure(image)
    colorfulness = calculate_colorfulness(image)
    contrast = calculate_contrast(image)
    
    combined_score = (0.35 * sharpness + 
                      0.25 * exposure + 
                      0.20 * colorfulness + 
                      0.15 * contrast)

rank images by combined_score
```

Results:
- Top-1 Accuracy: 42.05%

Surprising Finding:
Combining multiple factors made it WORSE than just using sharpness alone!

Why?
- Sharpness is what people care about most
- Adding other factors "diluted" the sharpness signal
- Other factors (color, exposure) may be uncorrelated or negatively correlated with human preference

## Complete Example: Evaluating a Method

Let's walk through evaluating a method on 3 series:

### The Dataset

```
Series 000001 (4 images):
- 000001-01.JPG
- 000001-02.JPG (BEST - human labeled)
- 000001-03.JPG
- 000001-04.JPG

Series 000002 (8 images):
- 000002-01.JPG
- 000002-02.JPG
- 000002-03.JPG
- 000002-04.JPG (BEST - human labeled)
- 000002-05.JPG
- 000002-06.JPG
- 000002-07.JPG
- 000002-08.JPG

Series 000010 (2 images):
- 000010-01.JPG (BEST - human labeled)
- 000010-02.JPG
```

### Your Algorithm Runs

Series 000001 - Your ranking:
```
1. 000001-03.JPG (score: 0.89)
2. 000001-02.JPG (score: 0.85) <- Human pick at rank 2
3. 000001-01.JPG (score: 0.72)
4. 000001-04.JPG (score: 0.61)

Top-1: 0 (wrong)
Top-2: 1 (correct)
Top-3: 1 (correct)
Reciprocal Rank: 1/2 = 0.5
```

Series 000002 - Your ranking:
```
1. 000002-04.JPG (score: 0.94) <- Human pick at rank 1!
2. 000002-03.JPG (score: 0.91)
3. 000002-01.JPG (score: 0.88)
4. 000002-06.JPG (score: 0.82)
5. 000002-05.JPG (score: 0.79)
6. 000002-08.JPG (score: 0.73)
7. 000002-02.JPG (score: 0.68)
8. 000002-07.JPG (score: 0.65)

Top-1: 1 (correct!)
Top-2: 1 (correct)
Top-3: 1 (correct)
Reciprocal Rank: 1/1 = 1.0
```

Series 000010 - Your ranking:
```
1. 000010-01.JPG (score: 0.87) <- Human pick at rank 1!
2. 000010-02.JPG (score: 0.79)

Top-1: 1 (correct!)
Top-2: 1 (correct)
Top-3: 1 (correct)
Reciprocal Rank: 1/1 = 1.0
```

### Final Metrics

Top-1 Accuracy:
- Series with correct pick: 2 (Series 000002, 000010)
- Total series: 3
- Top-1 Accuracy: 2/3 = 66.67%

Top-2 Accuracy:
- Series with best image in top 2: 3 (all of them)
- Total series: 3
- Top-2 Accuracy: 3/3 = 100%

Top-3 Accuracy:
- Same as Top-2: 100%

Mean Reciprocal Rank:
- Series 000001: 0.5
- Series 000002: 1.0
- Series 000010: 1.0
- MRR: (0.5 + 1.0 + 1.0) / 3 = 0.833

Mean Rank:
- Series 000001: rank 2
- Series 000002: rank 1
- Series 000010: rank 1
- Mean Rank: (2 + 1 + 1) / 3 = 1.333

## Key Takeaways

1. Not Pairwise: You rank ALL images in each series, not compare pairs
2. Variable Sizes: Series have 1-8 images (average 2.6)
3. One Ground Truth: Each series has exactly ONE human-labeled best image
4. Top-1 is Key: The most important metric - did you pick the same image as humans?
5. Sharpness Wins: On PhotoTriage, sharpness alone beats complex composite methods
6. People Prefer Sharp Photos: This is the main insight from the dataset

## How to Use This Dataset

### For Development
```bash
# Use sampling for quick testing (20 series)
python run_quality_benchmark.py configs/quality_benchmark.quick.yaml
```

### For Final Evaluation
```bash
# Full dataset (4,986 series)
python run_quality_benchmark.py configs/quality_benchmark.phototriage.yaml
```

### For Custom Methods
```python
from sim_bench.quality_assessment import QualityEvaluator
from sim_bench.datasets import load_dataset

# Load dataset
dataset = load_dataset('phototriage', config)
dataset.load_data()

# Your custom method
class MyQualityMethod:
    def assess_image(self, image_path):
        # Return quality score
        return score

# Evaluate
evaluator = QualityEvaluator(dataset, MyQualityMethod())
results = evaluator.evaluate()

print(f"Top-1 Accuracy: {results['metrics']['top1_accuracy']}")
```

## Related Documentation

- Quality Assessment Quickstart: quickstart.md
- Benchmark Guide: benchmark.md
- PhotoTriage Dataset: ../image_similarity/datasets_phototriage.md




