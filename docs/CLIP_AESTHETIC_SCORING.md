# How CLIP Aesthetic Scoring Works

## Overview

CLIP aesthetic assessment doesn't use a single "good vs bad" prompt. Instead, it:
1. Uses **multiple text prompts** (9-18 prompts depending on variant)
2. Computes **similarity scores** between the image and each prompt
3. **Aggregates** all similarities into a single quality score

## Step-by-Step Process

### Step 1: Encode Image and Prompts

For each image (e.g., `image_a.jpg`):
- CLIP encodes the image → `image_embedding` (512-dim vector)
- CLIP encodes all text prompts → `prompt_embeddings` (array of 512-dim vectors)

### Step 2: Compute Similarities

For each prompt, compute cosine similarity:
```
similarity = cosine_similarity(image_embedding, prompt_embedding)
```

This gives a similarity score (typically -1 to 1) for each prompt.

### Step 3: Aggregate into Single Score

The similarities are aggregated using one of three methods:

#### Method 1: Contrastive Only
For each contrastive pair (positive, negative):
```
contrastive_score = similarity(positive) - similarity(negative)
```

Then average all contrastive scores:
```
final_score = mean([contrastive_score_1, contrastive_score_2, ...])
```

#### Method 2: Weighted (Default)
Combines three components:

1. **Contrastive component** (50% weight):
   - For each pair: `pos_sim - neg_sim`
   - Average all pairs

2. **Positive component** (30% weight):
   - Average similarities to positive attributes
   - Higher = better

3. **Negative component** (20% weight):
   - Average similarities to negative attributes
   - Inverted (multiply by -1): lower similarity = better
   - So: `-mean(negative_similarities)`

Final formula:
```
final_score = 0.5 * contrastive + 0.3 * positive + 0.2 * negative
```

#### Method 3: Mean
Simple average of all similarities.

## Example: Learned Variant (9 Pairs)

For `image_a.jpg`:

1. **Encode image**: `image_a_embedding = [0.1, 0.2, ..., 0.5]` (512 dims)

2. **Encode 18 prompts** (9 pairs × 2):
   - Pair 1: "sharp and well-focused" → embedding_1
   - Pair 1: "blurry and out-of-focus" → embedding_2
   - Pair 2: "well-composed and uncluttered" → embedding_3
   - Pair 2: "cluttered and poorly-composed" → embedding_4
   - ... (9 pairs total = 18 prompts)

3. **Compute 18 similarities**:
   ```
   sim_1 = cosine(image_a_embedding, embedding_1) = 0.65  # "sharp"
   sim_2 = cosine(image_a_embedding, embedding_2) = 0.20  # "blurry"
   sim_3 = cosine(image_a_embedding, embedding_3) = 0.70  # "well-composed"
   sim_4 = cosine(image_a_embedding, embedding_4) = 0.15  # "cluttered"
   ... (14 more similarities)
   ```

4. **Aggregate** (weighted method):
   ```
   # Contrastive component (9 pairs)
   contrast_1 = sim_1 - sim_2 = 0.65 - 0.20 = 0.45
   contrast_2 = sim_3 - sim_4 = 0.70 - 0.15 = 0.55
   ... (7 more contrasts)
   contrastive = mean([0.45, 0.55, ...]) = 0.50
   
   # No positive/negative attributes in learned variant
   positive = 0
   negative = 0
   
   # Final score
   score_a = 0.5 * 0.50 + 0.3 * 0 + 0.2 * 0 = 0.25
   ```

5. **Repeat for image_b**:
   ```
   score_b = 0.15  # (example)
   ```

6. **Compare**:
   - `score_a = 0.25` > `score_b = 0.15`
   - Prediction: Image A is better quality

## Why Multiple Prompts?

Using multiple prompts captures different aspects of quality:
- **Sharpness**: "sharp" vs "blurry"
- **Composition**: "well-composed" vs "cluttered"
- **Framing**: "good framing" vs "bad framing"
- **Exposure**: "well-exposed" vs "dark"
- **Color**: "good color" vs "bad color"
- etc.

A single "good vs bad" prompt would be too vague. Multiple specific prompts give a more nuanced assessment.

## Visual Summary

```
Image A
  ↓
[CLIP Encoder]
  ↓
Image Embedding (512-dim)
  ↓
    ├─→ Similarity with "sharp" prompt → 0.65
    ├─→ Similarity with "blurry" prompt → 0.20
    ├─→ Similarity with "well-composed" prompt → 0.70
    ├─→ Similarity with "cluttered" prompt → 0.15
    └─→ ... (14 more similarities)
  ↓
[Aggregation]
  ↓
Final Score: 0.25
```

## Key Points

1. **One image → one score**: Despite using 9-18 prompts, we get a single quality score per image
2. **Contrastive pairs**: Each pair gives a relative score (positive - negative)
3. **Aggregation**: All prompt similarities are combined into one number
4. **Comparison**: For pairwise tasks, we compare `score_a` vs `score_b`

