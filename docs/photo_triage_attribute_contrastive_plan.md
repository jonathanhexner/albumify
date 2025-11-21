# PhotoTriage Attribute-Based Contrastive Learning

**Status**: 🚧 In Development
**Last Updated**: 2025-01-19
**Version**: 1.0

## Table of Contents
1. [Overview](#overview)
2. [Attribute Schema](#attribute-schema)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Data Formats](#data-formats)
5. [Model Architecture](#model-architecture)
6. [Training Strategy](#training-strategy)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Implementation Status](#implementation-status)

---

## Overview

### Goal
Build an attribute-aware photo selection system that:
1. Learns from **PhotoTriage user feedback** (34,827+ pairwise comparisons with textual reasons)
2. Trains a **CLIP-based contrastive model** with per-attribute heads
3. Enables **fine-grained photo selection** (e.g., "find the sharpest", "best composed", "most interesting")
4. Outperforms baseline methods (current best: sharpness-only at 64.95% top-1 accuracy)

### Approach
- **Weakly supervised learning**: Extract attribute labels from free-form user reasons
- **Contrastive training**: Learn from pairwise preferences rather than absolute scores
- **Multi-task learning**: Global preference + per-attribute preferences
- **Zero-shot backbone**: Leverage pre-trained CLIP vision encoder

### Why This Matters
Current quality assessment methods either:
- Use hand-crafted features (sharpness, exposure) - limited expressiveness
- Train on generic aesthetic datasets (AVA, KonIQ) - don't match triage context
- Use hardcoded text prompts - don't align with real user preferences

Our approach learns **data-driven attributes** from actual user feedback in the triage context.

---

## Attribute Schema

### Design Principles
1. **Coverage**: Span the range of reasons users actually give
2. **Orthogonality**: Minimize overlap between attributes
3. **Actionability**: Attributes should be understandable and useful for filtering
4. **Measurability**: Attributes should be visually verifiable in images

### Initial Attribute Set (14 attributes)

Based on analysis of 34,827 user reasons, grouped into 6 categories:

#### 1. Focus & Clarity (3 attributes)
| Attribute | Description | Example Reasons |
|-----------|-------------|-----------------|
| `sharpness` | Overall image sharpness and focus quality | "blurry", "out of focus", "sharp", "clear" |
| `detail_visibility` | Ability to see important details | "can't see details", "shows details well", "hazy" |
| `motion_blur` | Presence/absence of motion blur | "motion blur", "blurry movement", "crisp" |

#### 2. Composition & Framing (4 attributes)
| Attribute | Description | Example Reasons |
|-----------|-------------|-----------------|
| `framing` | Quality of image framing and boundaries | "good framing", "bad framing", "well-framed" |
| `cropping_completeness` | Whether subject is cut off or complete | "cropped", "cut off", "shows full subject" |
| `subject_placement` | Positioning of main subject in frame | "not centered", "good positioning", "off-center" |
| `background_clutter` | Amount of distracting background elements | "cluttered", "clean background", "messy" |

#### 3. Exposure & Lighting (3 attributes)
| Attribute | Description | Example Reasons |
|-----------|-------------|-----------------|
| `exposure_quality` | Overall brightness appropriateness | "too dark", "too bright", "well-exposed" |
| `lighting_quality` | Quality and direction of lighting | "good lighting", "harsh shadows", "flat lighting" |
| `dynamic_range` | Preservation of detail in highlights/shadows | "washed out", "crushed blacks", "good contrast" |

#### 4. Perspective & Field of View (2 attributes)
| Attribute | Description | Example Reasons |
|-----------|-------------|-----------------|
| `field_of_view` | Width of view (narrow vs. wide) | "too narrow", "shows more", "limited view" |
| `distance_appropriateness` | Subject distance suitability | "too far", "too close", "good distance" |

#### 5. Content & Interest (1 attribute)
| Attribute | Description | Example Reasons |
|-----------|-------------|-----------------|
| `subject_interest` | How interesting/engaging the subject is | "boring", "interesting", "lacks subject" |

#### 6. Overall Quality (1 attribute)
| Attribute | Description | Example Reasons |
|-----------|-------------|-----------------|
| `global_preference` | Overall aesthetic/triage preference | "better overall", "nicer", "preferred" |

### Attribute Refinement Process

**Phase 1** (Initial): 14 attributes as listed above
**Phase 2** (After data analysis): May merge, split, or adjust based on:
- Attribute frequency in labeled data
- Inter-attribute correlations
- Model performance on per-attribute tasks
- User feedback on attribute meaningfulness

**Merge candidates**:
- `motion_blur` → `sharpness` (if rarely distinguished)
- `lighting_quality` + `dynamic_range` → combined lighting metric

**Split candidates**:
- `subject_interest` → `subject_clarity` + `moment_quality`

---

## Pipeline Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   PhotoTriage Dataset                         │
│  4,986 series × ~7.4 comparisons/series = 37K+ pairs         │
│  Each pair: {image_a, image_b, chosen, reason_text}          │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│           Step 1: Reason Text Analysis                        │
│  • Extract 34,827 reason texts from review JSONs             │
│  • Normalize and clean text                                  │
│  • Cluster similar reasons                                   │
│  • Identify common patterns and keywords                     │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│           Step 2: Attribute Mapping                           │
│  • Map each reason → 1+ attributes from schema               │
│  • Determine polarity (which image wins for each attribute)  │
│  • Assign confidence scores (high/medium/low)                │
│  • Method: Keyword matching + NLU + LLM assistance           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│           Step 3: Attribute Dataset Creation                  │
│  Output: phototriage_attribute_pairs.jsonl                   │
│  Format: {pair_id, images, chosen, reason, attributes[]}     │
│  ~37K pairs × ~1.5 attributes/pair = ~55K attribute labels   │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│           Step 4: Model Training                              │
│  • CLIP image encoder (frozen or lightly fine-tuned)         │
│  • Global head + 14 attribute heads                          │
│  • Pairwise ranking loss per head                            │
│  • Train/val/test split by series_id (80/10/10)              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│           Step 5: Evaluation & Analysis                       │
│  • Global: Pairwise accuracy, top-1 series accuracy, MRR     │
│  • Per-attribute: Accuracy on labeled pairs                  │
│  • Comparison to baselines (sharpness, NIMA, MUSIQ)          │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│           Step 6: Inference API                               │
│  selector.select_best(images, criterion='sharpness')         │
│  selector.filter_by(images, 'exposure_quality', threshold)   │
│  selector.score_all(images) → per-attribute scores           │
└──────────────────────────────────────────────────────────────┘
```

### Component Dependencies

```
phototriage/
├── attribute_mapper.py      (Step 2)
│   └── Requires: reason texts
│
├── data_loader.py           (Step 3)
│   └── Requires: review JSONs, pair lists, attribute_mapper
│
models/
├── clip_heads.py            (Step 4)
│   └── Requires: CLIP encoder, attribute schema
│
├── attribute_contrastive.py (Step 4)
│   └── Requires: clip_heads, data_loader
│
training/
├── pairwise_loss.py         (Step 4)
│   └── Requires: PyTorch
│
├── contrastive_trainer.py   (Step 4)
│   └── Requires: attribute_contrastive, pairwise_loss, data_loader
│
evaluation/
├── pairwise_accuracy.py     (Step 5)
│   └── Requires: trained model, test data
│
phototriage/
└── selector.py              (Step 6)
    └── Requires: trained model, attribute schema
```

---

## Data Formats

### Input: PhotoTriage Review JSON

**Location**: `D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval\*.json`

**Structure**:
```json
{
  "reviews": [
    {
      "compareID1": 0,
      "compareID2": 2,
      "compareFile1": "1-1.JPG",
      "compareFile2": "1-3.JPG",
      "userChoice": "RIGHT",
      "reason": ["", "too narrow view; doesn't show enough background"]
    }
  ]
}
```

**Notes**:
- `compareID` is 0-based index within series
- `reason[1]` contains the actual text (reason[0] is always empty)
- `userChoice` is "LEFT" or "RIGHT"
- Multiple reviews per series (average 7.4)

### Input: PhotoTriage Pair List

**Location**: `D:\Similar Images\automatic_triage_photo_series\train_val\train_pairlist.txt`

**Format**:
```
SERIES_ID PHOTO1_INDEX PHOTO2_INDEX PREFERENCE_RATIO RANK_PHOTO1 RANK_PHOTO2
1 1 2 0.949 1 4
1 1 3 0.636 1 2
```

**Notes**:
- Photo indices are 1-based (not 0-based!)
- PREFERENCE_RATIO from Bradley-Terry model (0.0-1.0)
- RANK: 1=best photo in series

### Output: Attribute-Labeled Pairs Dataset

**Location**: `data/phototriage_attribute_pairs.jsonl`

**Format** (one JSON object per line):
```json
{
  "pair_id": "000001_0_2",
  "series_id": "000001",
  "image_a_id": "000001-01",
  "image_b_id": "000001-03",
  "image_a_path": "D:/Similar Images/.../000001-01.JPG",
  "image_b_path": "D:/Similar Images/.../000001-03.JPG",
  "chosen_image": "B",
  "reason_raw": "too narrow view; doesn't show enough background",
  "preference_strength": 0.636,
  "rank_a": 1,
  "rank_b": 2,
  "attributes": [
    {
      "name": "field_of_view",
      "winner": "B",
      "confidence": 0.95,
      "reason_snippet": "too narrow view"
    },
    {
      "name": "background_clutter",
      "winner": "B",
      "confidence": 0.85,
      "reason_snippet": "doesn't show enough background"
    }
  ],
  "metadata": {
    "review_file": "000001.json",
    "review_index": 2,
    "num_photos_in_series": 4
  }
}
```

**Field Definitions**:
- `pair_id`: Unique identifier (series_compareID1_compareID2)
- `chosen_image`: "A" or "B"
- `preference_strength`: From Bradley-Terry model (ground truth)
- `rank_a`, `rank_b`: Ground truth rankings (1=best)
- `attributes[]`: Extracted attribute labels
  - `winner`: "A" or "B" (which image is better for this attribute)
  - `confidence`: 0.0-1.0 (how confident is the attribute extraction)
  - `reason_snippet`: Part of reason that triggered this attribute

### Output: Attribute Statistics

**Location**: `data/phototriage_attribute_stats.json`

**Format**:
```json
{
  "total_pairs": 37244,
  "total_attribute_labels": 54891,
  "attributes": {
    "sharpness": {
      "count": 8234,
      "percentage": 22.1,
      "avg_confidence": 0.87,
      "common_keywords": ["blur", "sharp", "focus", "clear"],
      "example_reasons": [
        "too blurry",
        "not in focus",
        "sharp and clear"
      ]
    },
    "field_of_view": {
      "count": 6123,
      "percentage": 16.4,
      "avg_confidence": 0.91,
      "common_keywords": ["narrow", "wide", "shows more", "limited view"],
      "example_reasons": [
        "too narrow view",
        "shows more of the scene"
      ]
    }
  },
  "multi_attribute_pairs": 14823,
  "single_attribute_pairs": 22421
}
```

---

## Model Architecture

### Overview

```
Input Image (RGB, 224×224)
         │
         ↓
┌────────────────────────┐
│   CLIP Image Encoder   │  ← Pre-trained (OpenAI or LAION)
│   (ViT-B/32 or L/14)   │  ← Frozen or lightly fine-tuned
└────────────────────────┘
         │
         ↓
   Image Embedding
   (512 or 768-dim)
         │
         ├─────────────────┬─────────────────┬───────────────┬─────────
         ↓                 ↓                 ↓               ↓
    Global Head      Sharpness Head    Framing Head   ... (14 heads)
    Linear(d→1)      Linear(d→1)       Linear(d→1)
         ↓                 ↓                 ↓
    s_global         s_sharpness        s_framing      ...
    (scalar)         (scalar)           (scalar)
```

### Component Specifications

#### 1. CLIP Image Encoder

**Options**:
- **ViT-B/32**: 512-dim embeddings, 86M params, fast
- **ViT-L/14**: 768-dim embeddings, 304M params, more accurate
- **ViT-H/14**: 1024-dim embeddings, 632M params, best quality

**Pre-trained checkpoint**: `laion2b_s34b_b79k` (recommended)

**Fine-tuning strategy**:
- **Option A** (Recommended): Freeze encoder, train heads only
  - Pros: Fast, stable, leverages pre-training
  - Cons: Limited adaptability to triage context
- **Option B**: Light fine-tuning with very low LR
  - Encoder LR: 1e-6
  - Head LR: 1e-4 (100× larger)
  - Pros: Can adapt to triage-specific features
  - Cons: Risk of overfitting, slower training

#### 2. Attribute Heads

**Architecture** (per head):
```python
# Option A: Single linear layer
head = nn.Linear(embed_dim, 1)

# Option B: Small MLP (if needed for non-linear relationships)
head = nn.Sequential(
    nn.Linear(embed_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1)
)
```

**Initialization**: Xavier uniform for linear layers

**Output**: Unbounded scalar (interpreted as preference score)

#### 3. Global Head

Same architecture as attribute heads, but trained on ALL pairs (not just attribute-labeled pairs).

### Model Configuration

**File**: `configs/training/phototriage_contrastive.yaml`

```yaml
model:
  backbone:
    type: clip
    architecture: ViT-B/32
    checkpoint: laion2b_s34b_b79k
    freeze: true

  heads:
    architecture: linear  # or 'mlp'
    embed_dim: 512
    dropout: 0.3  # if using MLP

    # Attribute-specific heads
    attributes:
      - sharpness
      - detail_visibility
      - motion_blur
      - framing
      - cropping_completeness
      - subject_placement
      - background_clutter
      - exposure_quality
      - lighting_quality
      - dynamic_range
      - field_of_view
      - distance_appropriateness
      - subject_interest

    # Global preference head
    global: true

  output:
    normalize: false  # Don't normalize scores (allow unbounded)
```

---

## Training Strategy

### Data Preparation

**Split Strategy**: By series_id to prevent data leakage
- Train: 80% (3,989 series)
- Val: 10% (499 series)
- Test: 10% (498 series)

**Data Augmentation**:
```python
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])
```

**Batch Construction**:
- Sample pairs (not individual images)
- Each batch: 32-64 pairs
- Balance: Mix attribute-labeled and global-only pairs

### Loss Function

**Pairwise Ranking Loss** (Logistic Loss):

```python
def pairwise_ranking_loss(score_winner, score_loser, margin=0.0):
    """
    Loss that enforces score_winner > score_loser

    L = log(1 + exp(-(score_winner - score_loser - margin)))

    Equivalent to binary cross-entropy on:
    P(winner) = sigmoid(score_winner - score_loser)
    """
    return torch.log(1 + torch.exp(-(score_winner - score_loser - margin)))
```

**Total Loss**:

```python
L_total = w_global * L_global + Σ_k w_k * L_attr_k

Where:
- L_global: Ranking loss for global head (computed on all pairs)
- L_attr_k: Ranking loss for attribute k (computed on pairs labeled with k)
- w_global, w_k: Loss weights (hyperparameters)
```

**Loss Weighting Strategy**:

**Option A**: Uniform weights
```python
w_global = 1.0
w_k = 1.0 for all k
```

**Option B**: Inverse frequency weighting
```python
w_global = 1.0
w_k = total_pairs / (2 * count_k)  # Give more weight to rare attributes
```

**Option C**: Manual tuning
```python
w_global = 2.0  # Emphasize global preference
w_sharpness = 1.5  # Emphasize important attributes
w_other = 1.0
```

### Hyperparameters

```yaml
training:
  # Optimization
  optimizer: AdamW
  learning_rate:
    backbone: 1.0e-6  # If fine-tuning
    heads: 1.0e-4
  weight_decay: 0.01

  # Learning rate schedule
  scheduler: cosine_with_warmup
  warmup_epochs: 2
  max_epochs: 20

  # Batch settings
  batch_size: 64  # Pairs per batch
  accumulation_steps: 1

  # Loss settings
  loss:
    type: pairwise_ranking
    margin: 0.0
    weighting: inverse_frequency  # or 'uniform', 'manual'
    global_weight: 1.0

  # Regularization
  dropout: 0.3  # If using MLP heads
  label_smoothing: 0.0  # Optional

  # Early stopping
  early_stopping:
    patience: 5
    metric: val_global_accuracy
    mode: max

  # Checkpointing
  save_best: true
  save_every_n_epochs: 5
```

### Training Procedure

1. **Initialization**:
   - Load pre-trained CLIP encoder
   - Initialize heads with Xavier uniform
   - Move to GPU if available

2. **Training Loop**:
   ```python
   for epoch in range(max_epochs):
       # Training phase
       model.train()
       for batch in train_loader:
           # Forward pass
           scores_a = model(images_a)  # Dict of head outputs
           scores_b = model(images_b)

           # Compute losses
           loss = 0.0

           # Global loss (all pairs)
           loss += w_global * pairwise_loss(
               scores_a['global'] if chosen=='A' else scores_b['global'],
               scores_b['global'] if chosen=='A' else scores_a['global']
           )

           # Attribute losses (only labeled pairs)
           for attr in attributes:
               if attr in pair_labels:
                   winner = pair_labels[attr]
                   loss += w_attr[attr] * pairwise_loss(
                       scores_a[attr] if winner=='A' else scores_b[attr],
                       scores_b[attr] if winner=='A' else scores_a[attr]
                   )

           # Backward pass
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()

       # Validation phase
       model.eval()
       val_metrics = evaluate(model, val_loader)

       # Early stopping check
       if early_stopping(val_metrics['global_accuracy']):
           break
   ```

3. **Monitoring**:
   - Log losses per head
   - Track accuracy per attribute on val set
   - Visualize score distributions
   - Monitor gradient norms

---

## Evaluation Metrics

### 1. Global Preference Metrics

**Pairwise Accuracy**:
```python
correct = (score_winner > score_loser).sum()
accuracy = correct / total_pairs
```

**Top-1 Series Accuracy** (Main metric):
```python
for series in test_series:
    predicted_best = argmax(scores)
    actual_best = image_with_rank_1
    correct += (predicted_best == actual_best)
top1_accuracy = correct / num_series
```

**Mean Reciprocal Rank**:
```python
MRR = mean(1 / rank_of_best_image)
```

**Kendall's Tau** (Ranking Correlation):
```python
tau = correlation(predicted_ranking, ground_truth_ranking)
```

### 2. Per-Attribute Metrics

For each attribute k:

**Attribute Pairwise Accuracy**:
```python
# On pairs labeled with attribute k
correct_k = (score_k_winner > score_k_loser).sum()
accuracy_k = correct_k / pairs_with_label_k
```

**Attribute Precision/Recall** (if using thresholds):
```python
precision_k = TP / (TP + FP)
recall_k = TP / (TP + FN)
```

### 3. Comparison to Baselines

**Baselines**:
1. Sharpness-only (current best: 64.95%)
2. NIMA MobileNet (~55-60%)
3. MUSIQ (~58-62%)
4. CLIP aesthetic (hardcoded prompts: ~45-50%)
5. CLIP learned prompts (~50-55%)
6. Random selection (25-30%)

**Comparison Metrics**:
- Top-1 accuracy improvement
- Win rate (how often our method beats baseline)
- Complementarity (when they disagree, who's right?)

### 4. Ablation Studies

**Study A**: Architecture variants
- Linear heads vs. MLP heads
- Different CLIP backbones (B/32, L/14, H/14)
- Frozen vs. fine-tuned encoder

**Study B**: Loss weighting
- Uniform vs. inverse frequency vs. manual
- Different global_weight values

**Study C**: Attribute subsets
- Global only (no attribute heads)
- Top-5 most frequent attributes only
- All 14 attributes

**Study D**: Data amount
- 25%, 50%, 75%, 100% of training data
- Learning curve analysis

### 5. Error Analysis

**Failure Cases**:
- Pairs where model strongly disagrees with ground truth
- Attributes with lowest accuracy
- Series where top-1 prediction is worst-ranked image

**Correlation Analysis**:
- Between-attribute correlations
- Which attributes predict global preference best

**Visualizations**:
- Confusion matrices per attribute
- Score distributions (winners vs. losers)
- t-SNE of learned embeddings colored by attributes

---

## Implementation Status

### ✅ Completed
- [x] Project architecture design
- [x] Attribute schema definition
- [x] Data format specifications
- [x] Model architecture design
- [x] Training strategy definition
- [x] Evaluation metrics definition

### 🚧 In Progress
- [ ] Design document (this file)

### 📋 To Do
1. **Data Pipeline** (Week 1)
   - [ ] Reason extraction script
   - [ ] Reason analysis and clustering
   - [ ] Attribute mapping implementation
   - [ ] Dataset generation

2. **Model Implementation** (Week 2)
   - [ ] CLIP head modules
   - [ ] Contrastive model class
   - [ ] Pairwise loss functions
   - [ ] Data loader for pairs

3. **Training Pipeline** (Week 2-3)
   - [ ] Trainer class
   - [ ] Training script
   - [ ] Hyperparameter configs
   - [ ] Logging and checkpointing

4. **Evaluation** (Week 3)
   - [ ] Evaluation metrics implementation
   - [ ] Baseline comparisons
   - [ ] Ablation studies
   - [ ] Error analysis tools

5. **Inference API** (Week 4)
   - [ ] PhotoSelector class
   - [ ] Example scripts
   - [ ] Documentation
   - [ ] Integration with existing quality assessment framework

---

## Design Decisions & Rationale

### Why CLIP?
- Pre-trained on 2B+ images → rich visual representations
- Zero-shot capable → can generalize to unseen attributes
- Efficient → ViT-B/32 runs at ~100 images/sec on GPU
- Proven → state-of-the-art on many vision tasks

### Why Contrastive Learning?
- Aligns with human feedback format (pairwise comparisons)
- More robust than absolute scoring (no need to calibrate scales)
- Handles noisy labels well (relative judgments more reliable)
- Data efficient (learns from preferences, not absolute labels)

### Why Per-Attribute Heads?
- Interpretability: Can explain why photo was chosen
- Flexibility: Users can select by specific criteria
- Multi-task learning: Attributes provide auxiliary supervision
- Diagnostic: Can identify which attributes model understands

### Why Not End-to-End Fine-Tuning?
- Risk of overfitting (only 37K pairs, vs. CLIP's 2B training images)
- Computationally expensive
- May forget general visual knowledge
- Heads-only training is more data-efficient

---

## Open Questions & Future Work

### Open Questions
1. Should we use margin in ranking loss? (margin=0.0 vs. margin=0.1)
2. How to handle multi-clause reasons? (Split or treat as multi-label)
3. Should confidence scores affect loss weighting?
4. What's the optimal number of attributes? (Too many → sparse labels, too few → low expressiveness)

### Future Enhancements
1. **Active learning**: Select most informative pairs for manual labeling
2. **Attribute discovery**: Automatically discover new attributes from reasons
3. **Cross-dataset transfer**: Train on PhotoTriage, test on other triage datasets
4. **Temporal modeling**: Account for sequence within burst (first/last photo bias)
5. **Ensemble methods**: Combine with rule-based methods (e.g., sharpness)
6. **Explainability**: Generate text explanations for predictions
7. **User study**: Validate attribute meaningfulness with real users

---

## References

### Datasets
- PhotoTriage: [Kong et al., 2016](http://vision.cs.utexas.edu/projects/rationales/)
- AVA: Aesthetic Visual Analysis dataset
- KonIQ-10k: Quality assessment dataset

### Methods
- CLIP: [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- NIMA: Neural Image Assessment [Talebi & Milanfar, 2018](https://arxiv.org/abs/1709.05424)
- MUSIQ: Multi-Scale Image Quality [Ke et al., 2021](https://arxiv.org/abs/2108.05997)
- Bradley-Terry Model: [Bradley & Terry, 1952](https://www.jstor.org/stable/2334029)

### Related Work
- Learning from pairwise preferences: [Burges et al., 2005](https://icml.cc/Conferences/2005/proceedings/papers/012_Learning_BurgesShawnikoletAl.pdf)
- Multi-task learning with shared representations: [Caruana, 1997](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.2702)
- Contrastive learning: [Chen et al., 2020](https://arxiv.org/abs/2002.05709)

---

## Changelog

**2025-01-19 - v1.0**
- Initial design document
- Defined 14-attribute schema
- Specified data formats
- Designed model architecture
- Planned training and evaluation strategy
