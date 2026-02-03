# PhotoTriage Binary Classifier Training Guide

## Overview

This guide explains how to train a binary classification model to predict which of two images a user would prefer, based on the PhotoTriage dataset. The model uses pre-trained CLIP embeddings as features and trains a lightweight MLP classifier on top.

## Model Architecture

```
Image 1 → CLIP Encoder (frozen) → Embedding 1 (512-dim)
                                        ↓
                                   [concat] → MLP Classifier → Binary Prediction
                                        ↓                       (0 or 1)
Image 2 → CLIP Encoder (frozen) → Embedding 2 (512-dim)
```

### Components:

1. **CLIP Encoder (Frozen)**
   - Model: ViT-B-32 pre-trained on LAION-2B
   - Output: 512-dimensional image embeddings
   - Weights are frozen (not trained)
   - Used only for feature extraction

2. **MLP Classifier (Trainable)**
   - Input: Concatenated embeddings [emb1; emb2] = 1024-dim
   - Architecture: 1024 → 256 → 128 → 2
   - Activation: ReLU
   - Output: 2-class logits (image 1 preferred vs image 2 preferred)
   - Only these weights are trained

### Why This Architecture?

- **Fast Training**: Only ~200K parameters to train (MLP), not millions (CLIP)
- **CPU Friendly**: Pre-compute embeddings once, train on CPU in 1-2 hours
- **Strong Baseline**: CLIP embeddings capture visual quality well
- **Interpretable**: Simple linear decision boundary on top of embeddings

## Dataset

### Source Data

- **Location**: `D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv`
- **Images**: `D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs\`
- **Total Pairs**: 24,186
- **Total Images**: 12,988

### Data Filtering

We use only high-agreement pairs to ensure label quality:
- **Agreement > 0.7**: At least 70% of reviewers agreed
- **num_reviewers >= 2**: At least 2 reviewers per pair
- **Filtered Dataset**: 12,070 pairs (49.9% of total)

### CSV Format

Key columns:
- `series_id`: Photo series identifier
- `compareID1`, `compareID2`: Image indices being compared
- `compareFile1`, `compareFile2`: Image filenames (e.g., "1-1.JPG", "1-2.JPG")
- `MaxVote`: Index of the preferred image
- `Agreement`: Fraction of reviewers who agreed (0.0-1.0)
- `num_reviewers`: Number of reviewers for this pair

### Label Creation

```python
# Label = 1 if image 2 was preferred, 0 if image 1 was preferred
label = int(MaxVote == compareID2)
```

### Data Split Strategy

**Split by series_id to prevent data leakage:**
- Images from the same series stay in the same split
- Ensures the model generalizes to new photo series
- Standard 80/10/10 split: Train/Val/Test

Example:
```
Series 1, 2, 3, ..., 3988 → Train (80%)
Series 3989, ..., 4487 → Val (10%)
Series 4488, ..., 4986 → Test (10%)
```

## Training Pipeline

### Step 1: Pre-compute CLIP Embeddings

To speed up training, we pre-compute embeddings for all images once:

```python
# Load CLIP model
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)

# Compute embeddings for all images
for image_path in all_images:
    image = preprocess(Image.open(image_path))
    embedding = model.encode_image(image)  # 512-dim
    # Save to cache
```

**Expected Time**: ~30-60 minutes on CPU for 12,988 images

### Step 2: Create Train/Val/Test Splits

Split pairs by series_id:

```python
# Get unique series IDs
unique_series = df['series_id'].unique()
np.random.shuffle(unique_series)

# Split series
n_series = len(unique_series)
train_series = unique_series[:int(0.8 * n_series)]
val_series = unique_series[int(0.8 * n_series):int(0.9 * n_series)]
test_series = unique_series[int(0.9 * n_series):]

# Filter pairs by series
train_pairs = df[df['series_id'].isin(train_series)]
val_pairs = df[df['series_id'].isin(val_series)]
test_pairs = df[df['series_id'].isin(test_series)]
```

### Step 3: Train MLP Classifier

```python
# Model
classifier = MLPBinaryClassifier(
    input_dim=1024,  # 2 * 512
    hidden_dims=[256, 128],
    output_dim=2
)

# Training
optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # Training loop
    for emb1, emb2, label in train_loader:
        concat_emb = torch.cat([emb1, emb2], dim=1)
        logits = classifier(concat_emb)
        loss = criterion(logits, label)
        # Backprop

    # Validation
    val_accuracy = evaluate(val_loader)

    # Early stopping
    if val_accuracy > best_accuracy:
        save_checkpoint()
```

**Expected Time**: ~1-2 hours on CPU

### Step 4: Evaluate on Test Set

Metrics:
- **Pairwise Accuracy**: % of pairs where model predicts correct winner
- **Confusion Matrix**: True positives, false positives, etc.

## Usage

### Training

```bash
# From project root
cd d:/sim-bench

# Activate virtual environment
.venv\Scripts\activate

# Run training script
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py \
    --csv_path "D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv" \
    --images_dir "D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs" \
    --output_dir "outputs/phototriage_binary" \
    --batch_size 128 \
    --num_epochs 50 \
    --learning_rate 1e-3 \
    --early_stopping_patience 5
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_path` | Required | Path to CSV file with pairs |
| `images_dir` | Required | Directory containing images |
| `output_dir` | `outputs/phototriage_binary` | Where to save models/logs |
| `clip_model` | `ViT-B-32` | CLIP model architecture |
| `clip_pretrained` | `laion2b_s34b_b79k` | CLIP checkpoint |
| `batch_size` | 128 | Batch size for training |
| `num_epochs` | 50 | Maximum training epochs |
| `learning_rate` | 1e-3 | Learning rate for AdamW |
| `weight_decay` | 0.01 | Weight decay for regularization |
| `hidden_dims` | [256, 128] | MLP hidden layer sizes |
| `early_stopping_patience` | 5 | Epochs to wait before early stopping |
| `agreement_threshold` | 0.7 | Minimum agreement for pairs |
| `min_reviewers` | 2 | Minimum reviewers per pair |
| `train_split` | 0.8 | Fraction for training |
| `val_split` | 0.1 | Fraction for validation |
| `test_split` | 0.1 | Fraction for test |
| `random_seed` | 42 | Random seed for reproducibility |

### Inference

```python
from sim_bench.quality_assessment.trained_models.phototriage_binary import PhotoTriageBinaryClassifier

# Load trained model
model = PhotoTriageBinaryClassifier.load("outputs/phototriage_binary/best_model.pt")

# Compare two images
from PIL import Image
img1 = Image.open("path/to/image1.jpg")
img2 = Image.open("path/to/image2.jpg")

prediction = model.predict_pair(img1, img2)
# Returns: 0 if image1 preferred, 1 if image2 preferred
```

## Expected Performance

### Baseline Comparisons

| Method | Pairwise Accuracy | Notes |
|--------|-------------------|-------|
| Random | 50.0% | Coin flip |
| Sharpness (Laplacian) | ~65% | Single metric |
| Rule-based (combined) | ~65% | Multiple heuristics |
| **CLIP + MLP (This)** | **~75-80%** (target) | Learned from preferences |

### Training Timeline (CPU)

| Stage | Time | Cumulative |
|-------|------|------------|
| Load data & setup | ~2 min | 2 min |
| Pre-compute CLIP embeddings | ~30-60 min | 32-62 min |
| Train MLP classifier | ~60-120 min | 92-182 min |
| Final evaluation | ~5 min | **~97-187 min (1.5-3 hours)** |

**Total: 1.5-3 hours on CPU** ✅ (Well within 4-6 hour budget)

## Output Files

After training, the following files are created:

```
outputs/phototriage_binary/
├── embeddings/
│   └── clip_vit_b_32_embeddings.pkl    # Pre-computed embeddings
├── splits/
│   ├── train_pairs.csv                 # Training pairs
│   ├── val_pairs.csv                   # Validation pairs
│   └── test_pairs.csv                  # Test pairs
├── checkpoints/
│   ├── best_model.pt                   # Best model checkpoint
│   └── last_model.pt                   # Last epoch checkpoint
├── logs/
│   └── training.log                    # Training logs
├── plots/
│   ├── training_curves.png             # Loss & accuracy curves
│   └── confusion_matrix.png            # Test set confusion matrix
└── results/
    └── test_results.json               # Final test metrics
```

## Implementation Details

### PyTorch Dataset Class

```python
class PhotoTriagePairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_df, embeddings_dict):
        """
        Args:
            pairs_df: DataFrame with columns [compareFile1, compareFile2, label]
            embeddings_dict: Dict mapping filename -> embedding tensor
        """
        self.pairs = pairs_df
        self.embeddings = embeddings_dict

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        emb1 = self.embeddings[row['compareFile1']]  # (512,)
        emb2 = self.embeddings[row['compareFile2']]  # (512,)
        label = row['label']  # 0 or 1
        return emb1, emb2, label
```

### MLP Classifier Architecture

```python
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[256, 128], output_dim=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)  # (batch, 2)
```

### Training Loop

```python
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for emb1, emb2, labels in train_loader:
        # Concatenate embeddings
        x = torch.cat([emb1, emb2], dim=1)  # (batch, 1024)

        # Forward pass
        logits = model(x)  # (batch, 2)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size
```bash
python train_binary_classifier.py --batch_size 64
```

### Issue: Embeddings computation too slow

**Solution**: Use smaller CLIP model or reduce image resolution
```python
# In embedding computation
preprocess = transforms.Compose([
    transforms.Resize(224),  # Reduce from default
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

### Issue: Training not improving

**Solutions**:
1. Check learning rate (try 1e-4 or 1e-2)
2. Increase model capacity (add more hidden layers)
3. Check data imbalance (should be ~50/50)
4. Verify embeddings are normalized

### Issue: Overfitting (train acc >> val acc)

**Solutions**:
1. Increase dropout (default 0.2 → 0.3 or 0.4)
2. Add weight decay (default 0.01 → 0.1)
3. Reduce model capacity (smaller hidden dims)
4. Early stopping (reduce patience)

## Next Steps

After training the binary classifier:

1. **Benchmark Integration**: Add as a method in pairwise benchmark
2. **Multi-task Extension**: Train separate heads for 14 quality attributes
3. **Fine-tuning**: Unfreeze CLIP and fine-tune on PhotoTriage (requires GPU)
4. **Ensemble**: Combine with rule-based methods for robustness
5. **Active Learning**: Use model to request labels for uncertain pairs

## References

- PhotoTriage Dataset: [Paper](https://arxiv.org/abs/2007.14905)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- OpenCLIP: [LAION OpenCLIP](https://github.com/mlfoundations/open_clip)

## See Also

- [QUALITY_ASSESSMENT_QUICKSTART.md](../QUALITY_ASSESSMENT_QUICKSTART.md) - Getting started guide
- [QUALITY_BENCHMARK_GUIDE.md](../QUALITY_BENCHMARK_GUIDE.md) - Benchmarking framework
- [phototriage_binary.py](../../sim_bench/quality_assessment/trained_models/phototriage_binary.py) - Model implementation
