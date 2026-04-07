# Face Clustering Labeling App

**File**: `app/face_clustering_labeling.py`

**Purpose**: Manual labeling interface for assigning corrected identities to face clusters

---

## Overview

After running the face clustering pipeline, use the labeling app to:
1. View clusters and their faces
2. Assign corrected person identities to clusters
3. Identify merge/split candidates
4. Generate training data for ML merge classifier

---

## Usage

### Step 1: Run Pipeline

```bash
python scripts/run_face_clustering_pipeline.py \
    --album /path/to/photos \
    --output results/my_album \
    --config configs/face_clustering_experiment.yaml
```

### Step 2: Launch Labeling App

```bash
streamlit run app/face_clustering_labeling.py -- --data-dir results/my_album
```

### Step 3: Label Clusters

Open browser to `http://localhost:8501`

---

## Interface

### Sidebar: Cluster Navigation

```
Cluster Browser
├── Progress: 5/23 clusters labeled (22%)
├── Filter:
│   ├── ○ All clusters
│   ├── ● Unlabeled only
│   └── ○ Labeled only
└── Cluster List:
    ├── Cluster 0 (8 faces) ✓ Alice
    ├── Cluster 1 (5 faces) [Unlabeled]
    ├── Cluster 2 (12 faces) ✓ Bob
    └── ...
```

### Main Panel: Cluster Details

```
Cluster 3 (7 faces)

[Exemplars]  [All Faces]  [Debug]

┌─────────────┬─────────────┬─────────────┐
│ Face 0012   │ Face 0015   │ Face 0019   │
│  (exemplar) │  (exemplar) │  (exemplar) │
└─────────────┴─────────────┴─────────────┘

Statistics:
  Size: 7 faces
  Diameter: 0.38
  Median distance: 0.22

Assign Identity:
  [Dropdown: Select person...]  [New Person...]

  [✓ Confirm]  [Next Cluster →]
```

### Debug Tab

```
Debug View - Cluster 3

Select face: [Dropdown]

Within-Cluster Neighbors (5 closest):
  1. Face 0015  distance: 0.12  🟢
  2. Face 0019  distance: 0.18  🟢
  3. Face 0021  distance: 0.24  🟢
  4. Face 0007  distance: 0.31  🟡
  5. Face 0004  distance: 0.35  🟡

Cross-Cluster Neighbors (5 closest from other clusters):
  1. Face 0042 (Cluster 5)  distance: 0.51  🔴
  2. Face 0038 (Cluster 4)  distance: 0.56  🔴
  3. Face 0029 (Cluster 2)  distance: 0.61  🔴

Exemplar Distances:
  To Cluster 5: 0.48  🔴
  To Cluster 4: 0.52  🔴
  To Cluster 2: 0.58  🔴
```

---

## Features

### 1. Progress Tracking

Shows labeled vs unlabeled clusters:
```
Progress: 15/23 clusters labeled (65%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2. Quick Navigation

Keyboard shortcuts:
- `N` - Next cluster
- `P` - Previous cluster
- `Enter` - Confirm label
- `1-9` - Quick select from recent labels

### 3. Duplicate Detection

Warns if multiple clusters have same label:
```
⚠️ Warning: 2 clusters labeled 'Alice'
  - Cluster 0 (8 faces)
  - Cluster 3 (7 faces)

These might need to be merged. Review merge candidates?
```

### 4. Distance Color Coding

Visual feedback on similarity:
- 🟢 Green (< 0.25): High confidence same person
- 🟡 Yellow (0.25-0.40): Uncertain
- 🔴 Red (> 0.40): High confidence different person

### 5. Cluster Statistics

Helps identify problematic clusters:
- **High diameter** (> 0.50): Might contain multiple people
- **Low exemplar count**: Sparse cluster, uncertain
- **Many cross-cluster neighbors**: Merge candidate

---

## Output

### corrected_identities.json

```json
{
  "0": "Alice",
  "1": "Bob",
  "2": "Alice",
  "3": "Charlie",
  "4": "Bob",
  "5": "Alice",
  ...
}
```

**Format**: `cluster_id → person_name`

### labeling_metadata.json

```json
{
  "timestamp": "2026-03-28T15:30:00",
  "n_clusters": 23,
  "n_labeled": 18,
  "n_unlabeled": 5,
  "unique_identities": 12,
  "merge_candidates": [
    {"cluster_a": 0, "cluster_b": 2, "reason": "same_label"},
    {"cluster_a": 1, "cluster_b": 4, "reason": "same_label"}
  ]
}
```

---

## Workflow

### 1. Initial Pass

Label all obvious clusters:
- Start with largest clusters (most faces)
- Skip uncertain clusters for now
- Use "Unknown" for unrecognizable faces

### 2. Review Pass

Go back to uncertain clusters:
- Compare with already-labeled clusters
- Use debug view to see cross-cluster neighbors
- Decide if they belong to existing person or new person

### 3. Merge Candidate Review

Review clusters with same label:
- Should they be merged? (same person, different initial clusters)
- Or keep separate? (different people with same name)

### 4. Export

Save labels and proceed to ML training:
```bash
python scripts/train_merge_classifier.py \
    --data results/my_album \
    --corrected-labels corrected_identities.json
```

---

## Best Practices

### Naming Convention

Use consistent naming:
- ✅ "Alice", "Bob", "Charlie" (first names)
- ✅ "Person_1", "Person_2", ... (if names unknown)
- ❌ "person 1", "Person1", "PERSON_1" (inconsistent)

### Handling Uncertainty

If unsure about a cluster:
- Use "Unknown" or "Unsure" label
- Add comment in separate notes file
- Come back after labeling more clusters

### Quality Check

After labeling, verify:
- No typos in person names
- All large clusters are labeled
- Merge candidates are reasonable

---

## Troubleshooting

### Issue: No faces displayed

**Cause**: `face_crops/` directory not found or empty

**Fix**: Re-run pipeline with `export_for_labeling` step

### Issue: Distance colors not showing

**Cause**: `debug_neighbors.json` missing

**Fix**: Re-run pipeline with `compute_debug_distances` step

### Issue: Slow loading

**Cause**: Large album with many faces

**Fix**:
- Filter to unlabeled clusters only
- Process album in smaller batches

---

## Integration with ML Training

After labeling, use corrected identities for ML training:

```python
# scripts/train_merge_classifier.py

# Load corrected labels
with open('corrected_identities.json') as f:
    corrected_labels = json.load(f)

# Generate training examples
for i, j in cluster_pairs:
    should_merge = (corrected_labels[i] == corrected_labels[j])
    features = compute_merge_features(i, j)

    training_data.append({
        **features,
        'should_merge': should_merge
    })
```

---

**See Also**:
- [ML Training Workflow](../workflows/ml_training_workflow.md) - Next steps after labeling
- [Workbench Guide](workbench_guide.md) - Main experimentation app
- [Debug View](debug_view.md) - Distance visualization details
