# AVA (Aesthetic Visual Analysis) Notebooks

This folder is for notebooks related to AVA aesthetic score prediction training and analysis.

## What is AVA?

AVA is a large-scale dataset for aesthetic visual analysis with:
- 250,000+ images
- Aesthetic scores from 1-10 (based on human ratings)
- Score distributions (not just mean scores)

## Related Code

**Training script**: [`sim_bench/training/train_ava_resnet.py`](../../sim_bench/training/train_ava_resnet.py)  
**Model**: [`sim_bench/models/ava_resnet.py`](../../sim_bench/models/ava_resnet.py)  
**Dataset**: [`sim_bench/datasets/ava_dataset.py`](../../sim_bench/datasets/ava_dataset.py)  
**Configs**: [`configs/ava/`](../../configs/ava/)

## Suggested Notebooks to Create

### 1. **AVA Training Analysis**
- Load training history from checkpoint
- Plot loss and Spearman correlation curves
- Analyze per-epoch metrics
- Compare different model configurations

### 2. **AVA Model Exploration**
Similar to [`siamese_e2e/explore_model_behavior.ipynb`](../siamese_e2e/explore_model_behavior.ipynb):
- Test model on synthetic degradations
- Compare with IQA baselines
- Validate predicted score distributions
- Analyze failure cases

### 3. **AVA Score Distribution Analysis**
- Analyze predicted vs ground truth distributions
- Visualize score histograms
- Identify over/under-confident predictions
- Per-category performance (if using semantic categories)

### 4. **AVA vs PhotoTriage Comparison**
- Compare aesthetic scores with pairwise preferences
- Correlation analysis
- Transfer learning potential

## Example: Creating an AVA Exploration Notebook

You can adapt the Siamese exploration notebook:

```bash
# Copy and modify the template
cp ../siamese_e2e/explore_model_behavior.ipynb ./explore_ava_model.ipynb
```

Then update:
- Checkpoint path to AVA model
- Load AVA-specific predictions (aesthetic scores instead of pairwise)
- Adapt visualizations for score distributions

## Quick Start

If you've trained an AVA model, create an exploration notebook here to validate it:

1. **Create notebook**: `ava/explore_ava_model.ipynb`
2. **Load checkpoint**: Point to `outputs/ava/your_run/best_model.pt`
3. **Run tests**: Use degradation testing similar to Siamese model
4. **Compare baselines**: Compare predicted scores with IQA metrics

## Path Setup

When creating notebooks in this folder, use:

```python
import sys
from pathlib import Path

# Add project root (two levels up from ava/ subfolder)
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))
```

This allows you to import from `sim_bench` and access configs/data.
