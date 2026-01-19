# Notebooks Organization

This directory contains Jupyter notebooks organized by topic area.

## Directory Structure

### 📁 `siamese_e2e/`
Siamese network training and model exploration for PhotoTriage pairwise ranking.

- **`explore_model_behavior.ipynb`** ⭐ - Comprehensive model validation using synthetic degradations and IQA baselines
- `analyze_siamese.ipynb` - Siamese model analysis
- `compare_dataloaders.ipynb` - Dataloader comparison and validation

**Related checkpoint**: `outputs/siamese_e2e/20260111_005327/`

### 📁 `pairwise_analysis/`
Pairwise comparison analysis and multi-feature model exploration.

- `analyze_pairwise_multifeature_traininig.ipynb` - Multi-feature model analysis
- `pairwise_attribute_analysis.ipynb` - Attribute-based pairwise analysis
- `pairwise_attribute_analysis_2.ipynb` - Extended attribute analysis

### 📁 `quality_assessment/`
Image quality assessment methods and synthetic degradation testing.

- `quality_assessment_analysis.ipynb` - IQA method comparison
- `quality_assessment_analysis_clean.ipynb` - Clean version of IQA analysis
- `synthetic_degradation_analysis.ipynb` - Synthetic degradation validation

### 📁 `ava/`
AVA (Aesthetic Visual Analysis) dataset training and analysis.

**Currently empty** - Add AVA-related notebooks here:
- AVA training analysis
- AVA model exploration
- AVA score distribution analysis

**Related code**: `sim_bench/training/train_ava_resnet.py`  
**Related configs**: `configs/ava/`

### 📁 `debug/`
Debugging and diagnostic notebooks.

- `debug_dataloader_issue.ipynb` - Dataloader debugging
- `debug_multi_feature_training.ipynb` - Multi-feature training debugging

### 📁 `experiments/`
Experimental notebooks and Kaggle training runs.

- `kaggle_siamese_training.ipynb`
- `kaggle_siamese_training_corrected.ipynb`

### 📁 `analysis/`
Miscellaneous analysis scripts and notebooks.

## Quick Links

### Model Exploration
- **Siamese E2E**: [`siamese_e2e/explore_model_behavior.ipynb`](siamese_e2e/explore_model_behavior.ipynb)
- **Quality Assessment**: [`quality_assessment/synthetic_degradation_analysis.ipynb`](quality_assessment/synthetic_degradation_analysis.ipynb)

### Training Analysis
- **Pairwise**: [`pairwise_analysis/analyze_pairwise_multifeature_traininig.ipynb`](pairwise_analysis/analyze_pairwise_multifeature_traininig.ipynb)
- **AVA**: *Coming soon* - Add to `ava/` folder

## Adding New Notebooks

When creating new notebooks, place them in the appropriate subfolder:

- **Siamese/E2E models** → `siamese_e2e/`
- **AVA training/analysis** → `ava/`
- **Pairwise comparisons** → `pairwise_analysis/`
- **Quality assessment** → `quality_assessment/`
- **Debugging** → `debug/`
- **Experiments** → `experiments/`

## Usage Tips

### Running notebooks from subfolders

Notebooks in subfolders can still access the project:

```python
import sys
from pathlib import Path

# Add project root (two levels up from subfolder)
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

# Now import normally
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker
```

The notebooks already include the appropriate path setup code.
