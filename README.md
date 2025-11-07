# sim-bench

A simple, lightweight image similarity benchmarking framework for evaluating image retrieval methods on standard datasets.

## ⚡ New: Performance & UX Improvements

- **Feature Caching**: 10-300x speedup for repeated experiments
- **Quick Test Mode**: `--quick` flag for rapid development iteration  
- **Progress Monitoring**: Real-time feedback with progress bars
- **Python Logging**: Automatic experiment logs to file (in addition to console)
- **Group-Based Sampling**: Sample by relevance groups, not total images (maintains valid metrics!)
- **Better Documentation**: See `docs/PERFORMANCE.md`, `docs/LOGGING_AND_SAMPLING.md`, and `docs/CACHE_STORAGE.md`

## Features

- 🚀 **Simple & Fast**: Unified CLI with comma-separated lists, runs out of the box
- 📊 **Universal Metrics**: Accuracy, Recall@k, Precision@k, mAP@k, N-S Score - work with any dataset
- 🔧 **Configurable**: YAML-based configuration with factory patterns
- 📈 **Multiple Outputs**: CSV metrics, per-query results, rankings, and comprehensive summaries
- 🎯 **Multi-Dataset**: Built-in support for UKBench and INRIA Holidays datasets
- 🔄 **Extensible**: Clean factory patterns for methods, datasets, and metrics
- 🏭 **Multiple Methods**: Chi-square, EMD (Wasserstein), Deep Learning (ResNet50), SIFT BoVW

## Quick Start

### ⚡ Fast Testing (New!)

```bash
# Quick test with 100 images (completes in seconds!)
python -m sim_bench.cli --quick --methods chi_square --datasets ukbench

# Custom quick test size
python -m sim_bench.cli --quick --quick-size 50 --methods resnet50
```

See `docs/PERFORMANCE.md` for optimization guide and caching details.

### 📊 Results Analysis (New!)

```bash
# Analyze results in Jupyter notebook
jupyter notebook analyze_results.ipynb
```

The notebook provides automatic analysis:
- Method comparisons with visualizations
- Per-query performance distributions
- Best/worst query identification  
- Multi-experiment comparisons

See `docs/ANALYSIS_NOTEBOOK.md` for guide.

## Installation & Setup

### Prerequisites
- Python 3.10+
- Dataset(s):
  - **UKBench**: Download from [archive.org](https://archive.org/details/ukbench)
  - **INRIA Holidays**: Download from [INRIA website](http://lear.inrialpes.fr/~jegou/data.php)
  - **Sample Images**: Included in `samples/` folder for testing (see [Sample Images](#-sample-images))

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sim-bench

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip

# Option 1: Full install (includes visualization, notebook, deep learning)
pip install -r requirements.txt

# Option 2: Minimal install (benchmarking only, no visualization)
pip install -r requirements-minimal.txt

# Option 3: Development install (includes testing, docs, jupyter)
pip install -r requirements-dev.txt
```

**Recommendation:**
- **Users**: Use `requirements.txt` (includes notebook for analysis)
- **CI/Testing**: Use `requirements-minimal.txt` (faster, smaller)
- **Developers**: Use `requirements-dev.txt` (includes all tools)

### Configuration

1. **Configure dataset path**:

   **For UKBench** (`configs/dataset.ukbench.yaml`):
   ```yaml
   name: ukbench
   root: "/path/to/your/ukbench/dataset"  # Update this path
   subdirs:
     images: "full"
   pattern: "ukbench*.jpg"
   assume_groups_of_four: true
   ```

   **For INRIA Holidays** (`configs/dataset.holidays.yaml`):
   ```yaml
   name: holidays
   root: "/path/to/holidays/dataset"  # Update this path
   pattern: "*.jpg"
   ```

2. **Adjust run settings** in `configs/run.yaml` (optional):
   ```yaml
   sampling:
     max_queries: 200  # Limit for faster testing (50 for Holidays)
   k: 4               # Top-k for N-S score computation
   ```

### Running Evaluation

**New Unified CLI** - Simple comma-separated lists:

```bash
# Single method, single dataset
python -m sim_bench.cli --methods chi_square --datasets ukbench

# Multiple methods, single dataset  
python -m sim_bench.cli --methods chi_square,emd,deep --datasets ukbench

# Single method, multiple datasets
python -m sim_bench.cli --methods chi_square --datasets ukbench,holidays

# Multiple methods, multiple datasets
python -m sim_bench.cli --methods chi_square,emd --datasets ukbench,holidays

# All methods, all datasets (default - no arguments needed!)
python -m sim_bench.cli

# Custom configuration
python -m sim_bench.cli --methods emd --datasets holidays --run-config configs/run.yaml
```

**Available Methods**: `chi_square`, `emd`, `deep`, `dinov2`, `openclip`, `sift_bovw`  
**Available Datasets**: `ukbench`, `holidays`

## Understanding Results

Results are saved to `outputs/<timestamp>/<method>/`:

### Universal Metrics (work with any dataset)
- **`metrics.csv`**: Overall performance summary
  - **Accuracy**: Recall@1 (fraction of queries with correct result in top-1)
  - **Recall@k**: Fraction of queries with relevant results in top-k
  - **Precision@k**: Average precision in top-k results
  - **mAP@k**: Mean Average Precision at k (ranking quality)
  - **N-S Score**: Normalized score (average relevant in top-k, works with all datasets)
- **`per_query.csv`**: Per-query detailed metrics
- **`rankings.csv`**: Full ranking lists for analysis
- **`summary.csv`**: Cross-method comparison (when running multiple methods)

### Example Output
```bash
============================================================
🚀 SIM-BENCH EVALUATION
============================================================
📊 Datasets: ukbench
🔧 Methods: chi_square
============================================================
Dataset: ukbench
Method: chi_square
Extracting HSV histograms ((16, 16, 16) bins)...

📊 Results:
   • N-S Score: 2.690
   • Accuracy: 0.850
   • Recall@4: 0.920
   • mAP@10: 0.756

Results written to: outputs/2025-10-04_14-22-32/chi_square
```

## Available Methods

1. **Chi-Square** (`chi_square`):
   - **Features**: HSV color histograms (16×16×16 bins)
   - **Distance**: Chi-square distance
   - **Speed**: Fast

2. **EMD/Wasserstein** (`emd`):
   - **Features**: HSV color histograms (16×16×16 bins)
   - **Distance**: Earth Mover's Distance (Wasserstein)
   - **Speed**: Medium, more accurate than chi-square

3. **Deep Learning** (`deep`):
   - **Features**: ResNet50 CNN features (2048-dim)
   - **Distance**: Cosine distance
   - **Speed**: Fast, requires PyTorch

4. **DINOv2** (`dinov2`) 🆕:
   - **Features**: Meta's self-supervised ViT (384-1536-dim)
   - **Distance**: Cosine distance
   - **Speed**: Fast, state-of-the-art quality
   - **Variants**: small, base, large, giant

5. **OpenCLIP** (`openclip`) 🆕:
   - **Features**: Vision-language model (512-1024-dim)
   - **Distance**: Cosine distance
   - **Speed**: Fast, excellent semantic understanding
   - **Models**: ViT-B-32, ViT-L-14, and more

6. **SIFT BoVW** (`sift_bovw`):
   - **Features**: SIFT local features + Bag-of-Visual-Words (512 clusters)
   - **Distance**: Cosine distance
   - **Speed**: Slow (codebook building), traditional CV approach

## Benchmark Results

Comprehensive evaluation of all methods on standard datasets (INRIA Holidays and UKBench):

### Overall Method Rankings

**INRIA Holidays Dataset** (1491 queries):

| Rank | Method | mAP | mAP@10 | Recall@10 |
|------|--------|-----|--------|-----------|
| 1 | **DINOv2** | 0.893 | 0.885 | 1.000 |
| 2 | **OpenCLIP** | 0.869 | 0.861 | 1.000 |
| 3 | **Deep (ResNet50)** | 0.832 | 0.820 | 1.000 |
| 4 | **SIFT BoVW** | 0.385 | 0.368 | 1.000 |
| 5 | **EMD** | 0.245 | 0.225 | 1.000 |

**UKBench Dataset** (10,200 queries):

| Rank | Method | mAP | mAP@10 | Recall@10 | N-S Score |
|------|--------|-----|--------|-----------|-----------|
| 1 | **DINOv2** | 0.958 | 0.958 | 0.998 | N/A |
| 2 | **OpenCLIP** | 0.946 | 0.947 | 0.996 | N/A |
| 3 | **Deep (ResNet50)** | 0.946 | 0.946 | 0.998 | 2.83 |
| 4 | **SIFT BoVW** | 0.499 | 0.500 | 0.784 | 1.52 |
| 5 | **EMD** | 0.263 | 0.265 | 0.617 | 0.84 |

### Key Findings

- **DINOv2 is the clear winner** across both datasets, achieving the highest mAP and mAP@10 scores
- **Self-supervised and vision-language models** (DINOv2, OpenCLIP) significantly outperform traditional methods
- **Deep learning methods** consistently achieve 0.82-0.96 mAP, while traditional methods range 0.24-0.50
- **UKBench is easier** than Holidays (avg mAP@10: 0.723 vs 0.632) due to controlled object views
- All methods achieve **perfect Recall@10 on Holidays**, but vary significantly in ranking quality (mAP)

### Dataset Difficulty

**Cross-Dataset Average mAP@10**:
- **UKBench**: 0.723 (easier - controlled object views, 4 images per object)
- **INRIA Holidays**: 0.632 (harder - real vacation photos with viewpoint/lighting variations)

### Per-Query Performance Statistics (AP@10)

**INRIA Holidays**:
| Method | Mean | Median | Std Dev |
|--------|------|--------|---------|
| DINOv2 | 0.8850 | 1.0000 | 0.2426 |
| OpenCLIP | 0.8607 | 1.0000 | 0.2619 |
| Deep (ResNet50) | 0.8203 | 0.9375 | 0.2815 |
| SIFT BoVW | 0.3679 | 0.3333 | 0.3260 |
| EMD | 0.2252 | 0.1667 | 0.2562 |

**UKBench**:
| Method | Mean | Median | Std Dev |
|--------|------|--------|---------|
| DINOv2 | 0.9582 | 1.0000 | 0.1279 |
| OpenCLIP | 0.9467 | 1.0000 | 0.1530 |
| Deep (ResNet50) | 0.9463 | 1.0000 | 0.1519 |
| SIFT BoVW | 0.5002 | 0.5000 | 0.3406 |
| EMD | 0.2647 | 0.2500 | 0.2282 |

### Method Correlation

Methods show **high agreement** on query difficulty (correlation 0.66-0.84 on Holidays, 0.68-0.94 on UKBench), indicating that queries that are hard for one method tend to be hard for others as well.

**Full Analysis**: See [methods_comparison_updated.html](docs/methods_comparison_updated.html) for complete analysis including visualizations, correlation matrices, and per-query performance distributions.

## 🖼️ Sample Images

The `samples/` folder contains representative images from both supported datasets to help you understand the data structure and similarity patterns **without needing to download the full datasets**.

### 📁 What's Included

```
samples/
├── ukbench/          # 10 sample images (2.5 groups)
│   ├── ukbench00000-00003.jpg  # Group 0: Same object, 4 views
│   ├── ukbench00004-00007.jpg  # Group 1: Another object, 4 views  
│   └── ukbench00008-00009.jpg  # Group 2: Third object, 2 views
└── holidays/         # 3 sample images (1 query group)
    ├── 100000.jpg    # Query: Holiday scene
    ├── 100001.jpg    # Similar scene, different angle
    └── 100002.jpg    # Similar scene, different framing
```

### 🚀 Quick Test with Samples

You can test the framework immediately using the sample images:

```bash
# Test on UKBench samples (easy dataset, expect high scores)
python -m sim_bench.cli --methods chi_square,deep --datasets ukbench

# Test on Holidays samples (harder dataset, expect lower scores)
python -m sim_bench.cli --methods emd,deep --datasets holidays

# Compare all methods on both sample sets
python -m sim_bench.cli --methods chi_square,emd,deep --datasets ukbench,holidays
```

### 📊 Expected Results on Samples

| Method | UKBench Samples (N-S) | Holidays Samples (mAP@10) |
|--------|----------------------|---------------------------|
| **Chi-Square** | ~2.5-3.0 | ~0.6-0.8 |
| **EMD** | ~2.7-3.0 | ~0.7-0.9 |
| **Deep** | ~2.9-3.0 | ~0.8-0.95 |

### 📖 Learn More

- **Detailed explanations**: See [`samples/README.md`](samples/README.md)
- **Dataset documentation**: See [`docs/DATASETS.md`](docs/DATASETS.md)
- **Visual inspection**: Open the sample images to see similarity patterns

## Configuration Files

### Dataset Configurations

**UKBench** (`configs/dataset.ukbench.yaml`):
```yaml
name: ukbench
root: "/path/to/ukbench"
subdirs:
  images: "full"
pattern: "ukbench*.jpg"
assume_groups_of_four: true
```

**INRIA Holidays** (`configs/dataset.holidays.yaml`):
```yaml
name: holidays
root: "/path/to/holidays/dataset"
pattern: "*.jpg"
evaluation:
  primary_metric: "map"
  use_all_relevant: true
  map_at_k: [10, 50, 100]
```

### Method Configurations

**Chi-Square** (`configs/methods/chi_square.yaml`):
```yaml
method: chi_square
features:
  color_space: "HSV"
  bins: [16, 16, 16]
  preproc:
    resize: [256, 256]
    center_crop: [224, 224]
distance: "chi_square"
cache_dir: "artifacts/chi_square"
```

**EMD/Wasserstein** (`configs/methods/emd.yaml`):
```yaml
method: emd
features:
  color_space: "HSV"
  bins: [16, 16, 16]
  preproc:
    resize: [256, 256]
    center_crop: [224, 224]
distance: "wasserstein"
cache_dir: "artifacts/emd"
```

### Run Configuration (`configs/run.yaml`)
```yaml
output_dir: "outputs"
sampling:
  max_queries: 200    # null for all queries
metrics:
  - "ns_score"        # N-S Score (normalized score)
  - "recall@1"        # Accuracy
  - "recall@4"        # Recall at 4
  - "recall@10"       # Recall at 10
  - "precision@10"    # Precision at 10
  - "map"             # Mean Average Precision (full)
  - "map@10"          # Mean Average Precision at 10
  - "map@50"          # Mean Average Precision at 50
  # Note: All metrics work with all datasets
save:
  per_query_csv: true
  rankings_csv: true
  metrics_csv: true
  topk_rankings_k: 10
summary_csv: true     # Generate summary across methods
```

## Project Structure

```
sim-bench/
├── sim_bench/           # Main package
│   ├── cli.py          # Unified command-line interface
│   ├── experiment_runner.py  # Experiment orchestration
│   ├── result_manager.py     # Result saving & summaries
│   ├── strategies.py   # Distance computation strategies
│   ├── metrics_api.py  # Metrics computation entry point
│   ├── datasets/       # Dataset implementations
│   │   ├── base.py     # BaseDataset + factory
│   │   ├── ukbench.py  # UKBench dataset
│   │   └── holidays.py # INRIA Holidays dataset
│   ├── methods/        # Method implementations
│   │   ├── base.py     # BaseMethod + factory
│   │   ├── chi_square.py  # HSV + Chi-square
│   │   ├── emd.py      # HSV + Wasserstein
│   │   ├── deep.py     # ResNet50 features
│   │   └── sift_bovw.py   # SIFT + BoVW
│   └── metrics/        # Metrics implementations
│       ├── factory.py  # MetricFactory
│       ├── base.py     # BaseMetric + helpers
│       ├── recall.py   # Recall@k
│       ├── precision.py # Precision@k
│       ├── average_precision.py # mAP@k
│       ├── normalized_score.py  # N-S Score
│       └── accuracy.py # Accuracy (Recall@1)
├── configs/            # Configuration files
│   ├── run.yaml        # Run settings & metrics
│   ├── dataset.ukbench.yaml
│   ├── dataset.holidays.yaml
│   └── methods/        # Method configurations
│       ├── chi_square.yaml
│       ├── emd.yaml
│       ├── deep.yaml
│       └── sift_bovw.yaml
├── outputs/            # Results directory
└── requirements.txt
```

## Extending sim-bench

### Adding New Methods

1. **Create method class** in `sim_bench/methods/your_method.py`:
   ```python
   from .base import BaseMethod
   
   class YourMethod(BaseMethod):
       def extract_features(self, image_paths):
           # Your feature extraction logic
           return features
   ```

2. **Create method config** in `configs/methods/your_method.yaml`:
   ```yaml
   method: your_method
   distance: "cosine"  # or custom strategy
   # Your method-specific parameters
   ```

3. **No CLI changes needed** - factory pattern handles loading automatically!

### Adding New Datasets

1. **Create dataset class** in `sim_bench/datasets/your_dataset.py`:
   ```python
   from .base import BaseDataset
   
   class YourDataset(BaseDataset):
       def load_data(self):
           # Your dataset loading logic
           return image_paths, queries, evaluation_data
   ```

2. **Create dataset config** in `configs/dataset.your_dataset.yaml`:
   ```yaml
   name: your_dataset
   root: "/path/to/dataset"
   # Your dataset-specific parameters
   ```

3. **No CLI changes needed** - factory pattern handles loading automatically!

### Adding New Metrics

1. **Create metric class** in `sim_bench/metrics/your_metric.py`:
   ```python
   from .base import BaseMetric
   
   class YourMetric(BaseMetric):
       def compute(self, ranking_indices, relevance_sets):
           # Your metric computation logic
           return metric_value
   ```

2. **No registration needed** - factory pattern discovers classes automatically!

## Troubleshooting

**Common Issues:**

- **FileNotFoundError**: Check dataset path in `configs/dataset.ukbench.yaml`
- **Memory errors**: Reduce `max_queries` in `configs/run.yaml`
- **Import errors**: Ensure virtual environment is activated and dependencies installed

**Performance Tips:**

- Use `max_queries: 200` for quick testing
- Results scale linearly with dataset size
- HSV histograms are fast but simple - consider deep features for better accuracy

## Requirements

**Core Dependencies:**
- numpy>=1.26
- scipy>=1.11  
- scikit-learn>=1.4
- opencv-contrib-python>=4.9
- Pillow>=10.2
- tqdm>=4.66
- PyYAML>=6.0
- pandas>=2.0  # For result summaries

**Optional (for specific methods):**
- torch>=2.0, torchvision>=0.15  # For deep learning methods (ResNet50, DINOv2, OpenCLIP)
- open-clip-torch>=2.20  # For OpenCLIP method
- matplotlib>=3.8  # For visualization (if needed)

## 📚 Documentation

### 📖 Comprehensive Guides
- **[Dataset Documentation](docs/DATASETS.md)** - Detailed information about UKBench and INRIA Holidays datasets
- **[DINOv2 & OpenCLIP Guide](docs/DINOV2_AND_OPENCLIP.md)** - 🆕 New state-of-the-art feature extractors
- **[Sample Images Guide](samples/README.md)** - Understanding the included sample images and how to use them

### 🔍 Quick References
- **Sample Images**: `samples/` - Representative images from both datasets for immediate testing
- **Configuration**: `configs/` - YAML files for datasets, methods, and run settings
- **Results**: `outputs/` - All evaluation results with timestamps and summaries
- **Documentation**: `docs/` - Detailed guides and references

### 🎯 Getting Started Paths

1. **Quick Test**: Use sample images to test the framework immediately
   ```bash
   python -m sim_bench.cli --methods deep --datasets ukbench
   ```

2. **Full Evaluation**: Download complete datasets and run comprehensive benchmarks
   ```bash
   python -m sim_bench.cli --methods chi_square,emd,deep --datasets ukbench,holidays
   ```

3. **Custom Dataset**: Add your own dataset following the factory pattern (see Extension section)

## License

MIT License - see LICENSE file for details.
