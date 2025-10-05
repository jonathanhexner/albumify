# Dataset Documentation

This document provides comprehensive information about the datasets supported by sim-bench.

## 📊 Supported Datasets

| Dataset | Images | Queries | Groups | Avg Group Size | Primary Metric | Difficulty |
|---------|--------|---------|--------|----------------|----------------|------------|
| **UKBench** | 10,200 | 2,550 | 2,550 | 4 (fixed) | N-S Score | Easy-Medium |
| **INRIA Holidays** | 1,491 | 500 | 500 | 2-20+ (variable) | mAP | Medium-Hard |

## 🎯 UKBench Dataset

### Overview
The University of Kentucky Benchmark (UKBench) is a standard dataset for evaluating image retrieval systems, particularly object recognition methods.

### Dataset Characteristics
- **Total Images**: 10,200 color images
- **Resolution**: Approximately 640×480 pixels
- **Format**: JPEG
- **Content**: Various objects (toys, books, CDs, etc.) photographed from different viewpoints
- **Structure**: Exactly 4 images per object (2,550 objects total)
- **Naming**: Sequential numbering `ukbench{NNNNN}.jpg` where groups are 0-3, 4-7, 8-11, etc.

### Ground Truth
- **Groups**: Every 4 consecutive images show the same object
- **Queries**: Any image can be a query
- **Relevant Results**: The other 3 images in the same group
- **Evaluation**: N-S Score (Normalized Score) = average number of relevant images found in top-4 results

### Download & Setup
1. **Download**: [UKBench on Archive.org](https://archive.org/details/ukbench)
2. **Extract**: Extract all images to a single folder
3. **Configure**: Update `configs/dataset.ukbench.yaml`:
   ```yaml
   name: ukbench
   root: "/path/to/ukbench/images"
   subdirs:
     images: "full"  # or "." if images are in root
   pattern: "ukbench*.jpg"
   assume_groups_of_four: true
   ```

### Expected Performance
| Method | N-S Score Range | Notes |
|--------|----------------|-------|
| **Random** | 0.75 | Baseline (3/4 chance) |
| **Chi-Square** | 2.5-2.8 | Color histograms work well |
| **EMD** | 2.7-2.9 | Better than chi-square |
| **Deep (ResNet50)** | 2.9-3.2 | Best performance |
| **SIFT BoVW** | 2.2-2.7 | Depends on vocabulary size |

### Sample Images
See `samples/ukbench/` for example groups showing the similarity patterns.

---

## 🏖️ INRIA Holidays Dataset

### Overview
The INRIA Holidays dataset consists of personal holiday photos, representing a more realistic and challenging image retrieval scenario.

### Dataset Characteristics
- **Total Images**: 1,491 high-resolution color images
- **Resolution**: Variable (typically 1024×768 or higher)
- **Format**: JPEG
- **Content**: Holiday photos including landmarks, natural scenes, objects, and people
- **Structure**: Variable group sizes (1-20+ images per query)
- **Naming**: Sequential numbering `{NNNNNN}.jpg`

### Ground Truth
- **Queries**: 500 query images (automatically detected as first image in each series)
- **Relevant Results**: Variable number per query (manually annotated)
- **Groups**: Identified by image series (e.g., 100000-100002, 100300-100305, etc.)
- **Evaluation**: mAP (mean Average Precision) at various k values

### Download & Setup
1. **Download**: [INRIA Holidays Dataset](http://lear.inrialpes.fr/~jegou/data.php)
2. **Extract**: Extract all images to a single folder
3. **Configure**: Update `configs/dataset.holidays.yaml`:
   ```yaml
   name: holidays
   root: "/path/to/holidays/images"
   pattern: "*.jpg"
   ```

### Query Detection
The framework automatically identifies queries as the first (lowest numbered) image in each series:
- `100000.jpg` → Query (relevant: 100001, 100002, ...)
- `100300.jpg` → Query (relevant: 100301, 100302, ...)
- `101500.jpg` → Query (relevant: 101501, 101502, ...)

### Expected Performance
| Method | mAP@10 Range | Notes |
|--------|--------------|-------|
| **Random** | ~0.05 | Very poor baseline |
| **Chi-Square** | 0.35-0.45 | Color helps but limited |
| **EMD** | 0.40-0.50 | Slight improvement |
| **Deep (ResNet50)** | 0.65-0.75 | Much better semantic understanding |
| **SIFT BoVW** | 0.45-0.60 | Good for textured scenes |

### Sample Images
See `samples/holidays/` for an example query group showing the similarity patterns.

---

## 🔧 Configuration Details

### Dataset Configuration Files
Each dataset requires a YAML configuration file in `configs/`:

#### UKBench Configuration (`configs/dataset.ukbench.yaml`)
```yaml
name: ukbench
root: "/path/to/ukbench/dataset"
subdirs:
  images: "full"  # subdirectory containing images
pattern: "ukbench*.jpg"
assume_groups_of_four: true  # Enable UKBench-specific grouping
```

#### Holidays Configuration (`configs/dataset.holidays.yaml`)
```yaml
name: holidays
root: "/path/to/holidays/dataset"
pattern: "*.jpg"
description: "INRIA Holidays dataset - 1,491 images, 500 queries"

# Dataset characteristics
total_images: 1491
num_queries: 500

# Evaluation settings
evaluation:
  primary_metric: "map"
  use_all_relevant: true
  map_at_k: [10, 50, 100]
```

### Run Configuration (`configs/run.yaml`)
```yaml
dataset: ukbench  # default dataset
output_dir: "outputs"

sampling:
  max_queries: 50      # Limit queries for testing
  max_gallery: 2000    # Limit gallery size
  random_seed: 42      # Reproducible sampling

k: 4                   # Top-k for N-S score (UKBench)
metrics: [ns, recall@1, recall@4, map@10]
```

## 📈 Evaluation Metrics

### N-S Score (UKBench)
- **Range**: 0.0 to 4.0
- **Perfect Score**: 4.0 (all 4 group members in top-4)
- **Good Score**: >3.0
- **Formula**: Average number of relevant images in top-4 results across all queries

### mAP (Mean Average Precision)
- **Range**: 0.0 to 1.0
- **Perfect Score**: 1.0 (all relevant images ranked first)
- **Good Score**: >0.6 for Holidays
- **Formula**: Average of AP scores across all queries

### Universal Metrics
- **Recall@k**: Fraction of queries with at least one relevant result in top-k
- **Precision@k**: Average precision in top-k results
- **Accuracy**: Recall@1 (fraction with correct result in top-1)

## 🚀 Usage Examples

### Single Dataset Evaluation
```bash
# UKBench with chi-square
python -m sim_bench.cli --methods chi_square --datasets ukbench

# Holidays with deep learning
python -m sim_bench.cli --methods deep --datasets holidays
```

### Cross-Dataset Comparison
```bash
# Same method on both datasets
python -m sim_bench.cli --methods emd --datasets ukbench,holidays

# Multiple methods on single dataset
python -m sim_bench.cli --methods chi_square,emd,deep --datasets ukbench
```

### Custom Configuration
```bash
# With custom run settings
python -m sim_bench.cli --methods deep --datasets holidays --run-config configs/run.yaml
```

## 📚 References

1. **UKBench**: Nister, D. & Stewenius, H. "Scalable Recognition with a Vocabulary Tree" (CVPR 2006)
2. **INRIA Holidays**: Jegou, H. et al. "Hamming Embedding and Weak Geometric Consistency for Large Scale Image Search" (ECCV 2008)
3. **Evaluation**: Müller, H. et al. "Performance evaluation in content-based image retrieval: overview and proposals" (Pattern Recognition Letters 2001)

## 🔍 Troubleshooting

### Common Issues

1. **"No images found"**
   - Check dataset path in configuration file
   - Verify image pattern matches your files
   - Ensure images are in the specified subdirectory

2. **"No queries detected"**
   - For Holidays: Check if images follow the numbering pattern
   - For UKBench: Ensure `assume_groups_of_four: true` is set

3. **Poor performance**
   - Check if ground truth is correctly loaded
   - Verify image quality and resolution
   - Consider different feature extraction methods

### Performance Tips

1. **Sampling**: Use `max_queries` and `max_gallery` for faster testing
2. **Caching**: Features are cached automatically for repeated runs
3. **Memory**: Large datasets may require sufficient RAM for feature storage

