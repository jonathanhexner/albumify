# Sample Images

This folder contains representative sample images from the supported datasets to help you understand the data structure and expected image similarity patterns.

## 📁 Folder Structure

```
samples/
├── ukbench/          # UKBench dataset samples
│   ├── ukbench00000.jpg  # Group 0: Image 1 (query)
│   ├── ukbench00001.jpg  # Group 0: Image 2 (similar)
│   ├── ukbench00002.jpg  # Group 0: Image 3 (similar)
│   ├── ukbench00003.jpg  # Group 0: Image 4 (similar)
│   ├── ukbench00004.jpg  # Group 1: Image 1 (query)
│   ├── ukbench00005.jpg  # Group 1: Image 2 (similar)
│   ├── ukbench00006.jpg  # Group 1: Image 3 (similar)
│   ├── ukbench00007.jpg  # Group 1: Image 4 (similar)
│   ├── ukbench00008.jpg  # Group 2: Image 1 (query)
│   └── ukbench00009.jpg  # Group 2: Image 2 (similar)
└── holidays/         # INRIA Holidays dataset samples
    ├── 100000.jpg    # Query image
    ├── 100001.jpg    # Related image 1
    └── 100002.jpg    # Related image 2
```

## 🎯 UKBench Dataset

**Source**: [University of Kentucky Benchmark](https://archive.org/details/ukbench)  
**Total Images**: 10,200 images  
**Groups**: 2,550 groups of 4 similar images each  
**Image Size**: ~640×480 pixels  
**Format**: JPEG  

### Characteristics:
- **Structured Groups**: Every 4 consecutive images belong to the same object/scene
- **Naming Convention**: `ukbench{NNNNN}.jpg` where NNNNN is zero-padded
- **Group Pattern**: Images 0-3 are similar, 4-7 are similar, 8-11 are similar, etc.
- **Content**: Various objects photographed from different angles/lighting
- **Evaluation**: N-S Score (Normalized Score) - average number of relevant images in top-4 results

### Sample Groups in this folder:
- **Group 0** (ukbench00000-00003): 4 images of the same object
- **Group 1** (ukbench00004-00007): 4 images of another object  
- **Group 2** (ukbench00008-00009): First 2 images of a third object

### Expected Similarity:
When querying with `ukbench00000.jpg`, the top results should be:
1. `ukbench00001.jpg` ✅
2. `ukbench00002.jpg` ✅  
3. `ukbench00003.jpg` ✅
4. Other images ❌

## 🏖️ INRIA Holidays Dataset

**Source**: [INRIA Holidays Dataset](http://lear.inrialpes.fr/~jegou/data.php)  
**Total Images**: 1,491 images  
**Queries**: 500 query images  
**Image Size**: Variable (high resolution)  
**Format**: JPEG  

### Characteristics:
- **Variable Groups**: Each query has a different number of relevant images (1-20+)
- **Naming Convention**: `{NNNNNN}.jpg` where NNNNNN is the image ID
- **Query Pattern**: Queries are typically the first (lowest numbered) image in each series
- **Content**: Holiday photos - landmarks, scenes, objects from personal photo collections
- **Evaluation**: mAP (mean Average Precision) - ranking quality metric

### Sample Group in this folder:
- **Query**: `100000.jpg` - The query image for this group
- **Relevant**: `100001.jpg`, `100002.jpg` - Images similar to the query
- **Note**: There may be additional relevant images not included in this sample

### Expected Similarity:
When querying with `100000.jpg`, relevant results should include:
1. `100001.jpg` ✅
2. `100002.jpg` ✅
3. Additional images from the same scene/location (not in this sample)

## 🔍 How to Use These Samples

### Visual Inspection
Open the images in any image viewer to see the similarity patterns:
- **UKBench**: Look for same objects from different angles
- **Holidays**: Look for same scenes/locations with different framing/lighting

### Testing with sim-bench
You can test the framework with just these samples by:

1. **Create a local dataset config** (e.g., `configs/dataset.samples.yaml`):
```yaml
name: samples
root: "samples/ukbench"  # or "samples/holidays"
pattern: "*.jpg"
assume_groups_of_four: true  # for UKBench samples only
```

2. **Run evaluation**:
```bash
python -m sim_bench.cli --methods chi_square --datasets samples
```

### Understanding Results
- **Perfect UKBench Score**: N-S Score = 3.0 (all 3 other group members in top-4)
- **Good Holidays Score**: High mAP@10 (>0.8) indicates good ranking of relevant images

## 📊 Expected Performance on Samples

Based on the sample images, you should expect:

| Method | UKBench Samples | Holidays Samples |
|--------|----------------|------------------|
| **Chi-Square** | N-S: ~2.5-3.0 | mAP@10: ~0.6-0.8 |
| **EMD** | N-S: ~2.7-3.0 | mAP@10: ~0.7-0.9 |
| **Deep (ResNet50)** | N-S: ~2.9-3.0 | mAP@10: ~0.8-0.95 |
| **SIFT BoVW** | N-S: ~2.0-2.8 | mAP@10: ~0.5-0.8 |

## 🚀 Next Steps

1. **Download Full Datasets**: Use these samples to understand the data structure, then download the complete datasets
2. **Configure Paths**: Update the dataset configuration files with your local paths
3. **Run Full Evaluation**: Execute comprehensive benchmarks on the complete datasets
4. **Compare Methods**: Use the summary CSV to compare different similarity methods

## 📚 References

- **UKBench Paper**: Nister, D. & Stewenius, H. "Scalable Recognition with a Vocabulary Tree" (CVPR 2006)
- **Holidays Paper**: Jegou, H. et al. "Hamming Embedding and Weak Geometric Consistency for Large Scale Image Search" (ECCV 2008)
- **Evaluation Metrics**: See `sim_bench/metrics/` for implementation details

