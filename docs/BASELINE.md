# Current Baseline Performance

## Dataset Information
- **Datasets**: 
  - Holidays
  - UKBench
- **Total Images**: 
  - Holidays: [To be filled]
  - UKBench: [To be filled]
- **Number of Groups**: 
  - Holidays: [To be filled]
  - UKBench: [To be filled]

## Feature Extraction Methods

### 1. HSV Histogram (Chi-Square)
- **Configuration**:
  ```yaml
  color_space: HSV
  bins: [16, 16, 16]
  distance: chi2
  ```
- **Performance**:
  - Recall@1: 66.79%
  - Recall@4: 79.93%
  - mAP@10: 61.31%
  - N-S Score: 1.274

### 2. SIFT BoVW
- **Configuration**:
  ```yaml
  local_features:
    type: SIFT
    n_features_per_image: 800
  codebook:
    size: 512
    kmeans_max_iter: 100
  ```
- **Performance**:
  - Recall@1: 58.03%
  - Recall@4: 71.53%
  - mAP@10: 53.47%
  - N-S Score: 1.157

### 3. ResNet50 (Deep Features)
- **Configuration**:
  ```yaml
  backbone: ResNet50
  pretrained: ImageNet
  ```
- **Performance**:
  - Recall@1: 91.97%
  - Recall@4: 97.08%
  - mAP@10: 90.53%
  - N-S Score: 1.905

### 4. EMD (Earth Mover's Distance)
- **Performance**:
  - Recall@1: 45.62%
  - Recall@4: 62.77%
  - mAP@10: 39.36%
  - N-S Score: 0.843

## Computational Characteristics
- **Average Feature Extraction Time**: [To be measured]
- **Memory Usage**: [To be measured]
- **Feature Vector Sizes**:
  - HSV Histogram: 4,096
  - SIFT BoVW: 512
  - ResNet50: 2,048
  - EMD: [To be determined]

## Sampling Configuration
- **max_groups**: 100
- **max_queries**: Unlimited

## Logging Configuration
- **Experiment Log**: Enabled
- **Detailed Log**: Enabled

## Recommendations for Future Iterations
1. Improve SIFT BoVW performance
2. Investigate EMD method
3. Add more noise robustness tests
4. Expand feature extraction techniques

---

**Note**: This baseline serves as a reference point for future improvements and comparisons.
