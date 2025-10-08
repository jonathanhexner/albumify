# Baseline Performance Tracking

## Purpose
This document tracks the performance baseline for our image similarity benchmark across different versions and configurations.

## Version Tracking Template

### Version [X.Y.Z]
- **Date**: YYYY-MM-DD
- **Commit Hash**: 
- **Software Environment**:
  - Python version
  - OpenCV version
  - NumPy version
  - scikit-learn version

### Dataset Performance

#### Holidays Dataset
| Method | Recall@1 | Recall@4 | mAP@10 | N-S Score | Notes |
|--------|----------|----------|--------|-----------|-------|
| HSV    |          |          |        |           |       |
| SIFT   |          |          |        |           |       |
| ResNet |          |          |        |           |       |

#### UKBench Dataset
| Method | Recall@1 | Recall@4 | mAP@10 | N-S Score | Notes |
|--------|----------|----------|--------|-----------|-------|
| HSV    |          |          |        |           |       |
| SIFT   |          |          |        |           |       |
| ResNet |          |          |        |           |       |

## Experimental Configurations

### Feature Extraction
- **HSV Histogram**:
  - Bins: 
  - Color Space: 
  - Preprocessing: 

- **SIFT BoVW**:
  - Codebook Size: 
  - Features per Image: 
  - Clustering Method: 

- **ResNet50**:
  - Backbone: 
  - Preprocessing: 
  - Normalization: 

## Performance Insights
- Key observations
- Potential improvements
- Limitations discovered

## Reproducibility
- Random Seeds Used
- Hardware Specifications
- Exact Command Used

## Changelog
- [X.Y.Z] Initial baseline established
- [Next Version] Improvements planned

---

**Note**: Update this document with each significant experimental run or version change.
