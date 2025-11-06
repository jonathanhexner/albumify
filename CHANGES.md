# Project Changes and Versions

## Version History

### v0.2.0 (Current - DINOv2 & OpenCLIP)
**Date**: October 25, 2025

#### New Methods Added 🆕
- **DINOv2**: Meta's self-supervised vision transformer
  - Multiple variants: small, base, large, giant
  - State-of-the-art visual similarity performance
  - 384-1536 dimensional embeddings
- **OpenCLIP**: Open-source CLIP implementation
  - Multiple architectures (ViT-B, ViT-L, ViT-H)
  - Vision-language pre-training
  - 512-1024 dimensional embeddings

#### Implementation Details
- Clean factory pattern integration
- Flexible configuration via YAML
- Batch processing with progress bars
- Automatic model downloading
- Feature caching support
- GPU acceleration support

#### Documentation
- Created comprehensive guide: `docs/DINOV2_AND_OPENCLIP.md`
- Updated README with new methods
- Added usage examples: `examples/dinov2_openclip_examples.sh`
- Updated requirements.txt

#### Dependencies
- Updated torch requirements (removed upper bound for Python 3.14+ compatibility)
- Added open-clip-torch>=2.20

### v0.1.0 (Baseline)
**Date**: October 8, 2025

#### Major Changes
- Implemented two-tier logging system
- Added feature caching mechanism
- Refactored sampling logic
- Created comprehensive EDA framework
- Added Jupyter notebook for results analysis

#### Performance Improvements
- Added progress bars (tqdm)
- Optimized feature extraction
- Improved distance computation

#### Documentation
- Created detailed markdown guides
  - PERFORMANCE.md
  - LOGGING_AND_SAMPLING.md
  - CACHE_STORAGE.md
  - EXPLORATORY_DATA_ANALYSIS.md

### Planned Improvements
- Expand noise robustness testing
- Develop more sophisticated evaluation metrics
- Create automated EDA pipeline

## Versioning Strategy
- Semantic Versioning (MAJOR.MINOR.PATCH)
- Major: Significant architectural changes
- Minor: New features, substantial improvements
- Patch: Bug fixes, minor optimizations

## Contribution Guidelines
1. Update this file with each significant change
2. Include:
   - Version number
   - Date of change
   - Brief description of modifications
   - Performance impact
   - Any breaking changes

---

**Note**: This is a living document tracking the project's evolution.
