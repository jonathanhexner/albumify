# sim-bench Documentation

Documentation organized by task and use case.

## Quick Navigation by Task

### Image Similarity / Retrieval
Find similar images, group duplicates, retrieve images by visual similarity.
- [Image Similarity Overview](image_similarity/README.md)
- [Datasets for Similarity](image_similarity/datasets.md) (UKBench, Holidays, PhotoTriage)
- [Methods Comparison](image_similarity/methods_comparison.md)
- [Performance Benchmarks](image_similarity/performance.md)
- [Quick Start Guide](image_similarity/quickstart.md)

### Clustering
Automatically group similar images without ground truth.
- [Clustering Overview](clustering/README.md)
- [Clustering Methods](clustering/methods.md) (KMeans, DBSCAN, HDBSCAN)
- [Datasets for Clustering](clustering/datasets.md) (Budapest, PhotoTriage)
- [Gallery Visualization](clustering/gallery.md)
- [Quick Start Guide](clustering/quickstart.md)

### Quality Assessment
Select the best photo from a series of similar images.
- [Quality Assessment Overview](quality_assessment/README.md)
- [Benchmark Guide](quality_assessment/benchmark.md)
- [Methods Survey](quality_assessment/literature_survey.md)
- [PhotoTriage Results](quality_assessment/phototriage_results.md)
- [Quick Start Guide](quality_assessment/quickstart.md)

### Architecture & Development
System design, codebase structure, and development guides.
- [Architecture Overview](architecture/README.md)
- [Caching System](architecture/caching.md)
- [Logging System](architecture/logging.md)
- [Dataset Abstraction](architecture/datasets.md)
- [Refactoring Notes](architecture/refactoring.md)

## Quick Start by Use Case

### I want to find similar images
Start here: [Image Similarity Quick Start](image_similarity/quickstart.md)

### I want to group my photos automatically
Start here: [Clustering Quick Start](clustering/quickstart.md)

### I want to pick the best photo from bursts
Start here: [Quality Assessment Quick Start](quality_assessment/quickstart.md)

### I want to understand how it all works
Start here: [Architecture Overview](architecture/README.md)

## Documentation Structure

```
docs/
├── README.md (this file)
│
├── image_similarity/      # Image retrieval and similarity search
│   ├── README.md
│   ├── quickstart.md
│   ├── datasets.md (UKBench, Holidays, PhotoTriage)
│   ├── datasets_phototriage.md
│   ├── datasets_budapest.md
│   ├── methods_comparison.md
│   ├── methods_deep_learning.md (DINOv2, OpenCLIP)
│   ├── performance.md
│   ├── analysis.md
│   ├── multi_experiment_analysis.md
│   ├── notebooks.md
│   ├── troubleshooting.md
│   ├── research_literature_review.md
│   ├── research_datasets.xlsx
│   └── research_eda.docx
│
├── clustering/            # Automatic image grouping
│   ├── README.md
│   └── methods.md (KMeans, DBSCAN, HDBSCAN)
│
├── quality_assessment/    # Best photo selection
│   ├── README.md
│   ├── quickstart.md
│   ├── benchmark.md
│   └── literature_survey.md
│
└── architecture/          # System design and development
    ├── README.md
    ├── overview.md
    ├── caching.md
    ├── logging.md
    ├── logging_detailed.md
    ├── orchestration.md
    └── refactoring.md
```

## All Documentation Files

### By Task

**Image Similarity:**
- Datasets: UKBench (10,200 images), Holidays (1,491 images), PhotoTriage (12,988 images)
- Methods: SIFT BoVW, Chi-Square, EMD, DINOv2, OpenCLIP
- Analysis: Multi-experiment comparison, performance benchmarks

**Clustering:**
- Methods: KMeans, DBSCAN, HDBSCAN
- Visualization: HTML gallery generator
- Datasets: Budapest (310 images), PhotoTriage (12,988 images)

**Quality Assessment:**
- Methods: Rule-based (sharpness, contrast, exposure), CNN (NIMA), Transformers (MUSIQ)
- Benchmarks: Comprehensive comparison framework
- Best result: Sharpness-only (64.95% on PhotoTriage)

**Architecture:**
- Feature caching for speed
- Unified logging system
- Dataset abstraction layer
- Modular design patterns

## Troubleshooting

Common issues and solutions:
- [Image Similarity Troubleshooting](image_similarity/troubleshooting.md)
- [Clustering Troubleshooting](clustering/troubleshooting.md)
- [Quality Assessment Issues](quality_assessment/troubleshooting.md)

## Contributing

When adding documentation:
1. Choose the appropriate task directory
2. Use clear, descriptive filenames
3. Update the task's README.md
4. Update this main README.md
5. Include examples and use cases
