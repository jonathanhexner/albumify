# Image Similarity and Retrieval

Find similar images, match duplicates, and retrieve visually similar content.

## Quick Links

- [Quick Start](quickstart.md) - Get started in 5 minutes
- [Datasets](datasets.md) - UKBench, Holidays, PhotoTriage
- [Methods Comparison](methods_comparison.md) - Compare different algorithms
- [Performance](performance.md) - Speed and accuracy benchmarks
- [Troubleshooting](troubleshooting.md) - Common issues

## What is Image Similarity?

Given a query image, find other images that are visually similar:
- Same object from different angles (UKBench)
- Same scene/location (Holidays)
- Same photo burst/moment (PhotoTriage)

## Available Datasets

- **UKBench**: 10,200 images, 2,550 objects, 4 images each
- **Holidays**: 1,491 vacation photos, 500 queries
- **PhotoTriage**: 12,988 images, 4,986 bursts

See [datasets.md](datasets.md) for details.

## Methods

- **SIFT BoVW**: Fast, no GPU required
- **Chi-Square**: Color-based similarity
- **EMD**: Earth Mover's Distance
- **DINOv2**: Deep learning, best results
- **OpenCLIP**: Image-text embeddings

See [methods_comparison.md](methods_comparison.md) and [methods_deep_learning.md](methods_deep_learning.md) for details.

## Analysis Tools

- [Multi-experiment comparison](multi_experiment_analysis.md)
- [Jupyter notebooks](notebooks.md)
- [Result analysis](analysis.md)

## Research Materials

- [Literature Review](research_literature_review.md) - Academic papers and methods
- [Datasets Research](research_datasets.xlsx) - Dataset comparison spreadsheet
- [EDA](research_eda.docx) - Exploratory data analysis

