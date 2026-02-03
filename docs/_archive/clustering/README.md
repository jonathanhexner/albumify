# Image Clustering

Automatically group similar images without needing ground truth labels.

## Quick Links

- [Methods](methods.md) - KMeans, DBSCAN, HDBSCAN
- [Datasets](../image_similarity/datasets_budapest.md) - Budapest photos
- Quick Start - See methods.md

## What is Clustering?

Automatically organize images into groups based on visual similarity:
- Group photos from same location
- Find duplicate/near-duplicate images  
- Organize photo collections by content
- Discover patterns in unlabeled data

## When to Use Clustering

- No ground truth labels available
- Exploratory analysis of photo collection
- Automatic photo organization
- Finding groups in large datasets

## Available Methods

- **KMeans**: Fixed number of clusters, fast
- **DBSCAN**: Density-based, finds outliers
- **HDBSCAN**: Hierarchical DBSCAN, automatic cluster count

See [methods.md](methods.md) for details and examples.

## Visualization

Clustering results can be visualized in HTML galleries showing each cluster.






