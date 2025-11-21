"""
Clustering tools for grouping images by visual similarity.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

from sim_bench.agent.tools.base import BaseTool, ToolCategory
from sim_bench.clustering import load_clustering_method
from sim_bench.feature_extraction.base import load_method

logger = logging.getLogger(__name__)


class ClusterImagesTool(BaseTool):
    """
    Cluster images into groups based on visual similarity.

    Uses various clustering algorithms (DBSCAN, HDBSCAN, KMeans) with
    different feature extractors (DINOv2, OpenCLIP, histograms).
    """

    def setup(self):
        """Initialize clustering method (lazy loaded)."""
        self.clustering_method = None

    def execute(
        self,
        image_paths: List[str],
        method: str = 'dbscan',
        feature_type: str = 'dinov2',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Cluster images into groups.

        Args:
            image_paths: List of image file paths
            method: Clustering algorithm (dbscan, hdbscan, kmeans)
            feature_type: Feature extractor (dinov2, openclip, histogram)
            **kwargs: Method-specific parameters (min_cluster_size, eps, etc.)

        Returns:
            {
                'success': True,
                'data': {
                    'clusters': {cluster_id: [image_paths]},
                    'num_clusters': int,
                    'noise_images': [image_paths],  # DBSCAN/HDBSCAN only
                    'cluster_sizes': [int, ...],
                    'labels': [int, ...]  # cluster label per image
                },
                'message': str,
                'metadata': {...}
            }
        """
        logger.info(f"Clustering {len(image_paths)} images with {method}/{feature_type}")

        # Extract features
        logger.info(f"Extracting {feature_type} features from {len(image_paths)} images")
        feature_config = {'method': feature_type}
        feature_method = load_method(feature_type, feature_config)
        features = feature_method.extract_features(image_paths)

        # Build clustering config
        cluster_config = {
            'algorithm': method,
            'params': kwargs,
            'output': {'save_csv': False, 'save_stats': False, 'save_galleries': False}
        }

        # Create clustering method
        clusterer = load_clustering_method(cluster_config)

        # Execute clustering
        labels, stats = clusterer.cluster(features)

        # Organize results
        clusters = {}
        noise_images = []

        for img_path, label in zip(image_paths, labels):
            if label == -1:  # Noise (DBSCAN/HDBSCAN)
                noise_images.append(img_path)
            else:
                clusters.setdefault(int(label), []).append(img_path)

        num_clusters = stats.get('n_clusters', len(clusters))
        cluster_sizes = list(stats.get('cluster_sizes', {}).values())

        # Calculate statistics
        avg_cluster_size = sum(cluster_sizes) / num_clusters if num_clusters > 0 else 0

        message = (
            f"Found {num_clusters} clusters from {len(image_paths)} images. "
            f"Average cluster size: {avg_cluster_size:.1f}"
        )

        if noise_images:
            message += f". {len(noise_images)} noise images"

        return {
            'success': True,
            'data': {
                'clusters': clusters,
                'num_clusters': num_clusters,
                'noise_images': noise_images,
                'cluster_sizes': cluster_sizes,
                'labels': labels.tolist() if hasattr(labels, 'tolist') else list(labels),
                'avg_cluster_size': avg_cluster_size,
                'stats': stats
            },
            'message': message,
            'metadata': {
                'method': method,
                'feature_type': feature_type,
                'params': kwargs,
                'num_images': len(image_paths)
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema for LLM."""
        return {
            'name': 'cluster_images',
            'description': 'Cluster images into groups based on visual similarity',
            'category': ToolCategory.CLUSTERING,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of image file paths to cluster'
                    },
                    'method': {
                        'type': 'string',
                        'enum': ['dbscan', 'hdbscan', 'kmeans'],
                        'description': 'Clustering algorithm (dbscan=density-based, hdbscan=hierarchical, kmeans=k-means)',
                        'default': 'dbscan'
                    },
                    'feature_type': {
                        'type': 'string',
                        'enum': ['dinov2', 'openclip', 'histogram'],
                        'description': 'Feature extraction method (dinov2=best quality, openclip=semantic, histogram=fast)',
                        'default': 'dinov2'
                    },
                    'min_cluster_size': {
                        'type': 'integer',
                        'description': 'Minimum images per cluster (DBSCAN/HDBSCAN only)',
                        'default': 5
                    },
                    'eps': {
                        'type': 'number',
                        'description': 'Maximum distance for neighborhood (DBSCAN only)',
                        'default': 0.5
                    },
                    'n_clusters': {
                        'type': 'integer',
                        'description': 'Number of clusters (KMeans only)',
                        'default': 10
                    }
                },
                'required': ['image_paths']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get usage examples."""
        return [
            {
                'query': 'Group my vacation photos by event',
                'params': {
                    'image_paths': ['photo1.jpg', 'photo2.jpg', '...'],
                    'method': 'dbscan',
                    'feature_type': 'dinov2',
                    'min_cluster_size': 5
                },
                'description': 'Clusters vacation photos into events using DBSCAN with DINOv2 features'
            },
            {
                'query': 'Find similar photos and group them',
                'params': {
                    'image_paths': ['...'],
                    'method': 'hdbscan',
                    'feature_type': 'openclip'
                },
                'description': 'Groups similar photos using HDBSCAN with semantic CLIP features'
            },
            {
                'query': 'Organize photos into 8 categories',
                'params': {
                    'image_paths': ['...'],
                    'method': 'kmeans',
                    'n_clusters': 8
                },
                'description': 'Creates exactly 8 clusters using KMeans'
            }
        ]


class FindSimilarImagesTool(BaseTool):
    """Find images similar to a reference image."""

    def execute(
        self,
        reference_image: str,
        candidate_images: List[str],
        feature_type: str = 'dinov2',
        top_k: int = 10,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Find similar images to a reference.

        Args:
            reference_image: Path to reference image
            candidate_images: Paths to candidate images
            feature_type: Feature extractor
            top_k: Number of similar images to return
            threshold: Optional similarity threshold

        Returns:
            Dictionary with similar images and similarity scores
        """
        logger.info(f"Finding images similar to {Path(reference_image).name}")

        # Load similarity method
        feature_config = {'method': feature_type}
        method = load_method(feature_type, feature_config)

        # Extract features
        all_images = [reference_image] + candidate_images
        features = method.extract_features(all_images)

        # Compute similarities
        ref_feature = features[0:1]
        candidate_features = features[1:]

        from sim_bench.strategies import CosineDistanceStrategy
        strategy = CosineDistanceStrategy()

        distances = strategy.compute_distances(ref_feature, candidate_features)[0]

        # Convert distances to similarities
        similarities = 1 - distances

        # Rank by similarity
        ranked_indices = similarities.argsort()[::-1]

        # Apply threshold if provided
        if threshold is not None:
            ranked_indices = [
                idx for idx in ranked_indices
                if similarities[idx] >= threshold
            ]

        # Get top-k
        top_indices = ranked_indices[:top_k]

        similar_images = []
        for idx in top_indices:
            similar_images.append({
                'path': candidate_images[idx],
                'similarity': float(similarities[idx])
            })

        return {
            'success': True,
            'data': {
                'reference_image': reference_image,
                'similar_images': similar_images,
                'num_found': len(similar_images)
            },
            'message': f"Found {len(similar_images)} similar images",
            'metadata': {
                'feature_type': feature_type,
                'top_k': top_k,
                'threshold': threshold
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'find_similar_images',
            'description': 'Find images visually similar to a reference image',
            'category': ToolCategory.SIMILARITY,
            'parameters': {
                'type': 'object',
                'properties': {
                    'reference_image': {
                        'type': 'string',
                        'description': 'Path to reference image'
                    },
                    'candidate_images': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Paths to candidate images to search'
                    },
                    'feature_type': {
                        'type': 'string',
                        'enum': ['dinov2', 'openclip', 'deep'],
                        'default': 'dinov2'
                    },
                    'top_k': {
                        'type': 'integer',
                        'description': 'Number of similar images to return',
                        'default': 10
                    },
                    'threshold': {
                        'type': 'number',
                        'description': 'Minimum similarity threshold (0-1)',
                        'default': None
                    }
                },
                'required': ['reference_image', 'candidate_images']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get examples."""
        return [
            {
                'query': 'Find photos similar to this one',
                'params': {
                    'reference_image': 'favorite.jpg',
                    'candidate_images': ['photo1.jpg', '...'],
                    'top_k': 5
                },
                'description': 'Finds 5 most similar images'
            }
        ]
