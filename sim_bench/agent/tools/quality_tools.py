"""
Quality assessment tools for evaluating and selecting best photos.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging
import numpy as np

from sim_bench.agent.tools.base import BaseTool, ToolCategory
from sim_bench.quality_assessment.registry import create_quality_assessor

logger = logging.getLogger(__name__)


class AssessQualityBatchTool(BaseTool):
    """
    Assess quality of multiple images using various methods.
    """

    def setup(self):
        """Initialize quality assessor (lazy loaded)."""
        self.assessor = None

    def execute(
        self,
        image_paths: List[str],
        method: str = 'sharpness_only',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Assess quality of multiple images.

        Args:
            image_paths: List of image paths
            method: Quality assessment method (rule_based, nima_mobilenet,
                   vit_base, clip_aesthetic, clip_learned, sharpness_only, etc.)
            **kwargs: Method-specific parameters

        Returns:
            {
                'success': True,
                'data': {
                    'scores': {image_path: score},
                    'ranked_images': [(path, score), ...],  # Best first
                    'statistics': {...}
                },
                'message': str,
                'metadata': {...}
            }
        """
        logger.info(f"Assessing quality of {len(image_paths)} images with {method}")

        # Load quality method using registry
        config = {'type': method}
        config.update(kwargs)
        self.assessor = create_quality_assessor(config)

        # Assess all images
        scores = {}
        for img_path in image_paths:
            try:
                score = self.assessor.assess_image(img_path)
                scores[img_path] = float(score)
            except Exception as e:
                logger.warning(f"Failed to assess {img_path}: {e}")
                scores[img_path] = 0.0

        # Rank by score (highest first)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Calculate statistics
        score_values = list(scores.values())
        statistics = {
            'mean': float(np.mean(score_values)),
            'std': float(np.std(score_values)),
            'min': float(np.min(score_values)),
            'max': float(np.max(score_values)),
            'median': float(np.median(score_values))
        }

        return {
            'success': True,
            'data': {
                'scores': scores,
                'ranked_images': ranked,
                'statistics': statistics,
                'num_images': len(image_paths)
            },
            'message': f"Assessed {len(image_paths)} images. Top score: {ranked[0][1]:.3f}",
            'metadata': {
                'method': method,
                'params': kwargs
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'assess_quality_batch',
            'description': 'Assess image quality for multiple images using various methods',
            'category': ToolCategory.QUALITY,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of image paths to assess'
                    },
                    'method': {
                        'type': 'string',
                        'enum': [
                            'sharpness_only', 'contrast_only', 'exposure_only',
                            'rule_based', 'nima_mobilenet', 'nima_resnet50',
                            'vit_base', 'clip_aesthetic', 'clip_learned'
                        ],
                        'description': 'Quality assessment method',
                        'default': 'sharpness_only'
                    },
                    'device': {
                        'type': 'string',
                        'enum': ['cpu', 'cuda'],
                        'description': 'Device for deep learning methods',
                        'default': 'cpu'
                    }
                },
                'required': ['image_paths']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get examples."""
        return [
            {
                'query': 'Rate the quality of these photos',
                'params': {
                    'image_paths': ['photo1.jpg', 'photo2.jpg', '...'],
                    'method': 'sharpness_only'
                },
                'description': 'Assesses quality using sharpness (fast, accurate)'
            },
            {
                'query': 'Use AI to evaluate photo quality',
                'params': {
                    'image_paths': ['...'],
                    'method': 'clip_learned',
                    'device': 'cpu'
                },
                'description': 'Uses learned CLIP prompts for quality assessment'
            }
        ]


class SelectBestFromGroupTool(BaseTool):
    """
    Select best N images from each group.
    """

    def execute(
        self,
        groups: Dict[Any, List[str]],
        quality_scores: Dict[str, float],
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Select best N images from each group based on quality scores.

        Args:
            groups: Dictionary mapping group_id -> list of image paths
            quality_scores: Dictionary mapping image_path -> quality score
            top_n: Number of best images to select per group

        Returns:
            {
                'success': True,
                'data': {
                    'selected': {group_id: [(path, score), ...]},
                    'total_selected': int,
                    'selection_stats': {...}
                },
                'message': str,
                'metadata': {...}
            }
        """
        logger.info(f"Selecting top {top_n} from {len(groups)} groups")

        selected = {}
        total_selected = 0

        for group_id, image_paths in groups.items():
            # Get scores for this group's images
            group_scores = [
                (path, quality_scores.get(path, 0.0))
                for path in image_paths
            ]

            # Sort by score (highest first)
            group_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top N
            selected[group_id] = group_scores[:top_n]
            total_selected += len(selected[group_id])

        # Calculate statistics
        selection_stats = {
            'num_groups': len(groups),
            'total_selected': total_selected,
            'avg_per_group': total_selected / len(groups) if groups else 0,
            'top_n': top_n
        }

        return {
            'success': True,
            'data': {
                'selected': selected,
                'total_selected': total_selected,
                'selection_stats': selection_stats
            },
            'message': f"Selected {total_selected} best images from {len(groups)} groups",
            'metadata': {
                'top_n': top_n
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'select_best_from_group',
            'description': 'Select top N best images from each group based on quality scores',
            'category': ToolCategory.QUALITY,
            'parameters': {
                'type': 'object',
                'properties': {
                    'groups': {
                        'type': 'object',
                        'description': 'Dictionary mapping group_id to list of image paths'
                    },
                    'quality_scores': {
                        'type': 'object',
                        'description': 'Dictionary mapping image_path to quality score'
                    },
                    'top_n': {
                        'type': 'integer',
                        'description': 'Number of best images to select per group',
                        'default': 3
                    }
                },
                'required': ['groups', 'quality_scores']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get examples."""
        return [
            {
                'query': 'Pick the 3 best from each event',
                'params': {
                    'groups': {0: ['img1.jpg', 'img2.jpg'], 1: ['...']},
                    'quality_scores': {'img1.jpg': 0.8, 'img2.jpg': 0.6},
                    'top_n': 3
                },
                'description': 'Selects top 3 images from each group by quality'
            }
        ]


class RankImagesTool(BaseTool):
    """Rank images by quality score."""

    def execute(
        self,
        image_paths: List[str],
        quality_scores: Dict[str, float],
        descending: bool = True
    ) -> Dict[str, Any]:
        """
        Rank images by quality score.

        Args:
            image_paths: List of image paths to rank
            quality_scores: Quality scores for images
            descending: If True, rank best first

        Returns:
            Ranked list of images with scores
        """
        # Create (path, score) tuples
        scored_images = [
            (path, quality_scores.get(path, 0.0))
            for path in image_paths
        ]

        # Sort
        scored_images.sort(key=lambda x: x[1], reverse=descending)

        # Add ranks
        ranked = [
            {
                'rank': i + 1,
                'path': path,
                'score': score,
                'filename': Path(path).name
            }
            for i, (path, score) in enumerate(scored_images)
        ]

        return {
            'success': True,
            'data': {
                'ranked_images': ranked,
                'num_images': len(ranked)
            },
            'message': f"Ranked {len(ranked)} images by quality",
            'metadata': {
                'descending': descending
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'rank_images',
            'description': 'Rank images by quality score (best first)',
            'category': ToolCategory.QUALITY,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Images to rank'
                    },
                    'quality_scores': {
                        'type': 'object',
                        'description': 'Quality scores for images'
                    },
                    'descending': {
                        'type': 'boolean',
                        'description': 'If true, rank best first',
                        'default': True
                    }
                },
                'required': ['image_paths', 'quality_scores']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get examples."""
        return [
            {
                'query': 'Show me my photos ranked by quality',
                'params': {
                    'image_paths': ['img1.jpg', '...'],
                    'quality_scores': {'img1.jpg': 0.8, '...': 0.6}
                },
                'description': 'Ranks photos from best to worst'
            }
        ]
