"""
Photo analysis tools for tagging, face detection, and content understanding.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

from sim_bench.agent.tools.base import BaseTool, ToolCategory
from sim_bench.photo_analysis.factory import create_photo_analyzer

logger = logging.getLogger(__name__)


class CLIPTagImagesTool(BaseTool):
    """
    Tag images using CLIP zero-shot classification.

    Uses 57 prompts covering: people, activities, locations, lighting, etc.
    """

    def setup(self):
        """Initialize CLIP tagger."""
        self.tagger = None

    def execute(
        self,
        image_paths: List[str],
        batch_size: int = 8,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Tag images with CLIP.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            device: Device (cpu/cuda)

        Returns:
            {
                'success': True,
                'data': {
                    'tags': {image_path: {
                        'tags': [...],
                        'confidences': [...],
                        'routing': {...},
                        'importance_score': float
                    }},
                    'summary': {
                        'common_tags': [...],
                        'num_with_people': int,
                        'num_with_landmarks': int
                    }
                },
                'message': str,
                'metadata': {...}
            }
        """
        logger.info(f"Tagging {len(image_paths)} images with CLIP")

        # Load CLIP tagger
        if self.tagger is None:
            self.tagger = create_photo_analyzer(
                analyzer_type='clip',
                config={'device': device}
            )

        # Analyze images
        results = self.tagger.analyze_batch(
            image_paths,
            batch_size=batch_size,
            verbose=False
        )

        # Extract summary statistics
        num_with_people = sum(
            1 for r in results.values()
            if r.get('routing', {}).get('needs_face_detection', False)
        )

        num_with_landmarks = sum(
            1 for r in results.values()
            if r.get('routing', {}).get('needs_landmark_detection', False)
        )

        # Find common tags
        all_tags = []
        for result in results.values():
            all_tags.extend(result.get('tags', []))

        from collections import Counter
        tag_counts = Counter(all_tags)
        common_tags = [tag for tag, count in tag_counts.most_common(10)]

        summary = {
            'common_tags': common_tags,
            'num_with_people': num_with_people,
            'num_with_landmarks': num_with_landmarks,
            'total_images': len(results)
        }

        return {
            'success': True,
            'data': {
                'tags': results,
                'summary': summary
            },
            'message': f"Tagged {len(results)} images. Common tags: {', '.join(common_tags[:3])}",
            'metadata': {
                'device': device,
                'batch_size': batch_size
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'clip_tag_images',
            'description': 'Tag images with zero-shot CLIP classification (people, places, activities, etc.)',
            'category': ToolCategory.ANALYSIS,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of image paths to tag'
                    },
                    'batch_size': {
                        'type': 'integer',
                        'description': 'Batch size for processing',
                        'default': 8
                    },
                    'device': {
                        'type': 'string',
                        'enum': ['cpu', 'cuda'],
                        'description': 'Device to use',
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
                'query': 'Analyze what is in these photos',
                'params': {
                    'image_paths': ['photo1.jpg', 'photo2.jpg', '...'],
                    'batch_size': 8
                },
                'description': 'Tags images with content descriptors'
            },
            {
                'query': 'Find all photos with people',
                'params': {
                    'image_paths': ['...']
                },
                'description': 'Tags images and identifies those with people'
            }
        ]


class FilterByTagsTool(BaseTool):
    """Filter images based on CLIP tags."""

    def execute(
        self,
        image_tags: Dict[str, Dict],
        required_tags: List[str] = None,
        excluded_tags: List[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Filter images by tags.

        Args:
            image_tags: Results from CLIP tagging
            required_tags: Tags that must be present
            excluded_tags: Tags that must NOT be present
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of image paths
        """
        required_tags = required_tags or []
        excluded_tags = excluded_tags or []

        filtered_images = []

        for img_path, tag_data in image_tags.items():
            tags = tag_data.get('tags', [])
            confidences = tag_data.get('confidences', {})

            # Check required tags
            has_required = all(
                tag in tags and confidences.get(tag, 0) >= min_confidence
                for tag in required_tags
            )

            # Check excluded tags
            has_excluded = any(tag in tags for tag in excluded_tags)

            if has_required and not has_excluded:
                filtered_images.append(img_path)

        return {
            'success': True,
            'data': {
                'filtered_images': filtered_images,
                'num_filtered': len(filtered_images),
                'num_original': len(image_tags)
            },
            'message': f"Filtered to {len(filtered_images)} images from {len(image_tags)}",
            'metadata': {
                'required_tags': required_tags,
                'excluded_tags': excluded_tags,
                'min_confidence': min_confidence
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'filter_by_tags',
            'description': 'Filter images based on presence/absence of tags',
            'category': ToolCategory.ANALYSIS,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_tags': {
                        'type': 'object',
                        'description': 'Tag results from CLIP tagging'
                    },
                    'required_tags': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Tags that must be present',
                        'default': []
                    },
                    'excluded_tags': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Tags to exclude',
                        'default': []
                    },
                    'min_confidence': {
                        'type': 'number',
                        'description': 'Minimum tag confidence',
                        'default': 0.5
                    }
                },
                'required': ['image_tags']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get examples."""
        return [
            {
                'query': 'Show me only landscape photos',
                'params': {
                    'image_tags': {'...': {'tags': [...]}},
                    'required_tags': ['landscape', 'outdoor']
                },
                'description': 'Filters for landscape/outdoor photos'
            },
            {
                'query': 'Find portraits without groups',
                'params': {
                    'image_tags': {'...': {...}},
                    'required_tags': ['person'],
                    'excluded_tags': ['group']
                },
                'description': 'Filters for individual portraits'
            }
        ]
