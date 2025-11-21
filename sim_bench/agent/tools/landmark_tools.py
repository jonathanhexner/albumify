"""
Landmark and place recognition tools for photo organization.

Provides tools for identifying landmarks and grouping photos by location.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

from sim_bench.agent.tools.base import BaseTool, ToolCategory
from sim_bench.photo_analysis.factory import create_photo_analyzer

logger = logging.getLogger(__name__)


class DetectLandmarksTool(BaseTool):
    """
    Detect landmarks and places in images.

    Uses vision models to identify famous landmarks, monuments, and locations.
    """

    def setup(self):
        """Initialize landmark detector (lazy loaded)."""
        self.landmark_detector = None

    def execute(
        self,
        image_paths: List[str],
        device: str = 'cpu',
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect landmarks in images.

        Args:
            image_paths: List of image paths
            device: Device (cpu/cuda)
            min_confidence: Minimum detection confidence

        Returns:
            {
                'success': True,
                'data': {
                    'landmarks': {
                        image_path: {
                            'detected': bool,
                            'landmark': str,
                            'confidence': float,
                            'location': str
                        }
                    },
                    'summary': {
                        'total_images': int,
                        'images_with_landmarks': int,
                        'unique_landmarks': [str, ...],
                        'locations': {location: count}
                    }
                },
                'message': str,
                'metadata': {...}
            }
        """
        logger.info(f"Detecting landmarks in {len(image_paths)} images")

        # Load landmark detector
        if self.landmark_detector is None:
            self.landmark_detector = create_photo_analyzer(
                analyzer_type='landmark',
                config={'device': device}
            )

        # Analyze images
        landmark_results = {}
        images_with_landmarks = 0
        all_landmarks = []
        location_counts = {}

        for img_path in image_paths:
            try:
                result = self.landmark_detector.analyze_single(
                    img_path,
                    min_confidence=min_confidence
                )

                landmark_results[img_path] = result

                if result.get('detected', False):
                    images_with_landmarks += 1
                    landmark_name = result.get('landmark', 'Unknown')
                    location = result.get('location', 'Unknown')

                    all_landmarks.append(landmark_name)
                    location_counts[location] = location_counts.get(location, 0) + 1

            except Exception as e:
                logger.warning(f"Failed to process {Path(img_path).name}: {e}")
                landmark_results[img_path] = {
                    'detected': False,
                    'error': str(e)
                }

        unique_landmarks = list(set(all_landmarks))

        summary = {
            'total_images': len(image_paths),
            'images_with_landmarks': images_with_landmarks,
            'unique_landmarks': unique_landmarks,
            'locations': location_counts
        }

        message = (
            f"Detected landmarks in {images_with_landmarks}/{len(image_paths)} images. "
            f"Found {len(unique_landmarks)} unique landmarks."
        )

        return {
            'success': True,
            'data': {
                'landmarks': landmark_results,
                'summary': summary
            },
            'message': message,
            'metadata': {
                'device': device,
                'min_confidence': min_confidence
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'detect_landmarks',
            'description': 'Detect landmarks and famous places in images',
            'category': ToolCategory.ANALYSIS,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of image paths to analyze'
                    },
                    'device': {
                        'type': 'string',
                        'enum': ['cpu', 'cuda'],
                        'description': 'Device to use',
                        'default': 'cpu'
                    },
                    'min_confidence': {
                        'type': 'number',
                        'description': 'Minimum detection confidence',
                        'default': 0.5
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
                'query': 'Find photos of famous landmarks',
                'params': {
                    'image_paths': ['photo1.jpg', 'photo2.jpg', '...'],
                    'min_confidence': 0.5
                },
                'description': 'Detects famous landmarks in vacation photos'
            },
            {
                'query': 'Identify places in my travel photos',
                'params': {
                    'image_paths': ['...']
                },
                'description': 'Identifies landmarks and locations'
            }
        ]


class GroupByLocationTool(BaseTool):
    """Group photos by detected landmarks/locations."""

    def execute(
        self,
        landmark_results: Dict,
        group_by: str = 'landmark'
    ) -> Dict[str, Any]:
        """
        Group photos by landmark or location.

        Args:
            landmark_results: Output from detect_landmarks tool
            group_by: Group by 'landmark' or 'location'

        Returns:
            Dictionary with location groups
        """
        logger.info(f"Grouping photos by {group_by}")

        groups = {}

        for img_path, result in landmark_results.items():
            if not result.get('detected', False):
                groups.setdefault('Unknown', []).append(img_path)
                continue

            if group_by == 'landmark':
                key = result.get('landmark', 'Unknown')
            else:  # location
                key = result.get('location', 'Unknown')

            groups.setdefault(key, []).append(img_path)

        message = f"Grouped {len(landmark_results)} images into {len(groups)} {group_by} groups"

        return {
            'success': True,
            'data': {
                'groups': groups,
                'num_groups': len(groups),
                'group_sizes': {k: len(v) for k, v in groups.items()}
            },
            'message': message,
            'metadata': {
                'group_by': group_by
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'group_by_location',
            'description': 'Group photos by detected landmarks or locations',
            'category': ToolCategory.ORGANIZATION,
            'parameters': {
                'type': 'object',
                'properties': {
                    'landmark_results': {
                        'type': 'object',
                        'description': 'Landmark detection results from detect_landmarks tool'
                    },
                    'group_by': {
                        'type': 'string',
                        'enum': ['landmark', 'location'],
                        'description': 'Group by specific landmark or general location',
                        'default': 'landmark'
                    }
                },
                'required': ['landmark_results']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get usage examples."""
        return [
            {
                'query': 'Organize my travel photos by place',
                'params': {
                    'landmark_results': {'...': {...}},
                    'group_by': 'landmark'
                },
                'description': 'Groups photos by specific landmarks visited'
            },
            {
                'query': 'Group photos by city/country',
                'params': {
                    'landmark_results': {'...': {...}},
                    'group_by': 'location'
                },
                'description': 'Groups photos by general location'
            }
        ]


class FilterByLandmarksTool(BaseTool):
    """Filter images based on landmark detection."""

    def execute(
        self,
        landmark_results: Dict,
        require_landmark: bool = True,
        specific_landmarks: List[str] = None
    ) -> Dict[str, Any]:
        """
        Filter images by landmark detection.

        Args:
            landmark_results: Output from detect_landmarks tool
            require_landmark: Only include images with detected landmarks
            specific_landmarks: List of specific landmarks to filter for

        Returns:
            Filtered list of image paths
        """
        specific_landmarks = specific_landmarks or []

        logger.info(f"Filtering images by landmarks (require={require_landmark})")

        filtered = []

        for img_path, result in landmark_results.items():
            has_landmark = result.get('detected', False)

            # Check if landmark is required
            if require_landmark and not has_landmark:
                continue

            # Check for specific landmarks
            if specific_landmarks:
                landmark_name = result.get('landmark', '')
                if landmark_name not in specific_landmarks:
                    continue

            filtered.append(img_path)

        message = f"Filtered to {len(filtered)} images from {len(landmark_results)}"

        return {
            'success': True,
            'data': {
                'filtered_images': filtered,
                'num_filtered': len(filtered),
                'num_original': len(landmark_results)
            },
            'message': message,
            'metadata': {
                'require_landmark': require_landmark,
                'specific_landmarks': specific_landmarks
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'filter_by_landmarks',
            'description': 'Filter images based on landmark detection',
            'category': ToolCategory.ANALYSIS,
            'parameters': {
                'type': 'object',
                'properties': {
                    'landmark_results': {
                        'type': 'object',
                        'description': 'Landmark detection results from detect_landmarks tool'
                    },
                    'require_landmark': {
                        'type': 'boolean',
                        'description': 'Only include images with detected landmarks',
                        'default': True
                    },
                    'specific_landmarks': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Filter for specific landmarks',
                        'default': []
                    }
                },
                'required': ['landmark_results']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get usage examples."""
        return [
            {
                'query': 'Show me only photos of landmarks',
                'params': {
                    'landmark_results': {'...': {...}},
                    'require_landmark': True
                },
                'description': 'Filters for images with detected landmarks'
            },
            {
                'query': 'Find photos of the Eiffel Tower',
                'params': {
                    'landmark_results': {'...': {...}},
                    'specific_landmarks': ['Eiffel Tower']
                },
                'description': 'Filters for specific landmark'
            }
        ]
