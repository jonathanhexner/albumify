"""
Face detection and recognition tools for photo analysis.

Provides tools for detecting faces, extracting embeddings, and grouping photos by people.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

from sim_bench.agent.tools.base import BaseTool, ToolCategory
from sim_bench.photo_analysis.factory import create_photo_analyzer

logger = logging.getLogger(__name__)


class DetectFacesTool(BaseTool):
    """
    Detect faces in images and extract face embeddings.

    Uses DeepFace for face detection and embedding extraction.
    Can be used to identify photos with people, count faces, or group by person.
    """

    def setup(self):
        """Initialize face detector (lazy loaded)."""
        self.face_detector = None

    def execute(
        self,
        image_paths: List[str],
        backend: str = 'retinaface',
        device: str = 'cpu',
        min_confidence: float = 0.9
    ) -> Dict[str, Any]:
        """
        Detect faces in images.

        Args:
            image_paths: List of image paths
            backend: Face detection backend (retinaface, opencv, ssd, dlib, mtcnn)
            device: Device (cpu/cuda)
            min_confidence: Minimum detection confidence

        Returns:
            {
                'success': True,
                'data': {
                    'faces': {
                        image_path: {
                            'num_faces': int,
                            'faces': [
                                {
                                    'bbox': [x, y, w, h],
                                    'confidence': float,
                                    'embedding': [...]
                                }
                            ]
                        }
                    },
                    'summary': {
                        'total_images': int,
                        'images_with_faces': int,
                        'total_faces': int,
                        'avg_faces_per_image': float
                    }
                },
                'message': str,
                'metadata': {...}
            }
        """
        logger.info(f"Detecting faces in {len(image_paths)} images")

        # Load face detector
        if self.face_detector is None:
            self.face_detector = create_photo_analyzer(
                analyzer_type='face',
                config={
                    'backend': backend,
                    'device': device
                }
            )

        # Analyze images
        face_results = {}
        total_faces = 0
        images_with_faces = 0

        for img_path in image_paths:
            try:
                result = self.face_detector.analyze_single(
                    img_path,
                    min_confidence=min_confidence
                )

                num_faces = result.get('num_faces', 0)
                face_results[img_path] = result

                if num_faces > 0:
                    images_with_faces += 1
                    total_faces += num_faces

            except Exception as e:
                logger.warning(f"Failed to process {Path(img_path).name}: {e}")
                face_results[img_path] = {
                    'num_faces': 0,
                    'faces': [],
                    'error': str(e)
                }

        avg_faces = total_faces / len(image_paths) if image_paths else 0

        summary = {
            'total_images': len(image_paths),
            'images_with_faces': images_with_faces,
            'total_faces': total_faces,
            'avg_faces_per_image': avg_faces
        }

        message = (
            f"Detected {total_faces} faces in {images_with_faces}/{len(image_paths)} images. "
            f"Average: {avg_faces:.1f} faces per image."
        )

        return {
            'success': True,
            'data': {
                'faces': face_results,
                'summary': summary
            },
            'message': message,
            'metadata': {
                'backend': backend,
                'device': device,
                'min_confidence': min_confidence
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'detect_faces',
            'description': 'Detect faces in images and extract embeddings for face recognition',
            'category': ToolCategory.ANALYSIS,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of image paths to analyze'
                    },
                    'backend': {
                        'type': 'string',
                        'enum': ['retinaface', 'opencv', 'ssd', 'dlib', 'mtcnn'],
                        'description': 'Face detection backend',
                        'default': 'retinaface'
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
                        'default': 0.9
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
                'query': 'Find all photos with people',
                'params': {
                    'image_paths': ['photo1.jpg', 'photo2.jpg', '...'],
                    'backend': 'retinaface'
                },
                'description': 'Detects faces to identify photos containing people'
            },
            {
                'query': 'Detect faces for grouping by person',
                'params': {
                    'image_paths': ['...'],
                    'min_confidence': 0.95
                },
                'description': 'High-confidence face detection for person grouping'
            }
        ]


class GroupByPersonTool(BaseTool):
    """
    Group photos by the people in them using face embeddings.

    Uses face similarity to cluster photos that contain the same person.
    """

    def execute(
        self,
        face_results: Dict,
        similarity_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Group photos by person using face embeddings.

        Args:
            face_results: Output from detect_faces tool
            similarity_threshold: Face similarity threshold (0-1)

        Returns:
            Dictionary with person groups
        """
        logger.info("Grouping photos by person")

        # Extract all embeddings with metadata
        all_faces = []
        for img_path, result in face_results.items():
            for i, face in enumerate(result.get('faces', [])):
                if 'embedding' in face:
                    all_faces.append({
                        'image_path': img_path,
                        'face_idx': i,
                        'embedding': face['embedding'],
                        'bbox': face.get('bbox')
                    })

        if not all_faces:
            return {
                'success': True,
                'data': {
                    'person_groups': {},
                    'num_persons': 0,
                    'ungrouped': list(face_results.keys())
                },
                'message': "No faces detected for grouping",
                'metadata': {}
            }

        # Simple clustering by embedding similarity
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = np.array([f['embedding'] for f in all_faces])
        similarity_matrix = cosine_similarity(embeddings)

        # Greedy clustering
        person_groups = []
        assigned = set()

        for i, face in enumerate(all_faces):
            if i in assigned:
                continue

            # Start new group
            group = [i]
            assigned.add(i)

            # Find similar faces
            for j in range(i + 1, len(all_faces)):
                if j in assigned:
                    continue

                if similarity_matrix[i, j] >= similarity_threshold:
                    group.append(j)
                    assigned.add(j)

            person_groups.append(group)

        # Organize by image paths
        person_photos = {}
        for person_id, group in enumerate(person_groups):
            photo_set = set()
            for face_idx in group:
                photo_set.add(all_faces[face_idx]['image_path'])
            person_photos[f"person_{person_id}"] = list(photo_set)

        message = f"Found {len(person_groups)} different people across {len(face_results)} images"

        return {
            'success': True,
            'data': {
                'person_groups': person_photos,
                'num_persons': len(person_groups),
                'faces_per_person': {
                    f"person_{i}": len(group)
                    for i, group in enumerate(person_groups)
                }
            },
            'message': message,
            'metadata': {
                'similarity_threshold': similarity_threshold,
                'total_faces': len(all_faces)
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'group_by_person',
            'description': 'Group photos by the people in them using face recognition',
            'category': ToolCategory.ORGANIZATION,
            'parameters': {
                'type': 'object',
                'properties': {
                    'face_results': {
                        'type': 'object',
                        'description': 'Face detection results from detect_faces tool'
                    },
                    'similarity_threshold': {
                        'type': 'number',
                        'description': 'Face similarity threshold (0-1)',
                        'default': 0.6
                    }
                },
                'required': ['face_results']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get usage examples."""
        return [
            {
                'query': 'Group my photos by who is in them',
                'params': {
                    'face_results': {'...': {...}},
                    'similarity_threshold': 0.6
                },
                'description': 'Groups photos by person using face similarity'
            }
        ]


class FilterByFacesTool(BaseTool):
    """Filter images based on face detection criteria."""

    def execute(
        self,
        face_results: Dict,
        min_faces: int = 1,
        max_faces: int = None,
        require_faces: bool = True
    ) -> Dict[str, Any]:
        """
        Filter images by face count.

        Args:
            face_results: Output from detect_faces tool
            min_faces: Minimum number of faces
            max_faces: Maximum number of faces (None for unlimited)
            require_faces: If True, only include images with faces

        Returns:
            Filtered list of image paths
        """
        logger.info(f"Filtering images by face count (min={min_faces}, max={max_faces})")

        filtered = []

        for img_path, result in face_results.items():
            num_faces = result.get('num_faces', 0)

            # Apply filters
            if require_faces and num_faces == 0:
                continue

            if num_faces < min_faces:
                continue

            if max_faces is not None and num_faces > max_faces:
                continue

            filtered.append(img_path)

        message = f"Filtered to {len(filtered)} images from {len(face_results)} (face count: {min_faces}-{max_faces or 'unlimited'})"

        return {
            'success': True,
            'data': {
                'filtered_images': filtered,
                'num_filtered': len(filtered),
                'num_original': len(face_results)
            },
            'message': message,
            'metadata': {
                'min_faces': min_faces,
                'max_faces': max_faces,
                'require_faces': require_faces
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema."""
        return {
            'name': 'filter_by_faces',
            'description': 'Filter images based on number of faces detected',
            'category': ToolCategory.ANALYSIS,
            'parameters': {
                'type': 'object',
                'properties': {
                    'face_results': {
                        'type': 'object',
                        'description': 'Face detection results from detect_faces tool'
                    },
                    'min_faces': {
                        'type': 'integer',
                        'description': 'Minimum number of faces',
                        'default': 1
                    },
                    'max_faces': {
                        'type': 'integer',
                        'description': 'Maximum number of faces (null for unlimited)',
                        'default': None
                    },
                    'require_faces': {
                        'type': 'boolean',
                        'description': 'Only include images with faces',
                        'default': True
                    }
                },
                'required': ['face_results']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get usage examples."""
        return [
            {
                'query': 'Show me only portrait photos (single person)',
                'params': {
                    'face_results': {'...': {...}},
                    'min_faces': 1,
                    'max_faces': 1
                },
                'description': 'Filters for photos with exactly one person'
            },
            {
                'query': 'Find group photos',
                'params': {
                    'face_results': {'...': {...}},
                    'min_faces': 3
                },
                'description': 'Filters for photos with 3 or more people'
            }
        ]
