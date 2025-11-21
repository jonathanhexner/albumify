"""
Pre-defined workflow templates for common photo organization tasks.
"""

from sim_bench.agent.workflows.base import Workflow, WorkflowStep
import logging

logger = logging.getLogger(__name__)


class WorkflowTemplates:
    """Factory for creating common photo organization workflows."""

    @staticmethod
    def organize_by_event(
        top_n_per_event: int = 3,
        min_cluster_size: int = 5,
        quality_method: str = 'rule_based'
    ) -> Workflow:
        """
        Organize photos by event and select best from each.

        Workflow:
        1. Cluster images by visual similarity (events)
        2. Assess quality of all images
        3. Select top N from each event
        4. Return selected images

        Args:
            top_n_per_event: Number of best photos to keep per event
            min_cluster_size: Minimum photos per event cluster
            quality_method: Quality assessment method to use

        Returns:
            Workflow instance
        """
        return Workflow(
            name="organize_by_event",
            description=f"Organize photos into events and select top {top_n_per_event} from each",
            steps=[
                WorkflowStep(
                    name="cluster_by_event",
                    tool_name="cluster_images",
                    params={
                        'method': 'dbscan',
                        'feature_type': 'dinov2',
                        'min_cluster_size': min_cluster_size
                    }
                ),
                WorkflowStep(
                    name="assess_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': quality_method
                    },
                    dependencies=[]  # Can run in parallel with clustering
                ),
                WorkflowStep(
                    name="select_best_per_event",
                    tool_name="select_best_from_group",
                    params={
                        'top_n': top_n_per_event
                    },
                    dependencies=["cluster_by_event", "assess_quality"]
                )
            ]
        )

    @staticmethod
    def find_best_portraits(
        top_n: int = 10,
        min_confidence: float = 0.7
    ) -> Workflow:
        """
        Find and select best portrait photos.

        Workflow:
        1. Tag all images with CLIP
        2. Filter for portraits (person tag)
        3. Assess quality
        4. Rank and return top N

        Args:
            top_n: Number of best portraits to return
            min_confidence: Minimum confidence for person tag

        Returns:
            Workflow instance
        """
        return Workflow(
            name="find_best_portraits",
            description=f"Find and select top {top_n} portrait photos",
            steps=[
                WorkflowStep(
                    name="tag_images",
                    tool_name="clip_tag_images",
                    params={}
                ),
                WorkflowStep(
                    name="filter_portraits",
                    tool_name="filter_by_tags",
                    params={
                        'required_tags': ['person', 'portrait'],
                        'min_confidence': min_confidence
                    },
                    dependencies=["tag_images"]
                ),
                WorkflowStep(
                    name="assess_portrait_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': 'clip_aesthetic'
                    },
                    dependencies=["filter_portraits"]
                ),
                WorkflowStep(
                    name="rank_portraits",
                    tool_name="rank_images",
                    params={},
                    dependencies=["assess_portrait_quality"]
                )
            ]
        )

    @staticmethod
    def find_similar_and_best(
        reference_image: str,
        top_k_similar: int = 20,
        top_n_best: int = 5
    ) -> Workflow:
        """
        Find similar images and select best ones.

        Workflow:
        1. Find images similar to reference
        2. Assess quality of similar images
        3. Return top N by quality

        Args:
            reference_image: Path to reference image
            top_k_similar: Number of similar images to find
            top_n_best: Number of best to select from similar

        Returns:
            Workflow instance
        """
        return Workflow(
            name="find_similar_and_best",
            description=f"Find images similar to reference and select top {top_n_best}",
            steps=[
                WorkflowStep(
                    name="find_similar",
                    tool_name="find_similar_images",
                    params={
                        'reference_image': reference_image,
                        'top_k': top_k_similar
                    }
                ),
                WorkflowStep(
                    name="assess_similar_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': 'rule_based'
                    },
                    dependencies=["find_similar"]
                ),
                WorkflowStep(
                    name="rank_similar",
                    tool_name="rank_images",
                    params={},
                    dependencies=["assess_similar_quality"]
                )
            ]
        )

    @staticmethod
    def smart_selection(
        target_count: int = 50,
        quality_method: str = 'clip_learned',
        cluster_method: str = 'hdbscan'
    ) -> Workflow:
        """
        Smart selection ensuring diversity and quality.

        Workflow:
        1. Cluster for diversity
        2. Assess quality
        3. Select proportionally from clusters based on size
        4. Ensure target count is met

        Args:
            target_count: Target number of photos to select
            quality_method: Quality assessment method
            cluster_method: Clustering method for diversity

        Returns:
            Workflow instance
        """
        return Workflow(
            name="smart_selection",
            description=f"Smart selection of {target_count} diverse, high-quality photos",
            steps=[
                WorkflowStep(
                    name="cluster_for_diversity",
                    tool_name="cluster_images",
                    params={
                        'method': cluster_method,
                        'feature_type': 'dinov2'
                    }
                ),
                WorkflowStep(
                    name="assess_all_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': quality_method
                    },
                    dependencies=[]  # Parallel with clustering
                ),
                WorkflowStep(
                    name="select_diverse_best",
                    tool_name="select_best_from_group",
                    params={
                        'top_n': 3  # Will be adjusted based on cluster sizes
                    },
                    dependencies=["cluster_for_diversity", "assess_all_quality"]
                )
            ]
        )

    @staticmethod
    def organize_vacation_photos(
        quality_method: str = 'rule_based'
    ) -> Workflow:
        """
        Complete vacation photo organization workflow.

        Workflow:
        1. Tag all images
        2. Cluster into events
        3. Assess quality
        4. Select best from each event
        5. Filter landscapes separately
        6. Generate summary

        Args:
            quality_method: Quality assessment method

        Returns:
            Workflow instance
        """
        return Workflow(
            name="organize_vacation_photos",
            description="Complete vacation photo organization with tagging, clustering, and quality selection",
            steps=[
                WorkflowStep(
                    name="tag_all_photos",
                    tool_name="clip_tag_images",
                    params={}
                ),
                WorkflowStep(
                    name="cluster_into_events",
                    tool_name="cluster_images",
                    params={
                        'method': 'dbscan',
                        'feature_type': 'dinov2',
                        'min_cluster_size': 5
                    },
                    dependencies=[]
                ),
                WorkflowStep(
                    name="assess_all_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': quality_method
                    },
                    dependencies=[]
                ),
                WorkflowStep(
                    name="select_best_per_event",
                    tool_name="select_best_from_group",
                    params={
                        'top_n': 3
                    },
                    dependencies=["cluster_into_events", "assess_all_quality"]
                ),
                WorkflowStep(
                    name="filter_landscapes",
                    tool_name="filter_by_tags",
                    params={
                        'required_tags': ['landscape', 'outdoor'],
                        'min_confidence': 0.6
                    },
                    dependencies=["tag_all_photos"]
                )
            ]
        )

    @staticmethod
    def organize_by_people(
        similarity_threshold: float = 0.6,
        quality_method: str = 'rule_based'
    ) -> Workflow:
        """
        Organize photos by the people in them.

        Workflow:
        1. Detect faces in all images
        2. Group photos by person
        3. Assess quality within each person's photos
        4. Select best photos of each person

        Args:
            similarity_threshold: Face similarity threshold
            quality_method: Quality assessment method

        Returns:
            Workflow instance
        """
        return Workflow(
            name="organize_by_people",
            description="Group photos by person and select best of each",
            steps=[
                WorkflowStep(
                    name="detect_faces",
                    tool_name="detect_faces",
                    params={
                        'backend': 'retinaface',
                        'min_confidence': 0.9
                    }
                ),
                WorkflowStep(
                    name="group_by_person",
                    tool_name="group_by_person",
                    params={
                        'similarity_threshold': similarity_threshold
                    },
                    dependencies=["detect_faces"]
                ),
                WorkflowStep(
                    name="assess_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': quality_method
                    },
                    dependencies=[]
                ),
                WorkflowStep(
                    name="select_best_per_person",
                    tool_name="select_best_from_group",
                    params={
                        'top_n': 3
                    },
                    dependencies=["group_by_person", "assess_quality"]
                )
            ]
        )

    @staticmethod
    def organize_travel_photos(
        quality_method: str = 'rule_based'
    ) -> Workflow:
        """
        Complete travel photo organization with landmarks.

        Workflow:
        1. Tag all images with CLIP
        2. Detect landmarks and places
        3. Group by location
        4. Assess quality
        5. Select best from each location

        Args:
            quality_method: Quality assessment method

        Returns:
            Workflow instance
        """
        return Workflow(
            name="organize_travel_photos",
            description="Organize travel photos by landmarks and locations",
            steps=[
                WorkflowStep(
                    name="tag_photos",
                    tool_name="clip_tag_images",
                    params={
                        'batch_size': 8
                    }
                ),
                WorkflowStep(
                    name="detect_landmarks",
                    tool_name="detect_landmarks",
                    params={
                        'min_confidence': 0.5
                    }
                ),
                WorkflowStep(
                    name="group_by_location",
                    tool_name="group_by_location",
                    params={
                        'group_by': 'landmark'
                    },
                    dependencies=["detect_landmarks"]
                ),
                WorkflowStep(
                    name="assess_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': quality_method
                    }
                ),
                WorkflowStep(
                    name="select_best_per_location",
                    tool_name="select_best_from_group",
                    params={
                        'top_n': 5
                    },
                    dependencies=["group_by_location", "assess_quality"]
                )
            ]
        )

    @staticmethod
    def find_group_photos(
        min_people: int = 3,
        quality_method: str = 'rule_based'
    ) -> Workflow:
        """
        Find and select best group photos.

        Workflow:
        1. Detect faces
        2. Filter for group photos (multiple people)
        3. Assess quality
        4. Rank by quality

        Args:
            min_people: Minimum people for group photo
            quality_method: Quality assessment method

        Returns:
            Workflow instance
        """
        return Workflow(
            name="find_group_photos",
            description=f"Find and rank group photos ({min_people}+ people)",
            steps=[
                WorkflowStep(
                    name="detect_faces",
                    tool_name="detect_faces",
                    params={
                        'backend': 'retinaface'
                    }
                ),
                WorkflowStep(
                    name="filter_groups",
                    tool_name="filter_by_faces",
                    params={
                        'min_faces': min_people
                    },
                    dependencies=["detect_faces"]
                ),
                WorkflowStep(
                    name="assess_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': quality_method
                    },
                    dependencies=["filter_groups"]
                ),
                WorkflowStep(
                    name="rank_by_quality",
                    tool_name="rank_images",
                    params={},
                    dependencies=["assess_quality"]
                )
            ]
        )

    @staticmethod
    def list_templates() -> dict:
        """
        Get list of available templates with descriptions.

        Returns:
            Dictionary mapping template name to description
        """
        return {
            'organize_by_event': 'Cluster photos by event and select best from each',
            'find_best_portraits': 'Find and rank portrait photos',
            'find_similar_and_best': 'Find similar images and select highest quality',
            'smart_selection': 'Diverse selection ensuring quality and variety',
            'organize_vacation_photos': 'Complete vacation photo organization workflow',
            'organize_by_people': 'Group photos by person and select best of each',
            'organize_travel_photos': 'Organize travel photos by landmarks and locations',
            'find_group_photos': 'Find and rank group photos'
        }

    @staticmethod
    def get_template(name: str, **kwargs) -> Workflow:
        """
        Get workflow template by name.

        Args:
            name: Template name
            **kwargs: Template-specific parameters

        Returns:
            Workflow instance

        Raises:
            ValueError: If template not found
        """
        templates = {
            'organize_by_event': WorkflowTemplates.organize_by_event,
            'find_best_portraits': WorkflowTemplates.find_best_portraits,
            'find_similar_and_best': WorkflowTemplates.find_similar_and_best,
            'smart_selection': WorkflowTemplates.smart_selection,
            'organize_vacation_photos': WorkflowTemplates.organize_vacation_photos,
            'organize_by_people': WorkflowTemplates.organize_by_people,
            'organize_travel_photos': WorkflowTemplates.organize_travel_photos,
            'find_group_photos': WorkflowTemplates.find_group_photos
        }

        template_fn = templates.get(name)
        if not template_fn:
            raise ValueError(
                f"Unknown template: {name}. "
                f"Available: {', '.join(templates.keys())}"
            )

        return template_fn(**kwargs)
