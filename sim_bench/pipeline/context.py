"""Pipeline context - shared state passed through all steps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim_bench.pipeline.cache_handler import UniversalCacheHandler


@dataclass
class PipelineContext:
    """Shared mutable container that all steps read from and write to."""

    # Input
    source_directory: Path = None

    # Discovery
    image_paths: list[Path] = field(default_factory=list)

    # Analysis scores (keyed by image path string)
    iqa_scores: dict[str, float] = field(default_factory=dict)
    ava_scores: dict[str, float] = field(default_factory=dict)
    sharpness_scores: dict[str, float] = field(default_factory=dict)

    # Face-specific (keyed by image path string)
    faces: dict[str, list] = field(default_factory=dict)
    face_pose_scores: dict[str, list[float]] = field(default_factory=dict)
    face_eyes_scores: dict[str, list[float]] = field(default_factory=dict)
    face_smile_scores: dict[str, list[float]] = field(default_factory=dict)
    is_face_dominant: dict[str, bool] = field(default_factory=dict)

    # InsightFace pipeline data
    persons: dict[str, dict] = field(default_factory=dict)
    insightface_faces: dict[str, dict] = field(default_factory=dict)

    # Embeddings
    scene_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    face_embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # Filtering results
    quality_passed: set[str] = field(default_factory=set)
    portrait_passed: set[str] = field(default_factory=set)
    active_images: set[str] = field(default_factory=set)

    # Scene clustering
    scene_clusters: dict[int, list[str]] = field(default_factory=dict)
    scene_cluster_labels: dict[str, int] = field(default_factory=dict)

    # Face clustering (within scenes)
    face_clusters: dict[int, dict[int, list[str]]] = field(default_factory=dict)

    # People feature (global face clustering)
    all_faces: list = field(default_factory=list)
    all_face_embeddings: np.ndarray = None
    people_clusters: dict[int, list] = field(default_factory=dict)
    people_thumbnails: dict[int, Any] = field(default_factory=dict)
    people_best_images: dict[int, dict] = field(default_factory=dict)

    # Identity refinement outputs
    refined_people_clusters: dict[int, list] = field(default_factory=dict)
    unassigned_faces: list = field(default_factory=list)
    cluster_exemplars: dict[int, list] = field(default_factory=dict)
    cluster_centroids: dict[int, np.ndarray] = field(default_factory=dict)
    attachment_decisions: dict[str, dict] = field(default_factory=dict)

    # User overrides (loaded from DB before refinement)
    user_overrides: list = field(default_factory=list)

    # Composite scores (keyed by image path string, computed during select_best)
    composite_scores: dict[str, float] = field(default_factory=dict)

    # Siamese comparison log (list of comparison results for debugging/display)
    # Each entry: {cluster_id, img1, img2, winner, confidence, comparison_type}
    siamese_comparisons: list[dict] = field(default_factory=list)

    # Selection
    selected_images: list[str] = field(default_factory=list)

    # Progress callback
    on_progress: Callable[[str, float, str], None] = None

    # Step configurations (set by pipeline builder)
    step_configs: dict[str, dict] = field(default_factory=dict)
    
    # Cache handler (for persistent feature caching)
    cache_handler: Optional['UniversalCacheHandler'] = None

    def report_progress(self, step_name: str, progress: float, message: str = "") -> None:
        """Report progress if callback is set."""
        if self.on_progress is not None:
            self.on_progress(step_name, progress, message)

    def get_active_image_paths(self) -> list[str]:
        """Get list of images that passed all filters."""
        if self.active_images:
            return list(self.active_images)
        return [str(p) for p in self.image_paths]

    def get_image_score(self, image_path: str, weights: dict = None) -> float:
        """Calculate composite score for an image."""
        if weights is None:
            weights = {"iqa": 0.4, "ava": 0.6}

        score = 0.0
        total_weight = 0.0

        if image_path in self.iqa_scores and "iqa" in weights:
            score += self.iqa_scores[image_path] * weights["iqa"]
            total_weight += weights["iqa"]

        if image_path in self.ava_scores and "ava" in weights:
            # AVA scores are already normalized to 0-1 at storage time
            score += self.ava_scores[image_path] * weights["ava"]
            total_weight += weights["ava"]

        if total_weight > 0:
            return score / total_weight
        return 0.0
