"""Pipeline context - shared state passed through all steps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim_bench.pipeline.cache_handler import UniversalCacheHandler
    from geo_cluster.types import GeoMetadata

from face_cluster.filter_context import FilterContext


@dataclass
class StepDecision:
    """Record of a decision made by a pipeline step about an image or face.

    Steps emit these as they process items. Stored in DB and returned via API
    so the UI can display them without reimplementing pipeline logic.
    """
    item_id: str          # image path or face key
    item_type: str        # "image" or "face"
    step: str             # pipeline step name (e.g. "filter_quality", "select_best")
    decision: str         # "passed", "rejected", "selected", "detected", etc.
    reason: str           # human-readable, e.g. "IQA 0.08 < threshold 0.20"
    config_used: Dict[str, Any] = field(default_factory=dict)   # actual config for this decision
    metrics: Dict[str, Any] = field(default_factory=dict)       # measured values


@dataclass
class PipelineContext:
    """Shared mutable container that all steps read from and write to."""

    # Input
    source_directory: Path = None

    # Discovery
    image_paths: list[Path] = field(default_factory=list)

    # Geo-temporal metadata (spec-022): EXIF GPS + capture time, keyed by
    # image path string. Produced by the extract_geo_metadata step; absent
    # fields are None (graceful degradation, FR-011).
    geo_metadata: dict[str, "GeoMetadata"] = field(default_factory=dict)

    # Geo-temporal segmentation (spec-022): produced by geo_temporal_segment.
    # geo_segments is the winning axis's segments (empty if FLAT); geo_home is
    # the auto-detected home anchor (lat, lon) or None.
    geo_segments: list = field(default_factory=list)
    geo_home: Optional[tuple] = None

    # Vision-model outputs (spec-022): StreetCLIP city guesses + BLIP captions,
    # keyed by image path string.
    # geo_clip_predictions[path] = [{"label": "Budapest, Hungary", "score": 0.87}, ...] top-k (StreetCLIP)
    geo_clip_predictions: dict[str, list] = field(default_factory=dict)
    # geo_coord_predictions[path] = [{"lat":.., "lon":.., "prob":.., "place":"Budapest, HU"}, ...] (GeoCLIP)
    geo_coord_predictions: dict[str, list] = field(default_factory=dict)
    image_captions: dict[str, str] = field(default_factory=dict)

    # Analysis scores (keyed by image path string)
    iqa_scores: dict[str, float] = field(default_factory=dict)
    ava_scores: dict[str, float] = field(default_factory=dict)
    sharpness_scores: dict[str, float] = field(default_factory=dict)
    # spec-098: wavelet noise score [0,1], higher = cleaner
    noise_scores: dict[str, float] = field(default_factory=dict)

    # spec-094 follow-up: zero-shot CLIP scene tags, keyed by image path string.
    # scene_tags[path] = [{"label": "...", "score": 0.31}, ...] full ranked list.
    scene_tags: dict[str, list] = field(default_factory=dict)

    # spec-097 Stage 1: lens-occlusion detector (spec-096 winner), keyed by path.
    # occlusion_scores[path] = P(occluded); occlusion_tiles[path] = 9 tile scores
    # (3x3 row-major) — localization signal for UI + Stage-2 severity.
    occlusion_scores: dict[str, float] = field(default_factory=dict)
    occlusion_tiles: dict[str, list] = field(default_factory=dict)

    # spec-099/spec-100: crooked-photo (tilt) detector, GeoCalib backend, keyed by path.
    # tilt_angles[path] = signed roll deg (+ = content clockwise); tilt_confidences[path]
    # = [0,1] from GeoCalib roll uncertainty. Low confidence -> tilt_penalty ignores it.
    tilt_angles: dict[str, float] = field(default_factory=dict)
    tilt_confidences: dict[str, float] = field(default_factory=dict)

    # spec-101 (option A): terminal straighten_images repoints straightened winners in
    # selected_images to derived (leveled) files; this maps derived -> original for
    # provenance (display/export trace-back). Present only for straightened winners.
    straightened_from: dict[str, str] = field(default_factory=dict)

    # spec-093: generic multi-method quality scores from ScoreQualityStep.
    # In-run hand-off only (persistence is universal_cache); keyed
    # path -> {method: score} where score honors higher=better.
    method_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    # spec-040 Phase 3: canonical Pydantic representation for face state.
    # Producer steps (insightface_detect_faces, align_faces, score_*,
    # extract_face_embeddings, filter_faces) write/mutate this list
    # directly. Replaces context.insightface_faces / face_embeddings dicts.
    # During the strangler-fig window the dict-shaped fields below are
    # dual-written; Phase 7 removes them.
    face_records: list = field(default_factory=list)

    # Face-specific (keyed by image path string)
    # NOTE (spec-040): this dict-of-dicts representation is being retired.
    # See specs/040-unified-pipeline-framework/spec.md "Locked architectural
    # constraints" — every face-bearing step migrates to writing/reading
    # `context.face_records: List[FaceRecord]` directly (Pydantic, no
    # translator step). Both `faces` and `insightface_faces` become dead
    # state in Phase 3 and are deleted in Phase 7.
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

    # Face clustering export (for standalone Face Clustering App analysis)
    fc_export_dir: Optional[str] = None

    # User overrides (loaded from DB before refinement)
    user_overrides: list = field(default_factory=list)

    # Composite scores (keyed by image path string, computed during select_best)
    composite_scores: dict[str, float] = field(default_factory=dict)
    # spec-084: the two halves of composite_score = quality_score + person_penalty.
    # Stored so the Results table can explain *why* the composite is what it is.
    quality_scores: dict[str, float] = field(default_factory=dict)
    person_penalties: dict[str, float] = field(default_factory=dict)
    # spec-097: third composite component (0 unless P(occluded) >= gate).
    occlusion_penalties: dict[str, float] = field(default_factory=dict)
    # spec-099: fourth composite component (0 unless confident tilt > gate_deg).
    tilt_penalties: dict[str, float] = field(default_factory=dict)

    # Siamese comparison log (list of comparison results for debugging/display)
    # Each entry: {cluster_id, img1, img2, winner, confidence, comparison_type}
    siamese_comparisons: list[dict] = field(default_factory=list)

    # Selection
    selected_images: list[str] = field(default_factory=list)

    # Per-item decision records (emitted by steps, stored in DB, displayed by UI)
    step_decisions: list[StepDecision] = field(default_factory=list)

    # spec-032: named, enforced filter decisions.  Filter steps call
    # ctx.filters.record(...); downstream steps query ctx.filters.active(...).
    # Replaces ad-hoc patterns (quality_passed: set, face["filter_passed"]: bool).
    # Shared definition lives in face_cluster.filter_context so both pipeline
    # frameworks (Albumify and FC App) point at the same class.
    filters: FilterContext = field(default_factory=FilterContext)

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
