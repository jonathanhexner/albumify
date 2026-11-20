"""Everything the regular pages need from a finished run, as plain JSON (spec-116 R4).

The worker builds a ``RunOutput`` from its finished context and writes ``results/run_output.json``; the control
plane reads it back, points it at the original photos (:meth:`RunOutput.remap`) and hands it to ``RunPersister``.
No pictures, no framework objects: the same file comes back from this PC and from the L4.

Over 300 lines on purpose: ``build_image_metrics`` (about 150, moved as it was from ``PipelineService``) is the one
extraction of per-image metrics from a context, and it belongs with the output it fills.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sim_bench.api.schemas.result import ImageMetrics

__all__ = ["RunOutput", "build_image_metrics", "build_reason_by_path"]

_PX_FIELDS = ("x_px", "y_px", "w_px", "h_px")


def build_reason_by_path(step_decisions) -> dict:
    """Map each image path to its most informative "why" reason. spec-084.

    Images filtered early (e.g. ``filter_quality``) never reach ``select_best``,
    so a select_best-only reason would be blank for exactly the filtered images
    the user wants explained. We therefore keep the earliest image-level reason
    and let the final ``select_best`` decision override it when present.

    Decisions are appended in step order, so the earlier rejecting step's reason
    is recorded first and ``select_best`` (the last word) overwrites it.
    """
    reason_by_path: dict = {}
    for d in (step_decisions or []):
        if d.item_type != "image":
            continue  # face-level decisions aren't per-image reasons
        if d.step == "select_best" or d.item_id not in reason_by_path:
            reason_by_path[d.item_id] = d.reason
    return reason_by_path


def build_image_metrics(context, path: str, reason_by_path: dict = None, selected=None) -> dict:
    """Build complete metrics dict for a single image.

    Face scoring steps store scores keyed by cache key
    (``"<path>:face_<index>"``), not by image path.  This helper
    collects per-face values back into a list keyed by the image path
    so that they are persisted correctly in the database.

    ``reason_by_path`` (spec-084): optional {path: select_best reason} map
    so the Results table can show *why* an image was selected/filtered.
    ``selected`` (spec-116): the run's picks by input path; default ``context.selected_images``.
    """
    # Normalize path for cache key lookups (steps use forward slashes)
    path_normalized = path.replace('\\', '/')

    # MediaPipe faces (if available)
    faces = context.faces.get(path, [])

    # Aggregate face scores by iterating over detected faces
    pose_scores = []
    eyes_scores = []
    smile_scores = []
    for face in faces:
        face_path = str(face.original_path).replace('\\', '/')
        cache_key = f"{face_path}:face_{face.face_index}"
        pose = context.face_pose_scores.get(cache_key)
        if pose is not None:
            pose_scores.append(pose)
        eyes = context.face_eyes_scores.get(cache_key)
        if eyes is not None:
            eyes_scores.append(eyes)
        smile = context.face_smile_scores.get(cache_key)
        if smile is not None:
            smile_scores.append(smile)

    # InsightFace person detection (if available)
    person_data = context.persons.get(path_normalized, {}) if hasattr(context, 'persons') and context.persons else {}
    # Also try original path format
    if not person_data:
        person_data = context.persons.get(path, {}) if hasattr(context, 'persons') and context.persons else {}

    # InsightFace faces (if available) - try both path formats
    insightface_data = {}
    if hasattr(context, 'insightface_faces') and context.insightface_faces:
        insightface_data = context.insightface_faces.get(path_normalized, {})
        if not insightface_data:
            insightface_data = context.insightface_faces.get(path, {})
    insightface_faces = insightface_data.get('faces', [])

    # Get InsightFace face scores if MediaPipe faces not available
    if not pose_scores and insightface_faces:
        for face_info in insightface_faces:
            face_index = face_info.get('face_index', 0)
            cache_key = f"{path_normalized}:face_{face_index}"
            pose = context.face_pose_scores.get(cache_key)
            if pose is not None:
                pose_scores.append(pose)
            eyes = context.face_eyes_scores.get(cache_key)
            if eyes is not None:
                eyes_scores.append(eyes)
            smile = context.face_smile_scores.get(cache_key)
            if smile is not None:
                smile_scores.append(smile)

    # Extract face filtering and frontal scores from InsightFace faces
    filter_stats = insightface_data.get('filter_stats', {})
    frontal_stats = insightface_data.get('frontal_stats', {})

    # Get best frontal score, roll angle, and centrality from all faces
    best_frontal_score = None
    best_centrality = None
    roll_angles = []
    filter_scores_list = []
    frontal_scores_list = []

    for face_info in insightface_faces:
        # Filter scores
        filter_scores = face_info.get('filter_scores', {})
        face_bbox = face_info.get('bbox')  # {x, y, w, h, x_px, y_px, w_px, h_px}
        if filter_scores or face_bbox:
            entry = {
                'face_index': face_info.get('face_index', 0),
                'confidence': filter_scores.get('confidence') or face_info.get('confidence'),
                'bbox_ratio': filter_scores.get('bbox_ratio'),
                'relative_size': filter_scores.get('relative_size'),
                'eye_ratio': filter_scores.get('eye_ratio'),
                'filter_passed': face_info.get('filter_passed', True),
            }
            if face_bbox:
                entry['bbox'] = face_bbox  # Include face bounding box for UI overlay
            filter_scores_list.append(entry)

        # Frontal scores (only for faces that passed filtering)
        if face_info.get('filter_passed', True):
            frontal_score = face_info.get('frontal_score')
            if frontal_score is not None:
                frontal_scores_data = face_info.get('frontal_scores', {})
                frontal_scores_list.append({
                    'face_index': face_info.get('face_index', 0),
                    'frontal_score': frontal_score,
                    'eye_bbox_ratio': frontal_scores_data.get('eye_bbox_ratio'),
                    'asymmetry': frontal_scores_data.get('asymmetry'),
                    'is_clusterable': face_info.get('is_clusterable', True),
                })

                if best_frontal_score is None or frontal_score > best_frontal_score:
                    best_frontal_score = frontal_score

            centrality = face_info.get('centrality')
            if centrality is not None:
                if best_centrality is None or centrality > best_centrality:
                    best_centrality = centrality

            roll_angle = face_info.get('roll_angle')
            if roll_angle is not None:
                roll_angles.append(roll_angle)

    # spec-085 (C-lite): build the canonical ImageMetrics directly. The schema
    # is the single definition of the shape; this function only supplies the
    # values (the bespoke extraction from context). Returns a dict for the
    # JSON column. Field names/types are validated against the schema here.
    return ImageMetrics(
        path=path,
        iqa_score=context.iqa_scores.get(path),
        ava_score=context.ava_scores.get(path),
        sharpness=context.sharpness_scores.get(path),
        cluster_id=context.scene_cluster_labels.get(path),
        face_count=len(faces) or len(insightface_faces),
        face_pose_scores=pose_scores or None,
        face_eyes_scores=eyes_scores or None,
        face_smile_scores=smile_scores or None,
        composite_score=context.composite_scores.get(path),
        # spec-084: composite breakdown + the human-readable decision reason.
        quality_score=context.quality_scores.get(path),
        person_penalty=context.person_penalties.get(path),
        filter_reason=(reason_by_path or {}).get(path),
        is_selected=path in (context.selected_images if selected is None else selected),
        # InsightFace-specific metrics
        person_detected=person_data.get('person_detected'),
        body_facing_score=person_data.get('body_facing_score'),
        person_confidence=person_data.get('confidence'),
        # Face filtering metrics
        filter_stats=filter_stats or None,
        filter_scores=filter_scores_list or None,
        # Frontal scoring metrics
        frontal_stats=frontal_stats or None,
        frontal_scores=frontal_scores_list or None,
        best_frontal_score=best_frontal_score,
        best_centrality=best_centrality,
        roll_angles=roll_angles or None,
    ).model_dump()


def _face_subclusters(context) -> Optional[Dict]:
    if not context.face_clusters:
        return None
    return {
        str(scene_id): {
            str(sub_id): {
                "face_count": sub.get("face_count", "0"),
                "images": sub.get("images", []),
                "has_faces": sub.get("has_faces", False),
                "identity": sub.get("identity", ""),
            }
            for sub_id, sub in subclusters.items()
        }
        for scene_id, subclusters in context.face_clusters.items()
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _scaled_box(box: Optional[List[float]], scale: float) -> Optional[List[float]]:
    """``[x, y, w, h]``: pixels (any value over 1.5, the rule ``_normalized_bbox`` uses) scale; [0,1] stays."""
    if not box or max(box) <= 1.5:
        return box
    return [v * scale for v in box]


@dataclass
class RunOutput:
    total_images: int = 0
    filtered_images: int = 0
    image_metrics: Dict[str, Dict] = field(default_factory=dict)
    scene_clusters: Dict[str, List[str]] = field(default_factory=dict)
    face_subclusters: Optional[Dict] = None
    selected_images: List[str] = field(default_factory=list)
    people: List[Dict] = field(default_factory=list)
    step_decisions: Optional[List[Dict]] = None
    step_timings: Dict[str, int] = field(default_factory=dict)
    siamese_comparisons: List = field(default_factory=list)
    total_duration_ms: Optional[int] = None
    fc_export_dir: Optional[str] = None

    @classmethod
    def from_context(cls, context, result) -> "RunOutput":
        """``result`` is the executor's ``PipelineResult``. Picks are named by their input photo: a straightened
        pick's derived file stays on the worker, so ``straightened_from`` is folded back here (spec-116 R5)."""
        from sim_bench.api.services.people_service import people_payload

        folded = getattr(context, "straightened_from", {}) or {}
        selected = [folded.get(str(p), str(p)) for p in (context.selected_images or [])]
        reasons = build_reason_by_path(context.step_decisions)   # spec-084: built once
        people_clusters = context.refined_people_clusters or context.people_clusters
        return cls(
            total_images=len(context.image_paths),
            filtered_images=len(context.active_images),
            # Built once and reused: the blob column AND the normalized tables (spec-086) derive from it
            image_metrics={str(p): build_image_metrics(context, str(p), reasons, selected)
                           for p in context.image_paths},
            scene_clusters={str(k): [str(p) for p in v] for k, v in (context.scene_clusters or {}).items()},
            face_subclusters=_face_subclusters(context),
            selected_images=selected,
            people=people_payload(people_clusters, context.people_thumbnails or None,
                                  context.attachment_decisions or None),
            step_decisions=[
                {"item_id": d.item_id, "item_type": d.item_type, "step": d.step, "decision": d.decision,
                 "reason": d.reason, "config_used": d.config_used, "metrics": d.metrics}
                for d in (context.step_decisions or [])] or None,
            step_timings={r.step_name: r.duration_ms for r in result.step_results},
            siamese_comparisons=context.siamese_comparisons or [],
            total_duration_ms=result.total_duration_ms,
            fc_export_dir=context.fc_export_dir,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_jsonable)

    @classmethod
    def from_json(cls, text: str) -> "RunOutput":
        return cls(**json.loads(text))

    def remap(self, path_map) -> "RunOutput":
        """A copy that names the original photos, with pixel boxes in the original's pixels ([0,1] boxes stay).

        Paths also sit inside nested values (decisions, comparisons, ``<path>:face_N`` keys), so every string
        and dict key that IS a staged path is mapped; free text is left alone."""
        def walk(value: Any) -> Any:
            if isinstance(value, str):
                return path_map.original(value)
            if isinstance(value, list):
                return [walk(v) for v in value]
            if isinstance(value, dict):
                return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in value.items()}
            return value

        data = asdict(self)
        for path, metrics in data["image_metrics"].items():
            scale = path_map.scale(path)
            for face in metrics.get("filter_scores") or []:
                bbox = face.get("bbox") or {}
                for name in _PX_FIELDS:
                    if bbox.get(name) is not None:
                        bbox[name] = bbox[name] * scale
        for person in data["people"]:
            for face in person["face_instances"]:
                face["bbox"] = _scaled_box(face.get("bbox"), path_map.scale(face["image_path"]))
            thumb = person.get("thumbnail_image_path")
            person["thumbnail_bbox"] = _scaled_box(person.get("thumbnail_bbox"),
                                                   path_map.scale(thumb) if thumb else 1.0)
            if thumb and path_map.original(thumb) == thumb and person["face_instances"]:
                # a crop file the worker wrote (legacy pipelines): not returned, so use a face in its photo
                first = person["face_instances"][0]
                person["thumbnail_image_path"], person["thumbnail_bbox"] = first["image_path"], first.get("bbox")
        return RunOutput(**walk(data))
