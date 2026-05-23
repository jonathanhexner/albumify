"""InsightFace Detect Faces step - face detection with person association."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from face_cluster.types import FaceRecord
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext, StepDecision
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.insightface_pipeline.face_analyzer import InsightFaceFaceAnalyzer

logger = logging.getLogger(__name__)


@register_step
class InsightFaceDetectFacesStep(BaseStep):
    """Detect faces using InsightFace SCRFD with person association."""
    
    def __init__(self):
        self._metadata = StepMetadata(
            name="insightface_detect_faces",
            display_name="InsightFace Detect Faces",
            description="Detect faces using InsightFace and associate with persons.",
            category="people",
            requires={"image_paths"},
            produces={"faces"},  # Common interface
            depends_on=["detect_persons"],
            config_schema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "default": "buffalo_l"
                    },
                    "detection_threshold": {
                        "type": "number",
                        "default": 0.5
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda"],
                        "default": "cpu"
                    },
                    "associate_to_person": {
                        "type": "boolean",
                        "default": True
                    }
                }
            }
        )
        self._analyzer = None
    
    def _get_analyzer(self, config: dict) -> InsightFaceFaceAnalyzer:
        """Lazy load face analyzer."""
        self._analyzer = self._analyzer or InsightFaceFaceAnalyzer(config)
        return self._analyzer
    
    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face detection."""
        # Normalize paths to forward slashes for consistent keys
        image_paths = [str(p).replace('\\', '/') for p in context.image_paths]

        return {
            "items": image_paths,
            "feature_type": "insightface_detection",
            "model_name": config.get('model_name', 'buffalo_l'),
            "metadata": {"device": config.get("device", "cpu")}
        }
    
    def _process_uncached(self, items: List[str], context: PipelineContext, config: dict) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - detect faces."""
        analyzer = self._get_analyzer(config)
        results = {}
        
        for i, path_str in enumerate(items):
            person_data = context.persons.get(path_str) if hasattr(context, 'persons') else None
            faces = analyzer.detect_faces(Path(path_str), person_data)
            results[path_str] = self._serialize_faces(faces)
            
            progress = (i + 1) / len(items)
            context.report_progress("insightface_detect_faces", progress, f"Detecting {i + 1}/{len(items)}")
        
        return results
    
    def _serialize_faces(self, faces: List) -> Dict[str, Any]:
        """Serialize face detections to JSON-serializable dict."""
        return {
            'faces': [self._serialize_face(face) for face in faces]
        }
    
    def _serialize_face(self, face) -> Dict[str, Any]:
        """Serialize single face detection."""
        return {
            'face_index': face.face_index,
            'bbox': self._serialize_bbox(face.bbox),
            'confidence': float(face.confidence),
            'landmarks': face.landmarks.tolist(),
            'person_bbox': self._serialize_bbox(face.person_bbox) if face.person_bbox else None,
            'face_occluded': bool(face.face_occluded)
        }
    
    def _serialize_bbox(self, bbox) -> Dict[str, Any]:
        """Serialize BoundingBox to dict."""
        return {
            'x': float(bbox.x),
            'y': float(bbox.y),
            'w': float(bbox.w),
            'h': float(bbox.h),
            'x_px': int(bbox.x_px),
            'y_px': int(bbox.y_px),
            'w_px': int(bbox.w_px),
            'h_px': int(bbox.h_px)
        }
    
    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize face detections to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to face detections."""
        return Serializers.json_deserialize(data)
    
    def _store_results(self, context: PipelineContext, results: Dict[str, Dict[str, Any]], config: dict) -> None:
        """Store faces in context."""
        context.insightface_faces = results
        context.face_records = self._build_face_records(results)

        cfg = {"detection_threshold": config.get("detection_threshold", 0.5),
               "min_face_size": config.get("min_face_size", 50)}
        for img_path, data in results.items():
            faces = data.get("faces", [])
            n = len(faces)
            confs = [f.get("confidence", 0) for f in faces]
            context.step_decisions.append(StepDecision(
                item_id=img_path, item_type="image", step="insightface_detect_faces",
                decision=f"detected_{n}" if n > 0 else "no_faces",
                reason=f"{n} face(s) detected" + (f" (conf: {', '.join(f'{c:.2f}' for c in confs)})" if confs else ""),
                config_used=cfg,
                metrics={"face_count": n, "confidences": [round(c, 3) for c in confs]},
            ))

        total_faces = sum(len(r.get('faces', [])) for r in results.values())
        logger.info(f"Detected {total_faces} faces across {len(results)} images")

    # spec-040 A1: dual-write to context.face_records so the v2 clustering
    # chain (face_clustering_steps + FCAppRunner) can read the same detector
    # output without a translator step. Replaces the list rather than
    # appending so cache re-runs do not duplicate records.
    def _build_face_records(self, results: Dict[str, Dict[str, Any]]) -> List[FaceRecord]:
        records: List[FaceRecord] = []
        face_id = 0
        for image_path, data in results.items():
            for face in data.get("faces", []):
                bbox = face.get("bbox", {})
                x = float(bbox.get("x_px", 0))
                y = float(bbox.get("y_px", 0))
                w = float(bbox.get("w_px", 0))
                h = float(bbox.get("h_px", 0))
                # spec-040 Phase 4 (schema v5) — bbox dict already carries normalized
                # ratios (x, y, w, h are 0-1 image-relative); area_ratio = w_ratio * h_ratio.
                # Image dims derived from pixel/ratio (consistent for w>0 and h>0).
                #
                # spec-041 hotfix: InsightFace returns slightly-negative ratios when
                # a face's bbox extends past the image edge. Clamp to the visible
                # portion of the image so Pandera's in_range(0, 1) check passes —
                # the face is still detected, just its rectangle is reported as
                # what's actually within frame.
                x_raw = float(bbox.get("x", 0.0))
                y_raw = float(bbox.get("y", 0.0))
                w_raw = float(bbox.get("w", 0.0))
                h_raw = float(bbox.get("h", 0.0))
                x_ratio = max(0.0, min(1.0, x_raw))
                y_ratio = max(0.0, min(1.0, y_raw))
                # Shrink width/height by however much we clipped x/y so the bbox
                # stays inside [0, 1].
                w_ratio = max(0.0, min(1.0 - x_ratio, w_raw + (x_raw - x_ratio)))
                h_ratio = max(0.0, min(1.0 - y_ratio, h_raw + (y_raw - y_ratio)))
                area_ratio = w_ratio * h_ratio
                img_w = int(round(w / w_ratio)) if w_ratio > 0 else None
                img_h = int(round(h / h_ratio)) if h_ratio > 0 else None
                landmarks_raw = face.get("landmarks")
                landmarks = np.asarray(landmarks_raw, dtype=np.float32) if landmarks_raw else None
                # spec-040 A1 bugfix: the embedding dual-write in
                # extract_face_embeddings keys records by canonical forward-slash
                # paths (see _generate_cache_key). Store the same canonical form
                # here so the lookup succeeds on Windows — otherwise every
                # FaceRecord.embedding_normalized stays None and kNN crashes
                # with "inhomogeneous shape".
                canonical_path = str(image_path).replace("\\", "/")
                records.append(FaceRecord(
                    face_id=face_id,
                    image_id=Path(image_path).name,
                    bbox=(x, y, x + w, y + h),
                    landmarks=landmarks,
                    area=w * h,
                    image_path=canonical_path,
                    face_index=int(face.get("face_index", 0)),
                    det_score=float(face.get("confidence", 0.0)),
                    area_ratio=area_ratio,
                    bbox_x_ratio=x_ratio,
                    bbox_y_ratio=y_ratio,
                    bbox_w_ratio=w_ratio,
                    bbox_h_ratio=h_ratio,
                    image_width_px=img_w,
                    image_height_px=img_h,
                ))
                face_id += 1
        return records
