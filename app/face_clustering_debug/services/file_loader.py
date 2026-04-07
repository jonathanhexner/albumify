"""FileLoader — load clustering data from benchmark JSON/NPY files."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.face_clustering_debug.models.schemas import (
    AttachDecision,
    ClusterInfo,
    ClusteringResult,
    FaceInfo,
    MergeDecision,
)

logger = logging.getLogger(__name__)


class FileLoader:
    """Load clustering data from benchmark output files."""

    def __init__(self, results_dir: Path, specific_file: Optional[Path] = None):
        self.results_dir = Path(results_dir)
        self._specific_file = specific_file
        self._data: Optional[Dict[str, Any]] = None
        self._embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_benchmark_files(cls, results_dir: Path) -> List[Path]:
        """Return all benchmark JSON files, newest first."""
        return sorted(Path(results_dir).glob("benchmark_*.json"), reverse=True)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_json(self) -> Dict[str, Any]:
        if self._data is not None:
            return self._data

        if self._specific_file:
            source = Path(self._specific_file)
        else:
            files = self.list_benchmark_files(self.results_dir)
            if not files:
                raise FileNotFoundError(f"No benchmark JSON found in {self.results_dir}")
            source = files[0]

        with open(source, encoding="utf-8") as f:
            self._data = json.load(f)
        logger.info("Loaded benchmark data from %s", source)
        return self._data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_run_info(self) -> Dict[str, Any]:
        """Return album name, total faces and timestamp from the loaded file."""
        data = self._load_json()
        album_path = data.get("album_path", "")
        return {
            "album_name": Path(album_path).name if album_path else "unknown",
            "album_path": album_path,
            "total_faces": data.get("total_faces", 0),
            "timestamp": data.get("timestamp", ""),
        }

    def load_embeddings(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._embeddings

        data = self._load_json()
        # Prefer the filename stored in the JSON (new format)
        emb_filename = data.get("embeddings_file")
        if emb_filename:
            path = self.results_dir / emb_filename
            if path.exists():
                self._embeddings = np.load(path)
                logger.info("Loaded embeddings from %s  shape=%s", path, self._embeddings.shape)
                return self._embeddings

        # Fallback: scan for any embeddings_*.npy
        npy_files = sorted(self.results_dir.glob("embeddings_*.npy"), reverse=True)
        if not npy_files:
            raise FileNotFoundError(f"No embeddings NPY found in {self.results_dir}")
        self._embeddings = np.load(npy_files[0])
        logger.info("Loaded embeddings from %s  shape=%s", npy_files[0], self._embeddings.shape)
        return self._embeddings

    def load_faces(self) -> List[FaceInfo]:
        data = self._load_json()
        faces: List[FaceInfo] = []

        for i, meta in enumerate(data.get("face_metadata", [])):
            bbox_raw = meta.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox = (bbox_raw.get("x", 0), bbox_raw.get("y", 0), bbox_raw.get("w", 0), bbox_raw.get("h", 0))
                # Get pixel bbox for landmark normalization
                bbox_px = {
                    "x": bbox_raw.get("x_px", 0),
                    "y": bbox_raw.get("y_px", 0),
                    "w": bbox_raw.get("w_px", 1),
                    "h": bbox_raw.get("h_px", 1),
                }
            else:
                bbox = tuple(bbox_raw) if len(bbox_raw) >= 4 else (0, 0, 0, 0)
                bbox_px = None

            # Normalize landmarks to 0-1 range relative to face bbox
            landmarks = meta.get("landmarks")
            if landmarks and bbox_px and bbox_px["w"] > 0 and bbox_px["h"] > 0:
                normalized = []
                for p in landmarks[:5]:
                    # Transform pixel coords to normalized coords within bbox
                    x_norm = (p[0] - bbox_px["x"]) / bbox_px["w"]
                    y_norm = (p[1] - bbox_px["y"]) / bbox_px["h"]
                    # Clamp to [0, 1] range (landmarks should be within bbox)
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    normalized.append((x_norm, y_norm))
                landmarks = normalized
            elif landmarks:
                landmarks = [(p[0], p[1]) for p in landmarks[:5]]

            pose = None
            if all(k in meta for k in ["pitch_angle", "yaw_angle", "roll_angle"]):
                pose = (meta["pitch_angle"], meta["yaw_angle"], meta["roll_angle"])

            crop_path = self.results_dir / "face_crops" / f"face_{i:04d}.jpg"

            faces.append(FaceInfo(
                index=i,
                image_path=meta.get("image_path", ""),
                bbox=bbox,
                confidence=meta.get("confidence", 0.0),
                crop_path=str(crop_path) if crop_path.exists() else None,
                landmarks=landmarks,
                pose_angles=pose,
                frontal_score=float(meta.get("frontal_score", 0.0)),
                eye_bbox_ratio=float(meta.get("eye_bbox_ratio", 0.0)),
                asymmetry_ratio=float(meta.get("asymmetry_ratio", 0.0)),
            ))

        return faces

    def load_clustering_result(self, method: str) -> Optional[ClusteringResult]:
        data = self._load_json()
        methods = data.get("methods", {})
        if method not in methods:
            return None

        method_data = methods[method]
        labels = np.array(method_data.get("labels", []))
        stats = method_data.get("stats", {})
        debug = stats.get("debug", {})

        # JSON serialises int keys as strings — normalise to int
        cluster_thresholds = {int(k): v for k, v in debug.get("cluster_thresholds", {}).items()}
        cluster_exemplars = {int(k): v for k, v in debug.get("cluster_exemplars", {}).items()}
        cluster_d3_stats = {int(k): v for k, v in debug.get("cluster_d3_stats", {}).items()}

        clusters = self._parse_clusters(labels, cluster_thresholds, cluster_exemplars, cluster_d3_stats)
        merge_decisions = self._parse_merge_decisions(debug.get("merge_decisions", []))
        attach_decisions = self._parse_attach_decisions(debug.get("attach_decisions", []))

        embeddings = self.load_embeddings()
        faces = self.load_faces()

        return ClusteringResult(
            labels=labels,
            embeddings=embeddings,
            faces=faces,
            clusters=clusters,
            merge_decisions=merge_decisions,
            attach_decisions=attach_decisions,
            algorithm=method,
            params=stats.get("params", {}),
            n_clusters=stats.get("n_clusters", 0),
            n_noise=stats.get("n_noise", 0),
        )

    def _parse_clusters(
        self,
        labels: np.ndarray,
        thresholds: Dict[int, float],
        exemplars: Dict[int, List[int]],
        d3_stats: Dict[int, Dict],
    ) -> List[ClusterInfo]:
        clusters = []
        for label in sorted(set(labels) - {-1}):
            key = int(label)
            face_indices = [i for i, l in enumerate(labels) if l == label]
            d3 = d3_stats.get(key, {})
            clusters.append(ClusterInfo(
                cluster_id=key,
                face_indices=face_indices,
                threshold=thresholds.get(key, 0.0),
                exemplar_indices=exemplars.get(key, []),
                q1=d3.get("q1", 0.0),
                q3=d3.get("q3", 0.0),
                iqr=d3.get("iqr", 0.0),
                raw_threshold=d3.get("raw_threshold", 0.0),
            ))
        return clusters

    def _parse_merge_decisions(self, raw: List[Dict]) -> List[MergeDecision]:
        """Parse merge decisions - handles both hybrid_hdbscan_knn and hybrid_closest_face formats."""
        decisions = []
        for d in raw:
            cross = d.get("cross_distances", d.get("exemplar_cross_distances"))
            decisions.append(MergeDecision(
                cluster_a=d.get("cluster_a", 0),
                cluster_b=d.get("cluster_b", 0),
                threshold_a=d.get("threshold_a", d.get("threshold", 0.0)),
                threshold_b=d.get("threshold_b", d.get("threshold", 0.0)),
                merged=d.get("merged", False),
                reason=d.get("reason", ""),
                min_distance=d.get("min_distance", d.get("min_exemplar_dist", 0.0)),
                cross_distances=np.array(cross) if cross else None,
                # hybrid_hdbscan_knn specific
                threshold_used=d.get("threshold", 0.0),
                pairs_within_threshold=d.get("n_pairs_within", 0),
                exemplars_a_involved=d.get("exemplars_a_involved", 0),
                exemplars_b_involved=d.get("exemplars_b_involved", 0),
                min_dists_a=d.get("min_dists_a", d.get("exemplar_min_dists_a")),
                min_dists_b=d.get("min_dists_b", d.get("exemplar_min_dists_b")),
                # hybrid_closest_face specific
                d3_cross_a=d.get("d3_cross_a"),
                d3_cross_b=d.get("d3_cross_b"),
                fits_a=d.get("fits_a", 0),
                fits_b=d.get("fits_b", 0),
                n_fits_total=d.get("n_fits_total", 0),
                effective_threshold_a=d.get("effective_threshold_a", 0.0),
                effective_threshold_b=d.get("effective_threshold_b", 0.0),
                merge_threshold_multiplier=d.get("merge_threshold_multiplier", 1.0),
                merge_min_faces=d.get("merge_min_faces", 2),
                early_exit_threshold=d.get("early_exit_threshold", 0.0),
            ))
        return decisions

    def _parse_attach_decisions(self, raw: List[Dict]) -> List[AttachDecision]:
        decisions = []
        for d in raw:
            attached_to = d.get("attached_to")
            reason = d.get("reason") or ("attached" if attached_to is not None else "no_cluster_qualified")
            decisions.append(AttachDecision(
                face_index=d.get("face_idx", 0),
                attached_to=attached_to,
                reason=reason,
                candidates=d.get("candidates", []),
            ))
        return decisions

    def get_available_methods(self) -> List[str]:
        """Return methods sorted so debug-capable ones come first."""
        data = self._load_json()
        methods = list(data.get("methods", {}).keys())
        return sorted(methods, key=lambda m: not self.has_debug_data(m))

    def has_debug_data(self, method: str) -> bool:
        data = self._load_json()
        stats = data.get("methods", {}).get(method, {}).get("stats", {})
        debug = stats.get("debug", {})
        return bool(debug.get("merge_decisions") or debug.get("attach_decisions"))

    def get_face_crop(self, face_index: int) -> Optional[bytes]:
        """Get aligned face crop from saved benchmark results.

        Looks for the aligned version first (face_XXXX_aligned.jpg),
        then falls back to the legacy format (face_XXXX.jpg).
        """
        crops_dir = self.results_dir / "face_crops"

        # Try aligned version first (new format from benchmark script)
        aligned_path = crops_dir / f"face_{face_index:04d}_aligned.jpg"
        if aligned_path.exists():
            return aligned_path.read_bytes()

        # Fallback to legacy format
        legacy_path = crops_dir / f"face_{face_index:04d}.jpg"
        if legacy_path.exists():
            return legacy_path.read_bytes()

        return None

    def get_face_crop_raw(self, face_index: int) -> Optional[bytes]:
        """Get raw face crop (bbox only, no alignment) as JPEG bytes."""
        import cv2
        from sim_bench.pipeline.utils.image_cache import get_image_cache

        data = self._load_json()
        face_metadata = data.get("face_metadata", [])
        if face_index >= len(face_metadata):
            return None

        meta = face_metadata[face_index]
        image_path = meta.get("image_path")
        if not image_path:
            return None

        bbox_raw = meta.get("bbox", {})
        if isinstance(bbox_raw, dict):
            x = int(bbox_raw.get("x_px", 0))
            y = int(bbox_raw.get("y_px", 0))
            w = int(bbox_raw.get("w_px", 0))
            h = int(bbox_raw.get("h_px", 0))
        else:
            return None

        if w <= 0 or h <= 0:
            return None

        cache = get_image_cache()
        img = cache.get(image_path)
        if img is None:
            return None

        # Add margin (20%)
        margin = 0.2
        mw, mh = int(w * margin), int(h * margin)
        img_h, img_w = img.shape[:2]

        x1 = max(0, x - mw)
        y1 = max(0, y - mh)
        x2 = min(img_w, x + w + mw)
        y2 = min(img_h, y + h + mh)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to 256x256
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return buf.tobytes()

    def get_original_image_with_bbox(self, face_index: int, max_size: int = 800) -> Optional[bytes]:
        """Get original image with face bbox and landmarks drawn, resized for display."""
        import cv2
        from sim_bench.pipeline.utils.image_cache import get_image_cache

        data = self._load_json()
        face_metadata = data.get("face_metadata", [])
        if face_index >= len(face_metadata):
            return None

        meta = face_metadata[face_index]
        image_path = meta.get("image_path")
        if not image_path:
            return None

        bbox_raw = meta.get("bbox", {})
        landmarks = meta.get("landmarks")

        cache = get_image_cache()
        img = cache.get(image_path)
        if img is None:
            return None

        # Convert to BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw bbox
        if isinstance(bbox_raw, dict):
            x = int(bbox_raw.get("x_px", 0))
            y = int(bbox_raw.get("y_px", 0))
            w = int(bbox_raw.get("w_px", 0))
            h = int(bbox_raw.get("h_px", 0))
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw landmarks (pixel coordinates)
        if landmarks and len(landmarks) >= 5:
            colors = [(0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 0)]
            for i, (lx, ly) in enumerate(landmarks[:5]):
                color = colors[i] if i < len(colors) else (255, 255, 255)
                cv2.circle(img_bgr, (int(lx), int(ly)), 5, color, -1)
                cv2.circle(img_bgr, (int(lx), int(ly)), 7, (255, 255, 255), 2)

        # Resize for display
        img_h, img_w = img_bgr.shape[:2]
        scale = min(max_size / img_w, max_size / img_h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
