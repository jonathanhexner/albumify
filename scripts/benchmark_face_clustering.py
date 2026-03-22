"""
Benchmark face clustering methods.

Compares HDBSCAN vs Hybrid HDBSCAN+kNN on a photo album.
Generates visual comparison data for Streamlit app.

Usage:
    python scripts/benchmark_face_clustering.py --album-path D:\\Budapest2025_Google
"""

import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml

import numpy as np
from PIL import Image, ImageOps
import cv2

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.registry import get_registry
from sim_bench.clustering.base import load_clustering_method
from sim_bench.pipeline.steps.detect_face_orientation import detect_face_orientation
from sim_bench.pipeline.steps.align_faces import (
    rotate_image_and_landmarks,
    align_face_with_orientation,
    crop_face_generous,
)
from sim_bench.pipeline.utils.face_alignment import ARCFACE_REF_POINTS_112

# Import all steps to register them
import sim_bench.pipeline.steps.all_steps  # noqa: F401

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Configure logging to both console and file."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f'benchmark_{timestamp}.log'
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")
    return log_file


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load benchmark configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline_for_embeddings(album_path: Path, config: Dict[str, Any]) -> PipelineContext:
    """Run pipeline to extract face embeddings."""
    logger.info(f"Running pipeline on album: {album_path}")
    
    # Create context
    context = PipelineContext()
    context.source_directory = album_path
    
    # Get pipeline steps from config
    step_names = config['pipeline']['steps']
    
    # Build step configs
    step_configs = {
        'filter_faces': config['pipeline']['filter_faces'],
        'extract_face_embeddings': config['pipeline']['extract_face_embeddings'],
    }
    
    # Execute pipeline
    registry = get_registry()
    executor = PipelineExecutor(registry)
    pipeline_config = PipelineConfig(
        step_configs=step_configs,
        fail_fast=True
    )
    
    result = executor.execute(context, step_names, pipeline_config)
    
    if not result.success:
        raise RuntimeError(f"Pipeline failed: {result.error_message}")
    
    logger.info("Pipeline completed successfully")
    return context


def collect_face_data(context: PipelineContext) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract face embeddings and metadata from pipeline context."""
    embeddings = []
    metadata = []
    
    # Collect from insightface_faces (filtered faces)
    if hasattr(context, 'insightface_faces') and context.insightface_faces:
        for image_path, face_data in context.insightface_faces.items():
            for face_info in face_data.get('faces', []):
                # Only include faces that passed filtering
                if not face_info.get('filter_passed', True):
                    continue
                if not face_info.get('is_clusterable', True):
                    continue
                
                # Look up embedding
                path_str = str(image_path).replace('\\', '/')
                face_index = face_info.get('face_index', 0)
                cache_key = f"{path_str}:face_{face_index}"
                
                embedding = context.face_embeddings.get(cache_key)
                if embedding is None:
                    continue
                
                embeddings.append(embedding)
                metadata.append({
                    'image_path': str(image_path),
                    'face_index': face_index,
                    'bbox': face_info.get('bbox', {}),
                    'confidence': face_info.get('confidence', 0),
                    'landmarks': face_info.get('landmarks', []),
                    'roll_angle': face_info.get('roll_angle', 0.0),
                    'pitch_angle': face_info.get('pitch_angle', 0.0),
                    'yaw_angle': face_info.get('yaw_angle', 0.0),
                    'frontal_score': face_info.get('frontal_score', 0.0),
                    'eye_bbox_ratio': face_info.get('eye_bbox_ratio', 0.0),
                    'asymmetry_ratio': face_info.get('asymmetry_ratio', 0.0),
                })
    
    if len(embeddings) == 0:
        raise ValueError("No face embeddings found in context")
    
    embeddings_array = np.array(embeddings)
    logger.info(f"Collected {len(embeddings)} face embeddings")
    
    return embeddings_array, metadata


@dataclass
class CropConfig:
    """Configuration for face crop saving."""
    output_dir: Path
    crop_size: int = 112
    padding_ratio: float = 0.2


def is_valid_bbox(w_px: int, h_px: int) -> bool:
    """Check if bbox dimensions are valid."""
    return w_px > 0 and h_px > 0


def is_valid_crop_coords(left: int, top: int, right: int, bottom: int) -> bool:
    """Check if crop coordinates are valid."""
    return right > left and bottom > top


def compute_crop_coordinates(x_px: int, y_px: int, w_px: int, h_px: int, 
                            img_width: int, img_height: int, pad: int):
    """Compute crop coordinates with padding and boundary checks."""
    left = max(0, x_px - pad)
    top = max(0, y_px - pad)
    right = min(img_width, x_px + w_px + pad)
    bottom = min(img_height, y_px + h_px + pad)
    return left, top, right, bottom


def draw_landmarks_cv2(image: np.ndarray, landmarks: List, radius: int = 3) -> np.ndarray:
    """Draw landmarks on image (BGR format)."""
    img = image.copy()
    colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
    labels = ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']
    for pt, color, label in zip(landmarks[:5], colors, labels):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.putText(img, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return img


def save_single_face_crop(face_meta: Dict[str, Any], index: int, config: CropConfig) -> bool:
    """Save face crops with full debug artifacts. Returns True if successful."""
    image_path = Path(face_meta['image_path'])

    bbox = face_meta['bbox']
    x_px = int(bbox.get('x_px', 0))
    y_px = int(bbox.get('y_px', 0))
    w_px = int(bbox.get('w_px', 0))
    h_px = int(bbox.get('h_px', 0))
    landmarks = face_meta.get('landmarks', [])

    if not is_valid_bbox(w_px, h_px):
        return False

    # Load image with EXIF correction
    img_pil = ImageOps.exif_transpose(Image.open(image_path))
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Output prefix
    prefix = f'face_{index:04d}'
    crops_dir = config.output_dir / 'face_crops'

    # === 1. RAW CROP (bbox only) ===
    pad = int(min(w_px, h_px) * config.padding_ratio)
    left, top, right, bottom = compute_crop_coordinates(
        x_px, y_px, w_px, h_px, img_pil.width, img_pil.height, pad
    )

    if not is_valid_crop_coords(left, top, right, bottom):
        return False

    raw_crop = img_np[top:bottom, left:right].copy()
    cv2.imwrite(str(crops_dir / f'{prefix}_raw.jpg'), raw_crop)

    # === 2. RAW CROP + LANDMARKS ===
    if landmarks and len(landmarks) >= 5:
        # Transform landmarks to crop coordinates
        crop_landmarks = [[pt[0] - left, pt[1] - top] for pt in landmarks[:5]]
        raw_with_lm = draw_landmarks_cv2(raw_crop, crop_landmarks)
        cv2.imwrite(str(crops_dir / f'{prefix}_raw_landmarks.jpg'), raw_with_lm)

    # === 3. DETECT ORIENTATION ===
    orientation = 0
    if landmarks and len(landmarks) >= 5:
        orientation = detect_face_orientation(landmarks)

    # === 4. TWO-STAGE ALIGNMENT ===
    # Stage 1: Generous crop (50% margin for rotation room)
    generous_crop = None
    generous_landmarks = landmarks
    rotated_crop = None
    rotated_landmarks = landmarks
    aligned = None

    # Build bbox dict for margin calculation
    bbox = {'x': x_px, 'y': y_px, 'w': w_px, 'h': h_px}

    if landmarks and len(landmarks) >= 5:
        generous_crop, generous_landmarks = crop_face_generous(img_np, landmarks, margin=0.5, bbox=bbox)

        if generous_crop is not None:
            # Save generous crop
            generous_with_lm = draw_landmarks_cv2(generous_crop.copy(), generous_landmarks)
            cv2.putText(generous_with_lm, "Stage1: 50% margin", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(str(crops_dir / f'{prefix}_stage1_generous.jpg'), generous_with_lm)

            # Stage 2: Rotate if needed
            if orientation != 0:
                rotated_crop, rotated_landmarks = rotate_image_and_landmarks(
                    generous_crop, generous_landmarks, orientation
                )
                rotated_with_lm = draw_landmarks_cv2(rotated_crop.copy(), rotated_landmarks)
                cv2.putText(rotated_with_lm, f"Stage2: {orientation}deg rotation", (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imwrite(str(crops_dir / f'{prefix}_stage2_rotated.jpg'), rotated_with_lm)
            else:
                rotated_crop = generous_crop
                rotated_landmarks = generous_landmarks

            # Stage 3: 5-point affine alignment (with bbox for proper margin)
            aligned = align_face_with_orientation(
                img_np, landmarks, orientation, target_size=config.crop_size, bbox=bbox
            )

    if aligned is not None:
        cv2.imwrite(str(crops_dir / f'{prefix}_aligned.jpg'), aligned)

        # === 5. ALIGNED + REFERENCE LANDMARKS ===
        scale = config.crop_size / 112.0
        ref_landmarks = (ARCFACE_REF_POINTS_112 * scale).tolist()
        aligned_with_lm = draw_landmarks_cv2(aligned, ref_landmarks, radius=4)
        cv2.putText(aligned_with_lm, f"orient={orientation}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.imwrite(str(crops_dir / f'{prefix}_aligned_landmarks.jpg'), aligned_with_lm)

    # === 6. INFO FILE ===
    info_path = crops_dir / f'{prefix}_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Face Index (global): {index}\n")
        f.write(f"BBox: x={x_px}, y={y_px}, w={w_px}, h={h_px}\n")
        f.write(f"Confidence: {face_meta.get('confidence', 0):.3f}\n")
        f.write(f"Roll Angle: {face_meta.get('roll_angle', 0):.2f}\n")
        f.write(f"Orientation Detected: {orientation}°\n")
        f.write(f"\n=== PIPELINE STAGES ===\n")
        f.write(f"Stage 1: Generous crop (50% margin)\n")
        f.write(f"Stage 2: Rotate by {orientation}°\n")
        f.write(f"Stage 3: 5-point affine to {config.crop_size}x{config.crop_size}\n")
        f.write(f"\n=== LANDMARKS ===\n")
        f.write(f"Original (full image coords):\n")
        for pt, label in zip(landmarks[:5], ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']):
            f.write(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})\n")
        if generous_crop is not None:
            f.write(f"\nAfter Stage 1 (generous crop coords):\n")
            for pt, label in zip(generous_landmarks[:5], ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']):
                f.write(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})\n")
        if orientation != 0 and rotated_crop is not None:
            f.write(f"\nAfter Stage 2 ({orientation}° rotation):\n")
            for pt, label in zip(rotated_landmarks[:5], ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']):
                f.write(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})\n")

    return True


def save_face_crops(metadata: List[Dict[str, Any]], config: CropConfig) -> List[int]:
    """Save face crops with full debug artifacts. Returns list of successfully saved indices."""
    crops_dir = config.output_dir / 'face_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(metadata)} face crops with debug artifacts...")

    saved_indices = []
    orientation_counts = {0: 0, 90: 0, 180: 0, 270: 0}

    for i, face_meta in enumerate(metadata):
        # FIX for SIGHTING-006: Use metadata index i (not saved_count) for filename
        # This ensures crop filename matches metadata position even when some faces fail
        if save_single_face_crop(face_meta, i, config):
            saved_indices.append(i)
            # Count orientations
            landmarks = face_meta.get('landmarks', [])
            if landmarks and len(landmarks) >= 5:
                orient = detect_face_orientation(landmarks)
                orientation_counts[orient] = orientation_counts.get(orient, 0) + 1

    logger.info(f"Face crops saved: {len(saved_indices)}/{len(metadata)}")
    logger.info(f"Orientations: 0°={orientation_counts[0]}, 90°={orientation_counts[90]}, "
                f"180°={orientation_counts[180]}, 270°={orientation_counts[270]}")

    if len(saved_indices) < len(metadata):
        logger.warning(f"Skipped {len(metadata) - len(saved_indices)} faces due to invalid crops")
    
    return saved_indices


def validate_crop_filenames(
    metadata: List[Dict[str, Any]],
    saved_indices: List[int],
    crops_dir: Path
) -> None:
    """Validate that crop filenames match metadata indices.

    This catches the bug where saved_count is used instead of metadata index.

    Args:
        metadata: Full metadata array
        saved_indices: Indices of successfully saved faces
        crops_dir: Directory containing face crops

    Raises:
        ValueError: If validation fails (filename mismatch detected)
    """
    logger.info("Validating crop filename alignment...")

    errors = []

    # Check 1: For each saved face, verify filename matches metadata index
    for meta_idx in saved_indices:
        expected_file = crops_dir / f"face_{meta_idx:04d}_aligned.jpg"

        if not expected_file.exists():
            errors.append(
                f"  - metadata[{meta_idx}] was saved but expected file "
                f"{expected_file.name} doesn't exist"
            )

    # Check 2: Count actual crop files
    actual_crops = list(crops_dir.glob("face_*_aligned.jpg"))

    if len(actual_crops) != len(saved_indices):
        errors.append(
            f"  - Found {len(actual_crops)} crop files but saved_indices has "
            f"{len(saved_indices)} entries"
        )

    # Check 3: Detect sequential numbering (indicates saved_count bug)
    if actual_crops:
        # Get numeric IDs from filenames
        crop_ids = []
        for crop_file in actual_crops:
            match = re.search(r'face_(\d+)_aligned', crop_file.name)
            if match:
                crop_ids.append(int(match.group(1)))

        crop_ids_sorted = sorted(crop_ids)

        # If crops are 0,1,2,3... but saved_indices are 2,3,4,5...
        # then we have the saved_count bug
        if crop_ids_sorted == list(range(len(crop_ids_sorted))):
            # Sequential from 0 - could be the bug
            if saved_indices and saved_indices[0] != 0:
                errors.append(
                    f"  - DETECTED BUG: Crop files are sequential [0..{len(crop_ids_sorted)-1}] "
                    f"but first saved metadata index is {saved_indices[0]}. "
                    f"This indicates saved_count was used instead of metadata index!"
                )

    if errors:
        logger.error("Crop filename validation FAILED:")
        for error in errors:
            logger.error(error)

        # Include detailed errors in exception message
        error_details = "\n".join(errors)
        raise ValueError(
            f"Crop filename validation failed:\n{error_details}\n\n"
            "This usually means saved_count was used for filenames instead of metadata index."
        )

    logger.info(f"✓ Validation passed: {len(saved_indices)} crop filenames match metadata indices")


def calculate_cluster_statistics(embeddings: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Calculate detailed statistics for each cluster."""
    from scipy.spatial.distance import cdist, pdist, squareform
    
    cluster_stats = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        cluster_size = len(cluster_embeddings)
        
        if cluster_size < 2:
            cluster_stats.append({
                'cluster_id': int(label),
                'size': cluster_size,
                'intra_min': 0.0,
                'intra_max': 0.0,
                'intra_mean': 0.0,
                'intra_std': 0.0,
                'nearest_external_dist': None,
                'nearest_external_idx': None
            })
            continue
        
        # Intra-cluster distances
        intra_dists = pdist(cluster_embeddings, metric='cosine')
        
        # Find nearest external face
        external_mask = ~mask
        if np.any(external_mask):
            external_embeddings = embeddings[external_mask]
            dists_to_external = cdist(cluster_embeddings, external_embeddings, metric='cosine')
            min_external_dist = float(np.min(dists_to_external))
            min_idx_flat = np.argmin(dists_to_external)
            external_indices = np.where(external_mask)[0]
            nearest_external_idx = int(external_indices[min_idx_flat % len(external_indices)])
        else:
            min_external_dist = None
            nearest_external_idx = None
        
        cluster_stats.append({
            'cluster_id': int(label),
            'size': cluster_size,
            'intra_min': float(np.min(intra_dists)),
            'intra_max': float(np.max(intra_dists)),
            'intra_mean': float(np.mean(intra_dists)),
            'intra_std': float(np.std(intra_dists)),
            'nearest_external_dist': min_external_dist,
            'nearest_external_idx': nearest_external_idx
        })
    
    return cluster_stats


def run_clustering_method(
    method_name: str,
    method_config: Dict[str, Any],
    embeddings: np.ndarray
) -> Dict[str, Any]:
    """Run a single clustering method and return results with statistics."""
    logger.info(f"Running {method_name} clustering...")

    method = load_clustering_method(method_config)

    # Collect debug data for hybrid methods
    collect_debug = method_config.get('algorithm', '').startswith('hybrid')
    labels, stats = method.cluster(embeddings, collect_debug_data=collect_debug)
    labels_array = np.array(labels)

    result = {
        'labels': labels,
        'stats': stats,
        'cluster_stats': calculate_cluster_statistics(embeddings, labels_array)
    }

    n_clusters = stats.get('n_clusters', len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = stats.get('n_noise', 0)
    noise_info = f", noise={n_noise}" if n_noise > 0 else ""
    logger.info(f"{method_name}: {n_clusters} clusters{noise_info}")

    return result


def run_clustering_methods(
    embeddings: np.ndarray,
    method_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Run all clustering methods and return results with statistics."""
    results = {}

    for method_name, method_config in method_configs.items():
        results[method_name] = run_clustering_method(method_name, method_config, embeddings)

    return results


def get_clustering_methods_from_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract all clustering method configs from the benchmark config.

    A method config is any top-level key that has an 'algorithm' field.
    """
    methods = {}
    excluded_keys = {'output', 'pipeline'}

    for key, value in config.items():
        if key in excluded_keys:
            continue
        if isinstance(value, dict) and 'algorithm' in value:
            methods[key] = value

    return methods


@dataclass
class BenchmarkData:
    """Container for benchmark results to be saved."""
    timestamp: str
    album_path: str
    total_faces: int
    face_metadata: List[Dict[str, Any]]
    methods: Dict[str, Dict[str, Any]]
    embeddings_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'timestamp': self.timestamp,
            'album_path': self.album_path,
            'total_faces': self.total_faces,
            'face_metadata': self.face_metadata,
            'methods': self.methods
        }
        if self.embeddings_file:
            result['embeddings_file'] = self.embeddings_file
        return result


class NumpyTypeConverter:
    """Strategy for converting numpy types to JSON-serializable types."""
    
    CONVERTERS = {
        'integer': lambda x: int(x),
        'floating': lambda x: float(x),
        'boolean': lambda x: bool(x),
        'ndarray': lambda x: x.tolist(),
    }
    
    @classmethod
    def convert(cls, obj):
        """Convert numpy types recursively."""
        numpy_type = cls._get_numpy_type(obj)
        converter = cls.CONVERTERS.get(numpy_type)
        
        if converter:
            return converter(obj)
        if isinstance(obj, dict):
            return {key: cls.convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [cls.convert(item) for item in obj]
        return obj
    
    @staticmethod
    def _get_numpy_type(obj):
        """Determine numpy type of object."""
        if isinstance(obj, np.bool_):
            return 'boolean'
        if isinstance(obj, np.integer):
            return 'integer'
        if isinstance(obj, np.floating):
            return 'floating'
        if isinstance(obj, np.ndarray):
            return 'ndarray'
        return None


class AtomicJsonWriter:
    """Writes JSON atomically using temp file strategy."""
    
    def __init__(self, target_file: Path):
        self.target_file = target_file
        self.temp_file = target_file.with_suffix('.json.tmp')
    
    def write(self, data: Dict[str, Any]):
        """Write JSON data atomically."""
        logger.info(f"Writing to: {self.temp_file}")
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self._validate()
        self._commit()
    
    def _validate(self):
        """Validate JSON can be read back."""
        logger.info("Validating JSON...")
        with open(self.temp_file, 'r', encoding='utf-8') as f:
            json.load(f)
        logger.info("Validation successful")
    
    def _commit(self):
        """Commit temp file to final location."""
        self.temp_file.rename(self.target_file)
        logger.info(f"Saved to: {self.target_file}")


def save_embeddings(embeddings: np.ndarray, output_dir: Path) -> Path:
    """Save embeddings to numpy file."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    embeddings_file = output_dir / f'embeddings_{timestamp_str}.npy'
    np.save(embeddings_file, embeddings)
    logger.info(f"Saved embeddings to: {embeddings_file}")
    return embeddings_file


def save_benchmark_results(data: BenchmarkData, output_dir: Path) -> Path:
    """Save benchmark results to JSON."""
    logger.info("Converting numpy types...")
    clean_data = BenchmarkData(
        timestamp=data.timestamp,
        album_path=data.album_path,
        total_faces=data.total_faces,
        face_metadata=NumpyTypeConverter.convert(data.face_metadata),
        methods=NumpyTypeConverter.convert(data.methods),
        embeddings_file=data.embeddings_file
    )
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = output_dir / f'benchmark_{timestamp_str}.json'
    
    writer = AtomicJsonWriter(results_file)
    writer.write(clean_data.to_dict())
    return results_file


def main():
    parser = argparse.ArgumentParser(description='Benchmark face clustering methods')
    parser.add_argument(
        '--album-path',
        type=Path,
        required=True,
        help='Path to photo album directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/clustering_benchmark.yaml'),
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (defaults to config value)'
    )
    
    args = parser.parse_args()
    
    # Validate album path
    if not args.album_path.exists():
        logger.error(f"Album path does not exist: {args.album_path}")
        return 1
    
    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    
    # Set output directory
    output_dir = args.output_dir or Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file and console
    setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("FACE CLUSTERING BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Album: {args.album_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    # Step 1: Run pipeline to extract embeddings
    context = run_pipeline_for_embeddings(args.album_path, config)
    
    # Step 2: Collect face embeddings and metadata
    embeddings, metadata = collect_face_data(context)
    
    # Step 3: Save face crops and filter metadata
    crop_config = CropConfig(
        output_dir=output_dir,
        crop_size=config['output'].get('crop_size', 112)
    )
    saved_indices = save_face_crops(metadata, crop_config)

    # Validate crop filenames match metadata indices
    validate_crop_filenames(metadata, saved_indices, crop_config.output_dir / 'face_crops')

    # Filter to only faces with valid crops
    if len(saved_indices) < len(metadata):
        logger.info(f"Filtering data to {len(saved_indices)} faces with valid crops")
        filtered_metadata = [metadata[i] for i in saved_indices]
        filtered_embeddings = embeddings[np.array(saved_indices)]
    else:
        filtered_metadata = metadata
        filtered_embeddings = embeddings
    
    # Step 4: Save filtered embeddings (matching metadata)
    embeddings_file = save_embeddings(filtered_embeddings, output_dir)
    
    # Step 5: Get all clustering methods from config and run them
    method_configs = get_clustering_methods_from_config(config)
    logger.info(f"Found {len(method_configs)} clustering methods: {list(method_configs.keys())}")

    results = run_clustering_methods(filtered_embeddings, method_configs)

    # Step 6: Save results
    benchmark_data = BenchmarkData(
        timestamp=datetime.now().isoformat(),
        album_path=str(args.album_path),
        total_faces=len(filtered_metadata),
        face_metadata=filtered_metadata,
        methods=results,
        embeddings_file=embeddings_file.name  # Just the filename
    )
    results_file = save_benchmark_results(benchmark_data, output_dir)

    # Print summary
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total faces: {len(filtered_metadata)}")
    logger.info("")

    for method_name, method_result in results.items():
        stats = method_result['stats']
        n_clusters = stats.get('n_clusters', 0)
        n_noise = stats.get('n_noise', 0)
        n_singletons = stats.get('n_singletons', 0)

        logger.info(f"{method_name}:")
        logger.info(f"  Clusters: {n_clusters}")
        if n_noise > 0:
            logger.info(f"  Noise: {n_noise}")
        if n_singletons > 0:
            logger.info(f"  Singletons: {n_singletons}")
        if 'n_edges' in stats:
            logger.info(f"  Edges: {stats['n_edges']}")
        if 'merges' in stats:
            logger.info(f"  Merges: {stats['merges'].get('n_merges', 0)}")
        logger.info("")

    logger.info("=" * 70)
    logger.info(f"Results: {results_file}")
    logger.info(f"To view results, run: streamlit run app/face_clustering_comparison.py")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())
