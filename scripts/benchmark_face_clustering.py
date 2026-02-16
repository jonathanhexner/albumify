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


def align_crop_by_roll(crop_img: Image.Image, roll_angle: float, threshold: float = 5.0) -> Image.Image:
    """Rotate crop to align face (eyes horizontal)."""
    if abs(roll_angle) < threshold:
        return crop_img
    
    # Convert PIL to numpy for rotation
    crop_np = np.array(crop_img)
    h, w = crop_np.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotate to counter the roll
    rotation_matrix = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
    aligned = cv2.warpAffine(
        crop_np,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return Image.fromarray(aligned)


def save_single_face_crop(face_meta: Dict[str, Any], index: int, config: CropConfig) -> bool:
    """Save a single face crop with roll alignment. Returns True if successful."""
    image_path = Path(face_meta['image_path'])
    
    bbox = face_meta['bbox']
    x_px = int(bbox.get('x_px', 0))
    y_px = int(bbox.get('y_px', 0))
    w_px = int(bbox.get('w_px', 0))
    h_px = int(bbox.get('h_px', 0))
    
    if not is_valid_bbox(w_px, h_px):
        return False
    
    # CRITICAL: Apply EXIF rotation before cropping
    # Bbox coordinates are relative to the correctly-oriented image
    img = ImageOps.exif_transpose(Image.open(image_path))
    pad = int(min(w_px, h_px) * config.padding_ratio)
    left, top, right, bottom = compute_crop_coordinates(
        x_px, y_px, w_px, h_px, img.width, img.height, pad
    )
    
    if not is_valid_crop_coords(left, top, right, bottom):
        return False
    
    face_crop = img.crop((left, top, right, bottom))
    
    # Apply roll alignment
    roll_angle = face_meta.get('roll_angle', 0.0)
    face_crop = align_crop_by_roll(face_crop, roll_angle)
    
    face_crop = face_crop.resize((config.crop_size, config.crop_size), Image.Resampling.LANCZOS)
    
    crop_path = config.output_dir / 'face_crops' / f'face_{index:04d}.jpg'
    face_crop.save(crop_path, quality=95)
    return True


def save_face_crops(metadata: List[Dict[str, Any]], config: CropConfig) -> List[int]:
    """Save face crops for visualization. Returns list of successfully saved indices."""
    crops_dir = config.output_dir / 'face_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(metadata)} face crops...")
    
    saved_indices = []
    saved_count = 0
    
    for i, face_meta in enumerate(metadata):
        if save_single_face_crop(face_meta, saved_count, config):
            saved_indices.append(i)
            saved_count += 1
    
    logger.info(f"Face crops saved: {saved_count}/{len(metadata)}")
    if saved_count < len(metadata):
        logger.warning(f"Skipped {len(metadata) - saved_count} faces due to invalid crops")
    
    return saved_indices


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


def run_clustering_methods(
    embeddings: np.ndarray,
    hdbscan_config: Dict[str, Any],
    hybrid_config: Dict[str, Any],
    closest_config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Run all clustering methods and return results with statistics."""
    results = {}
    
    # HDBSCAN
    logger.info("Running HDBSCAN clustering...")
    hdbscan_method = load_clustering_method(hdbscan_config)
    hdbscan_labels, hdbscan_stats = hdbscan_method.cluster(embeddings)
    hdbscan_labels_array = np.array(hdbscan_labels)
    results['hdbscan'] = {
        'labels': hdbscan_labels,
        'stats': hdbscan_stats,
        'cluster_stats': calculate_cluster_statistics(embeddings, hdbscan_labels_array)
    }
    logger.info(f"HDBSCAN: {hdbscan_stats['n_clusters']} clusters")
    
    # Hybrid HDBSCAN+kNN (centroid-based)
    logger.info("Running Hybrid HDBSCAN+kNN (centroid) clustering...")
    hybrid_method = load_clustering_method(hybrid_config)
    # Collect debug data for detailed analysis in the comparison app
    hybrid_labels, hybrid_stats = hybrid_method.cluster(embeddings, collect_debug_data=True)
    hybrid_labels_array = np.array(hybrid_labels)
    results['hybrid_knn'] = {
        'labels': hybrid_labels,
        'stats': hybrid_stats,
        'cluster_stats': calculate_cluster_statistics(embeddings, hybrid_labels_array)
    }
    logger.info(f"Hybrid kNN: {hybrid_stats['n_clusters']} clusters")
    
    # Hybrid Closest-Face
    logger.info("Running Hybrid Closest-Face clustering...")
    closest_method = load_clustering_method(closest_config)
    closest_labels, closest_stats = closest_method.cluster(embeddings)
    closest_labels_array = np.array(closest_labels)
    results['hybrid_closest'] = {
        'labels': closest_labels,
        'stats': closest_stats,
        'cluster_stats': calculate_cluster_statistics(embeddings, closest_labels_array)
    }
    logger.info(f"Hybrid Closest: {closest_stats['n_clusters']} clusters")
    
    return results


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
    
    # Step 5: Run clustering methods on filtered data
    results = run_clustering_methods(
        filtered_embeddings,
        config['hdbscan'],
        config['hybrid_knn'],
        config['hybrid_closest']
    )
    
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
    logger.info(f"Total faces: {len(metadata)}")
    logger.info("")
    logger.info("HDBSCAN:")
    logger.info(f"  Clusters: {results['hdbscan']['stats']['n_clusters']}")
    if 'n_noise' in results['hdbscan']['stats']:
        logger.info(f"  Noise: {results['hdbscan']['stats']['n_noise']}")
    logger.info("")
    logger.info("Hybrid HDBSCAN+kNN:")
    logger.info(f"  Clusters: {results['hybrid_knn']['stats']['n_clusters']}")
    if 'merges' in results['hybrid_knn']['stats']:
        logger.info(f"  Merges: {results['hybrid_knn']['stats']['merges']['n_merges']}")
    if 'singletons' in results['hybrid_knn']['stats']:
        logger.info(f"  Attached: {results['hybrid_knn']['stats']['singletons']['n_attached']}")
        logger.info(f"  Singletons: {results['hybrid_knn']['stats']['singletons']['n_singletons']}")
    logger.info("=" * 70)
    logger.info(f"Results: {results_file}")
    logger.info(f"To view results, run: streamlit run app/face_clustering_comparison.py")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())
