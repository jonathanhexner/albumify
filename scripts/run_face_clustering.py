"""Run the face clustering pipeline on an image directory.

Usage:
    python scripts/run_face_clustering.py --images <dir> --output <dir>
    python scripts/run_face_clustering.py --images D:\\Google_Germany --output results\\Germany_clusters
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import FaceClusteringPipeline, PipelineConfig, PipelineStageError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def on_progress(stage: str, fraction: float, message: str):
    filled = int(fraction * 20)
    bar = "#" * filled + "-" * (20 - filled)
    # Only print start (0%) and done (100%) to keep console clean
    if fraction == 0.0 or fraction >= 1.0:
        print(f"  [{stage:10s}] [{bar}] {fraction:5.0%}  {message}")


def main():
    parser = argparse.ArgumentParser(description="Run face clustering pipeline")
    parser.add_argument("--images", required=True, type=Path, help="Source image directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for results")
    parser.add_argument("--distance-threshold", type=float, default=0.35)
    parser.add_argument("--blur-min", type=float, default=50.0)
    parser.add_argument("--require-pose", action="store_true", default=False)
    parser.add_argument("--max-faces-per-image", type=int, default=3)
    args = parser.parse_args()

    config = PipelineConfig(
        distance_threshold=args.distance_threshold,
        blur_min=args.blur_min,
        require_pose=args.require_pose,
        max_faces_per_image_core=args.max_faces_per_image,
    )

    print(f"\nFace Clustering Pipeline")
    print(f"  Images:  {args.images}")
    print(f"  Output:  {args.output}")
    print(f"  Config:  distance_threshold={config.distance_threshold}, blur_min={config.blur_min}, require_pose={config.require_pose}\n")

    pipeline = FaceClusteringPipeline(config)
    try:
        result = pipeline.run(args.images, args.output, on_progress=on_progress)
    except PipelineStageError as e:
        logger.error(str(e))
        sys.exit(1)

    print(f"\nDone!")
    print(f"  Faces:    {result.summary['n_faces']} total, {result.summary['n_core']} core")
    print(f"  Clusters: {result.summary['n_clusters']}")
    print(f"  Noise:    {result.summary['n_noise']}")
    print(f"  Output:   {result.output_dir}")


if __name__ == "__main__":
    main()
