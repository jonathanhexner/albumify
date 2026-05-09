#!/usr/bin/env python3
"""Save face debug artifacts for an album.

Usage:
    python scripts/save_face_debug.py /path/to/album
    python scripts/save_face_debug.py /path/to/album --output debug_output

Outputs folder structure:
    debug_faces/
      {image_name}/
        face_000_raw.jpg              # bbox crop only
        face_000_raw_landmarks.jpg    # bbox crop + landmarks
        face_000_aligned.jpg          # orientation-corrected + 5-point aligned
        face_000_aligned_landmarks.jpg
        face_000_info.txt             # orientation, confidence, etc.
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import pipeline components
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.all_steps import (
    DiscoverImagesStep,
    InsightFaceDetectFacesStep,
    DetectFaceOrientationStep,
    SaveFaceDebugArtifactsStep,
)


def main():
    parser = argparse.ArgumentParser(description="Save face debug artifacts")
    parser.add_argument("album", type=Path, help="Path to album directory")
    parser.add_argument("--output", "-o", type=Path, default=Path("debug_faces"),
                        help="Output directory (default: debug_faces)")
    args = parser.parse_args()

    if not args.album.exists():
        logger.error(f"Album not found: {args.album}")
        sys.exit(1)

    logger.info(f"Processing album: {args.album}")
    logger.info(f"Output directory: {args.output}")

    # Create context
    context = PipelineContext(source_directory=args.album)

    # Run steps
    steps = [
        (DiscoverImagesStep(), {}),
        (InsightFaceDetectFacesStep(), {
            "model_name": "buffalo_l",
            "detection_threshold": 0.5,
            "device": "cpu",
        }),
        (DetectFaceOrientationStep(), {
            "log_orientations": True,
        }),
        (SaveFaceDebugArtifactsStep(), {
            "output_dir": str(args.output),
            "target_size": 256,
            "save_raw": True,
            "save_aligned": True,
            "margin": 0.2,
        }),
    ]

    for step, config in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {step._metadata.display_name}")
        logger.info(f"{'='*60}")
        step.process(context, config)

    logger.info(f"\n{'='*60}")
    logger.info("DONE")
    logger.info(f"{'='*60}")
    logger.info(f"Debug artifacts saved to: {args.output.absolute()}")
    logger.info(f"Open the folder to inspect raw vs aligned crops")


if __name__ == "__main__":
    main()
