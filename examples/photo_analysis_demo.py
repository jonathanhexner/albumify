"""
Photo Analysis Demo

Demonstrates the new photo analysis capabilities:
1. Thumbnail generation (multi-resolution)
2. CLIP-based tagging (55 prompts)
3. Importance scoring
4. Routing decisions
5. Batch processing with HTML report
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.config import setup_logging
from sim_bench.photo_analysis import generate_html_report

logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for demo execution."""
    samples_dir: Path
    thumbnail_sizes: List[str]
    max_images: int
    min_images: int = 5


def _get_sample_images(samples_dir: Path, max_images: int) -> Optional[List[Path]]:
    """Get sample images from directory."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        logger.warning(f"Directory not found: {samples_dir}")
        return None
    
    images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg"))
    return images[:max_images] if images else None


def demo_thumbnail_generation(config: DemoConfig) -> Dict[str, Path]:
    """Demo 1: Multi-resolution thumbnail generation."""
    logger.info("Generating thumbnails...")

    from sim_bench.image_processing import ThumbnailGenerator

    generator = ThumbnailGenerator(cache_dir=".cache/demo_thumbnails")
    sample_images = _get_sample_images(config.samples_dir, config.max_images)
    
    if not sample_images:
        logger.error("No images found")
        return {}

    thumbnails = {}
    for image_path in sample_images:
        generated = generator.generate(image_path, sizes=config.thumbnail_sizes)
        thumbnails[str(image_path)] = generated

    logger.info(f"Generated thumbnails for {len(thumbnails)} images")
    return thumbnails


def demo_clip_tagging(config: DemoConfig, thumbnails: Optional[Dict[str, Dict[str, Path]]] = None) -> Dict[str, Dict]:
    """Demo 2: CLIP-based photo tagging."""
    logger.info("Analyzing images with CLIP...")

    from sim_bench.photo_analysis import CLIPTagger

    tagger = CLIPTagger(device='cpu')
    sample_images = _get_sample_images(config.samples_dir, config.max_images)
    
    if not sample_images:
        logger.error("No images found")
        return {}

    # Use thumbnails if available, otherwise use originals
    image_paths = []
    original_mapping = {}
    
    for img_path in sample_images:
        if thumbnails and str(img_path) in thumbnails:
            # Use smallest thumbnail for speed
            thumb_path = thumbnails[str(img_path)].get('tiny') or thumbnails[str(img_path)].get('small')
            if thumb_path:
                image_paths.append(thumb_path)
                original_mapping[str(thumb_path)] = img_path
        else:
            image_paths.append(img_path)
            original_mapping[str(img_path)] = img_path

    analysis_results = tagger.analyze_batch(
        image_paths,
        batch_size=8,
        verbose=False
    )
    
    logger.info(f"Analyzed {len(analysis_results)} images")
    return analysis_results, original_mapping


def demo_batch_processing(config: DemoConfig) -> Optional[Path]:
    """Demo 3: Complete batch processing pipeline with HTML report."""
    logger.info("Starting batch processing pipeline...")

    from sim_bench.image_processing import ThumbnailGenerator
    from sim_bench.photo_analysis import CLIPTagger

    sample_images = _get_sample_images(config.samples_dir, config.max_images)
    if not sample_images:
        logger.error("No images found")
        return None
        
    if len(sample_images) < config.min_images:
        logger.warning(f"Only {len(sample_images)} images found, need at least {config.min_images}")
        return None

    # Step 1: Generate thumbnails
    logger.info("Step 1/3: Generating thumbnails...")
    generator = ThumbnailGenerator(cache_dir=".cache/demo_thumbnails")
    thumbnail_results = generator.process_batch(
        sample_images,
        sizes=['tiny'],
        num_workers=4,
        verbose=False
    )

    # Step 2: Analyze with CLIP
    logger.info("Step 2/3: Analyzing with CLIP...")
    tagger = CLIPTagger(device='cpu')

    tiny_paths = [
        thumbnail_results[str(img)]['tiny']
        for img in sample_images
        if str(img) in thumbnail_results and 'tiny' in thumbnail_results[str(img)]
    ]

    if not tiny_paths:
        logger.error("No thumbnails generated")
        return None

    analysis_results = tagger.analyze_batch(
        tiny_paths,
        batch_size=8,
        verbose=False
    )

    # Step 3: Generate HTML report
    logger.info("Step 3/3: Generating HTML report...")
    
    # Map analysis paths to original images
    original_mapping = {
        str(thumb_path): img_path
        for img_path in sample_images
        if str(img_path) in thumbnail_results
        for thumb_path in [thumbnail_results[str(img_path)].get('tiny')]
        if thumb_path
    }

    output_file = Path("outputs/demo_analysis_report.html")
    report_path = generate_html_report(
        analysis_results=analysis_results,
        original_images=original_mapping,
        output_path=output_file,
        title="Photo Analysis Demo Report"
    )

    logger.info(f"Report generated: {report_path}")
    return report_path


def _run_demo(demo_func, config: DemoConfig, demo_name: str):
    """Run a demo function with error handling."""
    try:
        return demo_func(config)
    except Exception as e:
        logger.error(f"Demo {demo_name} failed: {e}", exc_info=True)
        return None


def main():
    """Run all demos and generate report."""
    logger.info("=" * 60)
    logger.info("Photo Analysis Demo")
    logger.info("=" * 60)

    config = DemoConfig(
        samples_dir=Path("D:\Budapest2025_Google"),
        thumbnail_sizes=['tiny', 'small'],
        max_images=20
    )

    # Run batch processing (includes all steps)
    report_path = _run_demo(demo_batch_processing, config, "Batch Processing")
    
    if report_path:
        logger.info("=" * 60)
        logger.info(f"[OK] Demo complete! View report: {report_path}")
        logger.info("=" * 60)
    else:
        logger.error("Demo failed - check logs for details")


if __name__ == "__main__":
    setup_logging()
    main()
