"""
Test image degradations with quality assessment methods.

Applies synthetic degradations (blur, exposure, compression) to test images
and evaluates them with multiple quality assessment methods.
"""

import argparse
from pathlib import Path
from datetime import datetime
import logging
import json

import pandas as pd
from tqdm import tqdm

from sim_bench.image_processing import create_degradation_processor
from sim_bench.quality_assessment import create_quality_assessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test synthetic image degradations with quality assessment'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or directory of images'
    )

    parser.add_argument(
        '--methods',
        type=str,
        default='all',
        help='Comma-separated list of quality assessment methods, or "all" for all 15 methods'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/degradation_test_TIMESTAMP)'
    )

    parser.add_argument(
        '--degradations',
        type=str,
        default='blur,exposure,jpeg',
        help='Comma-separated list of degradation types (blur, exposure, jpeg)'
    )

    parser.add_argument(
        '--blur-sigmas',
        type=str,
        default='0.5,1.0,2.0,4.0,8.0',
        help='Comma-separated blur sigma values'
    )

    parser.add_argument(
        '--exposure-stops',
        type=str,
        default='-3,-2,-1,1,2,3',
        help='Comma-separated exposure adjustment stops'
    )

    parser.add_argument(
        '--jpeg-qualities',
        type=str,
        default='95,80,60,40,20,10',
        help='Comma-separated JPEG quality levels'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for deep learning methods (cpu or cuda)'
    )

    return parser.parse_args()


def get_image_paths(input_path):
    """
    Get list of image paths from input.

    Args:
        input_path: Path to image or directory

    Returns:
        List of image paths
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        return [
            p for p in input_path.iterdir()
            if p.suffix.lower() in extensions
        ]

    return []


def create_method_configs(method_names, device):
    """
    Create quality method configurations - EXACT COPY from pairwise_benchmark.test.yaml

    Args:
        method_names: List of method names or 'all'
        device: Device for deep learning methods

    Returns:
        List of (method_name, config_dict) tuples
    """
    # EXACT COPY from configs/pairwise_benchmark.test.yaml
    all_methods_configs = [
        # Rule-based methods (5 methods)
        {'name': 'Sharpness', 'type': 'rule_based', 'weights': {'sharpness': 1.0, 'exposure': 0.0, 'colorfulness': 0.0, 'contrast': 0.0}},
        {'name': 'Exposure', 'type': 'rule_based', 'weights': {'sharpness': 0.0, 'exposure': 1.0, 'colorfulness': 0.0, 'contrast': 0.0}},
        {'name': 'Contrast', 'type': 'rule_based', 'weights': {'sharpness': 0.0, 'exposure': 0.0, 'colorfulness': 0.0, 'contrast': 1.0}},
        {'name': 'Colorfulness', 'type': 'rule_based', 'weights': {'sharpness': 0.0, 'exposure': 0.0, 'colorfulness': 1.0, 'contrast': 0.0}},
        {'name': 'Combined-RuleBased', 'type': 'rule_based', 'weights': {'sharpness': 0.4, 'exposure': 0.3, 'colorfulness': 0.2, 'contrast': 0.1}},

        # CLIP methods with attribute-specific prompts (7 methods)
        {'name': 'CLIP-Aesthetic-Overall', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a highly aesthetic, visually pleasing, beautiful photograph', 'an unattractive, poorly composed, ugly photograph'], 'positive': [], 'negative': []}},
        {'name': 'CLIP-Composition', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a well-composed photograph with excellent visual balance', 'a poorly-composed photograph with bad visual balance'], 'positive': [], 'negative': []}},
        {'name': 'CLIP-Subject-Placement', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a photo with the subject well placed in the frame', 'a photo with the subject not well placed in the frame'], 'positive': [], 'negative': []}},
        {'name': 'CLIP-Cropping', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a photo that is well cropped and shows the complete subject', 'a photo that is poorly cropped or cuts off the subject'], 'positive': [], 'negative': []}},
        {'name': 'CLIP-Sharpness', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a sharp, in-focus photograph with clear details', 'a blurry, out-of-focus photograph with unclear details'], 'positive': [], 'negative': []}},
        {'name': 'CLIP-Exposure', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a photo with good exposure and lighting', 'a photo with poor exposure, too dark or too bright'], 'positive': [], 'negative': []}},
        {'name': 'CLIP-Color', 'type': 'clip_aesthetic', 'model_name': 'ViT-B-32', 'device': device, 'aggregation_method': 'contrastive_only', 'custom_prompts': {'contrastive': ['a photo with vibrant, natural colors', 'a photo with dull, washed out colors'], 'positive': [], 'negative': []}},

        # Deep Learning Methods
        {'name': 'NIMA-MobileNet', 'type': 'nima', 'backbone': 'mobilenet_v2', 'device': device, 'batch_size': 8},
        {'name': 'NIMA-ResNet50', 'type': 'nima', 'backbone': 'resnet50', 'device': device, 'batch_size': 4},
        {'name': 'MUSIQ', 'type': 'musiq', 'use_pyiqa': True, 'device': device, 'batch_size': 1},
        {'name': 'CLIP-Aesthetic-LAION', 'type': 'clip_aesthetic', 'variant': 'laion', 'model_name': 'ViT-B-32', 'device': device},
    ]

    # Handle 'all' keyword
    if len(method_names) == 1 and method_names[0].strip().lower() == 'all':
        return [(cfg['name'], cfg) for cfg in all_methods_configs]

    # Build configs for specific methods
    method_configs = []
    for name in method_names:
        name = name.strip()

        # Find matching config
        matching_config = next((cfg for cfg in all_methods_configs if cfg['name'] == name), None)

        if matching_config:
            method_configs.append((matching_config['name'], matching_config))
        else:
            logger.warning(f"Unknown method: {name}, skipping")

    return method_configs


def main():
    """Main execution function."""
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("outputs") / f"degradation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Get input images
    image_paths = get_image_paths(args.input)
    logger.info(f"Found {len(image_paths)} input images")

    # Parse degradation parameters
    blur_sigmas = [float(x) for x in args.blur_sigmas.split(',')]
    exposure_stops = [float(x) for x in args.exposure_stops.split(',')]
    jpeg_qualities = [int(x) for x in args.jpeg_qualities.split(',')]

    # Create degradation processor
    degradation_processor = create_degradation_processor(output_dir=output_dir)

    # Generate degraded variants for each image
    logger.info("Generating degraded variants...")
    all_degraded_paths = {}

    for image_path in tqdm(image_paths, desc="Degrading images"):
        degraded_variants = degradation_processor.apply_degradation_suite(
            image_path=image_path,
            blur_sigmas=blur_sigmas,
            exposure_stops=exposure_stops,
            jpeg_qualities=jpeg_qualities,
            output_base_dir=output_dir
        )
        all_degraded_paths[image_path.stem] = degraded_variants

    # Create quality assessment methods
    method_names = args.methods.split(',')
    method_configs = create_method_configs(method_names, args.device)

    logger.info(f"Testing {len(method_configs)} quality assessment methods")

    # Assess quality for all variants
    results = []

    for method_name, config in method_configs:
        logger.info(f"Running method: {method_name}")

        assessor = create_quality_assessor(config)

        for image_name, degraded_paths in all_degraded_paths.items():
            for degradation_name, degraded_path in tqdm(
                degraded_paths.items(),
                desc=f"{method_name} - {image_name}",
                leave=False
            ):
                score = assessor.assess_image(str(degraded_path))

                # Parse degradation type and level
                degradation_type = degradation_name.split('_')[0]
                degradation_level = '_'.join(degradation_name.split('_')[1:]) \
                    if degradation_name != 'original' else 'original'

                results.append({
                    'image_name': image_name,
                    'degradation_type': degradation_type,
                    'degradation_level': degradation_level,
                    'degradation_full': degradation_name,
                    'method': method_name,
                    'score': score,
                    'file_path': str(degraded_path)
                })

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'results.csv'
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved results to {results_csv}")

    # Save metadata
    metadata = {
        'input_path': str(args.input),
        'num_images': len(image_paths),
        'image_names': [p.stem for p in image_paths],
        'methods': method_names,
        'degradations': {
            'blur_sigmas': blur_sigmas,
            'exposure_stops': exposure_stops,
            'jpeg_qualities': jpeg_qualities
        },
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }

    metadata_json = output_dir / 'metadata.json'
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_json}")

    # Print summary statistics
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Total assessments: {len(results_df)}")
    logger.info(f"Images tested: {len(image_paths)}")
    logger.info(f"Degradation variants per image: {len(all_degraded_paths[image_paths[0].stem])}")
    logger.info(f"Methods tested: {len(method_configs)}")

    # Show score ranges per method
    logger.info("\nScore ranges by method:")
    for method in method_names:
        method_scores = results_df[results_df['method'] == method]['score']
        if len(method_scores) > 0:
            logger.info(
                f"  {method}: {method_scores.min():.3f} - {method_scores.max():.3f} "
                f"(mean: {method_scores.mean():.3f})"
            )

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
