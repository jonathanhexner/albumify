"""
Simple test of degradation system with synthetic or provided images.

This script demonstrates the degradation system by:
1. Creating/using a test image
2. Applying degradations
3. Running a single quality method
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim_bench.image_processing import create_degradation_processor
from sim_bench.quality_assessment import create_quality_assessor


def create_test_image(output_path, size=(512, 512)):
    """
    Create a synthetic test image with various features.

    Args:
        output_path: Where to save the image
        size: Image dimensions
    """
    height, width = size

    # Create gradient background
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Add gradient
    for i in range(height):
        intensity = int((i / height) * 255)
        img[i, :] = [intensity, 128, 255 - intensity]

    # Add some shapes for detail
    cv2.circle(img, (width // 4, height // 4), 50, (255, 255, 0), -1)
    cv2.rectangle(img, (width // 2, height // 2), (3 * width // 4, 3 * height // 4), (0, 255, 255), -1)

    # Add text for sharpness testing
    cv2.putText(img, "TEST IMAGE", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imwrite(str(output_path), img)
    print(f"Created synthetic test image: {output_path}")
    return output_path


def main():
    """Run simple degradation test."""
    print("=== Simple Degradation Test ===\n")

    # Setup
    output_dir = project_root / 'outputs' / 'degradation_test_simple'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create or use test image
    test_image = output_dir / 'test_image.png'

    # Check if user provided an image path as argument
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
        print(f"Using provided image: {test_image}")
    else:
        test_image = create_test_image(test_image)

    # Create degradation processor
    print("\n1. Creating degradation processor...")
    processor = create_degradation_processor(output_dir=output_dir)

    # Apply a few degradations
    print("\n2. Applying degradations...")

    blur_mild = processor.apply_gaussian_blur(test_image, sigma=2.0)
    print(f"   Blur (mild): {blur_mild}")

    blur_severe = processor.apply_gaussian_blur(test_image, sigma=8.0)
    print(f"   Blur (severe): {blur_severe}")

    exposure_dark = processor.apply_exposure_adjustment(test_image, stops=-2)
    print(f"   Exposure (dark): {exposure_dark}")

    jpeg_low = processor.apply_jpeg_compression(test_image, quality=20)
    print(f"   JPEG (low quality): {jpeg_low}")

    # Assess quality with rule-based method
    print("\n3. Assessing quality with rule-based method...")

    config = {
        'type': 'rule_based',
        'weights': {
            'sharpness': 0.4,
            'exposure': 0.3,
            'colorfulness': 0.2,
            'contrast': 0.1
        }
    }

    assessor = create_quality_assessor(config)

    # Score all variants
    variants = {
        'Original': test_image,
        'Blur (mild, sigma=2.0)': blur_mild,
        'Blur (severe, sigma=8.0)': blur_severe,
        'Exposure (-2 stops)': exposure_dark,
        'JPEG (quality=20)': jpeg_low
    }

    print("\nQuality Scores:")
    print("-" * 50)

    for name, path in variants.items():
        score = assessor.assess_image(str(path))
        print(f"{name:25s} : {score:.4f}")

    print("\n" + "=" * 50)
    print("[OK] Test completed!")
    print(f"  Output directory: {output_dir}")
    print("\nExpected behavior:")
    print("  - Blur should decrease score (especially severe)")
    print("  - Dark exposure should decrease score")
    print("  - Low JPEG quality should decrease score")


if __name__ == '__main__':
    main()
