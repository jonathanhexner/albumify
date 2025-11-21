"""
Image degradation processor for synthetic quality testing.

Applies controlled degradations (blur, exposure, compression) to test
how quality assessment methods respond to known quality changes.
"""

from pathlib import Path
from typing import Union, Dict, List, Optional
import logging

import cv2
import numpy as np

from sim_bench.image_processing.base import ImageProcessor


logger = logging.getLogger(__name__)


class ImageDegradationProcessor(ImageProcessor):
    """
    Apply synthetic degradations to images for quality assessment validation.

    Supports three degradation types:
    - Gaussian blur (tests sharpness detection)
    - Exposure adjustment (tests exposure/histogram quality)
    - JPEG compression (tests artifact detection)
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = False
    ):
        """
        Initialize degradation processor.

        Args:
            output_dir: Directory for saving degraded images
            enable_cache: Whether to cache results (typically False for degradations)
        """
        super().__init__(cache_dir=output_dir, enable_cache=enable_cache)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/degraded")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(
        self,
        image_path: Union[str, Path],
        degradation_type: str = "blur",
        **kwargs
    ) -> Path:
        """
        Apply degradation to image.

        Args:
            image_path: Path to input image
            degradation_type: Type of degradation (blur, exposure, jpeg)
            **kwargs: Degradation-specific parameters

        Returns:
            Path to degraded image
        """
        degradation_map = {
            "blur": self.apply_gaussian_blur,
            "exposure": self.apply_exposure_adjustment,
            "jpeg": self.apply_jpeg_compression
        }

        handler = degradation_map.get(degradation_type)
        return handler(image_path, **kwargs)

    def apply_gaussian_blur(
        self,
        image_path: Union[str, Path],
        sigma: float = 2.0,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Apply Gaussian blur to image.

        Args:
            image_path: Path to input image
            sigma: Blur strength (standard deviation)
            output_path: Output path (auto-generated if None)

        Returns:
            Path to blurred image
        """
        img = cv2.imread(str(image_path))

        # Compute kernel size from sigma (ensure odd size)
        ksize = int(2 * np.ceil(3 * sigma) + 1)

        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        output_path = self._get_output_path(
            image_path, output_path, f"blur_sigma_{sigma}"
        )
        cv2.imwrite(str(output_path), blurred)

        return output_path

    def apply_exposure_adjustment(
        self,
        image_path: Union[str, Path],
        stops: float = 1.0,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Adjust image exposure by stops (EV).

        Args:
            image_path: Path to input image
            stops: Exposure adjustment in stops (positive=brighter, negative=darker)
            output_path: Output path (auto-generated if None)

        Returns:
            Path to adjusted image
        """
        img = cv2.imread(str(image_path))

        # Convert stops to multiplier: 1 stop = 2x brightness
        multiplier = 2.0 ** stops

        # Apply exposure adjustment with clipping
        adjusted = cv2.convertScaleAbs(img, alpha=multiplier, beta=0)

        sign = "plus" if stops >= 0 else "minus"
        abs_stops = abs(stops)
        output_path = self._get_output_path(
            image_path, output_path, f"exposure_{sign}_{abs_stops}"
        )
        cv2.imwrite(str(output_path), adjusted)

        return output_path

    def apply_jpeg_compression(
        self,
        image_path: Union[str, Path],
        quality: int = 80,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Apply JPEG compression to image.

        Args:
            image_path: Path to input image
            quality: JPEG quality (0-100, higher is better)
            output_path: Output path (auto-generated if None)

        Returns:
            Path to compressed image
        """
        img = cv2.imread(str(image_path))

        output_path = self._get_output_path(
            image_path, output_path, f"jpeg_q_{quality}"
        )

        # Ensure output has .jpg extension
        output_path = output_path.with_suffix('.jpg')

        cv2.imwrite(
            str(output_path),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        return output_path

    def apply_degradation_suite(
        self,
        image_path: Union[str, Path],
        blur_sigmas: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0],
        exposure_stops: List[float] = [-3, -2, -1, 1, 2, 3],
        jpeg_qualities: List[int] = [95, 80, 60, 40, 20, 10],
        output_base_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Path]:
        """
        Apply complete suite of degradations to image.

        Args:
            image_path: Path to input image
            blur_sigmas: List of blur sigma values to test
            exposure_stops: List of exposure adjustments to test
            jpeg_qualities: List of JPEG quality levels to test
            output_base_dir: Base directory for outputs

        Returns:
            Dict mapping degradation name to output path
        """
        image_path = Path(image_path)
        base_dir = Path(output_base_dir) if output_base_dir else self.output_dir

        results = {}

        # Save original
        original_dir = base_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        original_path = original_dir / image_path.name

        img = cv2.imread(str(image_path))
        cv2.imwrite(str(original_path), img)
        results['original'] = original_path

        # Apply blur degradations
        for sigma in blur_sigmas:
            blur_dir = base_dir / f"blur_sigma_{sigma}"
            blur_dir.mkdir(parents=True, exist_ok=True)
            blur_path = blur_dir / image_path.name

            self.apply_gaussian_blur(image_path, sigma=sigma, output_path=blur_path)
            results[f'blur_sigma_{sigma}'] = blur_path

        # Apply exposure degradations
        for stops in exposure_stops:
            sign = "plus" if stops >= 0 else "minus"
            abs_stops = abs(stops)
            exp_dir = base_dir / f"exposure_{sign}_{abs_stops}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            exp_path = exp_dir / image_path.name

            self.apply_exposure_adjustment(image_path, stops=stops, output_path=exp_path)
            results[f'exposure_{sign}_{abs_stops}'] = exp_path

        # Apply JPEG compression degradations
        for quality in jpeg_qualities:
            jpeg_dir = base_dir / f"jpeg_q_{quality}"
            jpeg_dir.mkdir(parents=True, exist_ok=True)
            jpeg_path = jpeg_dir / image_path.stem
            jpeg_path = jpeg_path.with_suffix('.jpg')

            self.apply_jpeg_compression(image_path, quality=quality, output_path=jpeg_path)
            results[f'jpeg_q_{quality}'] = jpeg_path

        logger.info(f"Generated {len(results)} degraded variants for {image_path.name}")
        return results

    def _get_output_path(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]],
        degradation_name: str
    ) -> Path:
        """
        Generate output path for degraded image.

        Args:
            input_path: Original image path
            output_path: Explicit output path (if provided)
            degradation_name: Name of degradation for directory

        Returns:
            Output path
        """
        # Return explicit path if provided
        if output_path:
            return Path(output_path)

        # Auto-generate path
        input_path = Path(input_path)
        degradation_dir = self.output_dir / degradation_name
        degradation_dir.mkdir(parents=True, exist_ok=True)

        return degradation_dir / input_path.name


def create_degradation_processor(
    output_dir: Optional[Union[str, Path]] = None
) -> ImageDegradationProcessor:
    """
    Factory function to create degradation processor.

    Args:
        output_dir: Directory for saving degraded images

    Returns:
        Configured ImageDegradationProcessor
    """
    return ImageDegradationProcessor(output_dir=output_dir)
