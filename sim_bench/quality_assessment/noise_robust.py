"""
Noise-robust sharpness and noise estimation (spec-098).

Sensor noise is high-frequency energy, so raw Laplacian variance INFLATES on grainy
images (39x on SIDD) and inverts quality ranking. These helpers fix that:

- noise_robust_laplacian_var: 3x3 median blur (kills isolated speckles), Laplacian
  variance, then subtraction of the noise's own residual contribution
  (K_SIGMA * sigma^2). Median blur alone is NOT enough: strong sensor noise still
  inflated the variance 11x on SIDD; with the sigma correction the inflation is
  gone while RealBlur blur-ranking stays at 99.3% (sweep: scripts/
  experiment_spec098_sweep2.py).
- estimate_noise_sigma: wavelet-based noise standard deviation (skimage); on SIDD
  clean-vs-noisy pairs it scored 100% pair accuracy / AUC 0.993 at ~20 ms/image
  (see reports/2026-07-10_defect_noise/).
- noise_sigma_to_score: maps sigma to a [0, 1] higher-is-better quality component.

Framework-agnostic (numpy in / float out) so notebooks can use them directly,
per the spec-053 helper convention.
"""

import cv2
import numpy as np

# Calibrated on SIDD-Small sigma distributions (spec-098 T0.3):
# clean median score 0.76 (target >= 0.7), ISO>=1600 noisy median 0.14 (target <= 0.2).
SIGMA_HALF = 2.0

# Sharpness correction coefficient (spec-098 sweep v2): additive noise with std
# sigma contributes ~sigma^2 * (kernel sum-of-squares) to Laplacian variance;
# after the 3x3 median a residual remains. K=5 empirically zeroes the SIDD
# inflation without moving RealBlur pair accuracy (0.9929 at k=0 and k=5).
K_SIGMA = 5.0

# Sigma estimation runs on a central crop for large images: noise statistics are
# spatially stationary, and resizing (unlike cropping) would smooth the noise away.
_SIGMA_CROP = 1024


def noise_robust_laplacian_var(gray: np.ndarray, sigma: float = None) -> float:
    """
    Sharpness as Laplacian variance, noise-corrected (spec-098).

    Pipeline: 3x3 median blur (removes isolated speckles) -> Laplacian variance
    -> subtract the residual noise contribution K_SIGMA * sigma^2, floored at 0.
    A grain-drowned image whose detail is below its own noise floor correctly
    reads as "not sharp".

    Args:
        gray: 2-D grayscale image (uint8 or float32).
        sigma: pre-computed noise sigma for this image; estimated from ``gray``
               when None (pass it in to avoid double estimation).

    Returns:
        Corrected Laplacian variance (raw, unnormalized, >= 0).
    """
    if gray.dtype not in (np.uint8, np.uint16, np.float32):
        gray = gray.astype(np.float32)
    if sigma is None:
        sigma = estimate_noise_sigma(gray)
    smoothed = cv2.medianBlur(gray, 3)
    lap_var = float(cv2.Laplacian(smoothed, cv2.CV_64F).var())
    return max(lap_var - K_SIGMA * sigma * sigma, 0.0)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """
    Wavelet-based noise standard deviation estimate.

    Args:
        img: BGR (H, W, 3) or grayscale (H, W) image.

    Returns:
        Estimated noise sigma in pixel-value units (higher = noisier).
    """
    from skimage.restoration import estimate_sigma

    h, w = img.shape[:2]
    if max(h, w) > _SIGMA_CROP:  # central crop: keeps noise stats, caps cost
        top = max(0, (h - _SIGMA_CROP) // 2)
        left = max(0, (w - _SIGMA_CROP) // 2)
        img = img[top:top + _SIGMA_CROP, left:left + _SIGMA_CROP]

    if img.ndim == 3:
        # Green channel only: standard luminance proxy, 6x faster than averaging
        # all 3 channels with identical pair discrimination on SIDD (spec-098 T0.4).
        sigma = estimate_sigma(img[:, :, 1])
    else:
        sigma = estimate_sigma(img)
    return float(sigma)


def noise_sigma_to_score(sigma: float, sigma_half: float = SIGMA_HALF) -> float:
    """
    Map noise sigma to a [0, 1] quality score (higher = cleaner).

    score = 1 / (1 + sigma / sigma_half); sigma_half is the sigma at which the
    score reaches 0.5.
    """
    return float(1.0 / (1.0 + max(sigma, 0.0) / sigma_half))
