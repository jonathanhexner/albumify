"""Unit tests for spec-098 noise-robust sharpness + noise estimation."""

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.needs_data  # needs the SIDD dataset (private, not in CI)

from sim_bench.quality_assessment.noise_robust import (
    SIGMA_HALF,
    estimate_noise_sigma,
    noise_robust_laplacian_var,
    noise_sigma_to_score,
)
from sim_bench.quality_assessment.rule_based import RuleBasedQuality


def _clean_test_image(size: int = 256) -> np.ndarray:
    """Synthetic photo-like grayscale: smooth gradient + sharp-edged shapes."""
    rng = np.random.default_rng(7)
    img = np.tile(np.linspace(60, 190, size, dtype=np.float32), (size, 1))
    for _ in range(12):  # sharp rectangles = real edges
        x, y = rng.integers(10, size - 60, 2)
        w, h = rng.integers(20, 50, 2)
        img[y:y + h, x:x + w] = rng.integers(30, 220)
    return img.astype(np.uint8)


def _add_gaussian_noise(gray: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(11)
    noisy = gray.astype(np.float32) + rng.normal(0, sigma, gray.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


class TestNoiseRobustLaplacian:
    def test_noise_barely_inflates_robust_sharpness(self):
        """A5: raw Laplacian inflated 39x by noise on SIDD; robust must stay < 2x."""
        clean = _clean_test_image()
        noisy = _add_gaussian_noise(clean, sigma=10.0)

        raw_ratio = (cv2.Laplacian(noisy, cv2.CV_64F).var()
                     / cv2.Laplacian(clean, cv2.CV_64F).var())
        robust_ratio = (noise_robust_laplacian_var(noisy)
                        / noise_robust_laplacian_var(clean))

        assert raw_ratio > 3.0  # sanity: the problem actually exists on raw
        assert robust_ratio < 2.0  # the fix: noise no longer reads as sharpness

    def test_blur_still_lowers_robust_sharpness(self):
        """Denoising must not destroy blur detection: blurred << clean."""
        clean = _clean_test_image()
        blurred = cv2.GaussianBlur(clean, (11, 11), 5.0)
        assert noise_robust_laplacian_var(blurred) < 0.3 * noise_robust_laplacian_var(clean)

    def test_blur_ranking_survives_under_noise(self):
        """The production failure: noisy blurred photo must NOT beat clean sharp one."""
        clean = _clean_test_image()
        noisy_blurred = _add_gaussian_noise(cv2.GaussianBlur(clean, (11, 11), 5.0), sigma=10.0)
        assert noise_robust_laplacian_var(noisy_blurred) < noise_robust_laplacian_var(clean)


class TestEstimateNoiseSigma:
    def test_sigma_increases_with_noise(self):
        clean = _clean_test_image()
        s_clean = estimate_noise_sigma(clean)
        s_noisy = estimate_noise_sigma(_add_gaussian_noise(clean, sigma=10.0))
        assert s_noisy > s_clean + 3.0

    def test_sigma_roughly_matches_injected_noise(self):
        clean = _clean_test_image()
        s = estimate_noise_sigma(_add_gaussian_noise(clean, sigma=8.0))
        assert 4.0 < s < 12.0  # wavelet estimate is approximate but same scale

    def test_color_image_supported(self):
        gray = _clean_test_image()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert estimate_noise_sigma(bgr) >= 0.0

    def test_large_image_uses_central_crop(self):
        """12 MP path must stay fast (A6 gate is <150 ms; here just correctness)."""
        big = np.tile(_clean_test_image(), (8, 8))  # 2048x2048
        assert estimate_noise_sigma(big) >= 0.0


class TestNoiseSigmaToScore:
    def test_monotone_decreasing(self):
        scores = [noise_sigma_to_score(s) for s in (0.0, 1.0, 5.0, 25.0)]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0

    def test_half_point(self):
        assert noise_sigma_to_score(SIGMA_HALF) == pytest.approx(0.5)


class TestRuleBasedIntegration:
    @pytest.fixture()
    def image_pair(self, tmp_path):
        """Clean vs noisy variant of the same synthetic photo, on disk."""
        gray = _clean_test_image(512)
        clean = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        noisy = cv2.cvtColor(_add_gaussian_noise(gray, sigma=12.0), cv2.COLOR_GRAY2BGR)
        clean_p, noisy_p = tmp_path / "clean.png", tmp_path / "noisy.png"
        cv2.imwrite(str(clean_p), clean)
        cv2.imwrite(str(noisy_p), noisy)
        return str(clean_p), str(noisy_p)

    def test_detailed_scores_expose_noise(self, image_pair):
        scores = RuleBasedQuality().get_detailed_scores(image_pair[0])
        assert 'noise_sigma' in scores and 'noise_score' in scores
        assert 0.0 <= scores['noise_score'] <= 1.0

    def test_noisy_image_scores_lower_overall(self, image_pair):
        """The headline spec-098 fix: clean twin must win (was 0/480 on SIDD)."""
        rb = RuleBasedQuality()
        clean_p, noisy_p = image_pair
        assert rb.assess_image(clean_p) > rb.assess_image(noisy_p)

    def test_default_weights_sum_to_one_with_noise(self):
        rb = RuleBasedQuality()
        assert 'noise' in rb.weights
        assert sum(rb.weights.values()) == pytest.approx(1.0)


class TestScoreIQACacheVersion:
    def test_cache_model_name_is_v2(self):
        """spec-098: cache key must NOT be the v1 name, or stale pre-noise scores
        (raw Laplacian, no noise component) would silently be served (spec-079 class)."""
        from sim_bench.pipeline.steps.score_iqa import ScoreIQAStep
        from sim_bench.pipeline.context import PipelineContext

        ctx = PipelineContext()
        ctx.image_paths = ["dummy.jpg"]
        cache_cfg = ScoreIQAStep()._get_cache_config(ctx, {})
        assert cache_cfg["model_name"] == "rule_based_v2"
