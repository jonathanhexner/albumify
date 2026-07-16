"""
Learned crooked-photo (tilt) estimation via GeoCalib -- spec-100.

spec-099 proved a classical Hough estimator cannot separate camera-tilt from
world-tilt (slanted scenery -> false positives, ~5% coverage). GeoCalib
(ECCV 2024) predicts a dense per-pixel "which way is up" field from *semantic*
cues (people/walls/trees are vertical), then a geometric optimizer fits the
single camera roll that explains it -- a learned prior a Hough transform lacks.

This module is a drop-in backend behind spec-099's `TiltResult` contract: on a
pass of the spec-100 gates, the pipeline step + penalty (spec-099 Phases 1-2)
consume it unchanged.

    estimate_tilt(image) -> TiltResult      # production: signed roll + confidence
    estimate_tilt_raw(image) -> RawTilt     # benchmark: signed roll + raw uncertainty

Confidence is derived from GeoCalib's native `roll_uncertainty` (radians): a
guessed tilt on a structureless scene comes back with large uncertainty and
therefore ~zero confidence -- honouring spec-099's rule that a guessed tilt
never moves a score.

Framework-agnostic (numpy/path in, dataclass out), spec-053 helper style.
CPU-capable; the GeoCalib model is loaded once at module level.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from sim_bench.quality_assessment.tilt import TiltResult  # reuse spec-099 contract

# Tilt is a global property; full resolution adds cost, not accuracy.
_MAX_SIDE = 1024

# confidence = exp(-roll_uncertainty_deg / _UNC_TAU): 0 deg unc -> 1.0, 2 deg -> 0.37,
# 5 deg -> 0.08. Monotonic-decreasing in uncertainty. The confidence GATE that turns
# this into a penalty is calibrated by the spec-100 benchmark sweep (T2.3), not here.
_UNC_TAU = 2.0

# GeoCalib's roll sign is OPPOSITE spec-099's "+ = content tilted clockwise":
# injecting +6 deg clockwise moves GeoCalib's raw roll by -5.5 deg. Flip it so the
# backend matches the contract. (Verified against injected rotations, spec-100 T1.)
_SIGN = -1.0

_MODEL = None


def _get_model():
    """Load GeoCalib once (CPU). Import is lazy so the dependency is optional."""
    global _MODEL
    if _MODEL is None:
        from geocalib import GeoCalib  # optional extra; see spec-100 T0
        _MODEL = GeoCalib().to("cpu").eval()
    return _MODEL


ImageLike = Union[np.ndarray, str, Path]


def _to_chw_tensor(image: ImageLike) -> torch.Tensor:
    """Accept an RGB/gray uint8 array or an image path -> CxHxW float tensor in [0,1]."""
    if isinstance(image, (str, Path)):
        from PIL import Image, ImageOps
        with Image.open(image) as pil:
            arr = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
    else:
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    scale = _MAX_SIDE / max(arr.shape[:2])
    if scale < 1.0:
        arr = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)))

    from geocalib.utils import numpy_image_to_torch
    return numpy_image_to_torch(arr)


@dataclass(frozen=True)
class RawTilt:
    """Primitive GeoCalib output, before the confidence transform.

    angle_deg: signed roll (spec-099 convention, + = content tilted clockwise).
    roll_uncertainty_deg: GeoCalib's estimated std of the roll, in degrees.
                          Small = confident; large = the model is abstaining.
    """
    angle_deg: float
    roll_uncertainty_deg: float


def estimate_tilt_raw(image: ImageLike) -> RawTilt:
    """Signed roll + raw uncertainty. Used by the spec-100 gate sweep."""
    model = _get_model()
    t = _to_chw_tensor(image).to("cpu")
    with torch.no_grad():
        res = model.calibrate(t)
    angle = _SIGN * float(torch.rad2deg(res["gravity"].roll))
    unc_deg = float(torch.rad2deg(res["roll_uncertainty"]))
    return RawTilt(angle_deg=angle, roll_uncertainty_deg=unc_deg)


def estimate_tilt(image: ImageLike) -> TiltResult:
    """Estimate photo roll via GeoCalib, mapped onto spec-099's TiltResult.

    confidence in [0,1] from the native roll uncertainty; n_lines is 0 (the
    learned backend has no line count -- the field is kept for contract parity).
    """
    raw = estimate_tilt_raw(image)
    confidence = float(np.exp(-max(raw.roll_uncertainty_deg, 0.0) / _UNC_TAU))
    return TiltResult(angle_deg=raw.angle_deg, confidence=confidence, n_lines=0)


# --- spec-053 pipeline helper (calc(inputs) -> result) -----------------------

# Bump when the serialized output shape or the estimator changes; drives
# universal_cache invalidation (see score_tilt step).
TILT_MODEL_VERSION = "geocalib_v1"


@dataclass(frozen=True)
class TiltInputs:
    """Per-call data for the pipeline step (framework-agnostic)."""
    image_paths: list


@dataclass(frozen=True)
class TiltScoreResult:
    """angles/confidences keyed by path; skipped = unreadable/failed images."""
    angles: dict
    confidences: dict
    skipped: list


class TiltScorer:
    """GeoCalib roll per image, behind the spec-053 calc() contract.

    Config lives here (none needed yet); per-call data is TiltInputs; output is
    TiltScoreResult. The individual estimate_tilt* functions stay public for
    notebook callers; the pipeline step uses calc().
    """

    version = TILT_MODEL_VERSION

    def calc(self, inputs: TiltInputs) -> TiltScoreResult:
        angles, confidences, skipped = {}, {}, []
        for path in inputs.image_paths:
            try:
                r = estimate_tilt(path)
                angles[path] = r.angle_deg
                confidences[path] = r.confidence
            except Exception as exc:  # unreadable image / inference failure
                logging.getLogger(__name__).warning("tilt skip %s: %s", path, exc)
                skipped.append(path)
        return TiltScoreResult(angles=angles, confidences=confidences, skipped=skipped)
