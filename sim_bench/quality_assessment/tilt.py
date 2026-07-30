"""
Crooked-photo (tilt) estimation — spec-099.

EXIF orientation only encodes 90-degree steps; a hand-tilted photo (3-10 deg roll)
carries no metadata about it. This estimator recovers the roll angle from the
photo's own line structure (horizons, buildings, door frames):

    Canny edges -> probabilistic Hough segments -> keep segments within
    FOLD_LIMIT_DEG of horizontal/vertical -> fold both families into one
    deviation-from-axis space -> angle = length-weighted median deviation.

Confidence combines support (total segment length vs image size) and agreement
(how much of that length votes within AGREE_TOL_DEG of the answer). Scenes with
no line structure (portraits, beaches) yield low confidence by construction —
downstream, low confidence MUST map to zero penalty (a guessed tilt never moves
a score).

Framework-agnostic (numpy in / dataclass out), spec-053 helper style.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Internal working width: tilt is a global property; full resolution adds cost,
# not accuracy.
_MAX_SIDE = 1024

# Segments further than this from the nearest axis are diagonals (stairs,
# rooflines) and vote for neither axis.
FOLD_LIMIT_DEG = 20.0

# Agreement window for the confidence term.
AGREE_TOL_DEG = 1.5

# Support saturation: agreeing segment length equal to 2x the image diagonal
# counts as full support. Deliberately strict: the 2026-07-12 benchmark showed
# relaxing to 0.75 raises coverage 1%->6% but degrades MAE 0.32->3.3 deg.
# For a penalty signal, precision beats coverage — abstaining is free.
_SUPPORT_SAT = 2.0


@dataclass(frozen=True)
class TiltResult:
    """Estimated roll of the photo content.

    angle_deg: signed; positive = content tilted clockwise (fix by rotating the
               image counter-clockwise by the same amount).
    confidence: [0,1]; product of line support and angular agreement.
    n_lines: number of near-axis segments that voted.
    """
    angle_deg: float
    confidence: float
    n_lines: int


def estimate_tilt(gray: np.ndarray) -> TiltResult:
    """Estimate the roll angle of a photo from its near-axis line segments."""
    if gray.ndim != 2:
        raise ValueError("estimate_tilt expects a 2-D grayscale image")

    scale = _MAX_SIDE / max(gray.shape)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    h, w = gray.shape
    diagonal = float(np.hypot(h, w))

    edges = cv2.Canny(gray, 50, 150)
    # threshold/minLineLength/maxLineGap tuned so pixel-grid artifacts in pure
    # noise yield ~no segments (1157 -> 7 at 640px) while real line structure
    # survives: only long, near-unbroken runs vote.
    segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360.0, threshold=80,
        minLineLength=int(min(h, w) / 6), maxLineGap=3,
    )
    if segments is None:
        return TiltResult(0.0, 0.0, 0)

    # Split votes into the two line families. Perspective convergence (looking
    # up at a building) makes verticals lean symmetrically while the horizon
    # stays true — so the families are estimated SEPARATELY and only trusted
    # together when they agree (disagreement = perspective, not tilt).
    fam_dev = {"h": [], "v": []}
    fam_len = {"h": [], "v": []}
    n_lines = 0
    for x1, y1, x2, y2 in segments[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        dev_h = (angle + 90.0) % 180.0 - 90.0          # deviation from horizontal
        dev_v = (angle % 180.0) - 90.0                 # deviation from vertical
        if abs(dev_h) <= FOLD_LIMIT_DEG:
            fam_dev["h"].append(dev_h)
            fam_len["h"].append(length)
            n_lines += 1
        elif abs(dev_v) <= FOLD_LIMIT_DEG:
            fam_dev["v"].append(dev_v)
            fam_len["v"].append(length)
            n_lines += 1

    def family_estimate(dev_list, len_list):
        """(angle, confidence) via histogram peak + concentration around it."""
        if not dev_list:
            return None
        dev = np.asarray(dev_list)
        wgt = np.asarray(len_list)
        hist, edges_ = np.histogram(dev, bins=np.arange(-FOLD_LIMIT_DEG,
                                                        FOLD_LIMIT_DEG + 0.5, 0.5),
                                    weights=wgt)
        peak = edges_[int(np.argmax(hist))] + 0.25
        near = np.abs(dev - peak) <= 2.0
        if not near.any():
            return None
        angle = float(np.average(dev[near], weights=wgt[near]))
        support = min(1.0, float(wgt[near].sum()) / (diagonal * _SUPPORT_SAT))
        concentration = float(wgt[np.abs(dev - angle) <= AGREE_TOL_DEG].sum() / wgt.sum())
        return angle, support * concentration

    est_h = family_estimate(fam_dev["h"], fam_len["h"])
    est_v = family_estimate(fam_dev["v"], fam_len["v"])

    if est_h is None and est_v is None:
        return TiltResult(0.0, 0.0, 0)
    if est_h is not None and est_v is not None:
        (a_h, c_h), (a_v, c_v) = est_h, est_v
        if abs(a_h - a_v) <= 2.0:
            # Families agree: perspective ruled out, confidence compounds.
            weight_sum = c_h + c_v
            angle_deg = (a_h * c_h + a_v * c_v) / weight_sum
            confidence = min(1.0, weight_sum)
        else:
            # Families disagree: perspective scene — abstain rather than guess.
            angle_deg, confidence = (a_h, c_h * 0.3) if c_h >= c_v else (a_v, c_v * 0.3)
    else:
        # One family only (pure horizon, or a single wall): usable but weaker.
        angle_deg, confidence = est_h if est_h is not None else est_v
        confidence *= 0.8

    return TiltResult(angle_deg=float(angle_deg), confidence=float(confidence),
                      n_lines=n_lines)
