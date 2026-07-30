# Spec 099: Crooked-Photo (Tilt) Detection + Scoring Penalty

**Status**: Code Review — 2026-07-13 Phases 1–2 implemented + REVIEW.md ACCEPT (no §1–§7 fail).
Awaiting user confirmation of 2 non-blocking decisions (kept score_tilt in default pipeline;
e2e_budapest waived as not-applicable) before flip to Implemented.
Phase 0 (classical) failed, but spec-100
validated a learned backend (GeoCalib) that fixes the slanted-scenery failure. Phases 1–2 now
proceed with `sim_bench/quality_assessment/tilt_geocalib.estimate_tilt` as the estimator (same
`TiltResult` contract), cached (`model_name="geocalib_v1"`, ~1.9 s/img CPU). Penalty design (S3)
unchanged; conf_gate 0.5 ≈ GeoCalib roll-uncertainty ~1.4° (spec-100 operating point).
**Created**: 2026-07-12

> **Phase-0 verdict** (`reports/2026-07-12_tilt_benchmark/`): the classical estimator is
> angularly precise when confident (MAE 0.32°) but covers only ~5% of real album photos, and the
> confident flags it does raise are upright photos with slanted *scenery* (tunnel perspective,
> illusion-room artwork) — pixels cannot distinguish camera-tilt from world-tilt without a gravity
> reference. Shipping the penalty would violate this spec's own first rule ("a guessed tilt never
> moves a score"). Estimator (`sim_bench/quality_assessment/tilt.py`) + 13 unit tests + benchmark
> retained for a future learned-horizon-model attempt. Phases 1–2 not started.
**Origin**: defect-scoring benchmarks (2026-07-10) + user discussion — "how should a crooked photo
impact the penalty". Auto-straightening is explicitly OUT of scope (future spec); this spec makes
the pipeline *see* tilt and prefer the straighter shot within otherwise-equal choices.

## Problem

A photo taken with a tilted hand (3–10° roll) is a common, user-visible defect. The pipeline is
completely blind to it:
- EXIF orientation (already applied everywhere via `exif_transpose`) only encodes 90° steps — it
  says nothing about tilt.
- No score, no penalty, no Studio column. Between a straight and a crooked shot of the same scene,
  best-shot selection currently flips a coin.

## Design principle (agreed with user)

**Penalize by the damage that remains after the cheapest fix.** Tilt is almost fully fixable
(rotate + small crop), so the penalty is a small tie-breaker — never a disqualifier. Ladder:
blur > noise > exposure > tilt.

## Solution

### S1. Tilt estimator — `sim_bench/quality_assessment/tilt.py`
Classical, framework-agnostic (spec-053 helper style), no new dependencies:

```python
@dataclass(frozen=True)
class TiltResult:
    angle_deg: float       # signed estimated roll; + = clockwise
    confidence: float      # [0,1] from line count + angular agreement
    n_lines: int

def estimate_tilt(gray: np.ndarray) -> TiltResult: ...
```

Method: Canny edges → probabilistic Hough segments → keep segments within ±20° of horizontal or
vertical → fold into deviation-from-axis space → angle = weighted circular median (weights = segment
length); confidence from (a) total supporting length, (b) inter-segment agreement (dispersion).
Portraits/beaches/close-ups yield few lines → low confidence by construction.

### S2. Pipeline step — `sim_bench/pipeline/steps/score_tilt.py` (one step per file, ≤80 LOC)
- `produces={"tilt_angles"}`; `context.tilt_angles[path] = {"angle_deg", "confidence"}`
  (+ spec-034 contract row; drift test will enforce).
- Cached via universal_cache (`feature_type="tilt"`, `model_name="hough_v1"`).
- Studio: "Tilt (deg)" column in the quality family (value = −|angle| so higher = straighter,
  matching the higher-is-better column contract; detail shows signed angle + confidence).

### S3. Penalty — `sim_bench/pipeline/scoring/tilt_penalty.py` (mirrors spec-097 occlusion_penalty)
Fourth composite component in `select_best`: `composite = quality + person_penalty +
occlusion_penalty + tilt_penalty`.

```
tilt_penalty = 0                                  if confidence < CONF_GATE (default 0.5)
             = 0                                  if |angle| < GATE_DEG (default 3.0)
             = -min(SLOPE * (|angle| - GATE_DEG), CAP)   otherwise
```
Defaults (to be fixed by the A-gates below, not hand-tuned): `GATE_DEG=3.0`, `CAP=0.15`,
`SLOPE=0.02/deg` (reaches cap ~10.5°). All in `configs/pipeline.yaml` under `select_best`,
like `occlusion_penalty`.

Rationale for each rule (user discussion): <3° is imperceptible AND within detection error — never
punish estimator noise; low confidence must contribute exactly 0 — a guessed tilt never moves a
score; cap keeps tilt subordinate to sharpness/occlusion (intentional angles exist).

## Acceptance criteria

| # | Gate | Threshold |
|---|---|---|
| A1 | Synthetic benchmark: rotate ~120 real album photos (Budapest) by known ±{2,4,6,8,10}° (crop-safe center region, EXIF-normalized) → detection error on confident (conf ≥ gate) estimates | MAE < 1.5°; ≥80% of ≥4° tilts detected within ±2° |
| A2 | False-positive control: original (upright) album photos flagged `|angle|>3°` at high confidence | ≤5% |
| A3 | Penalty unit tests: 0 below gates (angle & confidence), linear region, cap, sign-independence | pass |
| A4 | Composite regression: images with no confident tilt → composite scores bit-identical to pre-099 | pass |
| A5 | Perf: `estimate_tilt` on a 16 MP photo (downscaled internally to ~1024) | < 100 ms |
| A6 | Experiment report per mandate: `reports/<date>_tilt_benchmark/` with inline before/after samples + gate-value sweep justifying GATE_DEG/SLOPE/CAP | exists |

## Out of scope
- Auto-straightening (future spec; penalty design anticipates it — once a fix ships, the penalty
  is computed on the fixed image and ~vanishes).
- 90/180/270° content-rotation detection (defect finding #4, CLIP zero-shot — separate spec).
- Face-crop tilt (faces are alignment-normalized downstream already).
- Deep horizon-detection models.

## Risks
- **Architectural scenes with legitimate diagonals** (staircases, Dutch-angle art): mitigated by
  the agreement term in confidence + the cap; A2 measures the real false-positive rate.
- **Albums without line structure**: estimator abstains (low confidence) → penalty 0 — the feature
  degrades to exactly today's behavior, never worse.

## Docs to update (Code Review §7)
`classes.html` (tilt module + step), `data_flow.html` (context state + select_best composite),
spec-034 table (`tilt_angles`), `configs/pipeline.yaml` comments, CHANGES_LOG.
