# Spec 101: Auto-Straighten (roll correction + inscribed-rectangle crop)

**Status**: Implemented — 2026-07-17. Option A (terminal straighten + fixability-scaled tilt_penalty).
REVIEW.md ACCEPT, §5 blocker cleared via committed real-data E2E. 44 unit tests + 1 slow E2E green.
T4 before/after report done (`reports/2026-07-17_auto_straighten/`) + outcome artifact. Live behind
config (`straighten_images.enabled`). **Tracked follow-up (non-blocking):** wire the leveled derivative
into the app export/display via `straightened_from` (the pipeline produces it; the app not yet verified
to serve it).
**Created**: 2026-07-16
**Depends on**: spec-099/spec-100 (produces the roll: `context.tilt_angles` / `tilt_confidences`
from `score_tilt`, GeoCalib backend). This spec is the *remediation* half that spec-099 named as
out-of-scope-for-now: "once a fix ships, the penalty is computed on the fixed image and ~vanishes."

## Problem

The pipeline now *detects* crooked photos and *demotes* them (spec-099 tilt penalty) but never
*fixes* them. Rotating an image by `-roll` to level it exposes triangular corners. The two naive
fills are both wrong for a shipped asset:

```
   black corners (BORDER_CONSTANT)          smeared edges (BORDER_REPLICATE)
   ┌───────────┐                            ┌───────────┐
   ▟███████████▙  <- black wedges           ▟▓▓▓▓▓▓▓▓▓▓▓▙  <- replicated/streaked
   █           █                            █           █     pixels (what our
   ▜███████████▛                            ▜▓▓▓▓▓▓▓▓▓▓▓▛     report visuals do today)
```

Our report/tutorial `straighten()` uses `BORDER_REPLICATE` — fine for an illustration, unacceptable
for an output the user keeps. The correct fix is a **crop to the largest axis-aligned rectangle that
lies entirely inside the rotated frame**: zero invented pixels, at the cost of a small zoom-in.

## Solution

### S1. Core geometry — `sim_bench/quality_assessment/straighten.py`

Framework-agnostic (numpy in / numpy out), spec-053 helper style:

```python
def largest_inscribed_rect(w, h, angle_deg) -> tuple[float, float]:
    """Max upright rectangle fully inside a w x h image rotated by angle_deg.
    Classic 'rotatedRectWithMaxArea' closed form."""

def straighten(rgb, roll_deg, *, preserve_aspect=True) -> np.ndarray:
    """Rotate by -roll (high-quality interp), then crop to the inscribed rect.
    No BORDER fill is ever visible in the output."""
```

Algorithm (maximise-area variant; aspect-preserving variant scales the largest similar rectangle):

```
a = |roll| in radians;  sin_a, cos_a = |sin a|, |cos a|
long, short = (w,h) if w>=h else (h,w)
if short <= 2*sin_a*cos_a*long or |sin_a - cos_a| < eps:   # fully constrained by short side
    x = 0.5*short;  wr,hr = (x/sin_a, x/cos_a) if w>=h else (x/cos_a, x/sin_a)
else:
    cos2a = cos_a^2 - sin_a^2
    wr = (w*cos_a - h*sin_a)/cos2a;  hr = (h*cos_a - w*sin_a)/cos2a
crop the centred wr x hr box from the rotated image
```

`preserve_aspect=True` (photo default) instead returns the largest `W:H`-ratio box centred in the
rotated frame — no aspect distortion, slightly more zoom. Decision **D1** below picks the default.

### S2b. REVISED to option A (terminal) — 2026-07-17

The E2E (REVIEW.md §5) showed the framework orders steps by `depends_on` edges only, and forcing
scorers to run after an early straighten would cascade-inject `straighten_images` + its deps
(GeoCalib, YOLO) into every pipeline containing `score_iqa`. So the design reverted to **terminal**:
`tilt_penalty` (scaled by fixability, S4) makes *selection* account for the crop cost, and
`straighten_images` runs **after `select_best`** to remediate the chosen winners only — `depends_on
select_best`, nothing depends on it, no drag. The S2 "early/rebind" text below is superseded.

### S2. (SUPERSEDED — early design) Separate pipeline step `straighten_images`, EARLY (before scoring)

A dedicated step, running **early** — after `score_tilt` (roll) and `detect_persons` (person boxes
for the gate), but **before** `score_iqa`/`score_ava`/`score_occlusion` and the face steps. Rationale
(user): straightening crops the photo, and that framing change is part of the **final deliverable's
quality** — so every downstream scorer and `select_best` must judge the straightened result, not the
as-shot original. Comparing deliverables is the whole point of "pick the best photo."

Mechanics: for each image the step applies the subject-gated straighten (S3) and, on success, writes
a **derived asset** (spec-091 aligned-crop disk cache; key = src path, roll, crop-mode, subject-gate,
version) and **rebinds the working image set** — `context.active_images` / `image_paths` point at the
straightened derivative, with `context.straightened_from[derived] = original` for provenance.
Declined images pass through unchanged. Originals on disk are never mutated. `depends_on`:
`score_tilt`, `detect_persons`.

Pipeline order becomes: `… score_tilt → detect_persons → straighten_images → score_iqa → score_ava
→ score_occlusion → insightface_detect_faces → …`. All scoring/detection/selection then operate on
the straightened deliverable.

### S3. Subject-aware gate — reuse YOLO person boxes, no saliency (user-locked)

Straighten a winner **iff** all of:
1. `score_tilt` confident (`confidence >= conf_gate` 0.5) AND `|roll| >= gate_deg` (3°);
2. **area floor**: inscribed crop keeps `>= min_retained_area` of the frame (config, default 0.70);
3. **person preservation**: for every YOLO person box (from `detect_persons` — reused, not
   re-detected) whose area `>= prominent_person_frac` of the frame, the inscribed crop **fully
   contains** that box.

Else → **decline**: leave the winner untouched (tilted but whole). No saliency is used — person
detection is the trusted signal we already run. Crowds of small people (each `< prominent_person_frac`)
don't block straightening; no single one is the subject.

Config (every knob on the step, user-modifiable):
`conf_gate=0.5, gate_deg=3.0, min_retained_area=0.70, prominent_person_frac=0.15,
preserve_aspect=true, interp=cubic`.

### S4. tilt_penalty scaled by fixability (option A — the selection-side mechanism)

Selection accounts for the crop by scaling the penalty with the cheapest fix's residual damage,
using the SAME gate the terminal step uses (`straighten_gate.decide`):

```
tilt_penalty(p) = 0                                     if not confident / |roll| < gate_deg
                = −min(fov_weight·(1 − retained_area), cap)   if cleanly FIXABLE (small FOV cost)
                = −min(slope·(|roll| − gate_deg), cap)        if UNFIXABLE (person clip / area < floor)
```

Computed from roll + image size + person box (no actual crop — cheap). Selection thus prefers a
cleanly-straightenable tilt over one that can't be fixed, and a level shot over both. Defaults
`fov_weight=0.4, cap=0.15` (relative weight of FOV-cost vs angle tuned in the T4 report). Gate knobs
(`min_retained_area`, `prominent_person_frac`) must match `straighten_images`.

## Acceptance criteria

| # | Gate | Threshold |
|---|---|---|
| A1 | No invented pixels: every pixel of the output maps to a real source pixel (corner pixels are genuine content, not black/replicated) | 0 border pixels in output |
| A2 | Largest-rect optimality: `largest_inscribed_rect` matches a brute-force search over candidate boxes | area within 0.5% of numerical max |
| A3 | Round-trip: re-running GeoCalib on the straightened output reports `|roll| < 1°` | pass on the spec-100 confident set |
| A4 | Aspect handling: `preserve_aspect=True` output has ratio == input ratio (± rounding); `False` maximises area | pass both modes |
| A5 | FOV retention reported: median retained-area % across the confident Budapest tilts (so the "cost of the fix" is a real number) | reported, not gated |
| A6 | No-op below gate: `|roll| < 3°` or low confidence → output is the byte-identical original | pass |
| A7 | Experiment report: before/after gallery on ~10 real tilted photos, inline, with retained-area % per image | exists |

## Architecture: separate `straighten_images` step (user-confirmed 2026-07-16)

Straighten is its OWN pipeline step, separate from `score_tilt` (mirrors the
score_occlusion/occlusion_penalty and score_tilt/tilt_penalty split). Detection produces the roll;
policy decides what to do with it. This keeps the albumify app free to penalize, straighten, or both.

## Decisions (LOCKED 2026-07-16, user-confirmed)

- **D1 crop mode** = preserve_aspect (no distortion).
- **D2 stage** = EARLY (2026-07-17) — `straighten_images` runs before the scorers, rebinding the
  working image set so selection judges the straightened deliverable. (Reversed from terminal: user
  argued the crop is part of final quality and must count in selection.)
- **D3 penalty × straighten** = straightened images need no penalty (cost/benefit is in their real
  scores); `tilt_penalty` shrinks to **declined-only** (S4).
- **D4 gate** = subject-preservation AND area floor. Subject = **YOLO person boxes** from the
  existing `detect_persons` step (no saliency, no duplicate detector), guarded by
  `prominent_person_frac` so only a person filling ≥X% of the frame blocks the crop.
- **D5 prominence X** = step **config**, default 0.15, user-modifiable (every pipeline element has
  config). Area floor also config, default 0.70.
- **D6 interpolation** = INTER_CUBIC.

## Out of scope

- Perspective / keystone / vertical-line correction (only in-plane roll here).
- 90/180/270° content-rotation (separate; EXIF-less orientation is defect finding #4).
- Re-scoring / re-clustering on straightened images (S4: penalty stays; no re-score loop).
- Mutating original files (always a derived asset).

## Risks

- **Over-crop on large rolls**: a 16° roll on a portrait can lose ~30% area. A6/A5 make the cost
  visible; if unacceptable, gate straightening to a max roll (e.g. skip > 12°) — decision deferred
  to the A5 numbers.
- **Double transform**: EXIF orientation is already applied upstream (`exif_transpose`); straighten
  must run on the EXIF-normalised pixels, not re-apply orientation. Test guards this.
- **Cache key completeness**: crop-mode + version must be in the derived-asset key or a mode change
  serves stale straightened files (spec-091 cache-key lesson).

## Docs to update (Code Review §7)

`classes.html` (straighten helper), `data_flow.html` (export-time transform), CHANGES_LOG,
and — if D2 adds a pipeline/Studio surface — spec-034 contract + the relevant HTML.
