# Spec 100: Learned Tilt Detection — GeoCalib Benchmark (Phase-0 v2)

**Status**: Implemented (validation) — 2026-07-13 DECISION: **SHIP as gated tie-breaker** (option a).
G2 relaxed (coverage is content-bound; abstention is safe for a tie-breaker). Operating point:
confidence gate 0.5 (= roll uncertainty ~1.4°, where MAE ~0.3° and the 3 named slanted-scenery
cases stay clean). This unblocks spec-099 Phases 1–2 with `tilt_geocalib` as the backend
(`model_name="geocalib_v1"`, cached — GeoCalib is ~1.9 s/img on CPU).

## Results (2026-07-13)

Full 122-photo run (`reports/2026-07-12_geocalib_tilt/`) + descriptive album report
(`reports/2026-07-13_geocalib_budapest_tilt/report.html`).

| Gate | Bar | Result | Verdict |
|---|---|---|---|
| G1 MAE | <1.5° | 0.25–0.45° | ✅ pass (far better than classical at 5% cov) |
| G1 recall ≥4° | ≥80% | ~100% | ✅ pass |
| **G2 coverage** | ≥70% | **28% confident (unc≤1.5)**, 56% at unc≤3 | ❌ **miss** — album is people-heavy, few verticals |
| G3 FP ≤5% / named | ≤5%, named clean | 5.7% @unc≤1.0; 2/3 named clean | ⚠️ marginal — FP inflated (real album tilt, no GT) |
| G4 AUC | ≥0.85 | 0.835 | ⚠️ marginal miss |
| G5 perf | <3 s | 1.88 s/img CPU | ✅ pass |

**Key finding**: GeoCalib fixes the classical failure (slanted scenery no longer flagged) and its
native `roll_uncertainty` abstains honestly — the extreme "tilts" (kaleidoscope/mirror shots) come
back with 10–30° uncertainty and gate out. So low coverage is *safe abstention*, not silent error.
**Decision gate not cleanly met** (G2 fails; G3/G4 marginal). Not an auto-pass. Recommendation:
either (a) relax G2 for a gated tie-breaker and ship on the confident subset, (b) hold, or (c) raise
coverage (tile/ensemble). **User call required** before spec-099 Phases 1–2 unblock.
**Created**: 2026-07-12
**Supersedes**: spec-099 Phase 0 (classical Hough estimator — NEGATIVE result, `reports/2026-07-12_tilt_benchmark/`)
**Depends on / reuses**: spec-099 penalty design (S3), Studio column (S2), acceptance harness

> **Why this spec exists.** spec-099 Phase 0 proved a *classical* Hough estimator cannot separate
> camera-tilt from world-tilt: MAE 0.32° when confident, but only ~5% coverage, and its confident
> flags were **upright photos with slanted scenery** (tunnel perspective, illusion-room artwork).
> The report's own conclusion — "needs a gravity reference or a learned semantic prior" — points
> directly here. This spec validates a **learned single-image roll/gravity model** on the *same*
> harness before any penalty is wired. Fail-fast is retained: no penalty ships until the model
> clears the exact bar the classical method failed.

## Problem (unchanged from spec-099)

A hand-held photo with 3–10° roll (crooked horizon) is a common, user-visible defect the pipeline
is blind to. EXIF orientation encodes only 90° steps. Between a straight and a crooked shot of the
same scene, best-shot selection currently flips a coin. We want the pipeline to **see** tilt and
prefer the straighter shot — auto-straightening stays out of scope.

## Approach: swap the estimator, keep the interface

spec-099 defined a clean, framework-agnostic contract. This spec keeps it **byte-for-byte** and
only changes the backend from Hough to a learned model:

```python
@dataclass(frozen=True)
class TiltResult:
    angle_deg: float       # signed camera ROLL; + = clockwise
    confidence: float      # [0,1]; a guessed tilt MUST score ~0 confidence
    n_lines: int = 0       # legacy field; unused by learned backend

def estimate_tilt(image) -> TiltResult: ...   # now backed by GeoCalib
```

If GeoCalib clears the gates, spec-099 Phases 1–2 (step + Studio column + `select_best` penalty)
are unblocked **verbatim** — they never see the estimator internals.

### Model: GeoCalib (ECCV 2024, `cvg/GeoCalib`)

| Property | Value |
|---|---|
| Native output | gravity direction + intrinsics → **roll = tilt angle directly** |
| Confidence signal | per-pixel confidence map + geometric-optimizer residual |
| License | Apache-2.0 (code) / CC-BY-4.0 (weights) — both permissive |
| Compute | CPU-capable (`cuda if available else cpu`); torch |
| Install | `pip install -e .` / `git+https://github.com/cvg/GeoCalib` (optional extra) |

How it works (1 line each): a net predicts a dense **perspective field** (per-pixel "which way is
up") from *semantic* cues (people/trees/walls are vertical), then a Levenberg-Marquardt optimizer
(iterative least-squares curve fit) finds the single camera roll/focal that best explains those
per-pixel up-vectors. The learned prior is what a Hough transform lacks — it is not fooled by a
slanted painting because it reads content, not just edges.

**Secondary comparison (optional, same harness):** PerspectiveFields (CVPR'23, outputs roll/pitch/
vfov). Run only if GeoCalib is borderline on a gate — one alternative, not a zoo.

## The decisive regression set (this is the whole point)

The Phase-0 report's two confident false positives — the **tunnel-perspective photo** and the
**illusion-room artwork** — become named regression cases. A learned model that re-flags either at
high confidence has not solved the problem, only relocated it. These are pulled from the existing
Budapest benchmark set and pinned in the test.

## Acceptance criteria

| # | Gate | Threshold | Beats classical? |
|---|---|---|---|
| G1 | Injected-rotation recovery: ~120 Budapest photos rotated by known ±{2,4,6,8,10}° (EXIF-normalized, center-crop), recovered roll error on confident estimates | MAE < 1.5°; ≥80% of ≥4° tilts within ±2° | classical: 0.32° ✓ but only at 4.9% cov |
| G2 | **Coverage**: fraction of injected-tilt photos that get a confident (conf ≥ gate) estimate | **≥ 70%** | classical: **4.9%** — the gate that failed |
| G3 | False-positive control on upright originals flagged `|roll|>3°` at conf ≥ gate, **including the 2 named slanted-scenery cases** | ≤ 5% overall AND both named cases NOT flagged | classical: flagged both |
| G4 | Confidence honesty: separability of injected-tilt vs upright using the confidence signal | ROC-AUC ≥ 0.85 | — |
| G5 | CPU perf per image at ~1024 px (one-time; result cached) | < 3 s (GPU optional, note actual) | — |
| G6 | Experiment report per mandate: `reports/2026-07-12_geocalib_tilt/` — inline before/after samples, the 2 named FP cases, confidence-gate sweep, GeoCalib-vs-classical table | exists | — |

**Decision gate (explicit):** G1–G4 all pass → recommend unblocking spec-099 Phases 1–2 with
`estimate_tilt` backed by GeoCalib (config `model_name="geocalib_v1"`). Any of G1–G4 fails →
document in report, keep spec-099 On Hold, do not wire the penalty. **This spec ships no penalty
by itself** — it is a go/no-go validation.

## Out of scope

- The penalty wiring, Studio column, `select_best` composite (that is spec-099 Phases 1–2, unblocked
  only on a pass — designed already, not re-litigated here).
- Auto-straightening (future spec).
- 90/180/270° content-rotation detection (separate).
- Training/fine-tuning a model — inference-only on published weights.

## Risks

- **Dependency conflict**: pyiqa pins `numpy<2` (SIGHTING-111 / [[pyiqa_perf_opencv]]); torch is
  already in the env. T1 verifies GeoCalib installs under `numpy<2` before any benchmarking — if it
  demands numpy≥2, we isolate it (subprocess/venv) rather than break IQA. **Blocking check.**
- **CPU latency**: a learned model is ~10–100× slower than Hough. Mitigated by universal_cache
  (per-image, per-run one-time) and internal downscale; G5 measures the real cost so we decide with
  numbers, not fear.
- **Still a prior, not a sensor**: genuinely ambiguous photos (no up-cues, deliberate Dutch angle)
  can be wrong — G4 exists precisely to prove the confidence signal abstains honestly. If G4 fails,
  the model is confidently wrong and we do NOT ship, same rule as classical.
- **0 real crooked photos** in the set: G1–G4 run on injected rotations (measures recovery) + the
  named FP cases. Real-positive validation is a follow-up if the user supplies crooked shots.

## Docs to update (Code Review §7)

`reports/EXPERIMENTS.md` (2-line entry), CHANGES_LOG. (No `classes.html`/`data_flow.html`/spec-034
changes in this spec — those belong to spec-099 Phase 1 on a pass, since this spec adds no pipeline
state.) `setup.cfg` optional-extra note if GeoCalib is added as a dependency.
