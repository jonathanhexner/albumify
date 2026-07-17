"""Minimal smoke test: the occlusion detector fires on known finger-over-lens photos.

`examples/finger_occlusion/` holds two real phone photos where a finger partly
covers the lens (downsampled to 1280 px so they live in-repo). This test is the
one-glance proof that the production occlusion scorer (spec-097, the spec-096
winning CLIP probe) actually flags them — P(occluded) should clear the pipeline's
penalty gate (0.75). It is the clear, runnable example of "what the occlusion
detector does and that it works."

Observed scores (default v2 artifact, images at 1280 px):
    20250822_122617.jpg  P(occluded)=0.743  max tile=0.956
    20250822_122626.jpg  P(occluded)=0.770  max tile=0.848
Both sit at/near the 0.75 penalty gate (these are partial, borderline occlusions
by design), and both light up a strongly-occluded tile — so we assert on a robust
"clearly elevated + localized" signal, not the exact 0.75 threshold.

Marked slow: loads the CLIP model. Run with `pytest -m slow`.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "finger_occlusion"
ELEVATED = 0.60   # well above a 0.5 coin-flip -> the global score reacts to the finger
HOT_TILE = 0.70   # at least one 3x3 tile strongly flags the occluded corner


@pytest.mark.skipif(not EXAMPLES.exists(), reason="finger example images not present")
def test_finger_examples_are_flagged_occluded():
    from sim_bench.occlusion_bench.scorer import OcclusionInputs, OcclusionScorer

    imgs = sorted(str(p) for p in EXAMPLES.glob("*.jpg"))
    assert imgs, "no example images found"

    scorer = OcclusionScorer()  # default production artifact
    result = scorer.calc(OcclusionInputs(image_paths=imgs))

    for path in imgs:
        name = Path(path).name
        p_occluded = result.scores[path]
        hottest_tile = max(result.tiles[path])
        assert p_occluded >= ELEVATED, (
            f"{name}: P(occluded)={p_occluded:.3f} is not elevated — the detector "
            f"should react to a finger over the lens"
        )
        assert hottest_tile >= HOT_TILE, (
            f"{name}: hottest tile={hottest_tile:.3f} — expected a strongly-flagged "
            f"occluded region (localization)"
        )
