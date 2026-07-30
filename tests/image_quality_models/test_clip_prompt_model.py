"""ut for ClipPromptModel (spec-094 follow-up) — registration + availability.

No model download here (that needs network + ~338 MB). Scoring is covered by the
manual finger-occlusion experiment; these keep the wiring honest.
"""

from sim_bench.image_quality_models import MODEL_REGISTRY
from sim_bench.image_quality_models.clip_prompt_model import ClipPromptModel


def test_clip_occlusion_registered():
    assert MODEL_REGISTRY["clip_occlusion"] is ClipPromptModel


def test_is_available():
    # openai-clip is installed (came with pyiqa); scorer must report available.
    assert ClipPromptModel.is_available() is True


def test_in_engine_quality_family():
    from app.image_studio import engine
    keys = [m["key"] for m in engine.available_methods() if m["category"] == engine.CATEGORY_QUALITY]
    assert "clip_occlusion" in keys
