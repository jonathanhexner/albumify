"""Content-tier triage (spec-107 Tier 2): flag junk that metadata can't see.

Photographed documents, screenshots re-shared as photos, posters, receipts, maps — these are real
camera files (so Tier-1 metadata keeps them) but are not album moments. We read RAM++ content tags
(the content axis validated 2026-10-01) and classify.

Pure ``calc(tags) -> TriageResult`` (unit-testable without the model); ``RamTagger`` is the live
tag source for the pipeline. Conservative + recoverable: ``set_aside`` is default-excluded but shown
for restore; a printed item *within* a real scene downgrades to ``review`` (kept, flagged).
"""
from __future__ import annotations

from typing import List

from sim_bench.triage.metadata import TriageResult

# Document / screen content — almost never an album moment.
_STRONG_JUNK = {
    "screenshot", "document", "text", "text message", "website", "web page", "webpage",
    "computer screen", "screen", "note", "notepad", "paper", "checklist", "receipt", "invoice",
    "menu", "spreadsheet", "form", "letter", "resume", "boarding pass", "id card", "business card",
}
# Printed / graphic material — usually junk, but can legitimately appear inside a real scene.
_SOFT_JUNK = {
    "poster", "movie poster", "logo", "letter logo", "map", "sign", "road sign", "parking sign",
    "label", "book cover", "magazine", "flyer", "brochure", "diagram", "chart", "comic",
    "comic book", "cartoon", "flipchart", "whiteboard", "package",
}
# Strong "this is a real moment" indicators — presence downgrades soft-junk to review, not set_aside.
_MOMENT = {
    "person", "people", "man", "woman", "child", "boy", "girl", "family", "baby", "crowd", "group",
    "selfie", "portrait", "landscape", "mountain", "beach", "lake", "river", "sea", "ocean", "tree",
    "forest", "flower", "garden", "sunset", "food", "animal", "pet", "dog", "cat", "bird", "boat",
}


def calc(tags: List[str]) -> TriageResult:
    """Classify an image from its RAM++ content tags. Pure."""
    low = {t.strip().lower() for t in (tags or [])}
    strong = sorted(low & _STRONG_JUNK)
    soft = sorted(low & _SOFT_JUNK)

    if strong:
        return TriageResult("set_aside", f"document/screen content: {', '.join(strong)}", 0.85,
                            ["content:strong"] + strong)
    if soft:
        if low & _MOMENT:
            return TriageResult("review", f"printed item in a scene: {', '.join(soft)}", 0.4,
                                ["content:soft+moment"] + soft)
        return TriageResult("set_aside", f"printed material: {', '.join(soft)}", 0.6,
                            ["content:soft"] + soft)
    return TriageResult("keep", "no junk-content tags", 0.0, ["content:clean"])


class RamTagger:
    """Lazy RAM++ tag source for the live pipeline. Loads the model once; ``tag(path) -> [tags]``.

    Heavy + optional: the ``ram`` package and the ~2.8 GB checkpoint are only touched on construction,
    so importing this module stays cheap and environments without RAM++ are unaffected until used.
    """

    def __init__(self, checkpoint: str | None = None, image_size: int = 384, device: str = "cpu"):
        # transformers moved these helpers; RAM's bundled BERT imports them from the old location.
        import transformers.modeling_utils as mu, transformers.pytorch_utils as pu
        for n in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(mu, n) and hasattr(pu, n):
                setattr(mu, n, getattr(pu, n))
        from pathlib import Path
        from ram.models import ram_plus
        from ram import get_transform

        if checkpoint is None:
            root = Path(__file__).resolve().parents[2]
            checkpoint = str(root / "models" / "ram" / "ram_plus_swin_large_14m.pth")
        self._model = ram_plus(pretrained=checkpoint, image_size=image_size, vit="swin_l").eval().to(device)
        self._tf = get_transform(image_size=image_size)

    def tag(self, path) -> List[str]:
        import torch
        from PIL import Image
        from ram import inference_ram
        with torch.no_grad():
            res = inference_ram(self._tf(Image.open(path).convert("RGB")).unsqueeze(0), self._model)
        en = res[0] if isinstance(res, (list, tuple)) else res
        return [t.strip() for t in str(en).split("|") if t.strip()]

    def classify_file(self, path) -> TriageResult:
        return calc(self.tag(path))
