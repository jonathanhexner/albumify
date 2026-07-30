"""CLIP prompt-based scorer — e.g. finger/occlusion detection (spec-094 follow-up).

No-reference IQA metrics (BRISQUE/MANIQA/...) measure the quality of the VISIBLE
scene and miss localized content defects like a finger over the lens. This model
scores an image by CLIP similarity to a pair of antonym prompts (CLIP-IQA style):

    score = mean over prompt-pairs of  softmax([sim_good, sim_bad])[good]

so **higher = better = clearer / less occluded**. The finger-occlusion photos
should score LOW. Prompts are configurable so the same class can target other
content defects.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from sim_bench.image_quality_models.base_model import BaseQualityModel

logger = logging.getLogger(__name__)

# (clear / good prompt, occluded / bad prompt) antonym pairs.
_OCCLUSION_PROMPTS: List[Tuple[str, str]] = [
    ("a clear unobstructed photo", "a photo blocked by a finger over the lens"),
    ("a normal photo", "a photo with a blurry finger covering part of it"),
    ("a photo with a clear view", "a photo partly obstructed by something over the camera"),
]


class ClipPromptModel(BaseQualityModel):
    """Score = P(clear) vs P(occluded) via CLIP antonym prompts (higher = clearer)."""

    def __init__(self, device: str = "cpu", clip_model: str = "ViT-B/32",
                 prompts: List[Tuple[str, str]] = None):
        super().__init__(name="clip-occlusion", device=device)
        import clip  # openai-clip
        import torch
        self._torch = torch
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model.eval()
        self.prompts = prompts or _OCCLUSION_PROMPTS
        flat = [p for pair in self.prompts for p in pair]
        with torch.no_grad():
            toks = clip.tokenize(flat).to(device)
            tfeat = self.model.encode_text(toks)
            self._tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)  # (2*P, D)
        logger.info("Loaded CLIP occlusion model (%s, %d prompt pairs)", clip_model, len(self.prompts))

    def score_image(self, image_path: Path) -> float:
        """Mean P(clear) across prompt pairs. Higher = clearer/less occluded."""
        from PIL import Image
        torch = self._torch
        img = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ifeat = self.model.encode_image(img)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            sims = (100.0 * ifeat @ self._tfeat.T).squeeze(0)  # (2*P,)
        probs_good = []
        for i in range(len(self.prompts)):
            good, bad = sims[2 * i], sims[2 * i + 1]
            pair = torch.stack([good, bad]).softmax(dim=-1)
            probs_good.append(float(pair[0]))
        return sum(probs_good) / len(probs_good)

    def raw_score(self, image_path: Path) -> float:
        return self.score_image(image_path)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import clip  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def from_config(cls, config: Dict) -> "ClipPromptModel":
        return cls(device=config.get("device", "cpu"),
                   clip_model=config.get("clip_model", "ViT-B/32"),
                   prompts=config.get("prompts"))
