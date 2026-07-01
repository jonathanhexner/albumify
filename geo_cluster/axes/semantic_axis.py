"""SEMANTIC axis — group by scene content via cheap captions (spec-022, Slice 5).

Placeholder that registers the axis and returns ``None`` until per-image
captions are supplied. When wired up it will cluster caption embeddings (or
hand sampled captions to a local LLM) and is the flexible fallback used only
when the other axes score below the floor. Off by default.
"""

from __future__ import annotations

from typing import Optional

from geo_cluster.axes.base import AxisInputs, register_axis
from geo_cluster.types import Segmentation


@register_axis
class SemanticAxis:
    name = "semantic"
    needs = "per-image captions (ViT captioner)"

    def perturbed_config(self, config: dict) -> dict:
        return dict(config)

    def propose(self, inputs: AxisInputs) -> Optional[Segmentation]:
        if not inputs.captions:
            return None
        # Slice 5: caption-embedding clustering / LLM theming goes here.
        return None
