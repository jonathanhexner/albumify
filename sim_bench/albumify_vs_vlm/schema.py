"""Spec-102 — shared album-result schema, produced identically by both arms.

Both the Albumify arm and the VLM arm emit an `AlbumResult`: an ordered K-sequence of stems plus
per-pick detail. The blind A/B viewer reads both the same way, so it cannot tell which arm made
which album (fairness). `scene_clusters` is only populated by the Albumify arm (its scene grouping);
the VLM arm leaves it empty.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Pick:
    id: str                            # working-set stem (stable image ID)
    score: float                       # arm-native score (composite for Albumify; 0.0 for VLM)
    scene_cluster: int                 # Albumify scene id; -1 for VLM
    reason: str                        # one-clause justification
    role: str = ""                     # opener|hero|connective|detail|peak|closer (VLM only)


@dataclass
class AlbumResult:
    trip: str
    arm: str                           # "albumify" | "vlm"
    input_set_hash: str                # A1: ties the picks to the exact working set
    k: int
    pipeline: str                      # albumify: default/minimal; vlm: model string
    order: list[str] = field(default_factory=list)     # stems, presentation order
    picks: list[Pick] = field(default_factory=list)
    scene_clusters: dict[int, list[str]] = field(default_factory=dict)  # albumify only
    meta: dict = field(default_factory=dict)           # arm-specific (tokens, seeds, batches...)

    def to_json(self) -> dict:
        d = asdict(self)
        d["scene_clusters"] = {str(cid): stems for cid, stems in self.scene_clusters.items()}
        return d


def save_album_result(res: AlbumResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.to_json(), indent=2), encoding="utf-8")
    logger.info("wrote %s picks -> %s", res.arm, path)
