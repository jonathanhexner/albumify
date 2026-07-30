"""IDENTITY axis — group by the people present (spec-022).

Reuses the face-clustering output (``people_clusters``). Returns ``None`` when
no face clusters are supplied, so it plugs into the competition the moment that
signal exists without any change to the selector. Wins when one or two people
are the album's theme and place/time are flat.
"""

from __future__ import annotations

from typing import Optional

from geo_cluster.axes.base import AxisInputs, register_axis
from geo_cluster.types import Segment, SegmentKind, Segmentation


@register_axis
class IdentityAxis:
    name = "identity"
    needs = "face clusters (people_clusters)"

    def perturbed_config(self, config: dict) -> dict:
        return dict(config)  # categorical: nothing to perturb

    def propose(self, inputs: AxisInputs) -> Optional[Segmentation]:
        pc = inputs.people_clusters
        if not pc:
            return None
        segments = [
            Segment(
                image_paths=list(dict.fromkeys(paths)),
                kind=SegmentKind.PERSON,
                label=f"person {cid}",
                meta={"cluster_id": cid},
            )
            for cid, paths in pc.items()
            if paths
        ]
        if len(segments) < 2:
            return None
        # Categorical axis: separation is not geometric (metric="none").
        return Segmentation(
            axis=self.name, segments=segments, unsorted=[], score_space={}, metric="none",
        )
