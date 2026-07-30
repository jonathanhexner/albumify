"""TIME axis — group by capture-time gaps, ignoring location (spec-022)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from geo_cluster._clustering import split_on_time_gaps
from geo_cluster.axes.base import AxisInputs, register_axis
from geo_cluster.types import Segment, Segmentation


@register_axis
class TimeAxis:
    name = "time"
    needs = "capture timestamps"

    def perturbed_config(self, config: dict) -> dict:
        c = dict(config)
        c["time_gap_hours"] = config.get("time_gap_hours", 8.0) * 1.25
        return c

    def propose(self, inputs: AxisInputs) -> Optional[Segmentation]:
        meta = inputs.metadata
        paths = [p for p, m in meta.items() if m.has_time]
        if len(paths) < 2:
            return None

        gap = inputs.config.get("time_gap_hours", 8.0)
        times = [meta[p].timestamp for p in paths]
        labels = split_on_time_gaps(times, gap)

        t0 = min(times)
        score_space = {
            p: np.array([(meta[p].timestamp - t0).total_seconds()]) for p in paths
        }
        segments = [
            Segment(image_paths=[paths[i] for i in range(len(paths)) if labels[i] == s])
            for s in sorted(set(labels))
        ]
        unsorted = [p for p, m in meta.items() if not m.has_time]
        return Segmentation(
            axis=self.name, segments=segments, unsorted=unsorted,
            score_space=score_space, metric="euclidean",
        )
