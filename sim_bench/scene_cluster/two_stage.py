"""Spec-103 arm A3 — two-stage scene clustering: time/geo SEGMENT, then visual SUB-cluster.

Motivation (from eyeballing the flat-fusion report): a flat blended distance averages visual+time, so a
generous time scale can glue visually-distinct shots taken a few minutes apart into one scene, while a
tight scale would drop the near-duplicate rescues. Two-stage sidesteps the tradeoff by keeping the axes
separate:

  Stage 1  segment the trip into temporal events   (sort by capture time; new segment when the gap
           exceeds ``gap_threshold_min`` or GPS jumps > ``geo_jump_m``).  Time is the backbone.
  Stage 2  WITHIN each segment, connect photos whose visual cosine distance < ``visual_tau`` (single-
           linkage / connected components).  A component of >= ``min_scene_size`` photos is a scene;
           a lone photo in its segment stays unclustered (label -1), the two-stage analogue of noise.

Why connected-components not HDBSCAN in stage 2: segments are small (often 2-4 photos) and HDBSCAN is
degenerate at that size (it labels 2-photo groups as noise). A visual-distance threshold is robust for
small N and is exactly "same place/subject within the same moment". No imputation: a photo with no time
can't be segmented, so all untimed photos share one trailing segment and are sub-clustered visually.

spec-053 shape: config in __init__, per-call data in TwoStageInputs, calc() -> TwoStageResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from sim_bench.scene_cluster.geo_time_fusion import _geo_from, _haversine_m, _time_from

logger = logging.getLogger(__name__)


@dataclass
class TwoStageInputs:
    embeddings: dict            # image_id -> np.ndarray
    geo_metadata: dict          # image_id -> GeoMetadata (entries may be geo-less / time-less)
    image_ids: list


@dataclass
class TwoStageResult:
    image_ids: list
    labels: dict                # image_id -> cluster id (>=0) or -1 (unclustered / lone-in-segment)
    n_segments: int
    segment_of: dict            # image_id -> segment index (for inspection)


class TwoStageSceneClusterer:
    def __init__(self, gap_threshold_min: float = 60.0, geo_jump_m: float = 500.0,
                 visual_tau: float = 0.40, use_geo_in_segmentation: bool = True,
                 min_scene_size: int = 2, burst_sec: float = 60.0, burst_tau: float = 0.65):
        # visual_tau: photos join a scene only if visual distance < this (STRICT look-alike test).
        # burst_sec / burst_tau: time is a strong clue ONLY at very short range -- two photos taken
        # <= burst_sec apart may join at the looser burst_tau (a near-simultaneous shot where the
        # camera moved). Beyond burst_sec, time is ignored and only the strict visual_tau applies.
        self.gap_threshold_min = float(gap_threshold_min)
        self.geo_jump_m = float(geo_jump_m)
        self.visual_tau = float(visual_tau)
        self.use_geo_in_segmentation = bool(use_geo_in_segmentation)
        self.min_scene_size = int(min_scene_size)
        self.burst_sec = float(burst_sec)
        self.burst_tau = float(burst_tau)

    def calc(self, inputs: TwoStageInputs) -> TwoStageResult:
        ids = list(inputs.image_ids)
        times = {i: _time_from(i, inputs.geo_metadata.get(i)) for i in ids}
        geos = {i: _geo_from(inputs.geo_metadata.get(i)) for i in ids}

        timed = sorted([i for i in ids if times[i] is not None], key=lambda i: times[i])
        untimed = [i for i in ids if times[i] is None]

        # --- Stage 1: temporal (+optional geo) segmentation of the timed photos ---
        segments: list[list[str]] = []
        cur: list[str] = []
        prev = None
        for i in timed:
            if prev is not None:
                gap_min = abs((times[i] - times[prev]).total_seconds()) / 60.0
                jump = (_haversine_m(geos[prev], geos[i])
                        if (self.use_geo_in_segmentation and geos[prev] and geos[i]) else 0.0)
                if gap_min > self.gap_threshold_min or jump > self.geo_jump_m:
                    segments.append(cur)
                    cur = []
            cur.append(i)
            prev = i
        if cur:
            segments.append(cur)
        if untimed:
            segments.append(untimed)  # all time-less photos share one trailing segment

        segment_of = {img: s_idx for s_idx, seg in enumerate(segments) for img in seg}

        # --- Stage 2: visual connected-components within each segment ---
        labels: dict[str, int] = {}
        next_label = 0
        for seg in segments:
            comps = self._components(seg, inputs.embeddings, times)
            for comp in comps:
                if len(comp) >= self.min_scene_size:
                    for img in comp:
                        labels[img] = next_label
                    next_label += 1
                else:
                    for img in comp:
                        labels[img] = -1
        return TwoStageResult(image_ids=ids, labels=labels, n_segments=len(segments),
                              segment_of=segment_of)

    def _components(self, seg: list[str], emb: dict, times: dict) -> list[list[str]]:
        """Single-linkage connected components. Two photos join a scene if they LOOK alike
        (visual distance < visual_tau), OR they were taken within burst_sec of each other and are
        not wildly different (visual distance < burst_tau). Time only helps at very short range."""
        n = len(seg)
        if n == 1:
            return [seg]
        E = np.array([np.asarray(emb[i], dtype=np.float64) for i in seg])
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        cos = np.clip(E @ E.T, -1.0, 1.0)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(n):
            for b in range(a + 1, n):
                vd = 1.0 - cos[a, b]
                ta, tb = times.get(seg[a]), times.get(seg[b])
                dt_sec = abs((ta - tb).total_seconds()) if (ta and tb) else 9e18
                join = (vd < self.visual_tau) or (dt_sec <= self.burst_sec and vd < self.burst_tau)
                if join:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        groups: dict[int, list[str]] = {}
        for k in range(n):
            groups.setdefault(find(k), []).append(seg[k])
        return list(groups.values())
