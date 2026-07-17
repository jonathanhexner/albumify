"""Behaviour tests for the multi-axis segmentation competition (spec-022).

These pin the *architecture's* behaviour, not exact numbers: the right axis
wins for each album shape, and the selector declines to segment when it should.
"""

from datetime import datetime, timedelta

import numpy as np

from geo_cluster.axes.base import AxisInputs
from geo_cluster.selector import SegmentationSelector
from geo_cluster.types import GeoMetadata

_RNG = np.random.default_rng(7)
_Y = 2024
_CFG = {"geo_radius_km": 30.0, "time_gap_hours": 8.0}


def _block(center, n, jitter_km, start, end):
    dlat = _RNG.normal(0, jitter_km / 111.0, n)
    dlon = _RNG.normal(0, jitter_km / (111.0 * np.cos(np.radians(center[0]))), n)
    span = (end - start).total_seconds()
    out = []
    for i in range(n):
        ts = start + timedelta(seconds=float(_RNG.uniform(0, span)))
        out.append((center[0] + dlat[i], center[1] + dlon[i], ts))
    return out


def _album(blocks):
    meta = {}
    for j, (lat, lon, ts) in enumerate(blocks):
        p = f"img_{j:04d}.jpg"
        meta[p] = GeoMetadata(p, ts, lat, lon)
    return meta


def _run(meta, **kw):
    return SegmentationSelector(**kw).calc(AxisInputs(metadata=meta, config=_CFG))


class ut_SegmentationSelector:

    def test_traveler_picks_geo(self):
        tlv, dxb, alp = (32.08, 34.78), (25.20, 55.27), (47.26, 11.39)
        meta = _album(
            _block(tlv, 200, 12, datetime(_Y, 1, 1), datetime(_Y, 12, 31))
            + _block(dxb, 90, 8, datetime(_Y, 5, 12), datetime(_Y, 5, 18))
            + _block(alp, 60, 15, datetime(_Y, 7, 3), datetime(_Y, 7, 7))
        )
        out = _run(meta)
        assert not out.flat
        assert out.winning_axis == "geo"

    def test_kids_at_home_picks_time(self):
        tlv = (32.08, 34.78)
        blocks = []
        for mo in (1, 4, 7, 10):
            blocks += _block(tlv, 60, 10, datetime(_Y, mo, 5), datetime(_Y, mo, 8))
        out = _run(_album(blocks))
        assert not out.flat
        assert out.winning_axis == "time"

    def test_single_event_is_flat(self):
        venue = (47.50, 19.04)
        meta = _album(_block(venue, 600, 3, datetime(_Y, 6, 22, 10), datetime(_Y, 6, 22, 23)))
        out = _run(meta)
        assert out.flat
        assert out.winner is None

    def test_no_metadata_is_flat_and_insufficient(self):
        meta = {f"x_{i}.jpg": GeoMetadata(f"x_{i}.jpg", None, None, None) for i in range(50)}
        out = _run(meta)
        assert out.flat
        assert all(s.detail.get("insufficient") for s in out.scores)

    def test_identity_axis_plugs_in_when_people_clusters_present(self):
        # No geo/time at all -> geo/time insufficient; identity should win.
        meta = {f"p_{i}.jpg": GeoMetadata(f"p_{i}.jpg", None, None, None) for i in range(20)}
        people = {
            1: [f"p_{i}.jpg" for i in range(0, 10)],
            2: [f"p_{i}.jpg" for i in range(10, 20)],
        }
        out = SegmentationSelector().calc(
            AxisInputs(metadata=meta, people_clusters=people, config=_CFG)
        )
        assert not out.flat
        assert out.winning_axis == "identity"

    def test_floor_can_force_flat(self):
        # An impossibly high floor => even a good segmentation is declined.
        tlv, dxb = (32.08, 34.78), (25.20, 55.27)
        meta = _album(
            _block(tlv, 80, 10, datetime(_Y, 1, 1), datetime(_Y, 3, 1))
            + _block(dxb, 80, 8, datetime(_Y, 6, 1), datetime(_Y, 6, 5))
        )
        out = _run(meta, floor=0.99)
        assert out.flat
