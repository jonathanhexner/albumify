"""ut for app/geo_vision/geo_view.py (spec-095) — pure helpers, no Streamlit."""

import pytest

from app.image_studio.engine import AnalysisColumn, CATEGORY_GEO, COORD, LABEL_CONF
from app.geo_vision import geo_view


def _exif_col(lat, lon):
    return AnalysisColumn("exif", CATEGORY_GEO, COORD, None, "-",
                          [{"lat": lat, "lon": lon, "timestamp": None}])


def _geoclip_col(lat, lon, prob=0.4):
    return AnalysisColumn("geoclip", CATEGORY_GEO, LABEL_CONF, prob, "-",
                          [{"lat": lat, "lon": lon, "prob": prob, "place": "X"}])


# Budapest ~ (47.4979, 19.0402); Vienna ~ (48.2082, 16.3738) ~ 215 km away.
BUD = (47.4979, 19.0402)
VIE = (48.2082, 16.3738)


def test_haversine_known_distance():
    d = geo_view.haversine_km(*BUD, *VIE)
    assert 200 < d < 230  # ~215 km


def test_haversine_zero():
    assert geo_view.haversine_km(*BUD, *BUD) == pytest.approx(0.0, abs=1e-6)


def test_geoclip_accuracy_hit_and_miss():
    columns = {
        "a.jpg": {"exif": _exif_col(*BUD), "geoclip": _geoclip_col(47.50, 19.05)},  # ~1km -> hit
        "b.jpg": {"exif": _exif_col(*BUD), "geoclip": _geoclip_col(*VIE)},          # ~215km -> miss
    }
    acc = geo_view.geoclip_accuracy(columns, threshold_km=25.0)
    assert acc == {"hits": 1, "total": 2, "threshold_km": 25.0}


def test_geoclip_accuracy_ignores_exif_less():
    columns = {
        "a.jpg": {"geoclip": _geoclip_col(*BUD)},               # no EXIF -> ignored
        "b.jpg": {"exif": _exif_col(None, None), "geoclip": _geoclip_col(*BUD)},  # EXIF w/o GPS -> ignored
        "c.jpg": {"exif": _exif_col(*BUD), "geoclip": _geoclip_col(47.50, 19.05)},  # counted, hit
    }
    acc = geo_view.geoclip_accuracy(columns)
    assert acc["total"] == 1 and acc["hits"] == 1


def test_map_points_emits_exif_and_geoclip():
    columns = {"a.jpg": {"exif": _exif_col(*BUD), "geoclip": _geoclip_col(*VIE)}}
    pts = geo_view.map_points(columns)
    sources = sorted(p["source"] for p in pts)
    assert sources == ["exif", "geoclip"]
    assert all("lat" in p and "lon" in p for p in pts)


def test_map_points_skips_missing_coords():
    columns = {"a.jpg": {"exif": _exif_col(None, None)}}  # no GPS, no geoclip
    assert geo_view.map_points(columns) == []


def test_csv_rows_one_column_per_method():
    col = AnalysisColumn("blip", CATEGORY_GEO, "text", None, "a bridge", [])
    columns = {"a.jpg": {"blip": col}}
    rows = geo_view.csv_rows(columns, ["blip", "geoclip"])
    assert rows == [{"image": "a.jpg", "blip": "a bridge", "geoclip": ""}]
