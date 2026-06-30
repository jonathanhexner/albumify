"""Geo-Vision Studio — pure view helpers (spec-095).

No Streamlit. Operates on the ``{path: {method_key: AnalysisColumn}}`` structure
returned by ``app.image_studio.engine.run_methods`` (spec-094 engine), adding the
geo-specific bits 095 owns: EXIF-vs-GeoCLIP map points, a haversine accuracy
summary, and CSV rows. Kept pure so it is unit-testable without the UI.

AnalysisColumn is duck-typed here (``.topk`` / ``.display``) to avoid coupling.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

# Distance (km) within which a GeoCLIP top-1 guess counts as a "hit" vs EXIF GPS.
DEFAULT_HIT_KM = 25.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _exif_latlon(cols: dict) -> Optional[tuple]:
    """(lat, lon) from an image's EXIF column, or None if no GPS."""
    col = cols.get("exif")
    if not col or not col.topk:
        return None
    e = col.topk[0]
    lat, lon = e.get("lat"), e.get("lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def _geoclip_top1(cols: dict) -> Optional[dict]:
    col = cols.get("geoclip")
    if not col or not col.topk:
        return None
    return col.topk[0]


def geoclip_accuracy(
    columns: Dict[str, dict], threshold_km: float = DEFAULT_HIT_KM
) -> Dict[str, float]:
    """Hits/total for GeoCLIP top-1 vs EXIF GPS (images without GPS are ignored).

    A hit = haversine(EXIF, GeoCLIP#1) <= threshold_km. Returns
    ``{hits, total, threshold_km}``; ``total`` counts only EXIF-GPS images that
    also have a GeoCLIP prediction.
    """
    hits = total = 0
    for cols in columns.values():
        exif = _exif_latlon(cols)
        top1 = _geoclip_top1(cols)
        if exif is None or top1 is None:
            continue
        total += 1
        if haversine_km(exif[0], exif[1], float(top1["lat"]), float(top1["lon"])) <= threshold_km:
            hits += 1
    return {"hits": hits, "total": total, "threshold_km": threshold_km}


def map_points(columns: Dict[str, dict]) -> List[dict]:
    """Flat list of map markers: EXIF (truth) and GeoCLIP#1 (guess) per image.

    Each row: ``{path, lat, lon, source}`` where source is 'exif' or 'geoclip'.
    Only points with coordinates are emitted.
    """
    rows: List[dict] = []
    for path, cols in columns.items():
        exif = _exif_latlon(cols)
        if exif is not None:
            rows.append({"path": path, "lat": exif[0], "lon": exif[1], "source": "exif"})
        top1 = _geoclip_top1(cols)
        if top1 is not None and top1.get("lat") is not None:
            rows.append({"path": path, "lat": float(top1["lat"]),
                         "lon": float(top1["lon"]), "source": "geoclip"})
    return rows


def csv_rows(columns: Dict[str, dict], selected: List[str]) -> List[dict]:
    """Per-image flat rows for CSV export (one column per selected method)."""
    rows = []
    for path, cols in sorted(columns.items()):
        row = {"image": path}
        for key in selected:
            col = cols.get(key)
            row[key] = col.display if col else ""
        rows.append(row)
    return rows
