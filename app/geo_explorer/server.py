"""Standalone geo-segmentation explorer (spec-022).

Runs the REAL pipeline (`discover_images -> extract_geo_metadata ->
geo_temporal_segment`) once on a photo directory, then serves an interactive
Leaflet map + timeline so you can drag the geo radius / time gap / floor and
watch the segmentation change live. Same `geo_cluster` code that the pipeline
step uses powers the live re-segmentation, so anything you tune here ports
straight into Albumify.

Run (Windows):
    set GEO_EXPLORER_DIR=D:/Budapest2025_Google
    .venv/Scripts/python -m uvicorn app.geo_explorer.server:app --port 8077
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse

from geo_cluster.axes.base import AxisInputs, get_axes
from geo_cluster.home import HomeAnchor
from geo_cluster.selector import SegmentationSelector
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.run import execute_spec
from sim_bench.pipeline.spec import PipelineSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
SRC = Path(os.environ.get("GEO_EXPLORER_DIR", "D:/Budapest2025_Google"))


def _run_pipeline() -> PipelineContext:
    """The one real pipeline pass that feeds the explorer."""
    spec = PipelineSpec(steps=["discover_images", "extract_geo_metadata", "geo_temporal_segment"])
    ctx = PipelineContext(source_directory=SRC)
    logger.info("running geo pipeline on %s ...", SRC)
    execute_spec(spec, ctx)
    logger.info("done: %d images, %d with GPS", len(ctx.image_paths), len(ctx.geo_metadata))
    return ctx


CTX = _run_pipeline()
META = CTX.geo_metadata
HOME = CTX.geo_home
PATHS = list(META.keys())
PID = {p: i for i, p in enumerate(PATHS)}

app = FastAPI(title="Geo Segmentation Explorer")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/photos")
def photos():
    items = [
        {
            "id": PID[p],
            "lat": m.lat,
            "lon": m.lon,
            "t": m.timestamp.timestamp() if m.has_time else None,
            "has_geo": m.has_geo,
        }
        for p, m in META.items()
    ]
    times = [m.timestamp.timestamp() for m in META.values() if m.has_time]
    return {
        "photos": items,
        "home": list(HOME) if HOME else None,
        "t_min": min(times) if times else None,
        "t_max": max(times) if times else None,
        "n_total": len(META),
        "n_geo": sum(1 for m in META.values() if m.has_geo),
        "n_time": len(times),
        "source": str(SRC),
    }


def _round(v):
    return None if v is None else round(v, 3)


def _labels(seg):
    if seg is None:
        return None
    out = {}
    for idx, s in enumerate(seg.segments):
        for p in s.image_paths:
            out[str(PID[p])] = idx
    return {"n": len(seg.segments), "labels": out}


@app.get("/api/segment")
def segment(radius_km: float = 30.0, gap_hours: float = 8.0, floor: float = 0.45):
    inputs = AxisInputs(
        metadata=META, home=HOME,
        config={"geo_radius_km": radius_km, "time_gap_hours": gap_hours},
    )
    axes = get_axes()
    geo = _labels(axes["geo"].propose(inputs))
    time_ = _labels(axes["time"].propose(inputs))
    outcome = SegmentationSelector(floor=floor).calc(inputs)
    scores = [
        {
            "axis": s.axis, "overall": _round(s.overall),
            "separation": _round(s.separation), "coverage": _round(s.coverage),
            "balance": _round(s.balance), "stability": _round(s.stability),
            "parsimony": _round(s.parsimony),
            "n_segments": s.detail.get("n_segments"),
            "insufficient": bool(s.detail.get("insufficient")),
        }
        for s in outcome.scores
    ]
    return {"geo": geo, "time": time_, "winner": outcome.winning_axis,
            "flat": outcome.flat, "floor": floor, "scores": scores}


_THUMBS: dict[int, bytes] = {}


@app.get("/thumb/{pid}")
def thumb(pid: int):
    if pid not in _THUMBS:
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener
        register_heif_opener()
        try:
            with Image.open(PATHS[pid]) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((240, 240))
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80)
                _THUMBS[pid] = buf.getvalue()
        except Exception as e:
            logger.debug("thumb failed %s: %s", pid, e)
            _THUMBS[pid] = b""
    return Response(_THUMBS[pid], media_type="image/jpeg")
