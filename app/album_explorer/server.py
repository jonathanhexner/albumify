"""Album Pipeline Explorer (spec-022).

Point it at any folder, hit Run, and it executes the REAL pipeline via
`execute_spec` (same engine as Albumify):

    discover_images -> extract_geo_metadata -> infer_geo_clip (StreetCLIP)
    -> caption_images (BLIP) -> geo_temporal_segment

then shows a per-image table: thumbnail, time, GPS, StreetCLIP top-3 cities +
scores, BLIP caption. Results are disk-cached per image, so re-runs are fast.

Run (Windows):
    .venv/Scripts/python -m uvicorn app.album_explorer.server:app --port 8078
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.run import execute_spec
from sim_bench.pipeline.spec import PipelineSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
DEFAULT_DIR = os.environ.get("ALBUM_EXPLORER_DIR", "D:/Budapest2025_Google")

app = FastAPI(title="Album Pipeline Explorer")


class RunReq(BaseModel):
    folder: str
    limit: int = 12  # default small so the first (model-download) run is bearable


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/default")
def default():
    return {"folder": DEFAULT_DIR}


@app.post("/api/run")
def run(req: RunReq):
    folder = Path(req.folder)
    if not folder.exists():
        return JSONResponse({"error": f"folder not found: {folder}"}, status_code=400)

    spec = PipelineSpec(
        steps=[
            "discover_images", "extract_geo_metadata",
            "infer_geo_clip", "infer_geo_coords",
            "caption_images", "geo_temporal_segment",
        ],
        step_configs={
            "discover_images": {"limit": req.limit} if req.limit else {},
            "infer_geo_clip": {"top_k": 3},
            "infer_geo_coords": {"top_k": 3},
        },
    )
    ctx = PipelineContext(source_directory=folder)
    logger.info("running pipeline on %s (limit=%s) ...", folder, req.limit)
    result = execute_spec(spec, ctx)
    if not result.success:
        return JSONResponse({"error": result.error_message or "pipeline failed"}, status_code=500)

    rows = []
    for p in ctx.image_paths:
        path = str(p)
        m = ctx.geo_metadata.get(path)
        rows.append({
            "path": path,
            "name": Path(path).name,
            "time": m.timestamp.isoformat(sep=" ") if (m and m.has_time) else None,
            "lat": m.lat if (m and m.has_geo) else None,
            "lon": m.lon if (m and m.has_geo) else None,
            "streetclip": ctx.geo_clip_predictions.get(path, []),
            "geoclip": ctx.geo_coord_predictions.get(path, []),
            "caption": ctx.image_captions.get(path, ""),
        })

    n_gps = sum(1 for r in rows if r["lat"] is not None)
    n_time = sum(1 for r in rows if r["time"] is not None)
    return {
        "rows": rows,
        "summary": {
            "source": str(folder), "n": len(rows), "n_gps": n_gps, "n_time": n_time,
            "home": list(ctx.geo_home) if ctx.geo_home else None,
            "n_segments": len(ctx.geo_segments),
        },
    }


@app.get("/thumb")
def thumb(path: str):
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener
    register_heif_opener()
    try:
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
            img.thumbnail((150, 150))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            return Response(buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.debug("thumb failed %s: %s", path, e)
        return Response(b"", media_type="image/jpeg")
