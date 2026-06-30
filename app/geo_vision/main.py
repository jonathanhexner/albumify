"""Geo-Vision Studio (spec-095) — standalone Streamlit app.

Thin UI over the spec-094 engine (``app.image_studio.engine``): runs the geo /
caption family (EXIF, StreetCLIP, GeoCLIP, BLIP) over a folder and adds 095's
unique visual layer — confidence bars and an EXIF-vs-GeoCLIP map. No domain
logic here; scoring goes through ``run_methods`` (universal_cache-backed).

Run:  .venv/Scripts/streamlit run app/geo_vision/main.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Streamlit runs this script with app/geo_vision/ on sys.path, not the repo root,
# so `import app...` fails without this bootstrap (mirrors app/album/main.py).
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd
import streamlit as st

from app.image_studio.engine import available_methods, run_methods, CATEGORY_GEO
from app.geo_vision import geo_view

logger = logging.getLogger(__name__)

try:  # HEIC/HEIF thumbnails
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass

DEFAULT_FOLDER = r"D:\Budapest2025_Google"


def _cache_handler():
    """Real universal_cache handler against ~/.sim_bench/sim_bench.db (best-effort)."""
    try:
        from sim_bench.api.database.session import get_session_direct
        from sim_bench.pipeline.cache_handler import UniversalCacheHandler
        return UniversalCacheHandler(get_session_direct())
    except Exception as e:  # cache is an optimisation, never a hard requirement
        logger.warning("geo_vision: no cache handler (%s) — re-inferring each run", e)
        return None


def _geo_methods():
    return [m for m in available_methods() if m["category"] == CATEGORY_GEO]


def _confidence_bar(label: str, value: float):
    st.caption(f"{label}  ·  {value:.2f}")
    st.progress(min(max(value, 0.0), 1.0))


def _render_image_row(path: str, cols: dict, selected: list):
    left, right = st.columns([1, 3])
    with left:
        try:
            st.image(path, width=150)
        except Exception:
            st.write("(no preview)")
        st.caption(os.path.basename(path))
    with right:
        if "exif" in selected and "exif" in cols:
            st.write(f"**EXIF:** {cols['exif'].display}")
        for key, title in (("streetclip", "StreetCLIP"), ("geoclip", "GeoCLIP")):
            if key in selected and key in cols and cols[key].topk:
                st.write(f"**{title}** (confidence is relative ranking, not accuracy):")
                for entry in cols[key].topk[:3]:
                    lbl = entry.get("label") or entry.get("place") or \
                        f"{entry.get('lat'):.2f},{entry.get('lon'):.2f}"
                    conf = float(entry.get("score", entry.get("prob", 0.0)))
                    _confidence_bar(lbl, conf)
        if "blip" in selected and "blip" in cols:
            st.write(f"**Caption:** {cols['blip'].display}")
    st.divider()


def main():
    st.set_page_config(page_title="Geo-Vision Studio", layout="wide")
    st.title("Geo-Vision Studio")
    st.caption("Run geo/vision models on a folder and inspect each guess + confidence. "
               "spec-095, on the spec-094 engine.")

    methods = _geo_methods()
    with st.sidebar:
        folder = st.text_input("Folder path", value=DEFAULT_FOLDER)
        limit = st.number_input("Image limit", min_value=1, max_value=2000, value=12)
        st.markdown("**Models**")
        selected = []
        for m in methods:
            checked = st.checkbox(m["label"], value=m["available"], disabled=not m["available"],
                                  help=None if m["available"] else "dependency not installed")
            if checked and m["available"]:
                selected.append(m["key"])
        run = st.button("Run", type="primary", disabled=not selected)

    if not run:
        st.info("Pick a folder and models in the sidebar, then click Run.")
        return

    paths = geo_view_discover(folder, int(limit))
    if not paths:
        st.warning(f"No images found in: {folder}")
        return

    bar = st.progress(0.0, text="Starting...")
    columns = run_methods(
        paths, selected,
        cache_handler=_cache_handler(),
        progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg),
    )
    bar.empty()

    _render_summary(paths, columns, selected)
    st.subheader("Per-image results")
    for path in paths:
        _render_image_row(path, columns.get(path, {}), selected)


def geo_view_discover(folder: str, limit: int):
    from app.image_studio.engine import discover_images
    return discover_images(folder, limit)


def _render_summary(paths, columns, selected):
    n = len(paths)
    with_gps = sum(1 for p in paths if geo_view._exif_latlon(columns.get(p, {})) is not None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Images", n)
    c2.metric("With EXIF GPS", with_gps)
    if "geoclip" in selected:
        acc = geo_view.geoclip_accuracy(columns)
        label = f"{acc['hits']}/{acc['total']}" if acc["total"] else "n/a"
        c3.metric(f"GeoCLIP within {int(acc['threshold_km'])}km", label)

    pts = geo_view.map_points(columns)
    if pts:
        st.subheader("Map — EXIF (truth) vs GeoCLIP#1 (guess)")
        df = pd.DataFrame(pts)
        df["color"] = df["source"].map({"exif": "#2ca02c", "geoclip": "#ff7f0e"})
        try:
            st.map(df, latitude="lat", longitude="lon", color="color", size=40)
        except Exception:
            st.map(df[["lat", "lon"]])

    rows = geo_view.csv_rows(columns, selected)
    if rows:
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="geo_vision_results.csv",
                           mime="text/csv")


if __name__ == "__main__":
    main()
