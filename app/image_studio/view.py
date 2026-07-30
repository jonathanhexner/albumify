"""Image Analysis Studio — rendering layer (spec-094 Slice 3).

Streamlit rendering only; all data comes from ``engine.run_methods``. Clickable
thumbnails use a real ``st.button`` grid (v2 rule — never ``st.dataframe``
row-select). Confidence is shown as a bar labelled "relative, not accuracy".
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

from app.image_studio import engine
from app.image_studio.engine import METHODS, AnalysisColumn
from app.image_studio.method_info import METHOD_INFO


def render_legend(method_keys):
    """Expander explaining what each selected column measures + its range/direction."""
    with st.expander("ℹ️  What does each column mean? (measures · range · what's better)"):
        rows = ["| Column | What it measures | Range | Better |", "|---|---|---|---|"]
        for k in method_keys:
            info = METHOD_INFO.get(k)
            if not info:
                continue
            label = METHODS[k].label if k in METHODS else k
            measures, rng, _raw_dir, in_table = info
            rows.append(f"| **{label}** | {measures} | `{rng}` | {in_table} |")
        st.markdown("\n".join(rows))
        st.caption("Quality scores are shown so **higher = better for every column** — the app "
                   "negates distortion metrics (BRISQUE/NIQE), whose raw value is lower-is-better. "
                   "Geo confidence is the model's certainty, **not** accuracy.")

logger = logging.getLogger(__name__)


def _basename(path: str) -> str:
    return path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]


def load_thumb(path: str, size: int = 150):
    """Load + downscale an image to a PIL thumbnail. Never raises (-> None)."""
    try:
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener
        register_heif_opener()
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        img.thumbnail((size, size))
        return img
    except Exception as e:  # corrupt / unreadable -> no preview, never crash
        logger.debug("thumb failed for %s: %s", path, e)
        return None


def _bar_html(value: Optional[float], color: str = "#6ea8fe") -> str:
    if value is None:
        return ""
    v = max(0.0, min(1.0, float(value)))
    return (f"<div style='background:#2a2f3a;border-radius:4px;height:7px;width:96px'>"
            f"<div style='background:{color};width:{int(v * 100)}%;height:7px;"
            f"border-radius:4px'></div></div>")


def _sortable_keys(method_keys: List[str], columns: Dict[str, Dict[str, AnalysisColumn]]) -> List[str]:
    out = []
    for k in method_keys:
        if any((per.get(k) and per[k].sort_value is not None) for per in columns.values()):
            out.append(k)
    return out


def render_table(paths, columns, method_keys, thumbs, key_prefix,
                 default_sort=None, default_desc=True):
    """One sortable, clickable-thumbnail table for a set of method columns.

    ``default_sort`` — a method key to sort by initially (else file name);
    ``default_desc`` — initial sort direction.
    """
    sortable = _sortable_keys(method_keys, columns)
    ordered = sorted(paths, key=_basename)
    if sortable:
        options = ["(file name)"] + sortable
        idx = options.index(default_sort) if default_sort in options else 0
        c1, c2 = st.columns([3, 1])
        sort_key = c1.selectbox("Sort by", options, index=idx, key=f"{key_prefix}_sort")
        desc = c2.checkbox("Desc", value=default_desc, key=f"{key_prefix}_desc")
        if sort_key != "(file name)":
            def sv(p):
                c = columns.get(p, {}).get(sort_key)
                return c.sort_value if (c and c.sort_value is not None) else float("-inf")
            ordered = sorted(paths, key=sv, reverse=desc)

    st.caption("Confidence bars = model confidence (**relative, not accuracy**).")
    widths = [1.1, 1.5] + [1.7] * len(method_keys)
    head = st.columns(widths)
    head[0].markdown("**image**")
    head[1].markdown("**file**")
    for i, k in enumerate(method_keys):
        head[2 + i].markdown(f"**{METHODS[k].label}**")

    for p in ordered:
        row = st.columns(widths)
        img = thumbs.get(p)
        row[0].image(img) if img is not None else row[0].write("(no preview)")
        if row[1].button(_basename(p), key=f"{key_prefix}_sel_{_basename(p)}"):
            st.session_state["studio_selected"] = p
            st.rerun()
        for i, k in enumerate(method_keys):
            cell = row[2 + i]
            c = columns.get(p, {}).get(k)
            if not c:
                cell.write("-")
                continue
            cell.markdown(c.display)
            if c.sort_value is not None:
                cell.markdown(_bar_html(c.sort_value), unsafe_allow_html=True)


def render_selected(columns):
    """If a thumbnail was clicked, show it large with all its columns + top-k."""
    p = st.session_state.get("studio_selected")
    if not p:
        return
    st.markdown(f"### {_basename(p)}")
    left, right = st.columns([2, 3])
    with left:
        big = load_thumb(p, size=600)
        st.image(big, use_container_width=True) if big is not None else st.write("(no preview)")
    with right:
        for k, c in columns.get(p, {}).items():
            st.markdown(f"**{METHODS[k].label}** — {c.display}")
            if c.topk:
                with st.expander("top-k / detail"):
                    st.json(c.topk)
    if st.button("Close", key="studio_close"):
        st.session_state.pop("studio_selected", None)
        st.rerun()
    st.divider()


def build_csv(paths, columns, method_keys, folder: str = "") -> bytes:
    """CSV of the results. Metadata mandate: source path, run timestamp, spec/version."""
    run_ts = datetime.now().isoformat(timespec="seconds")
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["file", "source_path", "run_timestamp", "spec_version", "source_folder"]
    for k in method_keys:
        header += [k, f"{k}_score"]
    w.writerow(header)
    for p in sorted(paths, key=_basename):
        rowvals = [_basename(p), p, run_ts, "spec-094", folder]
        for k in method_keys:
            c = columns.get(p, {}).get(k)
            rowvals += [c.display if c else "",
                        c.sort_value if (c and c.sort_value is not None) else ""]
        w.writerow(rowvals)
    return buf.getvalue().encode("utf-8")


def quality_keys(selected):
    return [k for k in selected if METHODS[k].category == engine.CATEGORY_QUALITY]


def geo_keys(selected):
    return [k for k in selected if METHODS[k].category == engine.CATEGORY_GEO]


def render_quality(paths, columns, selected, thumbs, folder=""):
    """Image-quality family: a sortable ranking table (worst -> best)."""
    keys = quality_keys(selected)
    if not keys:
        st.info("This run has no image-quality methods. Configure a run with MANIQA / BRISQUE / "
                "NIQE / IQA / AVA to compare quality here.")
        return
    render_selected(columns)
    render_legend(keys)
    st.caption("Ranking table — sorted worst→best by the first metric (higher = better). "
               "Change the sort or click a file to enlarge.")
    first = next((k for k in keys if any(
        (per.get(k) and per[k].sort_value is not None) for per in columns.values())), None)
    render_table(paths, columns, keys, thumbs, key_prefix="q",
                 default_sort=first, default_desc=False)  # ascending = worst quality first
    st.download_button("Download CSV (quality)", build_csv(paths, columns, keys, folder),
                       file_name="image_quality.csv", mime="text/csv", key="csv_q")


def render_geo(paths, columns, selected, thumbs, folder=""):
    """Geo & caption family: EXIF-vs-GeoCLIP map + accuracy + a label/caption table."""
    keys = geo_keys(selected)
    if not keys:
        st.info("This run has no geo/caption methods. Configure a run with EXIF / StreetCLIP / "
                "GeoCLIP / BLIP to see the map here.")
        return
    render_selected(columns)
    render_legend(keys)
    from app.geo_vision import geo_view
    import pandas as pd

    n = len(paths)
    with_gps = sum(1 for p in paths if geo_view._exif_latlon(columns.get(p, {})) is not None)
    m = st.columns(3)
    m[0].metric("Images", n)
    m[1].metric("With EXIF GPS", with_gps)
    if "geoclip" in keys:
        acc = geo_view.geoclip_accuracy(columns)
        m[2].metric(f"GeoCLIP within {int(acc['threshold_km'])} km",
                    f"{acc['hits']}/{acc['total']}" if acc["total"] else "n/a")

    pts = geo_view.map_points(columns)
    if pts:
        st.markdown("**Map — EXIF (green) vs GeoCLIP #1 (orange)**")
        df = pd.DataFrame(pts)
        df["color"] = df["source"].map({"exif": "#2ca02c", "geoclip": "#e3903a"})
        try:
            st.map(df, latitude="lat", longitude="lon", color="color", size=8)
        except Exception:
            st.map(df[["lat", "lon"]])
    else:
        st.info("No EXIF GPS or GeoCLIP coordinates to map for this run.")

    render_table(paths, columns, keys, thumbs, key_prefix="g")
    st.download_button("Download CSV (geo)", build_csv(paths, columns, keys, folder),
                       file_name="image_geo.csv", mime="text/csv", key="csv_g")
