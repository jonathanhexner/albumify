"""Image Analysis Studio (spec-094 Slice 3).

Standalone Streamlit app: point at a folder, pick analysis methods across
families (image quality, geo-location, caption), run them, and compare every
model's output in a clickable-thumbnail, sortable table grouped into category
tabs. Results persist through ``universal_cache`` (shared with Albumify), so
re-runs are incremental.

Run (Windows):
    .venv/Scripts/streamlit run app/image_studio/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Streamlit runs this file directly, so the repo root is not on sys.path and
# `app.*` / `sim_bench.*` would not import. Add it before those imports.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from app.image_studio import engine, view

logger = logging.getLogger(__name__)

_MODEL_METHODS = {"streetclip", "geoclip", "blip", "ava"}  # heavy / first-run download
_DEFAULTS = {"exif", "streetclip", "blip", "iqa"}

st.set_page_config(page_title="Image Analysis Studio", layout="wide")


def _cache_handler():
    """Real shared DB session for universal_cache; None if unavailable (still runs)."""
    try:
        from sim_bench.api.database.session import get_session_direct
        from sim_bench.pipeline.cache_handler import UniversalCacheHandler
        return UniversalCacheHandler(get_session_direct())
    except Exception as e:  # no DB -> compute without persistence
        logger.warning("cache handler unavailable: %s", e)
        return None


st.title("Image Analysis Studio")
st.caption("spec-094 - run per-image models over a folder and compare outputs. "
           "Results cache in universal_cache (shared with Albumify); re-runs are incremental.")

# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Run")
    folder = st.text_input("Folder", value="D:/Budapest2025_Google")
    limit = st.number_input("Image limit", min_value=1, max_value=500, value=12, step=1)

    st.markdown("**Methods**")
    by_cat: dict = {}
    for m in engine.available_methods():
        by_cat.setdefault(m["category"], []).append(m)

    selected: list = []
    for cat in engine.categories():
        st.markdown(f"*{cat.replace('_', ' ')}*")
        for m in by_cat.get(cat, []):
            label = m["label"] if m["available"] else f"{m['label']} (unavailable)"
            checked = st.checkbox(
                label,
                value=(m["key"] in _DEFAULTS and m["available"]),
                disabled=not m["available"],
                key=f"chk_{m['key']}",
            )
            if checked and m["available"]:
                selected.append(m["key"])

    run = st.button("Run", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if run:
    paths = engine.discover_images(folder, int(limit))
    if not paths:
        st.warning(f"No images found in folder: {folder or '(empty)'}")
    elif not selected:
        st.warning("Select at least one method in the sidebar.")
    else:
        if any(k in _MODEL_METHODS for k in selected):
            st.info("First run of vision / AVA models downloads weights (~1.6 GB) and runs on "
                    "CPU - this can take a while. Cached results make re-runs instant.")
        bar = st.progress(0.0, "Starting...")

        def _cb(frac: float, msg: str) -> None:
            try:
                bar.progress(min(max(frac, 0.0), 1.0), msg)
            except Exception:
                pass

        columns = engine.run_methods(paths, selected, cache_handler=_cache_handler(), progress=_cb)
        bar.empty()

        st.session_state["studio_result"] = {
            "paths": paths, "columns": columns, "selected": selected, "folder": folder,
        }
        st.session_state["studio_thumbs"] = {p: view.load_thumb(p) for p in paths}
        st.session_state.pop("studio_selected", None)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
res = st.session_state.get("studio_result")
if res:
    n_with = sum(1 for per in res["columns"].values() if per)
    st.subheader(f"{len(res['paths'])} images - {len(res['selected'])} methods")
    view.render_results(
        res["paths"], res["columns"], res["selected"],
        st.session_state.get("studio_thumbs", {}), folder=res["folder"],
    )
else:
    st.info("Set a folder + methods in the sidebar, then click **Run**.")
