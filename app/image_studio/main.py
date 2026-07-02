"""Image Analysis Studio (spec-094).

Two pages via a top navbar:
  • Configure Run — pick a folder + methods across families, Run. Each run is
    saved to a RunFolder (<folder>/.studio_runs/<run_id>/) and cached in
    universal_cache.
  • Browse Run — open a saved run; a top-level Quality | Geo toggle shows each
    family's natural view (quality = ranking table; geo = map + accuracy).

Run (Windows):
    .venv/Scripts/streamlit run app/image_studio/main.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Streamlit runs this file directly, so the repo root is not on sys.path.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from app.image_studio import engine, view, run_folder

logger = logging.getLogger(__name__)

_MODEL_METHODS = {"streetclip", "geoclip", "blip", "ava", "maniqa", "musiq", "hyperiqa", "clipiqa"}
_DEFAULTS = {"exif", "streetclip", "iqa", "brisque"}
_DEFAULT_FOLDER = "D:/Budapest2025_Google"

st.set_page_config(page_title="Image Analysis Studio", layout="wide")


def _cache_handler():
    try:
        from sim_bench.api.database.session import get_session_direct
        from sim_bench.pipeline.cache_handler import UniversalCacheHandler
        return UniversalCacheHandler(get_session_direct())
    except Exception as e:  # no DB -> compute without persistence
        logger.warning("cache handler unavailable: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Shell — top navbar + shared folder
# --------------------------------------------------------------------------- #
st.title("Image Analysis Studio")
page = st.radio("nav", ["Configure Run", "Browse Run"], horizontal=True,
                label_visibility="collapsed", key="nav")
folder = st.text_input("Folder", value=st.session_state.get("folder", _DEFAULT_FOLDER), key="folder")
st.divider()


# --------------------------------------------------------------------------- #
def configure_run():
    with st.sidebar:
        st.header("Configure")
        limit = st.number_input("Image limit", min_value=1, max_value=1000, value=24, step=1)
        st.markdown("**Methods**")
        by_cat: dict = {}
        for mth in engine.available_methods():
            by_cat.setdefault(mth["category"], []).append(mth)
        selected: list = []
        for cat in engine.categories():
            st.markdown(f"*{cat.replace('_', ' ')}*")
            for mth in by_cat.get(cat, []):
                lbl = mth["label"] if mth["available"] else f"{mth['label']} (unavailable)"
                if st.checkbox(lbl, value=(mth["key"] in _DEFAULTS and mth["available"]),
                               disabled=not mth["available"], key=f"chk_{mth['key']}") and mth["available"]:
                    selected.append(mth["key"])
        run = st.button("▶ Run", type="primary", use_container_width=True)

    st.subheader("Configure a run")
    st.caption("Pick a folder + methods on the left, then Run. Each run is saved and opens in "
               "**Browse Run**. Scores cache in universal_cache, so re-running is incremental.")

    if not run:
        st.info("Set methods in the sidebar, then click **Run**.")
        return
    paths = engine.discover_images(folder, int(limit))
    if not paths:
        st.warning(f"No images found in: {folder or '(empty)'}")
        return
    if not selected:
        st.warning("Select at least one method.")
        return
    if any(k in _MODEL_METHODS for k in selected):
        st.info("Heavy models (vision / AVA / MANIQA / MUSIQ …) download weights on first use and run "
                "on CPU — this can take a while. Cached results make re-runs instant.")

    bar = st.progress(0.0, "Starting…")
    columns = engine.run_methods(
        paths, selected, cache_handler=_cache_handler(),
        progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
    bar.empty()

    created = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rid = run_folder.make_run_id(created, selected)
    run_folder.save(folder, rid, selected, paths, columns, created)
    st.success(f"Saved run **{rid}** ({len(paths)} images, {len(selected)} methods).")
    if st.button("Open in Browse Run →"):
        st.session_state["nav"] = "Browse Run"
        st.session_state["browse_run"] = rid
        st.rerun()


# --------------------------------------------------------------------------- #
def browse_run():
    runs = run_folder.list_runs(folder)
    if not runs:
        st.info(f"No saved runs in **{folder}** yet. Go to **Configure Run** and run one.")
        return

    with st.sidebar:
        st.header("Runs")
        ids = [r["run_id"] for r in runs]
        labels = {r["run_id"]: f"{r.get('created_ts', '?')} · {r['n_images']} imgs · "
                                f"{len(r['methods'])} methods" for r in runs}
        default = st.session_state.get("browse_run")
        idx = ids.index(default) if default in ids else 0
        rid = st.radio("Pick a run", ids, index=idx,
                       format_func=lambda x: labels[x], key="browse_run")

    data = run_folder.load(folder, rid)
    if not data:
        st.error(f"Could not load run {rid}.")
        return

    st.subheader(f"Run · {rid}")
    st.caption(f"{data['manifest']['n_images']} images · methods: {', '.join(data['methods'])}")

    fam = st.radio("family", ["Quality", "Geo"], horizontal=True, label_visibility="collapsed",
                   key=f"fam_{rid}")

    thumbs = st.session_state.setdefault(f"thumbs_{rid}",
                                         {p: view.load_thumb(p) for p in data["paths"]})
    if fam == "Quality":
        view.render_quality(data["paths"], data["columns"], data["methods"], thumbs, folder=folder)
    else:
        view.render_geo(data["paths"], data["columns"], data["methods"], thumbs, folder=folder)


if page == "Configure Run":
    configure_run()
else:
    browse_run()
