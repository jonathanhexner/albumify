"""Face Clustering Debug App — entry point.

Launch:
    streamlit run app/face_clustering_debug/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.face_clustering_debug.pages.algorithm_comparison import render_algorithm_comparison_page
from app.face_clustering_debug.pages.attach_decisions import render_attach_decisions_page
from app.face_clustering_debug.pages.distance_lookup import render_distance_lookup_page
from app.face_clustering_debug.pages.embedding_analysis import render_embedding_analysis_page
from app.face_clustering_debug.pages.merge_decisions import render_merge_decisions_page
from app.face_clustering_debug.pages.overview import render_overview_page
from app.face_clustering_debug.pages.parameter_tuning import render_parameter_tuning_page
from app.face_clustering_debug.services.db_loader import DBLoader
from app.face_clustering_debug.services.file_loader import FileLoader

st.set_page_config(page_title="Face Clustering Debug", page_icon="👥", layout="wide")

_DEFAULT_DIR = "results/face_clustering_benchmark"


def _build_loader():
    """Render sidebar data-source controls and return a configured loader."""
    source = st.sidebar.radio("Source", ["Benchmark Files", "Database"], label_visibility="collapsed")

    if source == "Database":
        album_id = st.sidebar.text_input("Album ID", "1")
        return DBLoader(album_id)

    # --- Benchmark Files path ---
    results_dir = Path(st.sidebar.text_input("Results directory", _DEFAULT_DIR))
    benchmark_files = FileLoader.list_benchmark_files(results_dir)

    if not benchmark_files:
        st.sidebar.error(f"No benchmark_*.json files in:\n`{results_dir}`")
        st.sidebar.info("Run the benchmark first:\n```\npython scripts/benchmark_face_clustering.py\n```")
        return None

    selected_file = st.sidebar.selectbox(
        "Run",
        benchmark_files,
        format_func=lambda p: p.name,
        label_visibility="collapsed",
    )
    return FileLoader(results_dir, specific_file=selected_file)


def main() -> None:
    st.title("👥 Face Clustering Debug")

    with st.sidebar:
        st.markdown("### 📂 Data Source")
        loader = _build_loader()

        if loader is None:
            return

        # Show run metadata
        info = loader.get_run_info()
        if info.get("album_name"):
            st.caption(f"📁 **{info['album_name']}**")
        if info.get("total_faces"):
            st.caption(f"👤 {info['total_faces']} faces")
        if info.get("timestamp"):
            st.caption(f"🕐 {info['timestamp']}")

        st.divider()
        st.markdown("### 🧮 Clustering Method")
        methods = loader.get_available_methods()
        method = st.selectbox("Method", methods, label_visibility="collapsed") if methods else None

        if method and not loader.has_debug_data(method):
            st.warning(
                f"**{method}** has no debug data.\n\n"
                "Merge/Attach Decisions tabs will be empty.\n"
                "Select **hybrid_knn** for full debug data."
            )

    if method is None:
        st.warning("⚠️ No clustering results found.")
        st.code("python scripts/benchmark_face_clustering.py --album-path <path/to/album>")
        return

    tab_overview, tab_merge, tab_attach, tab_dist, tab_tune, tab_compare, tab_embed = st.tabs([
        "📊 Overview",
        "🔗 Merge Decisions",
        "📎 Attach Decisions",
        "📏 Distance Lookup",
        "⚙️ Parameter Tuning",
        "🆚 Algorithm Comparison",
        "🔬 Embedding Analysis",
    ])

    with tab_overview:
        try:
            render_overview_page(loader, method)
        except Exception as e:
            st.error(f"Error rendering Overview: {e}")
            st.exception(e)

    with tab_merge:
        try:
            render_merge_decisions_page(loader, method)
        except Exception as e:
            st.error(f"Error rendering Merge Decisions: {e}")
            st.exception(e)

    with tab_attach:
        try:
            render_attach_decisions_page(loader, method)
        except Exception as e:
            st.error(f"Error rendering Attach Decisions: {e}")
            st.exception(e)

    with tab_dist:
        try:
            render_distance_lookup_page(loader, method)
        except Exception as e:
            st.error(f"Error rendering Distance Lookup: {e}")
            st.exception(e)

    with tab_tune:
        try:
            render_parameter_tuning_page(loader)
        except Exception as e:
            st.error(f"Error rendering Parameter Tuning: {e}")
            st.exception(e)

    with tab_compare:
        try:
            render_algorithm_comparison_page(loader)
        except Exception as e:
            st.error(f"Error rendering Algorithm Comparison: {e}")
            st.exception(e)

    with tab_embed:
        try:
            render_embedding_analysis_page(loader)
        except Exception as e:
            st.error(f"Error rendering Embedding Analysis: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
