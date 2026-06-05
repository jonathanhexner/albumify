"""spec-066 — v2 Gallery tab. Browse clusters as exemplar thumbnail strips.

Pure orchestration: resolve run -> service -> filter/paginate -> dispatch
each cluster to ``cluster_strip``. Read-only; disqualify is spec-069.
"""
from __future__ import annotations

import streamlit as st

from app.face_clustering_v2._run_context import cached_cluster_service, resolve_run_dir
from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from app.face_clustering_v2.components.cluster_strip import render_cluster_strip

PAGE_SIZE = 10


def render_gallery_tab() -> None:
    """Render the Gallery tab."""
    st.header("Gallery")
    run_dir = resolve_run_dir()
    if run_dir is None:
        tab_skipped("gallery", "no_run_loaded")
        st.info("No run loaded. Open a run from the History tab first.")
        return
    tab_start("gallery", run_dir)
    service = cached_cluster_service(run_dir, cache_prefix="_gallery_service")
    if service is None:
        tab_skipped("gallery", "repo_failed")
        return
    rows = service.list_clusters()
    if not rows:
        tab_skipped("gallery", "no_clusters")
        st.info("This run has no clusters to show.")
        return

    biggest = max(r.size for r in rows)
    c1, c2, c3 = st.columns(3)
    min_size = c1.slider("Min cluster size", 1, biggest, 1, key="v2_gal_min_size")
    # Cap at the largest cluster so "show every face of the biggest cluster" is
    # reachable; thumbnails wrap onto multiple rows (user feedback 2026-06-05).
    per_cluster = c2.slider("Faces per cluster", 1, biggest, min(8, biggest), key="v2_gal_per_cluster")
    descending = c3.selectbox("Sort by size", ["desc", "asc"], key="v2_gal_sort") == "desc"

    shown = sorted([r for r in rows if r.size >= min_size], key=lambda r: r.size, reverse=descending)
    pages = max(1, (len(shown) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = int(st.number_input("Page", 1, pages, 1, key="v2_gal_page")) if pages > 1 else 1
    page_rows = shown[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    # Cache the page's DB reads per (run, page, slider state): Streamlit
    # re-executes every tab body on each rerun, so without this the Gallery
    # would re-query the DB + reload all faces on every click anywhere.
    cache_key = f"_gallery_page::{run_dir}::{page}::{per_cluster}::{min_size}::{descending}"
    cached = st.session_state.get(cache_key)
    if cached is None:
        face_ids_by_cluster = {r.cluster_id: service.exemplar_face_ids(r.cluster_id, per_cluster) for r in page_rows}
        all_ids = [fid for ids in face_ids_by_cluster.values() for fid in ids]
        cached = (face_ids_by_cluster, all_ids, service.low_quality_face_ids(all_ids))
        st.session_state[cache_key] = cached
    face_ids_by_cluster, all_ids, low_q = cached

    for r in page_rows:
        st.divider()
        render_cluster_strip(
            cluster_id=r.cluster_id, size=r.size, face_ids=face_ids_by_cluster[r.cluster_id],
            run_dir=run_dir, low_quality=low_q,
        )
    tab_done("gallery", n_clusters=len(shown), page=int(page), thumbs_rendered=len(all_ids))
