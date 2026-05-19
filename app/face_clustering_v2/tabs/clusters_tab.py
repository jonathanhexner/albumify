"""spec-040 Phase 5b — Clusters viewer tab for FC App v2.

Read-only RunStore consumer. Points at a run directory (the last v2 run by
default, overridable via text input) and renders one expander per cluster
with face thumbnails. Deliberately minimal — the legacy app's deep cluster
analysis lives in tabs that aren't ported in this MVP.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_clusters_tab() -> None:
    st.subheader("Clusters — v2 run")
    st.caption("Read-only view backed by `face_cluster.run_store.RunStore`.")

    default = st.session_state.get("v2_last_run_dir", "")
    run_dir_str = st.text_input(
        "Run directory",
        value=default,
        help="Output dir of a previous v2 run (or any run with a face_clustering.db).",
        key="v2_clusters_dir",
    )
    if not run_dir_str:
        st.info("No run loaded yet. Run a pipeline from the Run tab, or paste a run directory above.")
        return
    run_dir = Path(run_dir_str)
    db_path = run_dir / "face_clustering.db"
    if not db_path.exists():
        st.error(f"face_clustering.db not found at {db_path}")
        return

    from face_cluster.run_store import RunStore

    try:
        store = RunStore(run_dir)
        clusters = store.list_clusters()
    except Exception as e:
        st.error(f"Could not read RunStore: {e}")
        return

    if not clusters:
        st.info("Run produced no clusters (all faces noise).")
        return

    # Summary row.
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", len(clusters))
    c2.metric("Faces", sum(int(c.get("size", 0)) for c in clusters))
    c3.metric("DB path", str(db_path.name))

    # One expander per cluster — keep it skim-able.
    for c in sorted(clusters, key=lambda r: -int(r.get("size", 0))):
        cid = int(c.get("cluster_id", -1))
        size = int(c.get("size", 0))
        with st.expander(f"Cluster {cid} — {size} face(s)"):
            try:
                rows = store.list_assignments(cluster_id=cid)
            except Exception as e:
                st.error(f"Could not list cluster {cid}: {e}")
                continue
            # Render thumbnails in a wrapping grid (4 per row).
            cols = st.columns(4)
            for i, r in enumerate(rows):
                with cols[i % 4]:
                    crop = r.get("crop_path")
                    if crop and Path(run_dir / crop).exists():
                        st.image(str(run_dir / crop), width=120)
                    st.caption(f"face_id={r.get('face_id')}")
