"""Tab 4: Clusters (Base) — run overview with UMAP."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist

from face_cluster.analysis_views import RunOverview
from face_cluster.pipeline import PipelineResult

from state import _AsyncState
from nav_helpers import _breadcrumb, _no_result
from cache_helpers import _crop_for_face, _load_faces_df
from gallery_panels import _render_cluster_gallery


def _render_worked_example(result: PipelineResult, overview):
    st.subheader("Worked Algorithm Example")
    st.caption("A concrete walkthrough of edge construction and cluster formation.")
    faces = result.faces
    cr    = result.cluster_result
    if not cr.clusters:
        st.info("No clusters in this run.")
        return
    good_ids = [cid for cid, members in cr.clusters.items() if 3 <= len(members) <= 6]
    if not good_ids:
        good_ids = [cid for cid, members in cr.clusters.items() if 2 <= len(members) <= 10]
    if not good_ids:
        st.info("No suitable small cluster found for a worked example.")
        return
    example_cid    = st.selectbox(
        "Pick a cluster for the worked example", good_ids,
        format_func=lambda c: f"Cluster {c} ({len(cr.clusters[c])} faces)",
        key="worked_example_cid",
    )
    member_indices = cr.clusters[example_cid]
    n              = len(member_indices)
    fid_list       = [faces[i].face_id for i in member_indices]
    run_cfg        = result.summary.get("config") or {}
    K              = run_cfg.get("K", 5)
    dist_thresh    = run_cfg.get("distance_threshold", 0.35)
    face_labels    = ", ".join(f"face\\_{fid}" for fid in fid_list)
    st.markdown(f"**Cluster {example_cid}** has **{n} faces**: {face_labels}")
    st.markdown(f"Config: `K={K}`, `distance_threshold={dist_thresh}`")
    crop_cols = st.columns(min(n, 8))
    for i, fid in enumerate(fid_list):
        with crop_cols[i % len(crop_cols)]:
            img = _crop_for_face(fid, result.output_dir)
            if img:
                st.image(img, caption=f"face_{fid:04d}", width=90)
    embs = []
    missing_emb = False
    for idx in member_indices:
        f   = faces[idx]
        emb = f.embedding_normalized if f.embedding_normalized is not None else f.embedding
        if emb is None:
            missing_emb = True
            break
        embs.append(emb / (np.linalg.norm(emb) + 1e-9))
    if missing_emb or len(embs) == 0:
        st.warning("Embeddings not available — cannot compute distances.")
        return
    mat = np.stack(embs).astype(np.float32)
    pw  = cdist(mat, mat, metric="cosine")
    st.markdown("---")
    st.markdown("#### Step 1: Pairwise cosine distances")
    dist_rows = [
        {"Face A": f"face_{fid_list[i]:04d}", "Face B": f"face_{fid_list[j]:04d}",
         "Distance": round(float(pw[i, j]), 4)}
        for i in range(n) for j in range(i + 1, n)
    ]
    st.dataframe(pd.DataFrame(dist_rows), hide_index=True)
    st.markdown("#### Step 2: K nearest neighbors (mutual kNN)")
    k_actual     = min(K, n - 1)
    neighbor_sets = []
    knn_rows      = []
    for i in range(n):
        dists_i = pw[i].copy()
        dists_i[i] = np.inf
        nearest = np.argsort(dists_i)[:k_actual]
        neighbor_sets.append(set(nearest.tolist()))
        nbr_str = ", ".join(f"face_{fid_list[j]:04d} ({pw[i,j]:.4f})" for j in nearest)
        knn_rows.append({"Face": f"face_{fid_list[i]:04d}", f"Top-{k_actual} neighbors": nbr_str})
    st.dataframe(pd.DataFrame(knn_rows), hide_index=True, use_container_width=True)
    st.markdown("#### Step 3: Edge creation (mutual + threshold)")
    edge_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            mutual  = j in neighbor_sets[i] and i in neighbor_sets[j]
            below   = pw[i, j] <= dist_thresh
            created = mutual and below
            edge_rows.append({
                "Face A": f"face_{fid_list[i]:04d}",
                "Face B": f"face_{fid_list[j]:04d}",
                "Distance": round(float(pw[i, j]), 4),
                "Mutual kNN?": "yes" if mutual else "no",
                f"<= {dist_thresh}?": "yes" if below else "no",
                "Edge created": "YES" if created else "no",
            })
    st.dataframe(pd.DataFrame(edge_rows), hide_index=True, use_container_width=True)
    n_created = sum(1 for r in edge_rows if r["Edge created"] == "YES")
    st.markdown(f"**Result: {n_created} edges created** out of {len(edge_rows)} possible pairs.")
    st.markdown("#### Step 4: Connected components -> cluster")
    st.markdown(
        f"All {n} faces reachable through {n_created} edges form **one component** = "
        f"**Cluster {example_cid}**."
    )


def _render_run_config(result: PipelineResult) -> None:
    """Show key thresholds/settings from the run config."""
    cfg = result.summary.get("config") or {}
    if not cfg:
        return
    with st.expander("Run Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Clustering**")
            st.text(f"K:                    {cfg.get('K', '?')}")
            st.text(f"distance_threshold:   {cfg.get('distance_threshold', '?')}")
            st.text(f"min_cluster_size:     {cfg.get('min_cluster_size', '?')}")
        with col2:
            st.markdown("**Quality Gate**")
            st.text(f"blur_min:             {cfg.get('blur_min', '?')}")
            st.text(f"yaw_max:              {cfg.get('yaw_max', '?')}")
            st.text(f"pitch_max:            {cfg.get('pitch_max', '?')}")
            st.text(f"roll_max:             {cfg.get('roll_max', '?')}")
            det = cfg.get('det_score_min')
            st.text(f"det_score_min:        {det if det is not None else 'disabled'}")
            st.text(f"max_faces_per_image:  {cfg.get('max_faces_per_image_core', '?')}")
        with col3:
            st.markdown("**Merge**")
            st.text(f"merge_enabled:        {cfg.get('merge_enabled', '?')}")
            st.text(f"merge_candidate_thr:  {cfg.get('merge_candidate_threshold', '?')}")
            st.text(f"merge_exemplar_thr:   {cfg.get('merge_exemplar_threshold', '?')}")
            st.text(f"merge_support_min:    {cfg.get('merge_support_min', '?')}")
            st.text(f"merge_support_frac:   {cfg.get('merge_support_frac', '?')}")
            st.text(f"merge_margin:         {cfg.get('merge_margin', '?')}")


def render_run_overview_tab():
    st.header("Clusters (Base)")
    _breadcrumb()
    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return
    worker: _AsyncState = st.session_state.overview_worker
    if worker is None:
        w = _AsyncState()
        st.session_state.overview_worker = w
        w.start(RunOverview.compute, result)
        st.rerun()
        return
    if worker.is_running:
        st.info("Computing run overview (UMAP may take ~10s)...")
        time.sleep(0.5)
        st.rerun()
        return
    if worker.has_error:
        st.error(f"Run overview failed: {worker.error}")
        return
    overview = worker.result
    _render_run_config(result)
    st.subheader("Quality Gate Funnel")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images",           overview.n_images)
    c2.metric("Faces detected",   overview.n_faces_detected)
    c3.metric("Core (passed)",    overview.n_core)
    c4.metric("Holdout (failed)", overview.n_holdout)
    pass_rate = 100 * overview.n_core / max(overview.n_faces_detected, 1)
    st.caption(f"Pass rate: {pass_rate:.0f}%")
    st.subheader("Clustering Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters",        overview.n_clusters)
    c2.metric("Noise faces",     overview.n_noise)
    c3.metric("Clustered faces", overview.n_core - overview.n_noise)
    if overview.stage_timings:
        with st.expander("Stage timings"):
            tdf = pd.DataFrame([{"stage": s, "seconds": f"{v:.1f}s" if v else "-"}
                                 for s, v in overview.stage_timings.items()])
            st.dataframe(tdf, hide_index=True)
    st.subheader("All Clusters")
    _render_cluster_gallery(overview, result, tab_key="base_gallery")
    if overview.cluster_rows:
        st.subheader("Cluster Size Distribution")
        st.bar_chart(pd.Series(
            {f"C{r.cluster_id}": r.size for r in overview.cluster_rows}
        ).sort_values(ascending=False))
    st.subheader("UMAP — All Faces")
    if overview.umap_coords is not None:
        import plotly.express as px
        udf = pd.DataFrame({
            "x":       overview.umap_coords[:, 0],
            "y":       overview.umap_coords[:, 1],
            "cluster": [f"C{l}" if l >= 0 else "noise" for l in overview.umap_labels],
            "face_id": overview.umap_face_ids or list(range(len(overview.umap_coords))),
        })
        fig = px.scatter(udf, x="x", y="y", color="cluster",
                         hover_data=["face_id", "cluster"], height=500)
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig)
    else:
        st.info("UMAP not available (embeddings not present in this run).")
    st.subheader("Face Mapping Tables")
    faces_df = _load_faces_df(result.output_dir)
    with st.expander("Face -> Cluster mapping", expanded=False):
        map_df = faces_df[["face_id", "cluster_id", "image_path", "is_core"]].copy()
        map_df["image_file"]   = map_df["image_path"].apply(lambda p: Path(str(p)).name if pd.notna(p) else "")
        map_df["cluster_label"] = map_df["cluster_id"].apply(lambda c: f"C{c}" if c >= 0 else "noise")
        st.dataframe(
            map_df[["face_id", "cluster_label", "cluster_id", "image_file", "is_core"]],
            hide_index=True, use_container_width=True,
            column_config={
                "face_id":       st.column_config.NumberColumn("Face ID", format="%d"),
                "cluster_label": "Cluster",
                "cluster_id":    st.column_config.NumberColumn("Cluster ID", format="%d"),
                "image_file":    "Source Image",
                "is_core":       "Core?",
            },
        )
        st.caption(f"{len(map_df)} faces  --  "
                   f"{(map_df['cluster_id'] >= 0).sum()} clustered  --  "
                   f"{(map_df['cluster_id'] == -1).sum()} noise")
    with st.expander("Face -> Source Image mapping", expanded=False):
        img_df = faces_df[["face_id", "image_path", "crop_path", "cluster_id", "blur_score", "area"]].copy()
        img_df["image_file"] = img_df["image_path"].apply(lambda p: Path(str(p)).name if pd.notna(p) else "")
        img_df["crop_file"]  = img_df["crop_path"].apply(lambda p: Path(str(p)).name if pd.notna(p) else "")
        st.dataframe(
            img_df[["face_id", "image_file", "crop_file", "cluster_id", "blur_score", "area"]],
            hide_index=True, use_container_width=True,
        )
    with st.expander("Cluster -> Faces summary", expanded=False):
        cluster_faces = (
            faces_df[faces_df["cluster_id"] >= 0]
            .groupby("cluster_id")
            .agg(
                n_faces=("face_id", "count"),
                face_ids=("face_id", lambda x: ", ".join(str(i) for i in sorted(x))),
                images=("image_path", lambda x: ", ".join(sorted(set(
                    Path(str(p)).name for p in x if pd.notna(p)
                )))),
            )
            .reset_index()
            .sort_values("n_faces", ascending=False)
        )
        st.dataframe(cluster_faces, hide_index=True, use_container_width=True)
    _render_worked_example(result, overview)
