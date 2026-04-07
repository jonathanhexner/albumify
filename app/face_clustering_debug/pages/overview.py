"""Cluster overview page — four sub-tabs matching the old app's capabilities."""

from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

from app.face_clustering_debug.components.algorithm_explanation import render_algorithm_explanation
from app.face_clustering_debug.components.face_detail import (
    render_face_detail,
    render_face_debug_panel,
    _ARCFACE_REF_LANDMARKS_NORMALIZED,
)
from app.face_clustering_debug.components.face_grid import render_face_grid
from app.face_clustering_debug.models.schemas import ClusteringResult, FaceInfo
from app.face_clustering_debug.services.protocols import DataLoaderProtocol

_LANDMARK_COLORS = ["red", "red", "green", "blue", "blue"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_overview_page(loader: DataLoaderProtocol, method: str) -> None:
    st.header(f"📊 Overview — {method}")

    result = loader.load_clustering_result(method)
    if result is None:
        st.error(f"No results found for method **{method}**")
        return

    # Show dynamic algorithm explanation with current params
    render_algorithm_explanation(algorithm=result.algorithm, params=result.params)

    tab_summary, tab_gallery, tab_explorer, tab_icd = st.tabs([
        "📋 Summary",
        "🖼️ Gallery",
        "🔍 Explorer",
        "🌐 Inter-Cluster Distances",
    ])

    with tab_summary:
        _render_summary(result)

    with tab_gallery:
        _render_gallery(result, loader)

    with tab_explorer:
        _render_explorer(result, loader)

    with tab_icd:
        _render_inter_cluster_distances(result)


# ---------------------------------------------------------------------------
# Tab 1 — Summary
# ---------------------------------------------------------------------------

def _render_summary(result: ClusteringResult) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters", result.n_clusters)
    m2.metric("Noise faces", result.n_noise)
    m3.metric("Total faces", len(result.faces))

    rows = [
        {
            "Cluster": c.cluster_id,
            "Faces": len(c.face_indices),
            "Exemplars": len(c.exemplar_indices),
            "Threshold T": round(c.threshold, 3),
            "Raw T": round(c.raw_threshold, 3),
            "Q1": round(c.q1, 3),
            "Q3": round(c.q3, 3),
            "IQR": round(c.iqr, 3),
        }
        for c in result.clusters
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2 — Gallery
# ---------------------------------------------------------------------------

def _render_gallery(result: ClusteringResult, loader: DataLoaderProtocol) -> None:
    st.caption("All clusters sorted by size. Click 🔍 on any face to see details with landmarks and filename.")

    # Build face lookup
    faces_by_idx: Dict[int, FaceInfo] = {f.index: f for f in result.faces}

    clusters_by_id: Dict[int, List[int]] = {
        c.cluster_id: c.face_indices for c in result.clusters
    }
    sorted_ids = sorted(clusters_by_id, key=lambda cid: len(clusters_by_id[cid]), reverse=True)

    # Initialize session state for selected face
    if "gallery_selected_face" not in st.session_state:
        st.session_state.gallery_selected_face = None

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        min_size = st.number_input("Min cluster size", min_value=1, value=2, key="gallery_min")
    with col2:
        n_show = len([cid for cid in sorted_ids if len(clusters_by_id[cid]) >= min_size])
        n_show = st.slider("Clusters to show", 1, max(1, len(sorted_ids)), min(20, n_show),
                           key="gallery_n_show")
    with col3:
        if st.session_state.gallery_selected_face is not None:
            if st.button("Clear selection"):
                st.session_state.gallery_selected_face = None
                st.rerun()

    displayed = 0
    for cid in sorted_ids:
        face_indices = clusters_by_id[cid]
        if len(face_indices) < min_size:
            continue
        if displayed >= n_show:
            break

        cluster = next(c for c in result.clusters if c.cluster_id == cid)
        t_str = f"T={cluster.threshold:.3f}" if cluster.threshold else ""
        label = f"**Cluster {cid}** — {len(face_indices)} faces  {t_str}"

        with st.expander(label, expanded=(displayed < 3)):
            # Get FaceInfo objects for this cluster
            cluster_faces = [faces_by_idx[idx] for idx in face_indices if idx in faces_by_idx]

            # Use face_grid component which shows filename and has 🔍 button
            selected = render_face_grid(
                faces=cluster_faces,
                get_crop_fn=loader.get_face_crop,
                highlight_indices=cluster.exemplar_indices,
                columns=10,
                key_prefix=f"ov_c{cid}_",
            )
            if selected is not None:
                st.session_state.gallery_selected_face = selected
                st.rerun()

        displayed += 1

    # Show face debug panel if a face is selected
    if st.session_state.gallery_selected_face is not None:
        selected_face_idx = st.session_state.gallery_selected_face
        st.divider()

        face = faces_by_idx.get(selected_face_idx)
        if face is None:
            st.warning(f"Could not load face #{selected_face_idx}")
        else:
            # Get all three versions of the face
            aligned_crop = loader.get_face_crop(selected_face_idx)
            raw_crop = loader.get_face_crop_raw(selected_face_idx) if hasattr(loader, 'get_face_crop_raw') else None
            original_with_bbox = loader.get_original_image_with_bbox(selected_face_idx) if hasattr(loader, 'get_original_image_with_bbox') else None

            render_face_debug_panel(
                face=face,
                aligned_crop=aligned_crop,
                raw_crop=raw_crop,
                original_with_bbox=original_with_bbox,
            )


# ---------------------------------------------------------------------------
# Tab 3 — Explorer
# ---------------------------------------------------------------------------

def _render_explorer(result: ClusteringResult, loader: DataLoaderProtocol) -> None:
    st.caption(
        "Detailed view for a selected cluster: faces with landmarks, quality metrics, "
        "intra-cluster distance matrix, and the 10 nearest faces outside the cluster."
    )

    clusters_by_id: Dict[int, List[int]] = {
        c.cluster_id: c.face_indices for c in result.clusters
    }
    sorted_ids = sorted(clusters_by_id, key=lambda cid: len(clusters_by_id[cid]), reverse=True)
    faces_by_idx: Dict[int, FaceInfo] = {f.index: f for f in result.faces}

    if not sorted_ids:
        st.warning("No clusters to explore.")
        return

    cluster_obj_map = {c.cluster_id: c for c in result.clusters}

    selected_cid = st.selectbox(
        "Select cluster",
        sorted_ids,
        format_func=lambda cid: (
            f"Cluster {cid} — {len(clusters_by_id[cid])} faces  "
            f"T={cluster_obj_map[cid].threshold:.3f}"
        ),
        key="explorer_select",
    )

    face_indices = clusters_by_id[selected_cid]
    cluster = cluster_obj_map[selected_cid]
    exemplar_set = set(cluster.exemplar_indices)

    st.markdown(f"### Cluster {selected_cid} — {len(face_indices)} faces  "
                f"T={cluster.threshold:.3f}")

    # --- Faces with landmarks ---
    st.markdown("#### Faces with Landmarks")
    cols_per_row = 10
    for row_start in range(0, len(face_indices), cols_per_row):
        row = face_indices[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, face_idx in enumerate(row):
            with cols[j]:
                crop = loader.get_face_crop(face_idx)
                face = faces_by_idx.get(face_idx)
                if crop and face:
                    # Use ArcFace reference landmarks since crops are 5-point aligned
                    img = _draw_landmarks(Image.open(BytesIO(crop)), _ARCFACE_REF_LANDMARKS_NORMALIZED)
                    caption = f"#{face_idx}" + (" ⭐" if face_idx in exemplar_set else "")
                    st.image(img, caption=caption, use_container_width=True)

    # --- Quality metrics table ---
    st.markdown("#### Face Quality Metrics")
    metrics_rows = []
    for face_idx in face_indices:
        face = faces_by_idx.get(face_idx)
        if face is None:
            continue
        pitch, yaw, roll = face.pose_angles if face.pose_angles else (0.0, 0.0, 0.0)
        metrics_rows.append({
            "Face": face_idx,
            "Confidence": f"{face.confidence:.2f}",
            "Frontal": f"{face.frontal_score:.2f}",
            "Roll": f"{roll:.1f}°",
            "Pitch": f"{pitch:.1f}°",
            "Yaw": f"{yaw:.1f}°",
            "Eye/W": f"{face.eye_bbox_ratio:.3f}",
            "Asymmetry": f"{face.asymmetry_ratio:.2f}",
            "Exemplar": "⭐" if face_idx in exemplar_set else "",
        })
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    # --- Intra-cluster distance matrix ---
    _MAX_HEATMAP = 50
    st.markdown("#### Intra-Cluster Distance Matrix")
    if len(face_indices) > 1:
        hm_indices = face_indices[:_MAX_HEATMAP]
        if len(face_indices) > _MAX_HEATMAP:
            st.caption(f"Showing first {_MAX_HEATMAP} of {len(face_indices)} faces.")
        embs = result.embeddings[hm_indices]
        dist_mat = cdist(embs, embs, metric="cosine")
        fig = _heatmap(dist_mat, [f"#{i}" for i in hm_indices],
                       title="Cosine Distance Matrix")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Only 1 face — no distance matrix.")

    # --- 10 nearest external faces ---
    st.markdown("#### 10 Nearest Faces Outside This Cluster")
    nearest = _find_nearest_external(face_indices, result.embeddings, result.labels, k=10)
    if nearest:
        cols = st.columns(10)
        for i, (ext_idx, dist, ext_cluster) in enumerate(nearest[:10]):
            with cols[i]:
                crop = loader.get_face_crop(int(ext_idx))
                label = f"#{ext_idx}\nd={dist:.3f}\nC{ext_cluster}"
                if crop:
                    st.image(crop, caption=label, use_container_width=True)
                else:
                    st.caption(label)
    else:
        st.info("No external faces found.")


# ---------------------------------------------------------------------------
# Tab 4 — Inter-Cluster Distances
# ---------------------------------------------------------------------------

def _render_inter_cluster_distances(result: ClusteringResult) -> None:
    st.caption(
        "Minimum embedding distance between each pair of clusters (exemplar-to-exemplar). "
        "**Gap to T** = min_distance − merge_threshold; negative = already within threshold."
    )

    cluster_ids = [c.cluster_id for c in result.clusters]
    if len(cluster_ids) < 2:
        st.info("Need at least 2 clusters.")
        return

    exemplars_map = {c.cluster_id: c.exemplar_indices for c in result.clusters}
    thresholds_map = {c.cluster_id: c.threshold for c in result.clusters}

    # Only show heatmap for ≤30 clusters (otherwise it's unreadable)
    if len(cluster_ids) <= 30:
        fig, dist_mat = _inter_cluster_heatmap(cluster_ids, exemplars_map, result.embeddings)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"{len(cluster_ids)} clusters — heatmap hidden (too large). See table below.")
        n = len(cluster_ids)
        dist_mat = np.zeros((n, n))
        for i, c1 in enumerate(cluster_ids):
            for j, c2 in enumerate(cluster_ids):
                if i < j and exemplars_map.get(c1) and exemplars_map.get(c2):
                    emb_a = result.embeddings[exemplars_map[c1]]
                    emb_b = result.embeddings[exemplars_map[c2]]
                    d = float(np.min(cdist(emb_a, emb_b, metric="cosine")))
                    dist_mat[i, j] = dist_mat[j, i] = d

    # Distance vs threshold table
    rows = []
    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            if i >= j:
                continue
            min_dist = dist_mat[i, j]
            t1 = thresholds_map.get(c1, 0.0)
            t2 = thresholds_map.get(c2, 0.0)
            merge_t = min(t1, t2) if t1 > 0 and t2 > 0 else 0.0
            gap = round(min_dist - merge_t, 3) if merge_t > 0 else None
            rows.append({
                "Cluster A": c1,
                "Cluster B": c2,
                "Min Dist": round(min_dist, 3),
                "T_A": round(t1, 3),
                "T_B": round(t2, 3),
                "Merge T": round(merge_t, 3),
                "Within T?": "✅" if (gap is not None and gap <= 0) else "❌",
                "Gap to T": gap,  # None when no threshold; pandas sorts NaN last
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Gap to T", ascending=True, na_position="last")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_landmarks(img: Image.Image, landmarks) -> Image.Image:
    if not landmarks or len(landmarks) != 5:
        return img
    result = img.copy()
    draw = ImageDraw.Draw(result)
    w, h = result.size
    r = max(2, int(min(w, h) * 0.025))
    for (x, y), color in zip(landmarks, _LANDMARK_COLORS):
        px, py = x * w, y * h
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white")
    return result


def _heatmap(matrix: np.ndarray, labels: List[str], title: str = ""):
    n = len(labels)
    side = min(30, max(6, n * 0.4 + 2))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Distance")
    plt.tight_layout()
    return fig


def _inter_cluster_heatmap(cluster_ids, exemplars_map, embeddings):
    n = len(cluster_ids)
    mat = np.zeros((n, n))
    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            if i < j and exemplars_map.get(c1) and exemplars_map.get(c2):
                emb_a = embeddings[exemplars_map[c1]]
                emb_b = embeddings[exemplars_map[c2]]
                d = float(np.min(cdist(emb_a, emb_b, metric="cosine")))
                mat[i, j] = mat[j, i] = d

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45 + 2), max(6, n * 0.45 + 1)))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], rotation=90, fontsize=8)
    ax.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=8)
    ax.set_title("Inter-Cluster Min Distance (Exemplar-to-Exemplar)")
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] > 0:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=6)
    plt.colorbar(im, ax=ax, label="Min Distance")
    plt.tight_layout()
    return fig, mat


def _find_nearest_external(cluster_indices, embeddings, labels, k=10):
    if len(cluster_indices) == 0:
        return []
    cluster_label = labels[cluster_indices[0]]
    external_mask = labels != cluster_label
    if not np.any(external_mask):
        return []
    ext_indices = np.where(external_mask)[0]
    emb_cluster = embeddings[cluster_indices]
    emb_external = embeddings[ext_indices]
    dists = cdist(emb_cluster, emb_external, metric="cosine")
    min_dists = np.min(dists, axis=0)
    top_k = np.argsort(min_dists)[:k]
    return [(int(ext_indices[i]), float(min_dists[i]), int(labels[ext_indices[i]])) for i in top_k]
