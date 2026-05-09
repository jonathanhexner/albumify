"""Private helpers for the merge analysis tab: distance matrix, feature computation, crop rendering."""
from __future__ import annotations

import numpy as np
import streamlit as st

from face_cluster.features import FeatureComputer, MergeFeatureContext
from face_cluster.pipeline import PipelineResult

from cache_helpers import _crop_for_face
from face_popup import face_detail_btn


def _get_distance_matrix(result: PipelineResult):
    faces = result.faces
    n     = len(faces)
    if n < 2:
        return None
    dim = next((f.embedding_normalized.shape[0] for f in faces if f.embedding_normalized is not None), None)
    if dim is None:
        return None
    embs = np.zeros((n, dim), dtype=np.float32)
    for i, f in enumerate(faces):
        if f.embedding_normalized is not None:
            embs[i] = f.embedding_normalized
    sims = embs @ embs.T
    return (1.0 - sims).clip(0.0, 2.0)


def _get_merge_candidate_threshold(result: PipelineResult) -> float:
    cfg = result.summary.get("config", {}) if result.summary else {}
    return float(cfg.get("merge_candidate_threshold", 0.45))


def _is_nan(v) -> bool:
    try:
        import math
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _compute_merge_features(result: PipelineResult):
    dm = _get_distance_matrix(result)
    if dm is None:
        return None
    ctx = MergeFeatureContext(cluster_result=result.cluster_result, faces=result.faces, distance_matrix=dm)
    fc  = FeatureComputer()
    return fc.compute_all_pairs(ctx)


def _render_pair_crops(row, result: PipelineResult, symbol: str = "?", key_suffix: str = ""):
    col_a, col_sep, col_b = st.columns([5, 1, 5])
    sfx = f"_{key_suffix}" if key_suffix else ""
    with col_a:
        st.caption(f"Cluster {row.cluster_a}  (size={row.cluster_a_size})")
        if row.exemplar_face_ids_a:
            img_cols = st.columns(len(row.exemplar_face_ids_a))
            for i, fid in enumerate(row.exemplar_face_ids_a):
                with img_cols[i]:
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        st.image(img, caption=f"face_{fid:04d}", width=80)
                    face_detail_btn(fid, key=f"mh_a_{row.cluster_a}_{row.cluster_b}_{i}{sfx}")
    with col_sep:
        st.markdown(f"<div style='text-align:center;font-size:24px;padding-top:32px'>{symbol}</div>",
                    unsafe_allow_html=True)
    with col_b:
        st.caption(f"Cluster {row.cluster_b}  (size={row.cluster_b_size})")
        if row.exemplar_face_ids_b:
            img_cols = st.columns(len(row.exemplar_face_ids_b))
            for i, fid in enumerate(row.exemplar_face_ids_b):
                with img_cols[i]:
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        st.image(img, caption=f"face_{fid:04d}", width=80)
                    face_detail_btn(fid, key=f"mh_b_{row.cluster_a}_{row.cluster_b}_{i}{sfx}")


def _exemplar_face_ids_for_cluster(cid: int, group, result: PipelineResult) -> list:
    for pair in group.pairs:
        if pair.cluster_a == cid:
            return pair.exemplar_face_ids_a[:2]
        if pair.cluster_b == cid:
            return pair.exemplar_face_ids_b[:2]
    return []


def _cluster_size(cid: int, group) -> int:
    for pair in group.pairs:
        if pair.cluster_a == cid:
            return pair.cluster_a_size
        if pair.cluster_b == cid:
            return pair.cluster_b_size
    return 0
