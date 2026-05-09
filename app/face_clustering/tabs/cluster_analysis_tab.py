"""Tab 5: Cluster Analysis — per-cluster drill-down."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from face_cluster import PipelineConfig, apply_manual_merges, save_manual_merge_snapshot
from face_cluster.analysis_views import ClusterView, ClusterDebugView
from face_cluster.features import FeatureComputer, MergeFeatureContext
from face_cluster.loader import load_pipeline_result
from face_cluster.pipeline import PipelineResult

from state import _AsyncState, _invalidate_run_caches
from nav_helpers import _breadcrumb, _no_result
from cache_helpers import _crop_for_face
from quality_panels import _render_cluster_provenance
from face_popup import face_detail_btn


def _render_cluster_debug(dbg: ClusterDebugView, result: PipelineResult):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Edges",      f"{dbg.n_edges} / {dbg.max_possible_edges}")
    c2.metric("Edge density", f"{dbg.edge_density:.1%}")
    c3.metric("Chain score",  f"{dbg.chain_score:.2f}",
              help="diameter / (2 * median_dist). >1.5 suggests chain structure.")
    c4.metric("Bridge faces", len(dbg.bridge_face_ids))
    if dbg.chain_score > 1.5:
        st.warning(
            f"**Chain structure detected** (score {dbg.chain_score:.2f}). "
            f"Diameter ({dbg.diameter:.3f}) much larger than median distance ({dbg.median_dist:.3f})."
        )
    elif dbg.edge_density < 0.15 and dbg.n_faces > 4:
        st.warning(f"**Sparse graph** (density {dbg.edge_density:.1%}).")
    if dbg.bridge_face_ids:
        st.markdown(
            f"**Bridge faces**: `{', '.join(f'face_{fid:04d}' for fid in dbg.bridge_face_ids)}`"
        )
        cols = st.columns(min(len(dbg.bridge_face_ids), 8))
        for i, fid in enumerate(dbg.bridge_face_ids[:8]):
            with cols[i]:
                img = _crop_for_face(fid, result.output_dir)
                if img:
                    st.image(img, caption=f"face_{fid:04d}")
    if dbg.distance_matrix is not None and len(dbg.distance_matrix) > 1:
        with st.expander("Pairwise distance heatmap", expanded=dbg.n_faces <= 30):
            import plotly.express as px
            labels = [f"f{fid}" for fid in dbg.face_ids_order]
            fig    = px.imshow(
                dbg.distance_matrix, x=labels, y=labels,
                color_continuous_scale="RdYlGn_r",
                zmin=0.0, zmax=min(0.6, float(dbg.distance_matrix.max()) + 0.05),
                labels=dict(color="cosine dist"), aspect="equal",
            )
            fig.update_layout(height=max(300, 18 * dbg.n_faces + 100))
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("Per-face graph connectivity"):
        rows = [{"face_id": f"face_{fg.face_id:04d}", "edges": fg.n_edges,
                 "bridge": "YES" if fg.is_bridge else "",
                 "neighbors (dist)": ", ".join(f"f{n}({d:.3f})" for n, d in zip(fg.neighbors, fg.neighbor_dists))}
                for fg in dbg.face_graph]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander("All edges"):
        if dbg.edges:
            edf = pd.DataFrame([{"face_a": f"face_{e.face_id_a:04d}",
                                   "face_b": f"face_{e.face_id_b:04d}",
                                   "distance": e.distance} for e in dbg.edges])
            st.dataframe(edf.sort_values("distance"), hide_index=True)
        else:
            st.info("No edges in this cluster.")


def _get_distance_matrix(result: PipelineResult):
    import numpy as np
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


def _compute_force_merge_preview(result: PipelineResult, cid_a: int, cid_b: int):
    dm = _get_distance_matrix(result)
    if dm is None:
        return None
    cr = result.cluster_result
    if cid_a not in cr.clusters or cid_b not in cr.clusters:
        return None
    ctx       = MergeFeatureContext(cluster_result=cr, faces=result.faces, distance_matrix=dm)
    fc        = FeatureComputer()
    t_global  = fc._compute_t_global(cr, dm)
    feat      = fc.compute_pair_features(cid_a, cid_b, ctx, t_global)
    threshold = _get_merge_candidate_threshold(result)
    ex_a      = cr.exemplars.get(cid_a, cr.clusters[cid_a])
    ex_b      = cr.exemplars.get(cid_b, cr.clusters[cid_b])
    exemplar_dist  = float(dm[ex_a[0], ex_b[0]]) if ex_a and ex_b else 1.0
    passes_exemplar = exemplar_dist <= threshold
    passes_support  = (feat.n_cross_pairs_below_threshold or 0) >= 1
    passes_diameter = (feat.post_merge_diameter or 0) <= (max(feat.diameter_a or 0, feat.diameter_b or 0) * 1.5 + 0.05)
    return {
        "feat": feat,
        "exemplar_dist": exemplar_dist,
        "threshold":        threshold,
        "is_candidate":     exemplar_dist <= threshold,
        "passes_exemplar":  passes_exemplar,
        "passes_support":   passes_support,
        "passes_diameter":  passes_diameter,
        "n_gates_passed":   sum([passes_exemplar, passes_support, passes_diameter]),
        "post_diameter":    feat.post_merge_diameter or 0.0,
        "support":          feat.n_cross_pairs_below_threshold or 0,
        "cluster_a_size":   len(cr.clusters[cid_a]),
        "cluster_b_size":   len(cr.clusters[cid_b]),
        "exemplar_face_ids_a": [result.faces[i].face_id for i in ex_a[:3]],
        "exemplar_face_ids_b": [result.faces[i].face_id for i in ex_b[:3]],
    }


def _render_force_merge_widget(result: PipelineResult):
    cr          = result.cluster_result
    cluster_ids = sorted(cr.clusters.keys())
    if len(cluster_ids) < 2:
        return
    with st.expander("Force Merge", expanded=False):
        st.caption("Select any two clusters to merge, even if they are not merge candidates.")
        col_a, col_b = st.columns(2)
        fm_a = col_a.selectbox("Cluster A", cluster_ids,
                               format_func=lambda c: f"C{c} ({len(cr.clusters[c])} faces)",
                               key="fm_cluster_a")
        fm_b = col_b.selectbox("Cluster B", [c for c in cluster_ids if c != fm_a],
                               format_func=lambda c: f"C{c} ({len(cr.clusters[c])} faces)",
                               key="fm_cluster_b")
        if st.button("Preview Merge", key="fm_preview_btn"):
            st.session_state.fm_preview = _compute_force_merge_preview(result, fm_a, fm_b)
            st.session_state.fm_pair    = (fm_a, fm_b)
        preview = st.session_state.get("fm_preview")
        pair    = st.session_state.get("fm_pair")
        if preview is None or pair != (fm_a, fm_b):
            return
        if preview is False:
            st.warning("Embeddings not available — cannot compute preview.")
            return
        cid_a, cid_b = pair
        dist         = preview["exemplar_dist"]
        threshold    = preview["threshold"]
        is_cand      = preview["is_candidate"]
        (st.info if is_cand else st.warning)(
            f"{'Merge candidate' if is_cand else 'Not a merge candidate'} — "
            f"exemplar dist `{dist:.3f}` {'<=' if is_cand else '>'} threshold `{threshold:.3f}`"
        )
        col_crops_a, col_sep, col_crops_b = st.columns([5, 1, 5])
        with col_crops_a:
            st.caption(f"Cluster {cid_a}  ({preview['cluster_a_size']} faces)")
            if preview["exemplar_face_ids_a"]:
                cc = st.columns(len(preview["exemplar_face_ids_a"]))
                for i, fid in enumerate(preview["exemplar_face_ids_a"]):
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        cc[i].image(img, caption=f"face_{fid:04d}", width=80)
        with col_sep:
            st.markdown("<div style='text-align:center;font-size:24px;padding-top:32px'>+</div>",
                        unsafe_allow_html=True)
        with col_crops_b:
            st.caption(f"Cluster {cid_b}  ({preview['cluster_b_size']} faces)")
            if preview["exemplar_face_ids_b"]:
                cc = st.columns(len(preview["exemplar_face_ids_b"]))
                for i, fid in enumerate(preview["exemplar_face_ids_b"]):
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        cc[i].image(img, caption=f"face_{fid:04d}", width=80)
        gate_cols = st.columns(3)
        for col, (name, passed, val) in zip(gate_cols, [
            ("Exemplar dist", preview["passes_exemplar"], f"{dist:.3f} / {threshold:.3f}"),
            ("Support pairs",  preview["passes_support"],  str(preview["support"])),
            ("Post diameter",  preview["passes_diameter"], f"{preview['post_diameter']:.3f}"),
        ]):
            color = "#4daa6e" if passed else "#cc6666"
            badge = "PASS" if passed else "FAIL"
            col.markdown(
                f"<div style='background:#1a1a2e;border-radius:4px;padding:4px 8px;"
                f"border-left:3px solid {color};font-size:12px'>"
                f"<span style='color:{color};font-weight:bold'>{badge}</span> "
                f"<span style='color:#aaa'>{name}</span><br/>"
                f"<span style='color:#ccc;font-size:11px'>{val}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("")
        if st.button(f"Confirm: Merge C{cid_a} + C{cid_b}", type="primary", key="fm_confirm_btn"):
            current_cr = result.merged_cluster_result or result.cluster_result
            rnd        = st.session_state.merge_round
            parent_dir = Path(result.output_dir).parent
            snap_dir   = parent_dir / f"merge_snap_{rnd}"
            save_manual_merge_snapshot(
                faces=result.faces, merged_cluster_result=current_cr,
                approved_pairs=[(cid_a, cid_b)], rejected_pairs=[],
                config=PipelineConfig(), output_dir=snap_dir,
                parent_output_dir=result.output_dir,
                parent_run_id=result.summary.get("run_id"), merge_round=rnd,
            )
            st.session_state.merge_round      = rnd + 1
            st.session_state.pipeline_result  = load_pipeline_result(snap_dir)
            st.session_state.fm_preview       = None
            st.session_state.cluster_worker   = None
            st.session_state.cluster_debug_worker = None
            _invalidate_run_caches()
            st.success(f"Merged C{cid_a} + C{cid_b} -> saved as `{snap_dir.name}`")
            st.rerun()


def render_cluster_analysis_tab():
    st.header("Cluster Analysis")
    _breadcrumb()
    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return
    cr          = result.cluster_result
    cluster_ids = sorted(cr.clusters.keys())
    if not cluster_ids:
        st.warning("No clusters found.")
        return
    default_idx = (cluster_ids.index(st.session_state.selected_cluster)
                   if st.session_state.selected_cluster in cluster_ids else 0)
    selected    = st.selectbox(
        "Cluster", cluster_ids, index=default_idx,
        format_func=lambda c: f"Cluster {c}  ({len(cr.clusters[c])} faces)",
    )
    if selected != st.session_state.selected_cluster:
        st.session_state.selected_cluster     = selected
        st.session_state.cluster_worker       = None
        st.session_state.cluster_debug_worker = None
    worker: _AsyncState = st.session_state.cluster_worker
    if worker is None:
        w = _AsyncState()
        st.session_state.cluster_worker = w
        w.start(ClusterView.compute, result, selected)
        st.rerun()
        return
    if worker.is_running:
        st.info(f"Analysing cluster {selected}...")
        time.sleep(0.4)
        st.rerun()
        return
    if worker.has_error:
        st.error(f"Cluster analysis failed: {worker.error}")
        return
    view = worker.result
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Faces",          view.size)
    c2.metric("Diameter",       f"{view.diameter:.3f}")
    c3.metric("Avg intra-dist", f"{view.avg_intra_dist:.3f}")
    c4.metric("Exemplars",      len(view.exemplar_face_ids))
    c5.metric("Outliers",       len(view.outlier_face_ids))
    # Provenance comes from the in-memory cluster_stats (populated by the loader
    # from DB.clusters.origin / parent_ids in v4 runs, or from clusters.csv on
    # legacy runs).  No direct file read here.
    final_cr = result.merged_cluster_result or result.cluster_result
    stats = (final_cr.cluster_stats or {}).get(selected) if final_cr else None
    if stats and stats.get("origin"):
        provenance_row = pd.Series({
            "cluster_id": selected,
            "origin": stats.get("origin"),
            "parent_cluster_ids": json.dumps(stats.get("parent_ids", [])),
        })
        _render_cluster_provenance(provenance_row, result.merge_log)
    if view.split_signal:
        st.warning("Split signal — bimodal distance distribution. This cluster may contain two people.")
    st.subheader("Exemplars (d10 — most central)")
    if view.exemplar_face_ids:
        n_total_ex = len(view.exemplar_face_ids)
        show_all   = (st.checkbox(f"Show all {n_total_ex} exemplars", key=f"show_all_ex_ca_{id(view)}")
                      if n_total_ex > 5 else False)
        n_show = n_total_ex if show_all else min(n_total_ex, 5)
        for row_start in range(0, n_show, 5):
            row_fids = view.exemplar_face_ids[row_start:row_start + 5]
            cols     = st.columns(len(row_fids))
            for idx, (col, fid) in enumerate(zip(cols, row_fids)):
                with col:
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        st.image(img, caption=f"face_{fid:04d}")
                    else:
                        st.caption(f"face_{fid:04d}")
                    if row_start == 0 and idx == 0:
                        st.caption("TOP EXEMPLAR")
                    face_detail_btn(fid, key=f"ca_ex_{selected}_{row_start}_{idx}")
    st.subheader("All Faces in Cluster")
    _GRID_COLS   = 8
    sorted_faces = sorted(
        view.faces,
        key=lambda fr: (0 if fr.face_id in view.exemplar_face_ids else 1,
                        fr.dist_to_exemplar if fr.dist_to_exemplar is not None else 9.0),
    )
    for row_start in range(0, len(sorted_faces), _GRID_COLS):
        row_faces = sorted_faces[row_start:row_start + _GRID_COLS]
        cols      = st.columns(_GRID_COLS)
        for col_i, (col, fr) in enumerate(zip(cols, row_faces)):
            with col:
                img = _crop_for_face(fr.face_id, result.output_dir)
                st.image(img) if img else st.markdown("_(no crop)_")
                is_ex    = fr.face_id in view.exemplar_face_ids
                tag      = "EX " if is_ex else ("!" if fr.is_outlier else "")
                dist_str = f"{fr.dist_to_exemplar:.3f}" if fr.dist_to_exemplar is not None else "-"
                st.caption(f"{tag}face_{fr.face_id:04d}\nd={dist_str}")
                face_detail_btn(fr.face_id, key=f"ca_af_{selected}_{row_start}_{col_i}")
    with st.expander("Face detail table"):
        fdf = pd.DataFrame([{
            "face_id":       fr.face_id,
            "image":         Path(fr.image_path).name if fr.image_path else "",
            "blur":          round(fr.blur_score, 1),
            "area":          int(fr.area),
            "dist_exemplar": fr.dist_to_exemplar,
            "dist_centroid": fr.dist_to_centroid,
            "role":          fr.role,
            "outlier":       "YES" if fr.is_outlier else "",
        } for fr in view.faces])
        st.dataframe(fdf, hide_index=True)
    sel_face = st.selectbox("Select face for Face Analysis", [fr.face_id for fr in view.faces],
                            format_func=lambda f: f"face_{f:04d}")
    if st.button("Open Face Analysis", type="primary"):
        st.session_state.selected_face = sel_face
        st.session_state.face_worker   = None
        st.info(f"face_{sel_face:04d} selected. Switch to **Face Analysis**.")
    st.subheader("Nearest Clusters")
    if view.nearest_clusters:
        for nc in view.nearest_clusters:
            badge = " -- MERGE CANDIDATE" if nc.merge_candidate else ""
            st.markdown(
                f"**Cluster {nc.cluster_id}** ({nc.size} faces){badge}  "
                f"&nbsp;&nbsp; min_dist `{nc.min_exemplar_dist:.3f}` "
                f"| p10 `{nc.p10_cross_dist:.3f}` | threshold `{nc.merge_threshold:.3f}`"
            )
            ex_indices = cr.exemplars.get(nc.cluster_id, [])
            ex_fids    = [result.faces[i].face_id for i in ex_indices[:4]]
            thumb_cols = st.columns([1, 1, 1, 1, 2])
            for col_i, fid in enumerate(ex_fids):
                with thumb_cols[col_i]:
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        st.image(img, width=70)
                    st.caption(f"face_{fid:04d}")
                    face_detail_btn(fid, key=f"ca_nc_{selected}_{nc.cluster_id}_{col_i}")
            with thumb_cols[4]:
                if st.button(f"Go to C{nc.cluster_id}", key=f"goto_nc_{nc.cluster_id}",
                             use_container_width=True):
                    st.session_state.selected_cluster     = nc.cluster_id
                    st.session_state.cluster_worker       = None
                    st.session_state.cluster_debug_worker = None
                    st.info(f"Cluster {nc.cluster_id} selected. Switch to **Cluster Analysis**.")
            st.divider()
    _render_force_merge_widget(result)
    st.subheader("Graph Debug")
    dbg_worker: _AsyncState = st.session_state.cluster_debug_worker
    if dbg_worker is None:
        w = _AsyncState()
        st.session_state.cluster_debug_worker = w
        w.start(ClusterDebugView.compute, result, selected)
        st.rerun()
        return
    if dbg_worker.is_running:
        st.info("Computing graph diagnostics...")
        time.sleep(0.4)
        st.rerun()
        return
    if dbg_worker.has_error:
        st.error(f"Debug diagnostics failed: {dbg_worker.error}")
    else:
        _render_cluster_debug(dbg_worker.result, result)
