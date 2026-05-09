"""Private panel functions for heuristic merge decision rendering and approval controls."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from face_cluster import FaceClusteringPipeline, PipelineConfig, save_manual_merge_snapshot
from face_cluster.analysis_views import MergeAnalysisView
from face_cluster.export import save_merge_decisions, save_merge_features
from face_cluster.features import FeatureComputer
from face_cluster.pipeline import PipelineResult
from face_cluster.run_naming import RunDirSpec, allocate_run_dir
from face_cluster.training_db import upsert_training_samples

from state import _AsyncState
from session_helpers import (
    _render_pending_labels_indicator, _save_pending_labels_to_session,
    _collect_all_merge_labels, _materialize_merge_step,
)
from constants import (
    _GALLERY_FILTERS, _GALLERY_SORTS, _GALLERY_PAGE_SIZE,
    _GROUP_FILTERS, _GROUP_PAGE_SIZE, _CONF_COLOR, _CONF_LABEL,
)
from _merge_helpers import (
    _is_nan, _render_pair_crops, _exemplar_face_ids_for_cluster, _cluster_size,
)
from face_cluster.analysis_views import compute_pair_feature_contributions


# ---------------------------------------------------------------------------
# Merge summary panels
# ---------------------------------------------------------------------------

def _render_criteria_reference():
    doc_path = Path(__file__).parents[3] / "docs" / "merge_criteria_reference.md"
    with st.expander("Merge Criteria Reference", expanded=False):
        if doc_path.exists():
            st.markdown(doc_path.read_text(encoding="utf-8"))
        else:
            st.warning(f"Reference doc not found at {doc_path}")


def _render_merge_run_provenance(result: PipelineResult) -> None:
    summary      = result.summary or {}
    mode         = summary.get("mode", "")
    source_type  = summary.get("source_type", "")
    run_dir_name = Path(result.output_dir).name
    if mode == "remerge":
        parent      = summary.get("source_run", "")
        parent_name = Path(parent).name if parent else "unknown"
        n_clusters  = summary.get("n_clusters_merged") or summary.get("n_clusters", "?")
        st.info(f"**Round {st.session_state.merge_round - 1} remerge**: `{run_dir_name}` "
                f"(built from `{parent_name}`).  Showing fresh candidates for {n_clusters} clusters.")
    elif source_type == "manual_merge":
        n_manual = summary.get("n_manual_merges", "?")
        st.info(f"**Snapshot**: `{run_dir_name}` — {n_manual} manual merge(s) applied.")
    else:
        st.caption(f"Run: **{run_dir_name}**")


def _render_merge_summary(view: MergeAnalysisView):
    st.subheader("Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Clusters (base)",      view.n_clusters_base)
    c2.metric("Clusters (merged)",    view.n_clusters_merged,
              delta=view.n_clusters_merged - view.n_clusters_base)
    c3.metric("Actual merges",        len(view.merges))
    c4.metric("Iterations run",       view.n_iterations)
    c5.metric("Rejected pairs",       len(view.rejections))
    c6.metric("Near misses (3/4)",    len(view.near_misses))


def _render_gate_bottleneck(view: MergeAnalysisView):
    if not view.rejections:
        return
    st.subheader("Gate Bottleneck Analysis")
    gate_labels = {"exemplar": "Exemplar dist", "support": "Support count",
                   "margin":   "Margin check",  "diameter": "Post-merge diameter"}
    rows = [
        {
            "Gate":             gate_labels[gate],
            "Total rejections": view.gate_rejection_counts.get(gate, 0),
            "Sole blocker":     view.gate_sole_blocker_counts.get(gate, 0),
        }
        for gate in gate_labels
    ]
    df       = pd.DataFrame(rows).sort_values("Total rejections", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)
    dominant = df.iloc[0]["Gate"] if not df.empty else ""
    if dominant:
        st.caption(f"**{dominant}** is the dominant bottleneck.")


def _render_iteration_timeline(view: MergeAnalysisView) -> None:
    """Render the per-iteration strip showing what was merged at each step."""
    if not view.iter_timeline:
        return
    st.subheader("Iteration Timeline")
    st.caption(
        "Each iteration evaluates all candidates, picks the single best merge, executes it, "
        "then re-evaluates from scratch.  Click to jump to that iteration in the pair list."
    )
    cols = st.columns(len(view.iter_timeline))
    for col, entry in zip(cols, view.iter_timeline):
        it = entry["iteration"]
        ma, mb = entry.get("merged_a"), entry.get("merged_b")
        nc = entry.get("n_candidates", "?")
        sa, sb = entry.get("size_a"), entry.get("size_b")
        with col:
            if ma is not None:
                label = f"C{ma}+C{mb}"
                sub   = f"sz {sa}+{sb}"
                color = "#4daa6e"
            else:
                label = "No merge"
                sub   = "stop"
                color = "#cc6666"
            st.markdown(
                f"<div style='background:#1a1a2e;border-radius:5px;padding:8px;text-align:center;"
                f"border-top:3px solid {color};cursor:pointer' "
                f"title='Iteration {it}: {nc} candidates'>"
                f"<div style='font-size:10px;color:#888'>Iter {it}</div>"
                f"<div style='font-size:13px;font-weight:700;color:{color}'>{label}</div>"
                f"<div style='font-size:10px;color:#888'>{sub} &middot; {nc} cands</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Filter", key=f"iter_btn_{it}", use_container_width=True):
                st.session_state.merge_iter_filter = str(it)
                st.session_state.merge_gallery_page = 0
                st.rerun()


def _render_pair_iter_history(
    pair_key: tuple,
    view: MergeAnalysisView,
) -> None:
    """Render the per-iteration evaluation history table inside a pair expander."""
    history = view.pair_history.get(pair_key)
    if not history or len(history) <= 1:
        return
    rows = [
        {
            "Iter":      r.iteration,
            "C_a size":  r.cluster_a_size,
            "C_b size":  r.cluster_b_size,
            "Exmp dist": round(r.exemplar_dist, 4),
            "Threshold": round(r.threshold_used, 3),
            "Delta":     round(r.exemplar_dist - r.threshold_used, 4),
            "Result":    "reject",
        }
        for r in history
    ]
    df = pd.DataFrame(rows)
    with st.expander(f"Evaluation history ({len(history)} iterations)", expanded=False):
        st.dataframe(df, hide_index=True, use_container_width=True)
        dist_values = [r.exemplar_dist for r in history]
        if len(set(round(d, 4) for d in dist_values)) == 1:
            st.caption(
                ":orange[Exemplar distance unchanged across all iterations] — "
                "no bridge node was discovered after each merge.  "
                "Consider inspecting these clusters visually or raising the threshold."
            )
        elif dist_values[-1] < dist_values[0]:
            diff = dist_values[0] - dist_values[-1]
            st.caption(f"Exemplar distance improved by {diff:.4f} across iterations.")


def _render_threshold_distribution(view: MergeAnalysisView, result: PipelineResult):
    pass  # Adaptive per-cluster thresholds removed; stub retained for API stability.


def _render_absorbed_clusters(view: MergeAnalysisView):
    if not view.cluster_mapping:
        return
    st.subheader("Absorbed Clusters")
    st.caption("Merged clusters that absorbed 2+ base clusters.")
    rows = [{"merged_id": mid, "absorbed_base_clusters": str(bids), "n_absorbed": len(bids)}
            for mid, bids in sorted(view.cluster_mapping.items())]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------------------------------------------------------------------
# Approval controls
# ---------------------------------------------------------------------------

def _save_merge_features_if_available(decisions, result, run_id, timestamp, sources=None) -> None:
    from face_cluster.features import VERSION as FEATURE_VERSION
    pair_features = st.session_state.merge_pair_features
    if not pair_features:
        return
    fc  = FeatureComputer()
    df  = fc.to_dataframe(pair_features)
    label_map = {key: (1 if v == "approve" else 0) for key, v in decisions.items()}
    df["label"] = df.apply(
        lambda r: label_map.get((int(r.cluster_a), int(r.cluster_b))), axis=1
    )
    human_label_map = {
        key: label_map[key] for key in label_map
        if (sources or {}).get(key) == "human"
    }
    df["run_id"]          = run_id
    df["timestamp"]       = timestamp
    df["feature_version"] = FEATURE_VERSION
    save_merge_features(df, result.output_dir)
    album_path = None
    run_json_path = result.output_dir / "pipeline_run.json"
    if run_json_path.exists():
        with open(run_json_path, encoding="utf-8") as fh:
            album_path = json.load(fh).get("source_album")
    exemplars  = result.cluster_result.exemplars
    feat_cols  = [c for c in df.columns
                  if c not in ("cluster_a", "cluster_b", "label", "run_id", "timestamp", "feature_version")]
    samples = []
    for _, row in df.iterrows():
        cid_a, cid_b = int(row["cluster_a"]), int(row["cluster_b"])
        pair_key     = (min(cid_a, cid_b), max(cid_a, cid_b))
        human_label  = human_label_map.get(pair_key)
        if human_label is None:
            continue
        feat_dict = {c: (None if _is_nan(row[c]) else row[c]) for c in feat_cols}
        samples.append({
            "run_id": run_id, "album_path": album_path,
            "cluster_a": cid_a, "cluster_b": cid_b,
            "label": human_label, "feature_version": int(FEATURE_VERSION),
            "features_json": json.dumps(feat_dict),
            "output_dir": str(result.output_dir),
            "exemplar_ids": json.dumps({"a": exemplars.get(cid_a, [])[:5], "b": exemplars.get(cid_b, [])[:5]}),
            "saved_at": timestamp,
        })
    if samples:
        n = upsert_training_samples(samples)
        st.toast(f"Saved {n} human-labeled rows to training DB")
    else:
        st.toast("No human-labeled pairs to save to training DB")


def _save_approval_decisions(view: MergeAnalysisView, result: PipelineResult) -> None:
    decisions = st.session_state.merge_approval_decisions
    all_rows  = {
        (min(r.cluster_a, r.cluster_b), max(r.cluster_a, r.cluster_b)): r
        for r in (view.merges + view.rejections)
    }
    run_id    = Path(result.output_dir).name
    timestamp = datetime.now().isoformat(timespec="seconds")
    entries   = []
    for key, decision in decisions.items():
        row = all_rows.get(key)
        entries.append({
            "cluster_a":       key[0], "cluster_b": key[1],
            "decision":        decision,
            "n_gates_passed":  row.n_gates_passed if row else None,
            "exemplar_dist":   row.exemplar_dist  if row else None,
            "threshold_used":  row.threshold_used if row else None,
            "support":         row.support        if row else None,
            "required_support": row.required_support if row else None,
            "margin_gap":      row.margin_gap     if row else None,
            "post_diameter":   row.post_diameter  if row else None,
            "run_id": run_id, "timestamp": timestamp,
        })
    save_merge_decisions(entries, result.output_dir)
    result.merge_decisions = entries
    st.toast(f"Saved {len(entries)} decisions to {result.output_dir.name}/merge_decisions.json")
    _save_merge_features_if_available(
        decisions, result, run_id, timestamp,
        sources=st.session_state.get("merge_decision_sources"),
    )


def _render_approval_controls(view: MergeAnalysisView, result: PipelineResult):
    all_rows = view.merges + view.rejections
    if not all_rows:
        return
    st.subheader("Merge Approval")
    decisions: dict = st.session_state.merge_approval_decisions
    total       = len(all_rows)
    n_approved  = sum(1 for v in decisions.values() if v == "approve")
    n_rejected  = sum(1 for v in decisions.values() if v == "reject")
    n_undecided = total - n_approved - n_rejected
    tally_c1, tally_c2, tally_c3 = st.columns(3)
    tally_c1.metric("Approved",  n_approved)
    tally_c2.metric("Rejected",  n_rejected)
    tally_c3.metric("Undecided", n_undecided)
    if view.merge_groups:
        n_ap_pairs = sum(len(g.pairs) for g in view.merge_groups if g.confidence == "auto_approve")
        n_rv_pairs = sum(len(g.pairs) for g in view.merge_groups if g.confidence == "review")
        n_ar_pairs = sum(len(g.pairs) for g in view.merge_groups if g.confidence == "auto_reject")
        st.caption(
            f"**{len(view.merge_groups)} groups**: "
            f"{view.n_auto_approve} auto-approve ({n_ap_pairs} pairs) &nbsp;|&nbsp; "
            f"{view.n_review} review ({n_rv_pairs} pairs) &nbsp;|&nbsp; "
            f"{view.n_auto_reject} auto-reject ({n_ar_pairs} pairs)"
        )
    col_accept, col_smart_ap, col_smart_rj, col_reset, _ = st.columns([1, 1, 1, 1, 2])
    with col_accept:
        if st.button("Accept all heuristic", key="bulk_accept_heuristic"):
            new_decisions, new_sources = {}, {}
            for row in all_rows:
                key = (min(row.cluster_a, row.cluster_b), max(row.cluster_a, row.cluster_b))
                new_decisions[key] = "approve" if row.action == "merged" else "reject"
                new_sources[key]   = "human"
            st.session_state.merge_approval_decisions = new_decisions
            st.session_state.merge_decision_sources   = new_sources
            st.session_state.merge_approval_result    = None
            st.rerun()
    with col_smart_ap:
        if st.button("Smart Approve", key="bulk_smart_approve"):
            new_decisions = dict(st.session_state.merge_approval_decisions)
            new_sources   = dict(st.session_state.merge_decision_sources)
            for group in view.merge_groups:
                if group.confidence == "auto_approve":
                    for pair in group.pairs:
                        key = (min(pair.cluster_a, pair.cluster_b), max(pair.cluster_a, pair.cluster_b))
                        new_decisions[key] = "approve"
                        new_sources[key]   = "human"
            st.session_state.merge_approval_decisions = new_decisions
            st.session_state.merge_decision_sources   = new_sources
            st.session_state.merge_approval_result    = None
            st.rerun()
    with col_smart_rj:
        if st.button("Smart Reject", key="bulk_smart_reject"):
            new_decisions = dict(st.session_state.merge_approval_decisions)
            new_sources   = dict(st.session_state.merge_decision_sources)
            for group in view.merge_groups:
                if group.confidence == "auto_reject":
                    for pair in group.pairs:
                        key = (min(pair.cluster_a, pair.cluster_b), max(pair.cluster_a, pair.cluster_b))
                        new_decisions[key] = "reject"
                        new_sources[key]   = "human"
            st.session_state.merge_approval_decisions = new_decisions
            st.session_state.merge_decision_sources   = new_sources
            st.session_state.merge_approval_result    = None
            st.rerun()
    with col_reset:
        if st.button("Reset all", key="bulk_reset_decisions"):
            st.session_state.merge_approval_decisions = {}
            st.session_state.merge_decision_sources   = {}
            st.session_state.merge_approval_result    = None
            st.rerun()
    st.divider()
    _render_pending_labels_indicator()
    st.checkbox("Re-run exemplar selection before merge", key="remerge_with_exemplars")
    if n_undecided > 0:
        st.info(f"{n_undecided} pair(s) undecided — re-evaluated in next round.")
    col_apply_only, col_apply_remerge, col_save = st.columns([1, 1, 1])
    with col_apply_only:
        if st.button("Apply Only", key="apply_only_btn", disabled=(n_approved == 0)):
            approved_pairs = [key for key, v in decisions.items() if v == "approve"]
            rejected_pairs = [key for key, v in decisions.items() if v == "reject"]
            _save_pending_labels_to_session(approved_pairs, rejected_pairs)
            st.session_state.merge_approval_decisions = {}
            st.session_state.merge_decision_sources   = {}
            st.session_state.merge_approval_result    = None
            st.rerun()
    with col_apply_remerge:
        disabled = n_approved == 0
        if st.button("Apply + Remerge", type="primary", key="apply_merges_btn", disabled=disabled):
            approved_pairs = [key for key, v in decisions.items() if v == "approve"]
            rejected_pairs = [key for key, v in decisions.items() if v == "reject"]
            all_approved, all_rejected = _collect_all_merge_labels(approved_pairs, rejected_pairs)
            if not all_approved:
                st.warning("No pairs approved — nothing to apply.")
            else:
                current_cr   = result.merged_cluster_result or result.cluster_result
                rnd          = st.session_state.merge_round
                source_album = st.session_state.get("current_source_album") or Path(result.output_dir).parent.name
                snap_dir     = allocate_run_dir(RunDirSpec(source_album, "merge_snap"), Path("results"))
                remerge_dir  = allocate_run_dir(RunDirSpec(source_album, "remerge"),    Path("results"))
                save_manual_merge_snapshot(
                    faces=result.faces, merged_cluster_result=current_cr,
                    approved_pairs=all_approved, rejected_pairs=all_rejected,
                    config=PipelineConfig(), output_dir=snap_dir,
                    parent_output_dir=result.output_dir,
                    parent_run_id=result.summary.get("run_id"), merge_round=rnd,
                )
                with_exemplars = st.session_state.get("remerge_with_exemplars", False)
                cfg = result.summary.get("config", {}) if result.summary else {}
                remerge_kwargs = {
                    k: cfg[k] for k in (
                        "merge_candidate_threshold", "merge_exemplar_threshold",
                        "merge_support_frac", "merge_support_min",
                        "merge_margin", "merge_diameter_expansion_factor",
                    ) if k in cfg
                }

                def _run_remerge():
                    return FaceClusteringPipeline().run(
                        PipelineConfig.remerge(snap_dir, remerge_dir,
                                               with_exemplars=with_exemplars,
                                               merge_enabled=True, **remerge_kwargs)
                    )

                _materialize_merge_step(
                    remerge_dir, all_approved, all_rejected,
                    sources=st.session_state.get("merge_decision_sources"),
                    n_undecided=n_undecided,
                )
                st.session_state.merge_decision_sources = {}
                w = _AsyncState()
                st.session_state.remerge_worker = w
                st.session_state.merge_round    = rnd + 1
                w.start(_run_remerge)
                st.rerun()
    with col_save:
        if st.button("Save Decisions", key="save_decisions_btn"):
            _save_approval_decisions(view, result)


# ---------------------------------------------------------------------------
# Gallery: grouped view
# ---------------------------------------------------------------------------

def _render_grouped_merge_gallery(view: MergeAnalysisView, result: PipelineResult):
    groups = view.merge_groups
    if not groups:
        st.info("No merge candidates found.")
        return
    chosen_filter = st.selectbox(
        "Filter groups", _GROUP_FILTERS,
        index=_GROUP_FILTERS.index(st.session_state.merge_group_filter),
        key="mg_group_filter_sel",
    )
    if chosen_filter != st.session_state.merge_group_filter:
        st.session_state.merge_group_filter = chosen_filter
        st.session_state.merge_group_page   = 0
        st.rerun()
    filter_map = {
        "Review Only": lambda g: g.confidence == "review",
        "Auto-Approve": lambda g: g.confidence == "auto_approve",
        "Auto-Reject":  lambda g: g.confidence == "auto_reject",
    }
    visible     = [g for g in groups if filter_map.get(chosen_filter, lambda _: True)(g)]
    total       = len(visible)
    if total == 0:
        st.info(f"No groups match filter: {chosen_filter}")
        return
    total_pages = max(1, (total + _GROUP_PAGE_SIZE - 1) // _GROUP_PAGE_SIZE)
    page        = max(0, min(st.session_state.merge_group_page, total_pages - 1))
    st.session_state.merge_group_page = page
    page_groups = visible[page * _GROUP_PAGE_SIZE: (page + 1) * _GROUP_PAGE_SIZE]
    total_pairs = sum(len(g.pairs) for g in visible)
    st.caption(
        f"Showing {page * _GROUP_PAGE_SIZE + 1}–{min((page + 1) * _GROUP_PAGE_SIZE, total)} "
        f"of {total} groups ({total_pairs} pairs)  |  Page {page + 1}/{total_pages}"
    )
    decisions: dict       = st.session_state.merge_approval_decisions
    sources: dict         = st.session_state.merge_decision_sources
    is_ml_mode            = view.ml_threshold is not None
    ml_pair_features: dict = st.session_state.get("ml_pair_features", {})
    ml_model_payload      = st.session_state.get("ml_model_payload")
    changed               = False
    for group in page_groups:
        conf_label = _CONF_LABEL[group.confidence]
        n_pairs    = len(group.pairs)
        group_keys = [(min(p.cluster_a, p.cluster_b), max(p.cluster_a, p.cluster_b)) for p in group.pairs]
        group_decs = [decisions.get(k) for k in group_keys]
        n_app      = group_decs.count("approve")
        n_rej      = group_decs.count("reject")
        dec_label  = (
            " | decision: **approve all**" if n_app == n_pairs else
            " | decision: **reject all**"  if n_rej == n_pairs else
            f" | decision: **mixed** ({n_app} approve, {n_rej} reject)" if n_app or n_rej else ""
        )
        cid_str      = ", ".join(f"C{cid}" for cid in group.cluster_ids)
        cohesion_pct = int(group.cohesion * 100)
        conf_color   = "green" if group.confidence == "auto_approve" else ("orange" if group.confidence == "review" else "red")
        header = (f":{conf_color}[{conf_label}]  |  {cid_str}  |  {group.total_faces} faces, {n_pairs} pairs"
                  f"  |  cohesion {cohesion_pct}%{dec_label}")
        with st.expander(header, expanded=(group.confidence == "review")):
            cluster_ids_to_show = group.cluster_ids[:6]
            crop_cols = st.columns(len(cluster_ids_to_show))
            for col, cid in zip(crop_cols, cluster_ids_to_show):
                face_ids = _exemplar_face_ids_for_cluster(cid, group, result)
                with col:
                    st.caption(f"C{cid} ({_cluster_size(cid, group)} faces)")
                    for fid in face_ids[:2]:
                        img = _crop_for_face_local(fid, result)
                        if img:
                            st.image(img, width=70)
            if is_ml_mode:
                ml_probs = [p.ml_prob for p in group.pairs if p.ml_prob is not None]
                if ml_probs:
                    avg_prob = sum(ml_probs) / len(ml_probs)
                    n_high   = sum(1 for p in ml_probs if p >= 0.8)
                    n_low    = sum(1 for p in ml_probs if p < 0.5)
                    st.caption(f"ML: avg prob={avg_prob:.2f}  |  high-conf ({n_high}) / low-conf ({n_low}) of {len(ml_probs)} pairs")
            else:
                gates_4   = sum(1 for p in group.pairs if p.n_gates_passed == 4)
                gates_3   = sum(1 for p in group.pairs if p.n_gates_passed == 3)
                gates_low = n_pairs - gates_4 - gates_3
                parts = (
                    ([f"4/4 on {gates_4} pair(s)"]   if gates_4   else []) +
                    ([f"3/4 on {gates_3} pair(s)"]   if gates_3   else []) +
                    ([f"<=2/4 on {gates_low} pair(s)"] if gates_low else [])
                )
                st.caption("Gates: " + ", ".join(parts))
            btn_col1, btn_col2, _ = st.columns([1, 1, 5])
            if btn_col1.button("Approve Group", key=f"grp_approve_{group.group_id}"):
                for k in group_keys:
                    decisions[k] = "approve"
                    sources[k]   = "human"
                st.session_state.merge_approval_decisions = decisions
                st.session_state.merge_decision_sources   = sources
                st.session_state.merge_approval_result    = None
                changed = True
            if btn_col2.button("Reject Group", key=f"grp_reject_{group.group_id}"):
                for k in group_keys:
                    decisions[k] = "reject"
                    sources[k]   = "human"
                st.session_state.merge_approval_decisions = decisions
                st.session_state.merge_decision_sources   = sources
                st.session_state.merge_approval_result    = None
                changed = True
            with st.expander(f"Show {n_pairs} individual pair(s)", expanded=False):
                for pair_i, pair in enumerate(group.pairs):
                    pair_key  = (min(pair.cluster_a, pair.cluster_b), max(pair.cluster_a, pair.cluster_b))
                    cur       = decisions.get(pair_key, "")
                    pair_src  = sources.get(pair_key)
                    outcome_label = "MERGE" if pair.action in ("merged", "proposed_merge") else "REJECT"
                    if cur == "approve":
                        pair_dec_label = " | **approve** *(ML)*" if pair_src == "ml" else " | **approve**"
                    elif cur == "reject":
                        pair_dec_label = " | **reject** *(ML)*"  if pair_src == "ml" else " | **reject**"
                    else:
                        pair_dec_label = ""
                    if is_ml_mode and pair.ml_prob is not None:
                        _pct   = int(pair.ml_prob * 100)
                        _pcolor = "green" if pair.ml_prob >= 0.8 else ("orange" if pair.ml_prob >= 0.5 else "red")
                        pair_header = (f"C{pair.cluster_a} vs C{pair.cluster_b}  dist={pair.exemplar_dist:.3f}"
                                       f"  :{_pcolor}[[{outcome_label}: {_pct}%]]{pair_dec_label}")
                    else:
                        pair_header = (f"C{pair.cluster_a} vs C{pair.cluster_b}  dist={pair.exemplar_dist:.3f}"
                                       f"  gates={pair.n_gates_passed}/4"
                                       f"  :{('green' if pair.action == 'merged' else 'red')}[{outcome_label}]"
                                       f"{pair_dec_label}")
                    with st.expander(pair_header, expanded=False):
                        _render_pair_crops(pair, result)
                        _render_gate_badges(pair)
                        if pair.rejection_reason and not is_ml_mode:
                            st.caption(f"Rejection: {pair.rejection_reason}")
                        if is_ml_mode and pair.ml_prob is not None:
                            ml_threshold_val = view.ml_threshold or 0.5
                            st.caption(f"ML: prob={pair.ml_prob:.3f}  threshold={ml_threshold_val:.2f}"
                                       f"  ({'above' if pair.ml_prob >= ml_threshold_val else 'below'} threshold)")
                            st.caption(f"Heuristic reference: {pair.n_gates_passed}/4 gates")
                            _feat = ml_pair_features.get(pair_key)
                            if _feat is not None and ml_model_payload is not None:
                                contribs = compute_pair_feature_contributions(_feat, ml_model_payload, top_n=3)
                                if contribs:
                                    st.dataframe(
                                        pd.DataFrame([{"Feature": n, "Value": f"{v:.4f}", "Direction": d}
                                                      for n, v, d in contribs]),
                                        hide_index=True, use_container_width=True,
                                    )
                        pbtn1, pbtn2, _ = st.columns([1, 1, 6])
                        approve_lbl = "OK (approved)" if cur == "approve" else "Approve"
                        reject_lbl  = "OK (rejected)"  if cur == "reject"  else "Reject"
                        if pbtn1.button(approve_lbl, key=f"pair_approve_{pair_key[0]}_{pair_key[1]}_{group.group_id}_{pair_i}"):
                            decisions[pair_key] = "approve"
                            st.session_state.merge_decision_sources[pair_key] = "human"
                            st.session_state.merge_approval_decisions = decisions
                            st.session_state.merge_approval_result    = None
                            changed = True
                        if pbtn2.button(reject_lbl, key=f"pair_reject_{pair_key[0]}_{pair_key[1]}_{group.group_id}_{pair_i}"):
                            decisions[pair_key] = "reject"
                            st.session_state.merge_decision_sources[pair_key] = "human"
                            st.session_state.merge_approval_decisions = decisions
                            st.session_state.merge_approval_result    = None
                            changed = True
    if changed:
        st.rerun()
    if total_pages > 1:
        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("Prev", key="mg_grp_prev", disabled=page == 0):
                st.session_state.merge_group_page -= 1
                st.rerun()
        with nav2:
            st.markdown(f"<div style='text-align:center;padding-top:6px'>Page {page+1} / {total_pages}</div>",
                        unsafe_allow_html=True)
        with nav3:
            if st.button("Next", key="mg_grp_next", disabled=page >= total_pages - 1):
                st.session_state.merge_group_page += 1
                st.rerun()


def _crop_for_face_local(fid, result):
    """Local wrapper to avoid circular import from cache_helpers."""
    from cache_helpers import _crop_for_face
    return _crop_for_face(fid, result.output_dir)


def _render_gate_badges(pair):
    gate_cols = st.columns(4)
    # Exemplar badge — show cross-dist when available
    exemplar_val = f"{pair.exemplar_dist:.3f}/{pair.threshold_used:.3f}"
    if pair.p25_cross_dist is not None:
        exemplar_val += f" | cross={pair.p25_cross_dist:.3f}"
    exemplar_delta = pair.exemplar_dist - pair.threshold_used

    # Support badge — show unique support when available
    support_val = f"{pair.support}/{pair.required_support}"
    if pair.unique_support is not None:
        support_val += f" (uniq={pair.unique_support})"

    gate_defs = [
        ("Exemplar", pair.passes_exemplar, exemplar_val, exemplar_delta),
        ("Support",  pair.passes_support, support_val,
         pair.support - pair.required_support),
        ("Margin",   pair.passes_margin,
         f"{pair.margin_gap:.3f}" if pair.margin_gap is not None else "n/a", None),
        ("Diameter", pair.passes_diameter,
         f"{pair.post_diameter:.3f}/{pair.max_allowed_diameter:.3f}" if pair.max_allowed_diameter else f"{pair.post_diameter:.3f}/n/a",
         pair.post_diameter - pair.max_allowed_diameter if pair.max_allowed_diameter else None),
    ]
    for gc, (gname, passed, gval, gdelta) in zip(gate_cols, gate_defs):
        color   = "#4daa6e" if passed else "#cc6666"
        badge   = "PASS" if passed else "FAIL"
        delta_s = f" ({gdelta:+.3f})" if gdelta is not None else ""
        gc.markdown(
            f"<div style='background:#1a1a2e;border-radius:4px;padding:4px 8px;"
            f"border-left:3px solid {color};font-size:12px'>"
            f"<span style='color:{color};font-weight:bold'>{badge}</span> "
            f"<span style='color:#aaa'>{gname}</span><br/>"
            f"<span style='color:#ccc;font-size:11px'>{gval}{delta_s}</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Gallery: flat view
# ---------------------------------------------------------------------------

def _render_iteration_grouped_gallery(view: MergeAnalysisView, result: PipelineResult):
    """Show every pair grouped under its iteration — default view."""
    if not view.iter_timeline:
        _render_flat_merge_gallery(view, result)
        return
    # Build iteration -> merge row lookup
    merge_by_iter: dict = {}
    for r in view.merges:
        merge_by_iter.setdefault(r.iteration, []).append(r)
    # Build iteration -> rejection rows lookup (sorted by exemplar_dist)
    reject_by_iter: dict = {}
    for r in view.all_rejection_rows:
        reject_by_iter.setdefault(r.iteration, []).append(r)

    decisions: dict = st.session_state.merge_approval_decisions
    changed = False

    for entry in view.iter_timeline:
        it   = entry["iteration"]
        ma, mb = entry.get("merged_a"), entry.get("merged_b")
        nc   = entry.get("n_candidates", 0)
        sa, sb = entry.get("size_a"), entry.get("size_b")

        iter_merges  = merge_by_iter.get(it, [])
        iter_rejects = sorted(reject_by_iter.get(it, []), key=lambda r: r.exemplar_dist)
        all_iter_rows = iter_merges + iter_rejects

        if ma is not None:
            iter_header = f"Iter {it} — :green[merged C{ma}+C{mb}] (sz {sa}+{sb}) | {nc} candidates"
        else:
            iter_header = f"Iter {it} — :red[no merge] | {nc} candidates"

        with st.expander(iter_header, expanded=(it == 1)):
            for i, d in enumerate(all_iter_rows):
                key              = (min(d.cluster_a, d.cluster_b), max(d.cluster_a, d.cluster_b))
                current_decision = decisions.get(key, "")
                outcome_label    = "MERGED" if d.action == "merged" else "REJECTED"
                decision_label   = (
                    " | decision: **approve**" if current_decision == "approve" else
                    " | decision: **reject**"  if current_decision == "reject"  else ""
                )
                color   = "green" if d.action == "merged" else "red"
                cross_part = f"  |  cross={d.p25_cross_dist:.3f}" if d.p25_cross_dist is not None else ""
                pair_hdr = (
                    f"C{d.cluster_a} ({d.cluster_a_size}) vs C{d.cluster_b} ({d.cluster_b_size})"
                    f"  |  p25_ex={d.exemplar_dist:.3f}{cross_part}"
                    f"  |  gates={d.n_gates_passed}/4"
                    f"  |  :{color}[{outcome_label}]{decision_label}"
                )
                with st.expander(pair_hdr, expanded=(d.action == "merged")):
                    _render_pair_crops(d, result, symbol="+" if d.action == "merged" else "x",
                                      key_suffix=f"it{it}_{i}")
                    _render_gate_badges(d)
                    if d.rejection_reason:
                        st.caption(f"Rejection reason: {d.rejection_reason}")
                    _render_pair_iter_history(key, view)
                    btn_col1, btn_col2, _ = st.columns([1, 1, 6])
                    approve_label = "OK (approved)" if current_decision == "approve" else "Approve"
                    reject_label  = "OK (rejected)"  if current_decision == "reject"  else "Reject"
                    if btn_col1.button(approve_label, key=f"igal_approve_{key[0]}_{key[1]}_{it}_{i}"):
                        st.session_state.merge_approval_decisions[key] = "approve"
                        st.session_state.merge_decision_sources[key]   = "human"
                        st.session_state.merge_approval_result         = None
                        changed = True
                    if btn_col2.button(reject_label, key=f"igal_reject_{key[0]}_{key[1]}_{it}_{i}"):
                        st.session_state.merge_approval_decisions[key] = "reject"
                        st.session_state.merge_decision_sources[key]   = "human"
                        st.session_state.merge_approval_result         = None
                        changed = True
    if changed:
        st.rerun()


def _render_flat_merge_gallery(view: MergeAnalysisView, result: PipelineResult):
    all_rows = view.merges + view.rejections
    if not all_rows:
        st.info("No merge candidates found.")
        return
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 3, 3])
    with ctrl_col1:
        chosen_filter = st.selectbox(
            "Filter", _GALLERY_FILTERS,
            index=_GALLERY_FILTERS.index(st.session_state.merge_gallery_filter),
            key="mg_filter_sel",
        )
    with ctrl_col2:
        chosen_sort = st.selectbox(
            "Sort by", _GALLERY_SORTS,
            index=_GALLERY_SORTS.index(
                "Exemplar distance"
                if st.session_state.merge_gallery_sort == "exemplar_dist"
                else "Gates passed (contested first)"
            ),
            key="mg_sort_sel",
        )
    with ctrl_col3:
        iter_options = ["Latest", "All"] + [str(e["iteration"]) for e in view.iter_timeline]
        cur_iter = st.session_state.merge_iter_filter
        if cur_iter not in iter_options:
            cur_iter = "Latest"
        chosen_iter = st.selectbox(
            "Iteration", iter_options,
            index=iter_options.index(cur_iter),
            key="mg_iter_sel",
        )

    new_sort_key = "exemplar_dist" if chosen_sort == "Exemplar distance" else "gates_passed"
    state_changed = (
        chosen_filter != st.session_state.merge_gallery_filter
        or new_sort_key != st.session_state.merge_gallery_sort
        or chosen_iter != st.session_state.merge_iter_filter
    )
    if state_changed:
        st.session_state.merge_gallery_filter = chosen_filter
        st.session_state.merge_gallery_sort   = new_sort_key
        st.session_state.merge_iter_filter    = chosen_iter
        st.session_state.merge_gallery_page   = 0
        st.rerun()

    # Apply iteration filter
    if chosen_iter == "Latest":
        rows_pool = all_rows  # view.rejections is already latest-per-pair; merges appear once
    elif chosen_iter == "All":
        rows_pool = view.merges + view.all_rejection_rows
    else:
        iter_num  = int(chosen_iter)
        rows_pool = (
            [r for r in view.merges if r.iteration == iter_num]
            + [r for r in view.all_rejection_rows if r.iteration == iter_num]
        )

    filter_map = {
        "Merged":            lambda d: d.action == "merged",
        "Rejected":          lambda d: d.action == "rejected",
        "Near Misses (3/4)": lambda d: d.n_gates_passed == 3,
        "Contested (1-3)":   lambda d: 1 <= d.n_gates_passed <= 3,
    }
    rows = [d for d in rows_pool if filter_map.get(chosen_filter, lambda _: True)(d)]
    rows.sort(key=lambda d: (d.n_gates_passed, d.exemplar_dist) if new_sort_key == "gates_passed" else d.exemplar_dist)
    total = len(rows)
    if total == 0:
        st.info(f"No decisions match filter: {chosen_filter}")
        return
    total_pages = max(1, (total + _GALLERY_PAGE_SIZE - 1) // _GALLERY_PAGE_SIZE)
    page        = max(0, min(st.session_state.merge_gallery_page, total_pages - 1))
    st.session_state.merge_gallery_page = page
    page_rows   = rows[page * _GALLERY_PAGE_SIZE: (page + 1) * _GALLERY_PAGE_SIZE]
    st.caption(
        f"Showing {page * _GALLERY_PAGE_SIZE + 1}–{min((page + 1) * _GALLERY_PAGE_SIZE, total)} "
        f"of {total}  |  Page {page + 1}/{total_pages}"
    )
    decisions: dict = st.session_state.merge_approval_decisions
    changed = False
    for i, d in enumerate(page_rows):
        key              = (min(d.cluster_a, d.cluster_b), max(d.cluster_a, d.cluster_b))
        current_decision = decisions.get(key, "")
        outcome_label    = "MERGED" if d.action == "merged" else "REJECTED"
        decision_label   = (
            " | decision: **approve**" if current_decision == "approve" else
            " | decision: **reject**"  if current_decision == "reject"  else ""
        )
        n_iters_for_pair = len(view.pair_history.get(key, []))
        iter_badge = f" | iter {d.iteration}" if d.iteration else ""
        multi_iter_note = f" | :blue[{n_iters_for_pair} iters]" if n_iters_for_pair > 1 else ""
        cross_part = f"  |  cross={d.p25_cross_dist:.3f}" if d.p25_cross_dist is not None else ""
        header = (
            f"C{d.cluster_a} ({d.cluster_a_size}) vs C{d.cluster_b} ({d.cluster_b_size})"
            f"  |  p25_ex={d.exemplar_dist:.3f}{cross_part}"
            f"  |  gates={d.n_gates_passed}/4"
            f"  |  :{('green' if d.action == 'merged' else 'red')}[{outcome_label}]"
            f"{iter_badge}{multi_iter_note}{decision_label}"
        )
        global_i = page * _GALLERY_PAGE_SIZE + i
        with st.expander(header, expanded=False):
            _render_pair_crops(d, result, symbol="+" if d.action == "merged" else "x")
            _render_gate_badges(d)
            if d.rejection_reason:
                st.caption(f"Rejection reason: {d.rejection_reason}")
            _render_pair_iter_history(key, view)
            btn_col1, btn_col2, _ = st.columns([1, 1, 6])
            approve_label = "OK (approved)" if current_decision == "approve" else "Approve"
            reject_label  = "OK (rejected)"  if current_decision == "reject"  else "Reject"
            if btn_col1.button(approve_label, key=f"ugal_approve_{key[0]}_{key[1]}_{global_i}"):
                st.session_state.merge_approval_decisions[key] = "approve"
                st.session_state.merge_decision_sources[key]   = "human"
                st.session_state.merge_approval_result         = None
                changed = True
            if btn_col2.button(reject_label, key=f"ugal_reject_{key[0]}_{key[1]}_{global_i}"):
                st.session_state.merge_approval_decisions[key] = "reject"
                st.session_state.merge_decision_sources[key]   = "human"
                st.session_state.merge_approval_result         = None
                changed = True
    if changed:
        st.rerun()
    if total_pages > 1:
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("Prev", key="mg_prev", disabled=page == 0):
                st.session_state.merge_gallery_page -= 1
                st.rerun()
        with nav_col2:
            st.markdown(f"<div style='text-align:center;padding-top:6px'>Page {page+1} / {total_pages}</div>",
                        unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next", key="mg_next", disabled=page >= total_pages - 1):
                st.session_state.merge_gallery_page += 1
                st.rerun()


def _render_merge_gallery(view: MergeAnalysisView, result: PipelineResult):
    st.subheader("Merge Decisions")
    _VIEW_OPTIONS = ["By iteration", "Flat list", "Transitive groups"]
    _MODE_KEY     = {"By iteration": "by_iteration", "Flat list": "flat", "Transitive groups": "groups"}
    _MODE_LABEL   = {v: k for k, v in _MODE_KEY.items()}

    cur_mode = st.session_state.get("merge_gallery_view_mode", "by_iteration")
    cur_label = _MODE_LABEL.get(cur_mode, "By iteration")
    chosen_label = st.radio(
        "View mode", _VIEW_OPTIONS,
        index=_VIEW_OPTIONS.index(cur_label),
        horizontal=True, key="mg_view_mode_radio",
    )
    chosen_mode = _MODE_KEY[chosen_label]
    if chosen_mode != cur_mode:
        st.session_state.merge_gallery_view_mode = chosen_mode
        st.rerun()

    if chosen_mode == "by_iteration":
        _render_iteration_grouped_gallery(view, result)
    elif chosen_mode == "groups" and view.merge_groups:
        _render_grouped_merge_gallery(view, result)
    else:
        _render_flat_merge_gallery(view, result)


def _render_merge_analysis(view: MergeAnalysisView, result: PipelineResult):
    _render_criteria_reference()
    _render_merge_summary(view)
    _render_iteration_timeline(view)
    _render_gate_bottleneck(view)
    _render_threshold_distribution(view, result)
    _render_approval_controls(view, result)
    _render_merge_gallery(view, result)
    _render_absorbed_clusters(view)
