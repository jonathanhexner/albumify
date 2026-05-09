"""Tab 9: Labeling Review — audit labeled merge decisions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from face_cluster.training_db import load_training_data, training_data_summary, update_label

from cache_helpers import _crop_for_face


def _load_gates_from_decisions(output_dir: Optional[Path]) -> dict:
    if not output_dir or not output_dir.exists():
        return {}
    decisions_path = output_dir / "merge_decisions.json"
    if not decisions_path.exists():
        return {}
    with open(decisions_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    return {
        (int(e["cluster_a"]), int(e["cluster_b"])): e.get("n_gates_passed")
        for e in entries if "cluster_a" in e and "cluster_b" in e
    }


def render_labeling_review_tab():
    import plotly.express as px

    st.header("Labeling Review")
    st.caption("Audit labeled merge decisions and spot human-heuristic disagreements.")

    summary = training_data_summary()
    total   = summary.get("total_rows") or 0
    if total == 0:
        st.info("No training data yet. Label decisions in **Merge Analysis** and click **Save Decisions**.")
        return

    n_approved  = summary.get("n_approved")  or 0
    n_rejected  = summary.get("n_rejected")  or 0
    n_unlabeled = summary.get("n_unlabeled") or 0
    n_runs      = summary.get("n_runs")      or 0
    labeled     = n_approved + n_rejected

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Samples", total)
    c2.metric("Approved",      n_approved)
    c3.metric("Rejected",      n_rejected)
    c4.metric("Unlabeled",     n_unlabeled)
    c5.metric("Runs",          n_runs)
    minority   = min(n_approved, n_rejected) if n_approved > 0 and n_rejected > 0 else 0
    majority   = max(n_approved, n_rejected)
    imbalanced = majority > 3 * minority if minority > 0 else True
    if labeled < 50:
        c6.metric("Readiness", "< 50 labeled",  delta="Need more data",    delta_color="inverse")
    elif labeled < 200 or imbalanced:
        c6.metric("Readiness", "Needs work",     delta="Imbalanced or <200", delta_color="off")
    else:
        c6.metric("Readiness", "Ready",          delta="200+ balanced",    delta_color="normal")

    df = load_training_data()
    if df.empty:
        return
    labeled_df = df[df["label"].notna()].copy()

    st.subheader("Per-Run Breakdown")
    if not labeled_df.empty:
        run_stats = (
            labeled_df.groupby("run_id")
            .agg(
                album    =("album_path", lambda x: Path(x.iloc[0]).name if pd.notna(x.iloc[0]) else ""),
                approved =("label", lambda x: int((x == 1).sum())),
                rejected =("label", lambda x: int((x == 0).sum())),
                total    =("label", "count"),
            )
            .reset_index()
            .sort_values("total", ascending=False)
        )
        st.dataframe(run_stats, hide_index=True, use_container_width=True)
        fig = px.bar(
            run_stats.melt(id_vars="run_id", value_vars=["approved", "rejected"],
                           var_name="Decision", value_name="Count"),
            x="run_id", y="Count", color="Decision", barmode="stack",
            color_discrete_map={"approved": "#2ecc71", "rejected": "#e74c3c"}, height=250,
        )
        fig.update_layout(margin=dict(t=20, b=40), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    key_feats = ["min_exemplar_dist", "post_merge_diameter", "shared_source_images"]
    available = [f for f in key_feats if f in df.columns]
    if available and not labeled_df.empty:
        with st.expander("Feature Distributions (labeled pairs)", expanded=False):
            plot_df = labeled_df[["label"] + available].copy()
            plot_df["Decision"] = plot_df["label"].map({1.0: "approve", 0.0: "reject"})
            for feat in available:
                fig = px.histogram(
                    plot_df, x=feat, color="Decision", barmode="overlay",
                    color_discrete_map={"approve": "#2ecc71", "reject": "#e74c3c"},
                    opacity=0.7, nbins=30, title=feat, height=220,
                )
                fig.update_layout(margin=dict(t=36, b=20), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Label Audit")

    audit_run_options   = ["All"] + sorted(labeled_df["run_id"].unique().tolist())
    audit_label_options = ["All", "approve", "reject"]
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    audit_run     = fc1.selectbox("Run",   audit_run_options,   key="lr_audit_run")
    audit_label   = fc2.selectbox("Label", audit_label_options, key="lr_audit_label")
    disagree_only = fc3.checkbox("Disagreements only", key="lr_disagree_only")

    audit_df = labeled_df.copy()
    if audit_run != "All":
        audit_df = audit_df[audit_df["run_id"] == audit_run]
    if audit_label != "All":
        lbl_val  = 1.0 if audit_label == "approve" else 0.0
        audit_df = audit_df[audit_df["label"] == lbl_val]

    _gates_cache: dict = {}

    def _get_gates(out_dir_str) -> dict:
        if not out_dir_str or (isinstance(out_dir_str, float) and pd.isna(out_dir_str)):
            return {}
        if out_dir_str not in _gates_cache:
            _gates_cache[out_dir_str] = _load_gates_from_decisions(Path(out_dir_str))
        return _gates_cache[out_dir_str]

    def _disagreement(label: float, n_gates) -> bool:
        if n_gates is None:
            return False
        return (label == 1 and n_gates <= 1) or (label == 0 and n_gates >= 3)

    audit_rows = []
    for _, row in audit_df.iterrows():
        gates_map = _get_gates(row.get("output_dir"))
        n_gates   = gates_map.get((int(row["cluster_a"]), int(row["cluster_b"])))
        label_val = float(row["label"])
        disagree  = _disagreement(label_val, n_gates)
        audit_rows.append({
            "run_id":    row["run_id"],
            "cluster_a": int(row["cluster_a"]),
            "cluster_b": int(row["cluster_b"]),
            "label":     "approve" if label_val == 1.0 else "reject",
            "n_gates":   n_gates,
            "dist":      round(row["min_exemplar_dist"], 3)
                         if "min_exemplar_dist" in row.index and pd.notna(row.get("min_exemplar_dist"))
                         else None,
            "agree":     "OK" if not disagree else "WARNING",
            "_disagree": disagree,
        })

    if disagree_only:
        audit_rows = [r for r in audit_rows if r["_disagree"]]

    if not audit_rows:
        st.info("No rows match the current filters.")
    else:
        for ar in audit_rows:
            warn_icon = ":warning:" if ar["_disagree"] else ""
            gates_str = f"{ar['n_gates']}/4" if ar["n_gates"] is not None else "n/a"
            dist_str  = str(ar["dist"]) if ar["dist"] is not None else "n/a"
            flip_key  = f"lr_flip_{ar['run_id']}_{ar['cluster_a']}_{ar['cluster_b']}"
            cols      = st.columns([2, 1, 1, 1, 1, 1])
            cols[0].write(f"{warn_icon} **{ar['run_id']}**  {ar['cluster_a']}<->{ar['cluster_b']}")
            cols[1].write(ar["label"])
            cols[2].write(f"gates: {gates_str}")
            cols[3].write(f"dist: {dist_str}")
            cols[4].write(ar["agree"])
            if cols[5].button("Flip", key=flip_key):
                new_label = 0 if ar["label"] == "approve" else 1
                update_label(ar["run_id"], ar["cluster_a"], ar["cluster_b"], new_label)
                st.toast(f"Label flipped: {ar['run_id']} {ar['cluster_a']}<->{ar['cluster_b']} "
                         f"-> {'approve' if new_label == 1 else 'reject'}")
                st.rerun()

    st.divider()
    st.subheader("Suggested Next Labels")
    unlabeled_df = df[df["label"].isna()].copy()
    if unlabeled_df.empty:
        st.info("All pairs are labeled — nothing to suggest.")
    else:
        sugg_rows = []
        for _, row in unlabeled_df.iterrows():
            gates_map = _get_gates(row.get("output_dir"))
            n_gates   = gates_map.get((int(row["cluster_a"]), int(row["cluster_b"])))
            gate_amb  = abs((n_gates or 2) - 2)
            dist_val  = (float(row["min_exemplar_dist"])
                         if "min_exemplar_dist" in row.index and pd.notna(row.get("min_exemplar_dist"))
                         else 0.35)
            dist_amb  = abs(dist_val - 0.35)
            sugg_rows.append({
                "run_id": row["run_id"],
                "cluster_a": int(row["cluster_a"]),
                "cluster_b": int(row["cluster_b"]),
                "n_gates":   n_gates,
                "dist":      round(dist_val, 3),
                "_sort":     (gate_amb, dist_amb),
                "_output_dir": row.get("output_dir"),
            })
        sugg_rows.sort(key=lambda r: r["_sort"])
        for sr in sugg_rows[:20]:
            gates_str = f"{sr['n_gates']}/4" if sr["n_gates"] is not None else "n/a"
            appr_key  = f"lr_sugg_a_{sr['run_id']}_{sr['cluster_a']}_{sr['cluster_b']}"
            rej_key   = f"lr_sugg_r_{sr['run_id']}_{sr['cluster_a']}_{sr['cluster_b']}"
            cols = st.columns([2, 1, 1, 1, 1])
            cols[0].write(f"**{sr['run_id']}**  {sr['cluster_a']}<->{sr['cluster_b']}")
            cols[1].write(f"gates: {gates_str}")
            cols[2].write(f"dist: {sr['dist']}")
            if cols[3].button("Approve", key=appr_key):
                update_label(sr["run_id"], sr["cluster_a"], sr["cluster_b"], 1)
                st.toast(f"Labeled approve: {sr['run_id']} {sr['cluster_a']}<->{sr['cluster_b']}")
                st.rerun()
            if cols[4].button("Reject", key=rej_key):
                update_label(sr["run_id"], sr["cluster_a"], sr["cluster_b"], 0)
                st.toast(f"Labeled reject: {sr['run_id']} {sr['cluster_a']}<->{sr['cluster_b']}")
                st.rerun()

    st.divider()
    st.subheader("Pair Inspector")
    run_ids = sorted(df["run_id"].unique())
    sel_run = st.selectbox("Run", run_ids, key="td_run_sel")
    run_df  = df[df["run_id"] == sel_run].reset_index(drop=True)

    def _pair_label(row) -> str:
        lbl = {1.0: "approve", 0.0: "reject"}.get(row.get("label"), "unlabeled")
        return f"Cluster {int(row['cluster_a'])} <-> {int(row['cluster_b'])}  [{lbl}]"

    pair_indices = list(run_df.index)
    sel_idx = st.selectbox("Pair", pair_indices, format_func=lambda i: _pair_label(run_df.loc[i]),
                           key="td_pair_sel")
    if sel_idx is not None:
        pair_row     = run_df.loc[sel_idx]
        out_dir_str  = pair_row.get("output_dir")
        out_dir      = Path(out_dir_str) if out_dir_str and pd.notna(out_dir_str) else None
        ex_ids       = {"a": [], "b": []}
        raw_ex       = pair_row.get("exemplar_ids")
        if raw_ex and pd.notna(raw_ex):
            ex_ids = json.loads(raw_ex)
        col_a, col_b = st.columns(2)
        for col, side, cid_key in [(col_a, "a", "cluster_a"), (col_b, "b", "cluster_b")]:
            with col:
                st.caption(f"Cluster {int(pair_row[cid_key])}")
                fids = ex_ids.get(side, [])[:5]
                if fids and out_dir and out_dir.exists():
                    img_cols = st.columns(len(fids))
                    for i, fid in enumerate(fids):
                        img = _crop_for_face(fid, out_dir)
                        img_cols[i].image(img, width=80) if img else img_cols[i].caption(f"#{fid}")
                elif not out_dir or not out_dir.exists():
                    st.caption("Run directory not found")
                else:
                    st.caption("No exemplars stored")
        show_feats = ["label", "min_exemplar_dist", "shared_source_images",
                      "post_merge_diameter", "support_fraction", "size_a", "size_b"]
        feat_rows = [{"feature": k, "value": pair_row.get(k)}
                     for k in show_feats if k in pair_row.index]
        if feat_rows:
            st.dataframe(pd.DataFrame(feat_rows), hide_index=True, use_container_width=True)

    st.subheader("Export")
    if not labeled_df.empty:
        export_df = labeled_df.drop(columns=["exemplar_ids"], errors="ignore")
        c_csv, c_pq = st.columns(2)
        c_csv.download_button("Download CSV",     data=export_df.to_csv(index=False),
                              file_name="merge_training_data.csv", mime="text/csv")
        c_pq.download_button("Download Parquet",  data=export_df.to_parquet(index=False),
                              file_name="merge_training_data.parquet",
                              mime="application/octet-stream")
    else:
        st.info("No labeled samples to export yet.")
