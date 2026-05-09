"""Tab 11: Label Verification — review merge candidates and assign human labels.

Spec: specs/017-merge-label-verification/spec.md

Layout:
  1. Controls: dataset dropdown + candidate threshold slider + Load button
  2. Metric cards
  3. Bulk apply section
  4. [Fragment] Filter controls + st.dataframe (sortable/filterable) +
               action buttons (Merge/Reject/Ignore selected) +
               detail panel (thumbnails for selected row) +
               save bar
  5. Label summary + CSV export

Performance note: The table is a single st.dataframe widget (not N×buttons),
so interactions trigger only a fragment-scoped rerun (~0.1s regardless of row count).
"""
from __future__ import annotations

import io
import time
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from face_cluster.label_verification import (
    LabelVerificationData,
    load_canonical_runs,
    load_crop,
    load_label_verification_data,
)
from face_cluster.training_db import (
    label_summary_by_run,
    load_training_data,
    save_human_label,
)

from state import _AsyncState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_THUMBS_DETAIL = 10
_THUMB_DETAIL_W = 72
_THUMB_TABLE_W = 52
_MAX_THUMBS_TABLE = 3

_DECISION_OPTIONS = ["All", "merge", "reject", "ignore", "unlabeled"]


# ---------------------------------------------------------------------------
# Crop cache (session-scoped lazy load)
# ---------------------------------------------------------------------------

def _get_crop(face_id: int, data: LabelVerificationData) -> Optional[Image.Image]:
    cache = st.session_state.lv_crop_cache
    if face_id in cache:
        return cache[face_id]
    img = load_crop(face_id, data.run_dir, data.crop_fallback_dir)
    cache[face_id] = img
    return img


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def _init_decisions_from_data(data: LabelVerificationData) -> None:
    """Populate lv_decisions from loaded pairs_df."""
    decisions: dict = {}
    for _, row in data.pairs_df.iterrows():
        key = (int(row["cluster_a"]), int(row["cluster_b"]))
        src = row.get("db_source")
        db_lbl = row.get("db_label")
        heur_lbl = row.get("heuristic_label")
        if src == "human":
            decisions[key] = "merge" if db_lbl == 1 else ("reject" if db_lbl == 0 else "ignore")
        elif heur_lbl == 1:
            decisions[key] = "merge"
        elif heur_lbl == 0:
            decisions[key] = "reject"
    st.session_state.lv_decisions = decisions
    st.session_state.lv_dirty = set()


# ---------------------------------------------------------------------------
# Display dataframe builder
# ---------------------------------------------------------------------------

def _build_display_df(pairs_df: pd.DataFrame, decisions: dict, dirty: set) -> pd.DataFrame:
    """Build a clean DataFrame for st.dataframe display.

    Columns:
      cluster_a, cluster_b, size_a, size_b,
      min_exemplar_dist, p10_cross_dist,
      heuristic  — algorithm decision (merge/reject/—)
      decision   — current human decision (merge/reject/ignore/'' for unlabeled)
    """
    keep = [c for c in [
        "cluster_a", "cluster_b", "size_a", "size_b",
        "min_exemplar_dist", "p10_cross_dist",
        "heuristic_label", "db_source",
    ] if c in pairs_df.columns]
    df = pairs_df[keep].copy()

    df["heuristic"] = df["heuristic_label"].map({1: "merge", 0: "reject"}).fillna("—")

    def _human_decision(r):
        key = (int(r["cluster_a"]), int(r["cluster_b"]))
        if r.get("db_source") == "human" or key in dirty:
            return decisions.get(key, "ignore")
        return ""  # heuristic only — not yet human-reviewed

    df["decision"] = df.apply(_human_decision, axis=1)
    df = df.drop(columns=["heuristic_label", "db_source"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Deferred save helpers
# ---------------------------------------------------------------------------

def _decision_to_label(decision: Optional[str]) -> Optional[int]:
    if decision == "merge":  return 1
    if decision == "reject": return 0
    return None


def _flush_dirty(run_id: str) -> int:
    """Write all dirty pairs to DB. Returns count saved."""
    dirty: set = st.session_state.lv_dirty
    if not dirty:
        return 0
    decisions = st.session_state.lv_decisions
    count = 0
    for pair_key in list(dirty):
        cid_a, cid_b = pair_key
        lbl = _decision_to_label(decisions.get(pair_key))
        if save_human_label(run_id, cid_a, cid_b, lbl, verified=True):
            count += 1
    st.session_state.lv_dirty = set()
    return count


# ---------------------------------------------------------------------------
# Detail panel (thumbnails + metrics)
# ---------------------------------------------------------------------------

def _render_detail_panel(pair_key: Tuple[int, int], row: pd.Series, data: LabelVerificationData) -> None:
    cid_a, cid_b = pair_key
    with st.container(border=True):
        st.markdown(f"**Cluster {cid_a}** ({row['size_a']} faces)  vs  **Cluster {cid_b}** ({row['size_b']} faces)")

        all_ids_a = data.cluster_to_face_ids.get(cid_a, [])[:_MAX_THUMBS_DETAIL]
        all_ids_b = data.cluster_to_face_ids.get(cid_b, [])[:_MAX_THUMBS_DETAIL]
        ex_ids_a = set(data.cluster_to_exemplar_ids.get(cid_a, []))
        ex_ids_b = set(data.cluster_to_exemplar_ids.get(cid_b, []))

        col_a, col_sep, col_b = st.columns([5, 0.4, 5])
        with col_a:
            st.caption(f"Cluster {cid_a} — {row['size_a']} faces  (* = exemplar)")
            if all_ids_a:
                sub = st.columns(len(all_ids_a))
                for i, fid in enumerate(all_ids_a):
                    with sub[i]:
                        img = _get_crop(fid, data)
                        lbl = f"*{fid}" if fid in ex_ids_a else str(fid)
                        if img:
                            st.image(img, width=_THUMB_DETAIL_W, caption=lbl)
                        else:
                            st.caption(f"[{lbl}]")
        with col_sep:
            st.markdown(
                "<div style='text-align:center;font-size:20px;padding-top:48px'>VS</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.caption(f"Cluster {cid_b} — {row['size_b']} faces  (* = exemplar)")
            if all_ids_b:
                sub = st.columns(len(all_ids_b))
                for i, fid in enumerate(all_ids_b):
                    with sub[i]:
                        img = _get_crop(fid, data)
                        lbl = f"*{fid}" if fid in ex_ids_b else str(fid)
                        if img:
                            st.image(img, width=_THUMB_DETAIL_W, caption=lbl)
                        else:
                            st.caption(f"[{lbl}]")

        st.divider()
        mc = st.columns(6)
        mc[0].metric("min_exemplar_dist",  f"{row.get('min_exemplar_dist'):.3f}"   if row.get("min_exemplar_dist")   is not None else "—")
        mc[1].metric("p10_cross_dist",     f"{row.get('p10_cross_dist'):.3f}"      if row.get("p10_cross_dist")      is not None else "—")
        mc[2].metric("p50_cross_dist",     f"{row.get('p50_cross_dist'):.3f}"      if row.get("p50_cross_dist")      is not None else "—")
        mc[3].metric("support_fraction",   f"{row.get('support_fraction'):.3f}"    if row.get("support_fraction")    is not None else "—")
        mc[4].metric("post_merge_diam",    f"{row.get('post_merge_diameter'):.3f}" if row.get("post_merge_diameter") is not None else "—")
        mc[5].metric("diam_expansion",     f"{row.get('diameter_expansion'):.3f}"  if row.get("diameter_expansion")  is not None else "—")

        st.divider()
        gc = st.columns([1, 1, 1, 1, 4])
        _gate_badge(gc[0], "Exemplar",   row.get("passes_exemplar"))
        _gate_badge(gc[1], "Cross-dist", row.get("passes_support"))
        _gate_badge(gc[2], "Margin",     row.get("passes_margin"))
        _gate_badge(gc[3], "Diameter",   row.get("passes_diameter"))
        if row.get("rejection_reason"):
            gc[4].caption(f"Rejected: {row['rejection_reason']}")


def _gate_badge(col, name: str, passed) -> None:
    if passed is True:
        col.markdown(f":green[**pass** {name}]")
    elif passed is False:
        col.markdown(f":red[**fail** {name}]")
    else:
        col.caption(f"— {name}")


def _render_thumbnail_row(pair_key: Tuple[int, int], row: pd.Series, data: LabelVerificationData) -> None:
    """Compact side-by-side thumbnail comparison for multi-row selection."""
    cid_a, cid_b = pair_key
    d = row.get("min_exemplar_dist")
    p = row.get("p10_cross_dist")
    dist_str = (f"min={d:.3f}" if d is not None else "") + (f"  p10={p:.3f}" if p is not None else "")
    with st.container(border=True):
        st.caption(f"C{cid_a} ({row['size_a']}) vs C{cid_b} ({row['size_b']})  —  {dist_str}")
        col_a, col_sep, col_b = st.columns([5, 0.3, 5])
        with col_a:
            ex_ids_a = data.cluster_to_exemplar_ids.get(cid_a, [])[:_MAX_THUMBS_TABLE]
            if ex_ids_a:
                sub = st.columns(len(ex_ids_a))
                for i, fid in enumerate(ex_ids_a):
                    img = _get_crop(fid, data)
                    if img:
                        sub[i].image(img, width=_THUMB_TABLE_W)
        with col_sep:
            st.markdown("<div style='text-align:center;padding-top:16px'>VS</div>", unsafe_allow_html=True)
        with col_b:
            ex_ids_b = data.cluster_to_exemplar_ids.get(cid_b, [])[:_MAX_THUMBS_TABLE]
            if ex_ids_b:
                sub = st.columns(len(ex_ids_b))
                for i, fid in enumerate(ex_ids_b):
                    img = _get_crop(fid, data)
                    if img:
                        sub[i].image(img, width=_THUMB_TABLE_W)


# ---------------------------------------------------------------------------
# Fragment: the entire interactive section
# ---------------------------------------------------------------------------

@st.fragment
def _render_table_section() -> None:
    data: Optional[LabelVerificationData] = st.session_state.lv_data
    if data is None:
        return

    pairs_df = data.pairs_df
    decisions: dict = st.session_state.lv_decisions
    dirty: set = st.session_state.lv_dirty

    # -- Save bar -----------------------------------------------------------
    if dirty:
        c1, c2 = st.columns([5, 1])
        c1.warning(f"{len(dirty)} unsaved change(s) — click Save to commit to DB.")
        if c2.button("Save", key="lv_save_btn", type="primary", use_container_width=True):
            saved = _flush_dirty(data.run_id)
            st.success(f"Saved {saved} pair(s).")
            st.rerun(scope="app")

    # -- Filter controls ----------------------------------------------------
    fc1, fc2, fc3 = st.columns([2, 1.5, 1.5])
    decision_filter = fc1.selectbox(
        "Filter by decision",
        _DECISION_OPTIONS,
        key="lv_decision_filter",
        label_visibility="collapsed",
        help="Filter by current human decision (or 'unlabeled' = no human label yet).",
    )

    p10_vals = pairs_df["p10_cross_dist"].dropna() if "p10_cross_dist" in pairs_df.columns else pd.Series(dtype=float)
    p10_data_min = float(p10_vals.min()) if not p10_vals.empty else 0.0
    p10_data_max = float(p10_vals.max()) if not p10_vals.empty else 1.0

    p10_min = fc2.number_input(
        "p10 >=", value=p10_data_min, min_value=0.0, max_value=2.0, step=0.01,
        format="%.3f", key="lv_p10_min",
    )
    p10_max = fc3.number_input(
        "p10 <=", value=p10_data_max, min_value=0.0, max_value=2.0, step=0.01,
        format="%.3f", key="lv_p10_max",
    )

    # -- Build display dataframe --------------------------------------------
    display_df = _build_display_df(pairs_df, decisions, dirty)

    # Apply decision filter
    if decision_filter != "All":
        if decision_filter == "unlabeled":
            display_df = display_df[display_df["decision"] == ""]
        else:
            display_df = display_df[display_df["decision"] == decision_filter]

    # Apply p10 filter (keep rows where p10 is NaN or within range)
    if "p10_cross_dist" in display_df.columns:
        p10_col = display_df["p10_cross_dist"]
        mask = p10_col.isna() | ((p10_col >= p10_min) & (p10_col <= p10_max))
        display_df = display_df[mask]

    display_df = display_df.reset_index(drop=True)

    st.caption(f"Showing {len(display_df)} pair(s)  — click a column header to sort, click rows to select.")

    # -- Dataframe ----------------------------------------------------------
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="lv_table",
        column_config={
            "cluster_a":         st.column_config.NumberColumn("Cluster A", format="%d", width="small"),
            "cluster_b":         st.column_config.NumberColumn("Cluster B", format="%d", width="small"),
            "size_a":            st.column_config.NumberColumn("Size A",    format="%d", width="small"),
            "size_b":            st.column_config.NumberColumn("Size B",    format="%d", width="small"),
            "min_exemplar_dist": st.column_config.NumberColumn("Min Dist",  format="%.3f"),
            "p10_cross_dist":    st.column_config.NumberColumn("p10",       format="%.3f"),
            "heuristic":         st.column_config.TextColumn("Heuristic",   width="small"),
            "decision":          st.column_config.TextColumn("Decision",    width="small"),
        },
    )

    selected_indices: list = event.selection.rows

    # -- Thumbnail preview + action buttons for selected rows ---------------
    if selected_indices:
        n = len(selected_indices)
        st.divider()

        # Thumbnails: full detail for single selection, compact rows for multi
        if n == 1:
            row_data = display_df.iloc[selected_indices[0]]
            pair_key = (int(row_data["cluster_a"]), int(row_data["cluster_b"]))
            full_row = pairs_df[
                (pairs_df["cluster_a"] == pair_key[0]) &
                (pairs_df["cluster_b"] == pair_key[1])
            ]
            if not full_row.empty:
                _render_detail_panel(pair_key, full_row.iloc[0], data)
        else:
            # Show compact thumbnail comparison for each selected pair (max 5)
            for i in selected_indices[:5]:
                row_data = display_df.iloc[i]
                pair_key = (int(row_data["cluster_a"]), int(row_data["cluster_b"]))
                full_row = pairs_df[
                    (pairs_df["cluster_a"] == pair_key[0]) &
                    (pairs_df["cluster_b"] == pair_key[1])
                ]
                if not full_row.empty:
                    _render_thumbnail_row(pair_key, full_row.iloc[0], data)
            if n > 5:
                st.caption(f"...and {n - 5} more selected (not shown)")

        # Action buttons below thumbnails
        st.divider()
        ac1, ac2, ac3, ac4 = st.columns([1.2, 1.2, 1.2, 4])

        def _apply(d: str) -> None:
            for i in selected_indices:
                row = display_df.iloc[i]
                key = (int(row["cluster_a"]), int(row["cluster_b"]))
                st.session_state.lv_decisions[key] = d
                st.session_state.lv_dirty.add(key)

        if ac1.button(f"Merge ({n})",  key="lv_act_merge",  type="primary"):
            _apply("merge");  st.rerun()
        if ac2.button(f"Reject ({n})", key="lv_act_reject", type="primary"):
            _apply("reject"); st.rerun()
        if ac3.button(f"Ignore ({n})", key="lv_act_ignore"):
            _apply("ignore"); st.rerun()
        ac4.caption(f"{n} row(s) selected")


# ---------------------------------------------------------------------------
# Main tab render
# ---------------------------------------------------------------------------

def render_label_verification_tab():
    st.header("Label Verification")
    st.caption("Review merge candidates and assign human labels for ML training.")

    # -- Controls -----------------------------------------------------------
    canonical_runs = load_canonical_runs()
    ctrl1, ctrl2 = st.columns([2, 3])
    with ctrl1:
        dataset_options = list(canonical_runs.keys())
        dataset_labels = [
            f"{ds}  [{canonical_runs[ds]['run']}]"
            for ds in dataset_options
        ]
        cur_idx = dataset_options.index(st.session_state.lv_dataset) \
            if st.session_state.lv_dataset in dataset_options else 0
        sel_idx = st.selectbox(
            "Source Dataset",
            options=range(len(dataset_options)),
            format_func=lambda i: dataset_labels[i],
            index=cur_idx,
            key="lv_dataset_select",
        )
        dataset_key = dataset_options[sel_idx]
    with ctrl2:
        threshold = st.slider(
            "Candidate Threshold",
            min_value=0.20, max_value=1.0, step=0.01,
            value=st.session_state.lv_threshold,
            key="lv_threshold_slider",
            help="Pairs with min_exemplar_dist below this threshold are shown as candidates.",
        )

    load_clicked = st.button(
        "Load Dataset",
        type="primary",
        key="lv_load_btn",
        disabled=st.session_state.lv_worker is not None
        and st.session_state.lv_worker.is_running,
    )
    if load_clicked:
        st.session_state.lv_dataset = dataset_key
        st.session_state.lv_threshold = threshold
        st.session_state.lv_data = None
        st.session_state.lv_crop_cache = {}
        w = _AsyncState()
        w.start(load_label_verification_data, dataset_key, threshold)
        st.session_state.lv_worker = w
        st.rerun()

    # -- Worker polling -----------------------------------------------------
    worker: Optional[_AsyncState] = st.session_state.lv_worker
    if worker is not None:
        if worker.is_running:
            elapsed = worker.elapsed_s() or 0
            st.info(f"Loading {st.session_state.lv_dataset}... ({elapsed:.0f}s)")
            time.sleep(0.4)
            st.rerun()
            return
        elif worker.has_error:
            st.error(f"Failed to load dataset: {worker.error}")
            st.session_state.lv_worker = None
            return
        elif worker.is_done and st.session_state.lv_data is None:
            st.session_state.lv_data = worker.result
            st.session_state.lv_worker = None
            _init_decisions_from_data(worker.result)
            st.rerun()

    data: Optional[LabelVerificationData] = st.session_state.lv_data
    if data is None:
        st.info("Select a dataset and click **Load Dataset** to begin.")
        _render_summary_section()
        return

    pairs_df = data.pairs_df
    dirty: set = st.session_state.lv_dirty
    n_total = len(pairs_df)
    n_labeled = min(
        int((pairs_df["db_source"] == "human").sum()) + len(dirty),
        n_total,
    )

    # -- Metric cards -------------------------------------------------------
    mc = st.columns(3)
    mc[0].metric("Clusters", len(data.cluster_to_face_ids))
    mc[1].metric("Candidate Pairs", n_total)
    mc[2].metric("Labeled", n_labeled)
    st.caption(
        f"Loaded: **{data.run_id}** — threshold {st.session_state.lv_threshold:.2f}. "
        "Pairs saved at a higher threshold won't appear here; raise the slider and reload to see them."
    )

    # -- Bulk apply ---------------------------------------------------------
    _render_bulk_apply_section(pairs_df, data.run_id)

    st.divider()

    # -- Interactive table (fragment) ---------------------------------------
    _render_table_section()

    st.divider()
    _render_summary_section()


# ---------------------------------------------------------------------------
# Bulk apply section
# ---------------------------------------------------------------------------

def _render_bulk_apply_section(pairs_df: pd.DataFrame, run_id: str) -> None:
    """Quick-label many pairs at once by p10 threshold range."""
    with st.expander("Quick Label by p10 threshold", expanded=False):
        c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 2])

        p10_vals = pairs_df["p10_cross_dist"].dropna() if "p10_cross_dist" in pairs_df.columns else pd.Series(dtype=float)
        p10_lo = float(p10_vals.min()) if not p10_vals.empty else 0.0
        p10_hi = float(p10_vals.max()) if not p10_vals.empty else 1.0

        bulk_min = c1.number_input(
            "p10 >=", value=round(p10_lo, 3), min_value=0.0, max_value=2.0, step=0.01,
            format="%.3f", key="lv_bulk_p10_min",
        )
        bulk_max = c2.number_input(
            "p10 <=", value=round(p10_hi, 3), min_value=0.0, max_value=2.0, step=0.01,
            format="%.3f", key="lv_bulk_p10_max",
        )
        bulk_decision = c3.selectbox(
            "Apply label", ["reject", "merge", "ignore"], key="lv_bulk_decision",
        )
        unlabeled_only = c4.checkbox("Unlabeled only", value=True, key="lv_bulk_unlabeled_only")

        dirty: set = st.session_state.lv_dirty
        mask = (
            pairs_df["p10_cross_dist"].notna() &
            (pairs_df["p10_cross_dist"] >= bulk_min) &
            (pairs_df["p10_cross_dist"] <= bulk_max)
        ) if "p10_cross_dist" in pairs_df.columns else pd.Series(False, index=pairs_df.index)
        if unlabeled_only:
            mask = mask & pairs_df.apply(
                lambda r: r.get("db_source") != "human"
                and (int(r["cluster_a"]), int(r["cluster_b"])) not in dirty,
                axis=1,
            )
        target_df = pairs_df[mask]
        n_target = len(target_df)
        st.caption(f"Matches **{n_target}** pair(s) — will label all as **{bulk_decision}**.")

        if st.button(
            f"Apply to {n_target} pair(s)", key="lv_bulk_apply",
            type="primary", disabled=n_target == 0,
        ):
            lbl_val = _decision_to_label(bulk_decision)
            for _, row in target_df.iterrows():
                pair_key = (int(row["cluster_a"]), int(row["cluster_b"]))
                st.session_state.lv_decisions[pair_key] = bulk_decision
                save_human_label(run_id, pair_key[0], pair_key[1], lbl_val, verified=True)
            st.session_state.lv_dirty -= {
                (int(r["cluster_a"]), int(r["cluster_b"])) for _, r in target_df.iterrows()
            }
            st.success(f"Applied '{bulk_decision}' to {n_target} pair(s).")
            st.rerun()


# ---------------------------------------------------------------------------
# Summary + export section
# ---------------------------------------------------------------------------

def _render_summary_section() -> None:
    st.subheader("Label Summary")
    st.caption("All labeled pairs across every run in the DB. Currently loaded dataset marked with *.")
    summary_df = label_summary_by_run()
    if summary_df.empty:
        st.info("No labeled pairs yet.")
    else:
        canonical_runs = load_canonical_runs()
        run_to_dataset = {v["run"]: k for k, v in canonical_runs.items()}
        lv_data = st.session_state.get("lv_data")
        active_run_id = lv_data.run_id if lv_data is not None else None

        def _label_row(row):
            ds = run_to_dataset.get(row["run_id"], "—")
            marker = " *" if row["run_id"] == active_run_id else ""
            return f"{ds}{marker}  [{row['run_id']}]"

        summary_df = summary_df.copy()
        summary_df["dataset"] = summary_df.apply(_label_row, axis=1)
        cols_order = ["dataset", "heuristic_merge", "heuristic_reject",
                      "human_merge", "human_reject", "human_ignore", "unverified"]
        avail = [c for c in cols_order if c in summary_df.columns]
        st.dataframe(summary_df[avail], hide_index=True, use_container_width=True)

        num_cols = [c for c in avail if c != "dataset"]
        if num_cols:
            totals = summary_df[num_cols].sum().to_dict()
            totals_str = "  |  ".join(f"{k}: {int(v)}" for k, v in totals.items())
            st.caption(f"Totals: {totals_str}")

    st.divider()
    st.subheader("Export")
    if st.button("Export Labeled Pairs as CSV", key="lv_export_btn"):
        df = load_training_data()
        if df.empty:
            st.warning("No labeled data to export.")
        else:
            export_df = df[df["label"].notna()].copy()
            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                label="Download CSV",
                data=buf.getvalue(),
                file_name="merge_training_labels.csv",
                mime="text/csv",
                key="lv_download_csv",
            )
            st.caption(f"Exporting {len(export_df)} labeled pairs (ignored pairs excluded).")
