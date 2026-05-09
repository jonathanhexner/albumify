"""Tab 3: History — searchable, filterable run history."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from face_cluster import run_history_db
from face_cluster.loader import load_pipeline_result
from face_cluster.run_history import HistoryFilters, RunRow, search, distinct_albums, get_run_by_id

from state import _invalidate_run_caches
from run_panels import _render_run_files_panel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_run_log_viewer(log_file: Optional[str]) -> None:
    if not log_file:
        st.caption("No log file path recorded.")
        return
    p = Path(log_file)
    if not p.exists():
        st.warning(f"Log file not found: `{log_file}`")
        return
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    st.code("\n".join(lines[-200:]), language=None)


def _render_run_payload(db_id: Optional[int]) -> None:
    if db_id is None:
        return
    row = run_history_db.get_action(db_id)
    if row:
        st.json(json.loads(row.get("payload_json") or "{}"))


def _run_row_to_display(r: RunRow) -> dict:
    return {
        "_id":         r.id,
        "Album":       r.display_album,
        "Run name":    r.run_name or r.run_id or f"#{r.id}",
        "Kind":        r.run_kind or r.action_type,
        "Output":      (
            f"{Path(r.output_dir).parts[-2]}/{Path(r.output_dir).parts[-1]}"
            if r.output_dir and len(Path(r.output_dir).parts) >= 2
            else Path(r.output_dir).name if r.output_dir else ""
        ),
        "Parent":      str(r.parent_run_id) if r.parent_run_id else "",
        "Created":     (r.started_at or "")[:16].replace("T", " "),
        "Faces":       r.n_faces,
        "Core":        r.n_core,
        "Clusters":    r.n_clusters,
        "Status":      r.status,
        "Comment":     r.comment or "",
    }


def _render_run_config_from_json(prun: dict) -> None:
    """Show key thresholds from pipeline_run.json config dict."""
    cfg = prun.get("config") or prun.get("summary", {}).get("config") or {}
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


def _render_run_summary_section(run_dir: Path) -> None:
    """Phase 9: Run Summary panel consuming spec-012 outputs."""
    pipeline_run_path = run_dir / "pipeline_run.json"
    merge_meta_path   = run_dir / "merge_metadata.json"
    merge_log_path    = run_dir / "merge_log.json"

    if not pipeline_run_path.exists():
        st.caption("Data not available (pre-012 run)")
        return

    prun = json.loads(pipeline_run_path.read_text(encoding="utf-8"))
    _render_run_config_from_json(prun)
    merge_meta = json.loads(merge_meta_path.read_text()) if merge_meta_path.exists() else {}
    merge_log  = json.loads(merge_log_path.read_text())  if merge_log_path.exists()  else None

    # Quality funnel
    quality = prun.get("quality_summary") or prun.get("summary", {})
    if quality:
        st.write("**Quality funnel**")
        funnels = {k: v for k, v in quality.items() if isinstance(v, (int, float))}
        if funnels:
            fc = st.columns(len(funnels))
            for i, (k, v) in enumerate(funnels.items()):
                fc[i].metric(k, v)

    # Cluster counts
    summary = prun.get("summary", {})
    if summary:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Faces",           summary.get("n_faces", "-"))
        sc2.metric("Clusters (base)", summary.get("n_clusters", "-"))
        sc3.metric("Clusters (merged)", summary.get("n_clusters_merged", "-"))

    # Merge candidates
    if merge_log is not None:
        if len(merge_log) == 0:
            threshold = merge_meta.get("merge_candidate_threshold", "?")
            st.info(f"0 candidates at threshold {threshold}")
        else:
            st.caption(f"{len(merge_log)} merge(s) applied")

    # Stage timeline
    stages = prun.get("stages", {})
    if stages:
        st.write("**Stage durations**")
        timing = [
            {"Stage": name, "Duration (s)": info.get("elapsed_s", "-")}
            for name, info in stages.items()
        ]
        st.dataframe(pd.DataFrame(timing), hide_index=True, use_container_width=False)


def _render_run_header(row: RunRow) -> None:
    """Phase 10: run header with album, run name, parent link, config diff."""
    from face_cluster.config_diff import compute as _config_diff

    album   = row.display_album
    name    = row.run_name or row.run_id or f"#{row.id}"
    st.markdown(f"**Album:** `{album}` &nbsp;|&nbsp; **Run:** `{name}`")

    if row.parent_run_id:
        parent = get_run_by_id(row.parent_run_id)
        parent_label = parent.run_name or parent.run_id or f"#{parent.id}" if parent else str(row.parent_run_id)
        st.caption(f"Parent: {parent_label}")

        with st.expander("Config Delta (vs parent)", expanded=False):
            child_cfg  = row.config
            parent_cfg = parent.config if parent else {}
            deltas     = _config_diff(parent_cfg, child_cfg)
            if not deltas:
                st.caption("No config differences from parent.")
            else:
                df = pd.DataFrame([
                    {"field": d.field, "parent": d.parent_value, "child": d.child_value}
                    for d in deltas
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.caption("Base run")

    comment_val = row.comment or ""
    new_comment = st.text_input(
        "Comment", value=comment_val, max_chars=2048,
        placeholder="Add a note about this run…",
        key=f"run_header_comment_{row.id}",
    )
    if new_comment != comment_val:
        run_history_db.update_comment(row.id, new_comment)


# ---------------------------------------------------------------------------
# Main tab renderer
# ---------------------------------------------------------------------------

def render_history_tab():
    st.header("History")

    # ---- Filter bar --------------------------------------------------------
    albums = distinct_albums()
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        album_filter = st.selectbox(
            "Album", ["(all)"] + albums, key="hist_album_filter"
        )
    with fc2:
        date_range = st.date_input(
            "Date range", value=[], key="hist_date_range"
        )
    with fc3:
        text_filter = st.text_input(
            "Search (album / run name / comment)",
            key="hist_text_filter",
        )

    filters = HistoryFilters(
        album  = album_filter if album_filter != "(all)" else None,
        date_from = date_range[0] if isinstance(date_range, (list, tuple)) and len(date_range) >= 1 else None,
        date_to   = date_range[1] if isinstance(date_range, (list, tuple)) and len(date_range) >= 2 else None,
        text   = text_filter.strip() or None,
    )
    run_rows = search(filters)

    # ---- Main table --------------------------------------------------------
    st.subheader(f"Pipeline Runs ({len(run_rows)})")
    if not run_rows:
        st.info("No runs match the current filters.")
    else:
        display = [_run_row_to_display(r) for r in run_rows]
        df      = pd.DataFrame(display)
        visible = ["Album", "Run name", "Kind", "Output", "Created",
                   "Faces", "Core", "Clusters", "Status", "Comment"]
        event = st.dataframe(
            df[visible],
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="hist_run_table",
        )
        sel_indices = event.selection.get("rows", []) if hasattr(event, "selection") else []
        selected_db_id = df.iloc[sel_indices[0]]["_id"] if sel_indices else None

        # ---- Comment editor in table (data_editor) --------------------------
        with st.expander("Edit comments inline", expanded=False):
            edit_df = df[["_id", "Album", "Run name", "Comment"]].copy()
            edited  = st.data_editor(
                edit_df, hide_index=True, use_container_width=True,
                disabled=["_id", "Album", "Run name"],
                key="hist_comment_editor",
            )
            if st.button("Save comments", key="hist_save_comments"):
                for _, erow in edited.iterrows():
                    orig = next((r for r in run_rows if r.id == erow["_id"]), None)
                    if orig and erow["Comment"] != (orig.comment or ""):
                        if len(str(erow["Comment"])) > 2048:
                            st.error(f"Comment for run #{erow['_id']} exceeds 2048 characters.")
                        else:
                            run_history_db.update_comment(erow["_id"], str(erow["Comment"]))
                st.success("Comments saved.")
                st.rerun()

        # ---- Selected run detail -------------------------------------------
        if selected_db_id is not None:
            row = get_run_by_id(int(selected_db_id))
            if row is not None:
                st.divider()
                _render_run_header(row)
                output_dir = Path(row.output_dir) if row.output_dir else None

                if output_dir:
                    with st.expander("Run Summary", expanded=True):
                        _render_run_summary_section(output_dir)

                col_log, col_payload = st.columns(2)
                with col_log:
                    with st.expander("View Log", expanded=False):
                        _render_run_log_viewer(row.log_file)
                with col_payload:
                    with st.expander("Full Payload (debug)", expanded=False):
                        _render_run_payload(row.id)

                if output_dir:
                    with st.expander("Data Files", expanded=True):
                        if output_dir.exists():
                            _render_run_files_panel(output_dir, key_prefix="hist_tab_")
                        else:
                            st.warning(f"Output directory not found: `{output_dir}`")

                st.divider()
                _render_load_button(row, output_dir)

    # ---- Other actions -----------------------------------------------------
    st.divider()
    st.subheader("Recent Actions")
    other_actions = run_history_db.list_actions(
        types=["merge_apply", "profile_save", "ml_train", "model_load"], limit=100
    )
    if not other_actions:
        st.info("No other actions recorded yet.")
        return

    rows_display = [_format_action_row(a) for a in other_actions]
    df_actions   = pd.DataFrame(rows_display)
    st.dataframe(
        df_actions[["when", "type", "status", "duration", "details", "error"]],
        hide_index=True, use_container_width=True,
    )
    sel_action = st.selectbox(
        "Inspect action payload",
        [r["_db_id"] for r in rows_display],
        format_func=lambda i: next(
            f"{r['when']}  {r['type']}" for r in rows_display if r["_db_id"] == i
        ),
        key="hist_action_selector",
    )
    with st.expander("Full Payload (debug)", expanded=False):
        _render_run_payload(sel_action)


def _render_load_button(row: RunRow, output_dir: Optional[Path]) -> None:
    if output_dir is None:
        return
    has_faces_csv = output_dir.exists() and (output_dir / "faces.csv").exists()
    is_incomplete = row.status != "complete" or not has_faces_csv
    already_loaded = (
        st.session_state.pipeline_result is not None
        and output_dir.exists()
        and Path(st.session_state.pipeline_result.output_dir) == output_dir
    )
    if already_loaded:
        st.success(f"Run `{row.run_name or row.run_id}` is currently loaded.")
        return
    if is_incomplete:
        missing = [f for f in ["faces.csv", "clusters.csv", "embeddings.npy"]
                   if not (output_dir / f).exists()]
        st.warning(
            f"Run is incomplete (status: `{row.status}`). "
            + (f"Missing: {', '.join(missing)}." if missing else "")
        )
        st.button("Load into analysis tabs", type="primary", disabled=True,
                  help="Cannot load — required files are missing")
        return
    if st.button("Load into analysis tabs", type="primary", key=f"hist_load_{row.id}"):
        result = load_pipeline_result(output_dir)
        st.session_state.pipeline_result      = result
        st.session_state.current_source_album = row.display_album
        _invalidate_run_caches()
        st.success("Loaded. Switch to **Clusters (Base)** to analyse.")
        st.rerun()


def _format_action_row(a: dict) -> dict:
    payload = json.loads(a["payload_json"]) if a.get("payload_json") else {}
    details = {
        "merge_apply":  lambda p: (f"round={p.get('round')}  "
                                   f"approved={p.get('n_approved')}  "
                                   f"clusters_before={p.get('clusters_before')} -> {a.get('n_clusters')}"),
        "profile_save": lambda p: f"profile={p.get('profile_name')}",
        "ml_train":     lambda p: f"model={p.get('model_type')}  acc={p.get('accuracy', '')}",
        "model_load":   lambda p: f"model={p.get('model_name')}",
    }.get(a["action_type"], lambda _: "")
    return {
        "when":     (a.get("started_at") or "")[:19],
        "type":     a["action_type"],
        "status":   a["status"],
        "duration": f"{a['duration_s']:.1f}s" if a.get("duration_s") else "-",
        "details":  details(payload),
        "error":    a.get("error") or "",
        "_db_id":   a["id"],
    }
