"""Run-related rendering helpers: run listing, log rendering, stage plan, run files panel."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from face_cluster import run_history_db

try:
    from constants import _RUN_FILE_DEFS, _LOG_LIVE_LINES, _STAGE_ORDER, _STATUS_ICON
except ImportError:
    from app.face_clustering.constants import _RUN_FILE_DEFS, _LOG_LIVE_LINES, _STAGE_ORDER, _STATUS_ICON


def _list_available_runs(complete_only: bool = False) -> list[dict]:
    rows   = run_history_db.list_actions(types=["pipeline_run", "recluster", "manual_merge", "remerge"])
    result = []
    for r in rows:
        if complete_only and r.get("status") != "complete":
            continue
        output_dir  = r.get("output_dir") or ""
        _type_label = {
            "pipeline_run": "Full Run",
            "recluster":    "Recluster",
            "manual_merge": "Manual Merge",
            "remerge":      "Remerge",
        }.get(r.get("action_type", ""), r.get("action_type", "?"))
        result.append({
            "run_id":        r.get("run_id") or Path(output_dir).name,
            "output_folder": Path(output_dir).name,
            "album":         r.get("album") or "",
            "started":       (r.get("started_at") or "")[:19],
            "type":          _type_label,
            "status":        r.get("status", "?"),
            "faces":         r.get("n_faces"),
            "clusters":      r.get("n_clusters"),
            "noise":         r.get("n_noise"),
            "duration_s":    r.get("duration_s"),
            "error":         r.get("error"),
            "log_file":      r.get("log_file"),
            "_dir":          output_dir,
            "_db_id":        r.get("id"),
        })
    return result


def _latest_log_file(run_dir: Path) -> Optional[Path]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return None
    files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _render_live_log(log_lines: list[str]):
    tail = log_lines[-_LOG_LIVE_LINES:]
    st.code("\n".join(tail) if tail else "(no log output yet)", language=None)


def _render_log_expander(log_lines: list[str], label: str = "Log"):
    if log_lines:
        with st.expander(label, expanded=True):
            st.code("\n".join(log_lines[-200:]), language=None)


def _render_stage_plan(run_dir: Optional[Path]):
    if run_dir is None:
        return
    run_json = run_dir / "pipeline_run.json"
    if not run_json.exists():
        st.caption("Waiting for pipeline to write stage data...")
        return
    try:
        with open(run_json, encoding="utf-8") as f:
            rec = json.load(f)
    except Exception:
        return
    stages = rec.get("stages", {})
    rows   = []
    for name in _STAGE_ORDER:
        if name not in stages:
            continue
        s           = stages[name]
        status      = s.get("status", "pending")
        started     = s.get("started_at", "")
        started_fmt = started[11:19] if len(started) >= 19 else ""
        elapsed     = s.get("elapsed_s")
        elapsed_fmt = f"{elapsed:.1f}s" if elapsed is not None else ("..." if status == "running" else "")
        rows.append({
            "Stage":   name,
            "Status":  _STATUS_ICON.get(status, status),
            "Started": started_fmt,
            "Elapsed": elapsed_fmt,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_run_files_panel(run_dir: Path, key_prefix: str = ""):
    st.markdown(f"**Run directory:** `{run_dir.resolve()}`")
    rows = []
    for fd in _RUN_FILE_DEFS:
        p          = run_dir / fd["file"]
        exists     = p.exists()
        size_str   = ""
        items_str  = ""
        if exists:
            if p.is_file():
                sz       = p.stat().st_size
                size_str = f"{sz / 1024:.1f} KB" if sz < 1_000_000 else f"{sz / 1_000_000:.1f} MB"
                if fd["file"].endswith(".csv"):
                    with open(p, encoding="utf-8") as fh:
                        items_str = f"{sum(1 for _ in fh) - 1} rows"
                elif fd["file"] == "crop_manifest.json":
                    with open(p, encoding="utf-8") as fh:
                        items_str = f"{len(json.load(fh))} entries"
                elif fd["file"].endswith(".npy"):
                    arr       = np.load(p)
                    items_str = f"shape {arr.shape}"
            elif p.is_dir():
                files     = list(p.rglob("*"))
                n_files   = sum(1 for f in files if f.is_file())
                total_sz  = sum(f.stat().st_size for f in files if f.is_file())
                items_str = f"{n_files} files"
                size_str  = f"{total_sz / 1_000_000:.1f} MB"
        rows.append({
            "File":        fd["file"],
            "Format":      fd["format"],
            "Status":      "OK" if exists else "MISSING",
            "Size":        size_str,
            "Items":       items_str,
            "Description": fd["description"],
            "Schema":      fd["schema"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    col1, col2 = st.columns(2)
    run_json   = run_dir / "pipeline_run.json"
    if run_json.exists():
        with col1:
            st.download_button(
                "Download pipeline_run.json",
                run_json.read_bytes(),
                file_name=f"{run_dir.name}_pipeline_run.json",
                mime="application/json",
                key=f"{key_prefix}dl_run_json",
            )
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("run_*.log"), reverse=True)
        if log_files:
            with col2:
                st.download_button(
                    f"Download {log_files[0].name}",
                    log_files[0].read_bytes(),
                    file_name=f"{run_dir.name}_{log_files[0].name}",
                    mime="text/plain",
                    key=f"{key_prefix}dl_log",
                )
