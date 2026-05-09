"""Session management helpers and sidebar panel."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from face_cluster.session_manager import SessionManager, Session
from face_cluster.chain_executor import ChainExecutor, write_step_labels
from face_cluster.pipeline import PipelineResult

_session_manager = SessionManager()
_chain_executor  = ChainExecutor()


def _create_session_from_result(result: PipelineResult) -> None:
    session_root = st.session_state.get("session_root")
    if session_root is None:
        return
    session_root = Path(session_root)
    source_album = result.summary.get("source_album", "")
    base_summary = {
        "params_summary": f"{result.summary.get('n_images', 0)} images, {result.summary.get('n_faces', 0)} faces",
        "n_faces":  result.summary.get("n_faces", 0),
        "n_images": result.summary.get("n_images", 0),
    }
    session = _session_manager.create(session_root, source_album, base_summary)
    _session_manager.create_chain(session)
    st.session_state.active_session  = session
    st.session_state.active_chain_id = session.active_chain


def _get_active_session() -> Optional[Session]:
    return st.session_state.get("active_session")


def _get_active_chain_id() -> Optional[str]:
    return st.session_state.get("active_chain_id")


def _save_pending_labels_to_session(approved_pairs: list, rejected_pairs: list) -> None:
    session  = _get_active_session()
    chain_id = _get_active_chain_id()
    if session is None or chain_id is None:
        return
    labels = (
        [{"pair": list(p), "decision": "approve"} for p in approved_pairs]
        + [{"pair": list(p), "decision": "reject"}  for p in rejected_pairs]
    )
    _session_manager.save_pending_labels(session, chain_id, labels)


def _collect_all_merge_labels(current_approved: list, current_rejected: list):
    session  = _get_active_session()
    chain_id = _get_active_chain_id()
    pending = _session_manager.consume_pending_labels(session, chain_id) if (session and chain_id) else []
    extra_approved = [tuple(d["pair"]) for d in pending if d.get("decision") == "approve"]
    extra_rejected  = [tuple(d["pair"]) for d in pending if d.get("decision") == "reject"]
    return list(current_approved) + extra_approved, list(current_rejected) + extra_rejected


def _materialize_merge_step(
    remerge_dir: Path,
    approved_pairs: list,
    rejected_pairs: list,
    sources: Optional[dict] = None,
    n_undecided: int = 0,
    n_training_samples_saved: int = 0,
) -> None:
    session  = _get_active_session()
    chain_id = _get_active_chain_id()
    if session is None or chain_id is None:
        return
    src = sources or {}
    n_approved_human = sum(1 for k in approved_pairs if src.get(k) == "human")
    n_approved_ml    = len(approved_pairs) - n_approved_human
    n_rejected_human = sum(1 for k in rejected_pairs if src.get(k) == "human")
    n_rejected_ml    = len(rejected_pairs) - n_rejected_human
    _, step_dir = _session_manager.append_step(
        session, chain_id, "merge",
        {
            "n_pairs":                  len(approved_pairs),
            "merge_dir":                str(remerge_dir),
            "merge_mode":               st.session_state.get("merge_mode", "heuristic"),
            "model_name":               None,
            "ml_threshold":             None,
            "n_approved_human":         n_approved_human,
            "n_approved_ml":            n_approved_ml,
            "n_rejected_human":         n_rejected_human,
            "n_rejected_ml":            n_rejected_ml,
            "n_undecided":              n_undecided,
            "n_training_samples_saved": n_training_samples_saved,
        },
    )
    write_step_labels(step_dir, approved_pairs, rejected_pairs)


def _append_recluster_step_to_session(result: PipelineResult) -> None:
    session  = _get_active_session()
    chain_id = _get_active_chain_id()
    if session is None or chain_id is None:
        return
    cr      = result.cluster_result
    summary = f"{cr.n_clusters} clusters, {cr.n_noise} noise"
    step, _ = _session_manager.append_step(session, chain_id, "recluster", {})
    _session_manager.update_step_result(session, chain_id, step.step_index, summary)


def _render_pending_labels_indicator() -> None:
    session  = _get_active_session()
    chain_id = _get_active_chain_id()
    if session is None or chain_id is None:
        return
    pending = _session_manager.get_pending_labels(session, chain_id)
    if pending:
        st.info(f"{len(pending)} pending label(s) accumulated — click 'Apply + Remerge' to materialize.")


def _render_session_panel() -> None:
    session: Optional[Session] = _get_active_session()
    if session is None:
        return
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Session:** `{session.session_id}`")
        base = session.base
        if base:
            st.caption(f"Base: {base.get('n_images', 0)} images, {base.get('n_faces', 0)} faces")
        chain_id = _get_active_chain_id()
        if chain_id is None:
            return
        chain = _session_manager.get_chain(session, chain_id)
        st.markdown(f"**Chain:** `{chain_id}`")
        if not chain.steps:
            st.caption("No steps yet — initial state is base/")
        else:
            for step in sorted(chain.steps, key=lambda s: s.step_index):
                summary = step.result_summary or "..."
                st.markdown(f"  `[{step.step_index}]` **{step.type}** — {summary}")
        pending = _session_manager.get_pending_labels(session, chain_id)
        if pending:
            st.caption(f"__{len(pending)} pending label(s)__")
        if len(session.chains) > 1:
            chain_options = [c.chain_id for c in session.chains]
            sel = st.selectbox(
                "Switch chain", chain_options,
                index=chain_options.index(chain_id),
                key="sidebar_chain_selector",
            )
            if sel != chain_id:
                if pending:
                    st.warning(f"{len(pending)} pending labels not yet remerged.")
                else:
                    _session_manager.set_active_chain(session, sel)
                    st.session_state.active_chain_id = sel
                    st.rerun()


def _try_restore_session() -> None:
    if st.session_state.active_session is not None:
        return
    session_root = st.session_state.get("session_root")
    if not session_root:
        return
    session = _session_manager.load(Path(session_root))
    if session is None:
        return
    st.session_state.active_session  = session
    st.session_state.active_chain_id = session.active_chain
