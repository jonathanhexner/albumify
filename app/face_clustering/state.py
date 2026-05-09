"""Async worker, session state initialisation, and state-mutation helpers."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import streamlit as st

from face_cluster.profile_store import ProfileStore


# ---------------------------------------------------------------------------
# Logging: QueueHandler routes face_cluster.* log records to the UI
# ---------------------------------------------------------------------------

class _QueueHandler(logging.Handler):
    """Puts formatted log records into a queue for the UI to consume."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(levelname)-8s %(name)s  %(message)s"))

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AsyncWorker
# ---------------------------------------------------------------------------

class _AsyncState:
    """Shared state between background thread and UI thread."""
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"

    def __init__(self):
        self.status     = self.IDLE
        self.result     = None
        self.error      = None
        self.started_at: Optional[float] = None
        self.ended_at:   Optional[float] = None
        self.log_q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self): return self.status == self.RUNNING
    @property
    def is_done(self):    return self.status == self.DONE
    @property
    def has_error(self):  return self.status == self.ERROR

    def elapsed_s(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.ended_at or time.time()
        return end - self.started_at

    def start(self, fn, *args, **kwargs):
        if self.is_running:
            return
        self.status     = self.RUNNING
        self.result     = None
        self.error      = None
        self.started_at = time.time()
        self.ended_at   = None
        while not self.log_q.empty():
            try: self.log_q.get_nowait()
            except queue.Empty: break

        handler = _QueueHandler(self.log_q)
        fc_logger = logging.getLogger("face_cluster")
        fc_logger.addHandler(handler)
        fc_logger.setLevel(logging.DEBUG)

        def _run():
            try:
                self.result = fn(*args, **kwargs)
                self.status = self.DONE
            except Exception as exc:
                self.error = str(exc)
                self.status = self.ERROR
                import traceback
                self.log_q.put_nowait("ERROR: " + traceback.format_exc())
            finally:
                self.ended_at = time.time()
                fc_logger.removeHandler(handler)

        self._thread = threading.Thread(target=_run, name="_run", daemon=True)
        self._thread.start()

    def drain_logs(self) -> list[str]:
        lines = []
        while True:
            try:
                lines.append(self.log_q.get_nowait())
            except queue.Empty:
                break
        return lines


# ---------------------------------------------------------------------------
# Recluster profile parameter keys
# ---------------------------------------------------------------------------
_RC_PARAM_KEYS: frozenset[str] = frozenset({
    "rc_K", "rc_dist", "rc_min_cluster", "rc_N_exemplars", "rc_d10_thresh", "rc_suppression",
    "rc_split", "rc_merge", "rc_attach",
    "rc_merge_use_adaptive", "rc_merge_exemplar_pct", "rc_merge_global_pct",
    "rc_merge_alpha", "rc_merge_beta", "rc_merge_candidate", "rc_merge_exemplar_thresh",
    "rc_merge_support_frac", "rc_merge_support_min", "rc_merge_margin", "rc_merge_diameter",
    "rc_merge_use_cross_gate", "rc_merge_cross_thresh", "rc_merge_cross_max_size", "rc_merge_support_unique",
})


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "pipeline_result":          None,
        "pipeline_worker":          None,
        "recluster_worker":         None,
        "pipeline_log":             [],
        "active_run_dir":           None,
        "overview_worker":          None,
        "merged_overview_worker":   None,
        "cluster_worker":           None,
        "merged_cluster_worker":    None,
        "cluster_debug_worker":     None,
        "face_worker":              None,
        "merge_analysis_worker":    None,
        "merge_pair_features":      None,
        "merge_approval_decisions": {},
        "merge_decision_sources":   {},
        "merge_approval_result":    None,
        "merge_round":              1,
        "pending_candidates":       [],
        "pending_decisions":        {},
        "merge_gallery_filter":     "All",
        "merge_gallery_sort":       "exemplar_dist",
        "merge_gallery_page":       0,
        "merge_iter_filter":        "Latest",
        "selected_cluster":         None,
        "selected_merged_cluster":  None,
        "fm_preview":               None,
        "fm_pair":                  None,
        "selected_face":            None,
        "manifest_cache":           None,
        "faces_df_cache":           None,
        "last_image_dir":           "",
        "last_output_dir":          "",
        "ml_train_worker":          None,
        "ml_train_result":          None,
        "recluster_promoted_dir":   None,
        "remerge_worker":           None,
        "remerge_with_exemplars":   False,
        "active_session":           None,
        "active_chain_id":          None,
        "session_root":             None,
        "merge_gallery_view_mode":  "by_iteration",
        "merge_group_view":         True,
        "merge_group_page":         0,
        "merge_group_filter":       "All",
        "merge_smart_prefilled":    False,
        "merge_mode":               "heuristic",
        "ml_predict_worker":        None,
        "ml_merge_view":            None,
        "ml_prefilled":             False,
        "ml_pair_features":         {},
        "ml_model_payload":         None,
        # spec-013: source album identity for the currently loaded run
        "current_source_album":     None,
        # spec-015: face detail popup
        "face_popup_id":            None,
        "face_popup_cache":         {},
        # cluster detail popup
        "cluster_popup_id":         None,
        "cluster_popup_cache":      {},
        # spec-017: label verification tab
        "lv_dataset":               "Google_Germany",
        "lv_threshold":             0.45,
        "lv_worker":                None,
        "lv_data":                  None,
        "lv_decisions":             {},
        "lv_dirty":                 set(),
        "lv_crop_cache":            {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _profile_defaults = ProfileStore().load("default")
    for k, v in _profile_defaults.items():
        if k not in st.session_state and k in _RC_PARAM_KEYS:
            st.session_state[k] = v

    # One-time startup: purge stale run history entries
    if "_db_purged" not in st.session_state:
        from face_cluster.run_history_db import purge_stale_runs
        purge_stale_runs()
        st.session_state._db_purged = True


def _collect_rc_params() -> dict:
    return {k: st.session_state[k] for k in _RC_PARAM_KEYS if k in st.session_state}


def _invalidate_run_caches():
    """Clear all per-run cached state when a new run is loaded."""
    st.session_state.overview_worker        = None
    st.session_state.merged_overview_worker = None
    st.session_state.cluster_worker         = None
    st.session_state.merged_cluster_worker  = None
    st.session_state.cluster_debug_worker   = None
    st.session_state.face_worker            = None
    st.session_state.merge_analysis_worker  = None
    st.session_state.merge_pair_features    = None
    st.session_state.merge_approval_decisions = {}
    st.session_state.merge_decision_sources  = {}
    st.session_state.ml_predict_worker      = None
    st.session_state.ml_merge_view          = None
    st.session_state.ml_prefilled           = False
    st.session_state.ml_pair_features       = {}
    st.session_state.ml_model_payload       = None
    st.session_state.merge_approval_result  = None
    st.session_state.merge_round            = 1
    st.session_state.pending_candidates     = []
    st.session_state.pending_decisions      = {}
    st.session_state.merge_gallery_filter   = "All"
    st.session_state.merge_gallery_sort     = "exemplar_dist"
    st.session_state.merge_gallery_page     = 0
    st.session_state.merge_iter_filter      = "Latest"
    st.session_state.merge_group_page       = 0
    st.session_state.merge_group_filter     = "All"
    st.session_state.merge_gallery_view_mode = "by_iteration"
    st.session_state.merge_smart_prefilled  = False
    st.session_state.selected_cluster       = None
    st.session_state.selected_merged_cluster = None
    st.session_state.selected_face          = None
    st.session_state.manifest_cache         = None
    st.session_state.faces_df_cache         = None
    st.session_state.face_popup_id          = None
    st.session_state.face_popup_cache       = {}
    st.session_state.cluster_popup_id       = None
    st.session_state.cluster_popup_cache    = {}
    _stale_suffixes = ("_selector", "_tab_selection", "_cluster_select")
    _gallery_prefixes = ("base_gallery_", "merged_gallery_")
    for key in [k for k in st.session_state
                if k.endswith(_stale_suffixes) or k.startswith(_gallery_prefixes)]:
        del st.session_state[key]


def _prefill_approval_decisions(result) -> None:
    """Restore saved merge decisions from a previous session (if any)."""
    if result.merge_decisions:
        decisions: dict = {}
        for d in result.merge_decisions:
            key = (min(d["cluster_a"], d["cluster_b"]),
                   max(d["cluster_a"], d["cluster_b"]))
            decisions[key] = d["decision"]
        st.session_state.merge_approval_decisions = decisions
        st.session_state.merge_decision_sources = {k: "human" for k in decisions}
