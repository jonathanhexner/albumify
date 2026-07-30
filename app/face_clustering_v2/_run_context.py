"""spec-066 — shared run-resolution + service caching for v2 analysis tabs.

Every analysis tab (Cluster Analysis, Face Analysis, Gallery, ...) answers
the same two questions: "which run dir am I looking at?" and "give me a
cached ClusterAnalysisService for it". This module owns both so the tabs
stay thin. Not a tab itself — free to construct the Repository (the
layering arch test scans only ``tabs/*.py``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, TypeVar

import streamlit as st

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.views.cluster_analysis import ClusterAnalysisService

logger = logging.getLogger(__name__)

# Resolution priority: History "Load" key, then the last fresh run, then the
# legacy History key. Mirrors cluster_analysis_tab._resolve_current_run_dir.
_RUN_DIR_KEYS = ("current_run_dir", "v2_last_run_dir", "active_run_dir")


def resolve_run_dir() -> Optional[Path]:
    """First session-state run dir that has a readable face_clustering.db."""
    for key in _RUN_DIR_KEYS:
        v = st.session_state.get(key)
        if v and Path(v).is_dir() and (Path(v) / "face_clustering.db").is_file():
            return Path(v)
    return None


_T = TypeVar("_T")


def cached_service(
    run_dir: Path, factory: Callable[[ClusterAnalysisRepository], _T], *, cache_prefix: str,
) -> Optional[_T]:
    """Build (or return cached) ``factory(repo)`` service for ``run_dir``.

    Generic version of :func:`cached_cluster_service` for tabs whose service is
    not ``ClusterAnalysisService`` (e.g. Merged Clusters / Images). Returns None
    and surfaces ``st.error`` if the run can't be opened. Cached per
    (prefix, run_dir) so each tab keeps its own slot.
    """
    key = f"{cache_prefix}::{run_dir}"
    cached = st.session_state.get(key)
    if cached is not None:
        return cached
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s: repo construction failed for %s", cache_prefix, run_dir)
        st.error(f"Cannot open run at `{run_dir}`: {exc}")
        return None
    service = factory(repo)
    st.session_state[key] = service
    return service


def cached_cluster_service(run_dir: Path, *, cache_prefix: str) -> Optional[ClusterAnalysisService]:
    """Build (or return cached) ClusterAnalysisService for ``run_dir``.

    Returns None and surfaces ``st.error`` if the Repository can't open the
    run (deleted dir, schema mismatch). Cached per (prefix, run_dir) so each
    tab keeps its own slot without colliding.
    """
    key = f"{cache_prefix}::{run_dir}"
    cached = st.session_state.get(key)
    if cached is not None:
        return cached
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s: repo construction failed for %s", cache_prefix, run_dir)
        st.error(f"Cannot open run at `{run_dir}`: {exc}")
        return None
    service = ClusterAnalysisService(repo)
    st.session_state[key] = service
    return service
