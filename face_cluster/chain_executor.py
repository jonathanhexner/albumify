"""Chain step execution engine.

Executes one or more steps in a clustering chain, chaining output directories.

Step type -> action mapping:
  cluster   → FaceClusteringPipeline.recluster() on base embeddings (step.params)
  recluster → FaceClusteringPipeline.recluster() on previous step's output
  merge     → load labels from labels.json, snapshot + remerge
  remerge   → remerge without new labels (catch centroid cascades)
  split     → placeholder (not yet implemented)

Usage (branching):
    executor = ChainExecutor()
    output_dir = executor.execute_chain_from(session, "chain_02", from_step=0)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from face_cluster.config import PipelineConfig
from face_cluster.pipeline import FaceClusteringPipeline
from face_cluster.session_manager import Session, Step, SessionManager

logger = logging.getLogger(__name__)

_LABELS_FILE = "labels.json"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    session: Session
    chain_id: str
    step: Step
    input_dir: Path
    output_dir: Path
    on_progress: Optional[object] = None


class ChainExecutor:
    """Re-executes steps in a chain.  Thread-safe (stateless)."""

    def __init__(self):
        self._mgr = SessionManager()

    def execute_chain_from(
        self,
        session: Session,
        chain_id: str,
        from_step: int,
        on_progress=None,
    ) -> Path:
        """Execute steps from_step..end of the chain.

        Creates each step's output folder, runs the appropriate pipeline stage,
        and updates result_summary in session.json after each step.

        Returns the final step's output directory.
        """
        chain = self._mgr.get_chain(session, chain_id)
        steps = sorted(
            (s for s in chain.steps if s.step_index >= from_step),
            key=lambda s: s.step_index,
        )

        current_dir = self._resolve_input_dir(session, chain_id, from_step)
        logger.info("execute_chain_from: chain=%s from_step=%d input=%s", chain_id, from_step, current_dir)

        for step in steps:
            output_dir = session.session_root / chain.folder / step.folder
            output_dir.mkdir(parents=True, exist_ok=True)
            ctx = ExecutionContext(session, chain_id, step, current_dir, output_dir, on_progress)
            _dispatch(ctx)
            current_dir = output_dir
            logger.info("Step %s/%s done -> %s", chain_id, step.folder, output_dir)

        return current_dir

    # ------------------------------------------------------------------ private

    def _resolve_input_dir(self, session: Session, chain_id: str, from_step: int) -> Path:
        """Return the input directory for from_step.

        - If from_step == 0: use session base dir.
        - Else: use the output dir of step (from_step - 1).
        """
        if from_step == 0:
            return self._mgr.get_base_dir(session)
        chain = self._mgr.get_chain(session, chain_id)
        return self._mgr.get_step_output_dir(session, chain_id, from_step - 1)


# ---------------------------------------------------------------------------
# Step dispatchers
# ---------------------------------------------------------------------------

def _dispatch(ctx: ExecutionContext) -> None:
    _STEP_HANDLERS[ctx.step.type](ctx)


def _execute_cluster(ctx: ExecutionContext) -> None:
    params = ctx.step.params or {}
    config = PipelineConfig.recluster(
        ctx.input_dir,
        ctx.output_dir,
        merge_enabled=params.get("merge_enabled", False),
        K=params.get("K", 5),
        distance_threshold=params.get("distance_threshold", 0.35),
        min_cluster_size=params.get("min_cluster_size", 2),
        on_progress=ctx.on_progress,
    )
    FaceClusteringPipeline().run(config)


def _execute_recluster(ctx: ExecutionContext) -> None:
    _execute_cluster(ctx)


def _execute_merge(ctx: ExecutionContext) -> None:
    from face_cluster.loader import load_pipeline_result
    from face_cluster.manual_merge_snapshot import save_manual_merge_snapshot

    labels = _load_labels(ctx)
    approved = [tuple(p) for p in labels.get("approved", [])]
    rejected = [tuple(p) for p in labels.get("rejected", [])]

    prev_result = load_pipeline_result(ctx.input_dir)
    current_cr = prev_result.merged_cluster_result or prev_result.cluster_result

    snap_dir = ctx.output_dir / "_snap"
    save_manual_merge_snapshot(
        faces=prev_result.faces,
        merged_cluster_result=current_cr,
        approved_pairs=approved,
        rejected_pairs=rejected,
        config=PipelineConfig(),
        output_dir=snap_dir,
        parent_output_dir=ctx.input_dir,
    )

    cfg = ctx.step.params.get("merge_config", {})
    FaceClusteringPipeline().run(
        PipelineConfig.remerge(
            snap_dir,
            ctx.output_dir,
            merge_enabled=True,
            **cfg,
        )
    )


def _execute_remerge(ctx: ExecutionContext) -> None:
    cfg = ctx.step.params.get("merge_config", {})
    FaceClusteringPipeline().run(
        PipelineConfig.remerge(
            ctx.input_dir,
            ctx.output_dir,
            merge_enabled=True,
            **cfg,
        )
    )


def _execute_split(ctx: ExecutionContext) -> None:
    raise NotImplementedError("Split step type is not yet implemented")


_STEP_HANDLERS = {
    "cluster":   _execute_cluster,
    "recluster": _execute_recluster,
    "merge":     _execute_merge,
    "remerge":   _execute_remerge,
    "split":     _execute_split,
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _load_labels(ctx: ExecutionContext) -> dict:
    """Load labels.json for a merge step.

    Search order:
    1. ctx.output_dir / labels.json  (already materialized in this chain)
    2. Source chain folder / step.folder / labels.json  (branched from another chain)
    """
    own_labels = ctx.output_dir / _LABELS_FILE
    if own_labels.exists():
        return _read_json(own_labels)

    chain = ctx.session.chains
    target = next((c for c in chain if c.chain_id == ctx.chain_id), None)
    if target and target.branched_from:
        src_id = target.branched_from["chain_id"]
        src_chain = next((c for c in chain if c.chain_id == src_id), None)
        if src_chain:
            src_labels = ctx.session.session_root / src_chain.folder / ctx.step.folder / _LABELS_FILE
            if src_labels.exists():
                return _read_json(src_labels)

    logger.warning("No labels.json found for merge step %s/%s; using empty labels", ctx.chain_id, ctx.step.folder)
    return {"approved": [], "rejected": []}


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_step_labels(step_dir: Path, approved_pairs: list, rejected_pairs: list) -> None:
    """Write labels.json to a step folder (called by app after Apply+Remerge)."""
    data = {
        "approved": [list(p) for p in approved_pairs],
        "rejected": [list(p) for p in rejected_pairs],
    }
    with open(step_dir / _LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
