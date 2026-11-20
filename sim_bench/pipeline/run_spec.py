"""What a run runs, built once (spec-116 R2): the steps and their parameters, whichever machine runs them.

``RunSpec.build`` is the ONE config interpreter for identity clustering (spec-079): the user-facing
``people_clustering`` block goes through the typed ``FCParams`` and is spread to the unified clustering steps.
The result is plain JSON, so the same spec reaches this PC's child process and the L4 worker.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
from face_cluster.fc_params import FCParams

logger = logging.getLogger(__name__)

__all__ = ["RunSpec", "spread_clustering_config"]


def spread_clustering_config(steps: List[str], step_configs: Dict[str, dict]) -> Dict[str, dict]:
    """``people_clustering`` (profiles saved before spec-113: ``cluster_people``) -> the unified steps' configs.

    Uses ``FCParams.to_step_configs()``, the same interpretation the face-clustering app uses. A config already
    set on a single unified step wins over the spread. No-op when the unified steps are not in the run.
    """
    if not any(s in steps for s in UNIFIED_CLUSTERING_STEPS):
        return step_configs

    raw = dict(step_configs.get("people_clustering") or {})
    legacy = step_configs.get("cluster_people") or {}
    if legacy:
        # Saved profiles / user overrides from before spec-113: their values win over the yaml defaults,
        # exactly as they did under the old key.
        logger.warning("clustering config: reading the pre-spec-113 key 'cluster_people'; "
                       "re-save the profile to store it as 'people_clustering'")
        raw.update(legacy)
    allowed = set(FCParams.model_fields)
    fcp = FCParams(**{k: v for k, v in raw.items() if k in allowed})
    dropped = sorted(set(raw) - allowed)
    if dropped:
        logger.info("clustering config: %d keys not on FCParams ignored: %s", len(dropped), dropped)
    for name, cfg in fcp.to_step_configs().items():
        step_configs[name] = {**cfg, **step_configs.get(name, {})}
    # spec-088 / SIGHTING-107: the "Export for analysis" toggle is an IO concern, not a clustering knob. Route it
    # to the analysis-export step, along with the clustering params it serializes into the export.
    if raw.get("export_for_analysis"):
        step_configs["face_cluster_analysis_export"] = {"export_for_analysis": True, **fcp.model_dump()}
    # The clustering block is a config source, not a step: drop it so the validator does not look for that step.
    step_configs.pop("people_clustering", None)
    step_configs.pop("cluster_people", None)
    logger.info("clustering config: spread FCParams to %d unified steps (K=%s yaw_max=%s blur_min=%s "
                "merge_enabled=%s)", len(UNIFIED_CLUSTERING_STEPS), fcp.K, fcp.yaw_max, fcp.blur_min,
                fcp.merge_enabled)
    return step_configs


@dataclass
class RunSpec:
    """The steps + parameters of one run, as plain JSON."""

    steps: List[str]
    step_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fail_fast: bool = True
    album_name: Optional[str] = None     # names the face-clustering export folder (spec-088)

    @classmethod
    def build(cls, steps: List[str], step_configs: Optional[Dict[str, dict]] = None, *,
              fail_fast: bool = True, album_name: Optional[str] = None) -> "RunSpec":
        steps = list(steps)
        return cls(steps=steps, step_configs=spread_clustering_config(steps, dict(step_configs or {})),
                   fail_fast=fail_fast, album_name=album_name)

    def to_pipeline_spec(self):
        from sim_bench.pipeline.spec import PipelineSpec   # lazy: registers every step
        return PipelineSpec(steps=list(self.steps), step_configs=dict(self.step_configs))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSpec":
        return cls(steps=list(d["steps"]), step_configs=dict(d.get("step_configs") or {}),
                   fail_fast=bool(d.get("fail_fast", True)), album_name=d.get("album_name"))
