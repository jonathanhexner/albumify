"""spec-088 / SIGHTING-107: re-enable "Export for analysis" for the unified chain.

The FC-app export used to live only inside the deprecated ``cluster_people`` step
(removed by spec-079). This thin step re-enables the toggle for the unified
clustering chain by reusing the existing ``export_for_analysis()`` helper with the
cluster artifacts already sitting on the context. It is READ-ONLY with respect to
clustering output — it only writes files + sets ``context.fc_export_dir``.
"""

import logging

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.steps.face_cluster_export import export_for_analysis
from face_cluster.fc_params import FCParams

logger = logging.getLogger(__name__)


@register_step
class FaceClusterAnalysisExportStep(BaseStep):
    """Write the FC-app-loadable export (face_clustering.db + crops + merge log)."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="face_cluster_analysis_export",
            display_name="Export for analysis (FC app)",
            description="Write face_clustering.db + crops so the run opens in the FC app.",
            category="export",
            requires={"face_records"},
            produces={"fc_export_dir"},
            depends_on=["assign_people_clusters"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if not config.get("export_for_analysis", False):
            return
        if context.cluster_result is None or not context.face_records:
            logger.info("face_cluster_analysis_export: no clustering result; skipping")
            return

        # The clustering params were broadcast into this step's config (see
        # PipelineService._broadcast_clustering_config); rebuild the typed config
        # the export helper serializes. Non-FCParams keys (the flag) are filtered.
        fcp = FCParams(**{k: v for k, v in config.items() if k in FCParams.model_fields})
        fc_cfg = fcp.to_fc_config()

        export_for_analysis(
            face_records=context.face_records,
            base_cluster_result=context.cluster_result,
            merged_cluster_result=context.merged_cluster_result or context.cluster_result,
            core_indices=context.core_indices,
            fc_cfg=fc_cfg,
            merge_log=context.merge_log,
            merge_metadata=getattr(context, "merge_metadata", None),
            context=context,
        )
        logger.info("face_cluster_analysis_export: wrote %s", context.fc_export_dir)
