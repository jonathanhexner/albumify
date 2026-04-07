"""Unified face clustering pipeline."""
from __future__ import annotations

import json
import logging
import time
import traceback
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from face_cluster.config import PipelineConfig
from face_cluster.types import FaceRecord, ClusterResult

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, float, str], None]]


def _setup_run_logging(output_dir: Path) -> tuple:
    """Attach a file handler to 'face_cluster' logger for this run."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    ))
    fc_logger = logging.getLogger("face_cluster")
    fc_logger.addHandler(handler)
    fc_logger.setLevel(logging.DEBUG)
    return handler, log_path


class PipelineStageError(Exception):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"Pipeline failed at stage '{stage}': {cause}")
        self.stage = stage
        self.cause = cause


@dataclass
class PipelineResult:
    faces: List[FaceRecord]
    cluster_result: ClusterResult
    output_dir: Path
    summary: dict


@dataclass
class _RunContext:
    """Mutable state passed between pipeline stages."""
    config: PipelineConfig
    image_dir: Path
    output_dir: Path
    on_progress: ProgressCallback
    run_record: Dict
    file_handler: logging.FileHandler
    log_path: Path

    image_paths: List[Path] = field(default_factory=list)
    faces: List[FaceRecord] = field(default_factory=list)
    core_indices: List[int] = field(default_factory=list)
    holdout_indices: List[int] = field(default_factory=list)
    crop_manifest: Dict = field(default_factory=dict)
    cluster_result: Optional[ClusterResult] = None
    graph_result: object = None

    def write_run_record(self):
        with open(self.output_dir / "pipeline_run.json", "w", encoding="utf-8") as f:
            json.dump(self.run_record, f, indent=2)

    def progress(self, stage: str, fraction: float, message: str):
        logger.info(f"[{stage}] {fraction:.0%}  {message}")
        if self.on_progress:
            self.on_progress(stage, fraction, message)

    def mark_start(self, name: str):
        self.run_record["stages"][name]["status"] = "running"
        self.run_record["stages"][name]["started_at"] = datetime.now().isoformat()
        self.write_run_record()

    def mark_done(self, name: str, t0: float, **counts):
        self.run_record["stages"][name].update(
            {"status": "done", "elapsed_s": round(time.time() - t0, 2), **counts}
        )
        self.write_run_record()

    def mark_failed(self, name: str, exc: Exception):
        self.run_record["stages"][name].update({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        self.run_record["status"] = "failed"
        self.run_record["error"] = f"Stage '{name}': {exc}"
        self.write_run_record()


def _execute_stage(ctx: _RunContext, name: str, stage_fn) -> dict:
    """Run a pipeline stage with start/done/fail tracking."""
    ctx.mark_start(name)
    t0 = time.time()
    try:
        counts = stage_fn(ctx) or {}
    except Exception as e:
        ctx.mark_failed(name, e)
        raise PipelineStageError(name, e) from e
    ctx.mark_done(name, t0, **counts)
    return counts


class FaceClusteringPipeline:
    """End-to-end face clustering pipeline.

    Usage:
        pipeline = FaceClusteringPipeline(PipelineConfig())
        result = pipeline.run("path/to/images", "path/to/output")
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

    STAGES = [
        ("discover",  "_discover"),
        ("embed",     "_embed"),
        ("quality",   "_quality_gate"),
        ("crops",     "_save_crops"),
        ("cluster",   "_cluster"),
        ("exemplars", "_select_exemplars"),
        ("export",    "_export"),
    ]

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    # -- Public API ----------------------------------------------------------

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        on_progress: ProgressCallback = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Stages: discover -> embed -> quality -> crops -> cluster -> exemplars -> export

        Each stage is tracked in pipeline_run.json with status
        (pending -> running -> done/failed).  On crash, _finalize marks
        remaining stages and writes the error.
        """
        ctx = self._init_context(image_dir, output_dir, on_progress)
        try:
            for stage_name, method_name in self.STAGES:
                _execute_stage(ctx, stage_name, getattr(self, method_name))
            return self._build_result(ctx)
        finally:
            self._finalize(ctx)

    # -- Context setup / teardown --------------------------------------------

    def _init_context(self, image_dir, output_dir, on_progress) -> _RunContext:
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_handler, log_path = _setup_run_logging(output_dir)

        run_record = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_album": str(image_dir),
            "output_dir": str(output_dir),
            "started_at": datetime.now().isoformat(),
            "config": {k: v for k, v in vars(self.config).items()
                       if not k.startswith("_")},
            "stages": {name: {"status": "pending"} for name, _ in self.STAGES},
            "status": "running",
            "error": None,
        }

        ctx = _RunContext(
            config=self.config,
            image_dir=image_dir,
            output_dir=output_dir,
            on_progress=on_progress,
            run_record=run_record,
            file_handler=file_handler,
            log_path=log_path,
        )
        ctx.write_run_record()

        logger.info("=" * 60)
        logger.info(f"Pipeline run {run_record['run_id']}")
        logger.info(f"  source : {image_dir}")
        logger.info(f"  output : {output_dir}")
        logger.info(f"  stages : {' -> '.join(n for n, _ in self.STAGES)}")
        logger.info("=" * 60)
        return ctx

    def _finalize(self, ctx: _RunContext):
        """Clean up log handler; mark interrupted runs as failed."""
        fc_logger = logging.getLogger("face_cluster")
        fc_logger.removeHandler(ctx.file_handler)
        ctx.file_handler.close()

        if ctx.run_record["status"] != "running":
            return

        pending = [n for n, s in ctx.run_record["stages"].items()
                   if s["status"] == "pending"]
        ctx.run_record["status"] = "failed"
        ctx.run_record["error"] = (
            ctx.run_record.get("error")
            or f"Pipeline interrupted. Stages not started: {pending}"
        )
        ctx.write_run_record()

    def _build_result(self, ctx: _RunContext) -> PipelineResult:
        total_s = sum(s.get("elapsed_s", 0)
                      for s in ctx.run_record["stages"].values())
        summary = {
            "n_faces": len(ctx.faces),
            "n_core": len(ctx.core_indices),
            "n_clusters": ctx.cluster_result.n_clusters,
            "n_noise": ctx.cluster_result.n_noise,
            "source_album": str(ctx.image_dir),
            "output_dir": str(ctx.output_dir),
            "log_file": str(ctx.log_path),
            "stages_timing": {
                k: v.get("elapsed_s")
                for k, v in ctx.run_record["stages"].items()
            },
        }

        ctx.run_record["status"] = "complete"
        ctx.run_record["finished_at"] = datetime.now().isoformat()
        ctx.run_record["summary"] = summary
        ctx.write_run_record()

        logger.info(
            f"Pipeline complete: {summary['n_faces']} faces, "
            f"{summary['n_clusters']} clusters, {summary['n_noise']} noise "
            f"in {total_s:.1f}s -> {ctx.output_dir}"
        )
        return PipelineResult(
            faces=ctx.faces,
            cluster_result=ctx.cluster_result,
            output_dir=ctx.output_dir,
            summary=summary,
        )

    # -- Stage implementations -----------------------------------------------
    # Each returns a dict of counts written into run_record for that stage.

    def _discover(self, ctx: _RunContext) -> dict:
        ctx.progress("discover", 0.0, "Scanning image directory...")
        ctx.image_paths = sorted(
            p for p in ctx.image_dir.rglob("*")
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        if not ctx.image_paths:
            raise ValueError(f"No supported images found in {ctx.image_dir}")
        ctx.progress("discover", 1.0, f"Found {len(ctx.image_paths)} images")
        return {"n_images": len(ctx.image_paths)}

    def _embed(self, ctx: _RunContext) -> dict:
        from face_cluster.embedding import InsightFaceEmbedder

        n = len(ctx.image_paths)
        ctx.progress("embed", 0.0, f"Detecting faces in {n} images...")
        embedder = InsightFaceEmbedder()

        for i, img_path in enumerate(ctx.image_paths):
            ctx.faces.extend(embedder.detect_and_embed([str(img_path)]))
            if (i + 1) % 5 == 0 or i == n - 1:
                ctx.progress("embed", (i + 1) / n,
                             f"{i+1}/{n} images, {len(ctx.faces)} faces")

        if not ctx.faces:
            raise ValueError("No faces detected in any image")

        for new_id, face in enumerate(ctx.faces):
            face.face_id = new_id

        missing = [f.face_id for f in ctx.faces if f.image_path is None]
        if missing:
            raise ValueError(f"{len(missing)} faces have null image_path")

        ctx.progress("embed", 1.0, f"Detected {len(ctx.faces)} faces")
        return {"n_faces": len(ctx.faces)}

    def _quality_gate(self, ctx: _RunContext) -> dict:
        from face_cluster.quality import QualityGater

        ctx.progress("quality", 0.0, "Applying quality gate...")
        gater = QualityGater(ctx.config)
        ctx.faces = gater.compute_blur_scores(ctx.faces)
        ctx.core_indices, ctx.holdout_indices = gater.select_core_set(ctx.faces)

        if not ctx.core_indices:
            raise ValueError(
                f"Quality gate rejected all {len(ctx.faces)} faces. "
                f"blur_min={ctx.config.blur_min}, require_pose={ctx.config.require_pose}. "
                f"Try lowering blur_min or check face crops."
            )

        ctx.progress("quality", 1.0,
                     f"{len(ctx.core_indices)} core, {len(ctx.holdout_indices)} holdout")
        return {"n_core": len(ctx.core_indices), "n_holdout": len(ctx.holdout_indices)}

    def _save_crops(self, ctx: _RunContext) -> dict:
        from face_cluster.crops import save_crops

        ctx.progress("crops", 0.0, "Saving aligned face crops...")
        ctx.crop_manifest = save_crops(ctx.faces, ctx.output_dir)
        ctx.progress("crops", 1.0, f"Saved {len(ctx.crop_manifest)} crops")
        return {"n_saved": len(ctx.crop_manifest)}

    def _cluster(self, ctx: _RunContext) -> dict:
        from face_cluster.knn_graph import KNNGraphBuilder
        from face_cluster.clustering import ConnectedComponentsClusterer

        ctx.progress("cluster", 0.0, "Building kNN graph...")
        builder = KNNGraphBuilder(ctx.config)
        ctx.graph_result = builder.build_graph(ctx.faces, ctx.core_indices)
        logger.info(
            f"  kNN graph: {len(ctx.graph_result.edges)} edges on "
            f"{len(ctx.core_indices)} core (threshold={ctx.config.distance_threshold})"
        )

        ctx.progress("cluster", 0.5, "Connected components...")
        clusterer = ConnectedComponentsClusterer(ctx.config)
        ctx.cluster_result = clusterer.cluster(ctx.graph_result, ctx.core_indices)

        if ctx.cluster_result.n_clusters == 0:
            logger.warning(
                f"ZERO clusters. {len(ctx.graph_result.edges)} edges, "
                f"{len(ctx.core_indices)} core. "
                f"Raise distance_threshold or lower blur_min."
            )

        ctx.progress(
            "cluster", 1.0,
            f"{ctx.cluster_result.n_clusters} clusters, "
            f"{ctx.cluster_result.n_noise} noise"
        )
        return {
            "n_clusters": ctx.cluster_result.n_clusters,
            "n_noise": ctx.cluster_result.n_noise,
            "n_edges": len(ctx.graph_result.edges),
        }

    def _select_exemplars(self, ctx: _RunContext) -> dict:
        from face_cluster.exemplars import D10ExemplarSelector

        ctx.progress("exemplars", 0.0, "Selecting exemplars...")
        selector = D10ExemplarSelector(ctx.config)
        ctx.cluster_result = selector.select_exemplars(
            ctx.cluster_result, ctx.graph_result
        )
        ctx.progress("exemplars", 1.0, "Exemplars selected")
        return {}

    def _export(self, ctx: _RunContext) -> dict:
        """Remap indices for app/analysis, then write all output files."""
        from face_cluster.export import export_results

        ctx.progress("export", 0.0, "Preparing export...")

        cluster_for_export = ClusterResult(
            labels=ctx.cluster_result.labels,
            clusters=ctx.cluster_result.clusters,
            cluster_stats=ctx.cluster_result.cluster_stats,
            exemplars=ctx.cluster_result.exemplars,
            n_clusters=ctx.cluster_result.n_clusters,
            n_noise=ctx.cluster_result.n_noise,
        )

        self._remap_to_face_indices(ctx)

        ctx.progress("export", 0.5, "Writing files...")
        export_results(
            faces=ctx.faces,
            cluster_result=cluster_for_export,
            crop_manifest=ctx.crop_manifest,
            output_dir=ctx.output_dir,
            config=ctx.config,
            source_album=str(ctx.image_dir),
            core_indices=ctx.core_indices,
        )
        ctx.progress("export", 1.0, "Export complete")
        return {}

    @staticmethod
    def _remap_to_face_indices(ctx: _RunContext):
        """Remap graph-local node indices (0..n_core-1) to face-list indices.

        Keeps live-run output consistent with loader.py (which rebuilds
        from CSV using face-list indices).
        """
        cr = ctx.cluster_result
        cr.clusters = {
            cid: [ctx.core_indices[node] for node in nodes]
            for cid, nodes in cr.clusters.items()
        }
        cr.exemplars = {
            cid: [ctx.core_indices[node] for node in nodes]
            for cid, nodes in cr.exemplars.items()
        }
        full_labels = np.full(len(ctx.faces), -1, dtype=np.int32)
        for cid, face_indices in cr.clusters.items():
            for fi in face_indices:
                full_labels[fi] = cid
        cr.labels = full_labels
