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
from face_cluster import run_history_db

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
    merged_cluster_result: Optional[ClusterResult] = None
    merge_log: Optional[List[Dict]] = None
    merge_metadata: Optional[Dict] = None
    merge_decisions: Optional[List[Dict]] = None  # user approve/reject decisions


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

    action_id: Optional[int] = None  # run_history_db row id

    image_paths: List[Path] = field(default_factory=list)
    faces: List[FaceRecord] = field(default_factory=list)
    core_indices: List[int] = field(default_factory=list)
    holdout_indices: List[int] = field(default_factory=list)
    crop_manifest: Dict = field(default_factory=dict)
    cluster_result: Optional[ClusterResult] = None
    graph_result: object = None
    merged_cluster_result: Optional[ClusterResult] = None
    merge_log: Optional[List[Dict]] = None
    merge_metadata: Optional[Dict] = None

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


def _copy_cluster_result(cr: ClusterResult) -> ClusterResult:
    """Shallow-copy a ClusterResult to snapshot dict references before in-place remap."""
    return ClusterResult(
        labels=cr.labels,
        clusters=cr.clusters,
        cluster_stats=cr.cluster_stats,
        exemplars=cr.exemplars,
        n_clusters=cr.n_clusters,
        n_noise=cr.n_noise,
    )


def _load_crop_manifest_as_paths(source_dir: Path) -> Dict[int, Path]:
    """Load crop_manifest.json from source_dir as {face_id: absolute_path}."""
    manifest_path = source_dir / "crop_manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(fid): (source_dir / rel).resolve() for fid, rel in raw.items()}


def _write_absolute_manifest(source_dir: Path, output_dir: Path):
    """Write crop_manifest.json in output_dir with absolute path values.

    This allows the app to load crops from the source directory when
    displaying a recluster run.
    """
    src_manifest = source_dir / "crop_manifest.json"
    if not src_manifest.exists():
        return
    with open(src_manifest, encoding="utf-8") as f:
        raw = json.load(f)
    abs_manifest = {fid: str((source_dir / rel).resolve()) for fid, rel in raw.items()}
    with open(output_dir / "crop_manifest.json", "w", encoding="utf-8") as f:
        json.dump(abs_manifest, f, indent=2)


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
        pipeline = FaceClusteringPipeline()
        result = pipeline.run(PipelineConfig.full_run("path/to/images", "path/to/output"))
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

    FULL_STAGES = [
        ("discover",  "_discover"),
        ("embed",     "_embed"),
        ("quality",   "_quality_gate"),
        ("crops",     "_save_crops"),
        ("cluster",   "_cluster"),
        ("exemplars", "_select_exemplars"),
        ("merge",     "_merge"),
        ("export",    "_export"),
    ]

    # DB action_type derived from the first stage in the config
    _MODE_FROM_FIRST_STAGE: Dict[str, str] = {
        "discover":  "pipeline_run",
        "cluster":   "recluster",
        "exemplars": "remerge",
        "merge":     "remerge",
    }

    # Source loader method name derived from the first stage
    _SOURCE_LOADERS: Dict[str, str] = {
        "cluster":   "_load_source_recluster",
        "exemplars": "_load_source_remerge",
        "merge":     "_load_source_remerge",
    }

    def __init__(self):
        pass

    # -- Public API ----------------------------------------------------------

    def run(self, config: "PipelineConfig") -> PipelineResult:
        """Execute the pipeline according to config.

        config.stages determines which stages run (None = full pipeline).
        config.source_dir is the input (raw images, previous run, or snapshot).
        config.output_dir is where results are written.

        Each stage is tracked in pipeline_run.json with status
        (pending -> running -> done/failed).  On crash, _finalize marks
        remaining stages and writes the error.
        """
        from face_cluster.config import PipelineConfig

        stage_map = dict(self.FULL_STAGES)
        stages_list = config.stages or [name for name, _ in self.FULL_STAGES]
        stages = [(name, stage_map[name]) for name in stages_list]

        source_dir = Path(config.source_dir) if config.source_dir else None
        output_dir = Path(config.output_dir)
        first_stage = stages_list[0]
        mode = self._MODE_FROM_FIRST_STAGE.get(first_stage, "pipeline_run")

        ctx = self._init_context(source_dir or output_dir, output_dir,
                                 config.on_progress, stages, mode=mode, config=config)

        loader_name = self._SOURCE_LOADERS.get(first_stage)
        if loader_name:
            getattr(self, loader_name)(ctx, source_dir)

        try:
            if first_stage == "discover":
                _execute_stage(ctx, "discover", self._discover)

                cache_hit = (
                    config.embed_cache_enabled
                    and self._try_load_embed_cache(ctx)
                )
                if cache_hit:
                    ctx.run_record["stages"]["embed"]["status"] = "cached"
                    ctx.write_run_record()
                else:
                    t_embed = time.time()
                    _execute_stage(ctx, "embed", self._embed)
                    embed_elapsed = time.time() - t_embed
                    if config.embed_cache_enabled:
                        self._write_embed_cache(ctx, embed_elapsed)

                for stage_name, method_name in stages:
                    if stage_name in ("discover", "embed"):
                        continue
                    _execute_stage(ctx, stage_name, getattr(self, method_name))
            else:
                for stage_name, method_name in stages:
                    _execute_stage(ctx, stage_name, getattr(self, method_name))
                if first_stage in ("cluster", "exemplars", "merge"):
                    _write_absolute_manifest(source_dir, ctx.output_dir)

            return self._build_result(ctx)
        finally:
            self._finalize(ctx)

    # -- Context setup / teardown --------------------------------------------

    def _init_context(self, image_dir, output_dir, on_progress, stages,
                      mode: str = "pipeline_run", config=None) -> _RunContext:
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_handler, log_path = _setup_run_logging(output_dir)

        cfg = config or {}
        config_dict = {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_") and not callable(v)
        } if cfg else {}

        run_record = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_album": str(image_dir),
            "output_dir": str(output_dir),
            "started_at": datetime.now().isoformat(),
            "config": config_dict,
            "stages": {name: {"status": "pending"} for name, _ in stages},
            "status": "running",
            "error": None,
        }

        from face_cluster.config import PipelineConfig
        ctx = _RunContext(
            config=config if config is not None else PipelineConfig(),
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
        logger.info(f"  stages : {' -> '.join(n for n, _ in stages)}")
        logger.info("=" * 60)

        run_record["mode"] = mode
        ctx.action_id = run_history_db.start_action(
            mode,
            payload={
                "run_id":       run_record["run_id"],
                "source_dir":   str(image_dir),
                "output_dir":   str(output_dir),
                "album":        image_dir.name,
                "source_album": image_dir.name,
                "run_kind":     mode,
                "config_json":  __import__("json").dumps(run_record.get("config", {})),
                "log_file":     str(ctx.log_path),
                "config":       run_record.get("config", {}),
                "stages":       list(run_record["stages"].keys()),
            },
        )
        return ctx

    def _finalize(self, ctx: _RunContext):
        """Clean up log handler; call fail_action for any non-complete run."""
        fc_logger = logging.getLogger("face_cluster")
        fc_logger.removeHandler(ctx.file_handler)
        ctx.file_handler.close()

        status = ctx.run_record["status"]
        if status == "complete":
            return  # _build_result already called complete_action

        if status == "running":
            # Pipeline interrupted before any stage could mark it failed
            pending = [n for n, s in ctx.run_record["stages"].items()
                       if s["status"] == "pending"]
            error_msg = f"Pipeline interrupted. Stages not started: {pending}"
            ctx.run_record["status"] = "failed"
            ctx.run_record["error"] = error_msg
            ctx.write_run_record()
        else:
            error_msg = ctx.run_record.get("error") or "Pipeline failed"

        if ctx.action_id is not None:
            run_history_db.fail_action(ctx.action_id, error_msg)

    def _build_result(self, ctx: _RunContext) -> PipelineResult:
        total_s = sum(s.get("elapsed_s", 0)
                      for s in ctx.run_record["stages"].values())
        n_merged = ctx.merged_cluster_result.n_clusters if ctx.merged_cluster_result else None
        summary = {
            "n_faces": len(ctx.faces),
            "n_core": len(ctx.core_indices),
            "n_clusters": ctx.cluster_result.n_clusters,
            "n_noise": ctx.cluster_result.n_noise,
            "n_clusters_merged": n_merged,
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

        if ctx.action_id is not None:
            run_history_db.complete_action(
                ctx.action_id,
                result_fields={
                    "run_id":       ctx.run_record["run_id"],
                    "source_dir":   str(ctx.image_dir),
                    "output_dir":   str(ctx.output_dir),
                    "album":        ctx.image_dir.name,
                    "source_album": ctx.image_dir.name,
                    "run_kind":     ctx.run_record.get("mode"),
                    "n_faces":      summary["n_faces"],
                    "n_core":       summary.get("n_core"),
                    "n_clusters":   summary["n_clusters"],
                    "n_noise":      summary["n_noise"],
                    "log_file":     str(ctx.log_path),
                },
                payload_update={"summary": summary},
            )

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
            merged_cluster_result=ctx.merged_cluster_result,
            merge_log=ctx.merge_log,
            merge_metadata=ctx.merge_metadata,
        )

    # -- Source loaders ------------------------------------------------------

    def _load_source_recluster(self, ctx: _RunContext, source_dir: Path) -> None:
        """Load faces + embeddings from a previous run for recluster mode."""
        from face_cluster.loader import load_pipeline_result
        source = load_pipeline_result(source_dir)
        ctx.run_record["source_run"] = str(source_dir)
        ctx.write_run_record()
        ctx.faces = source.faces
        ctx.core_indices = [i for i, f in enumerate(source.faces) if f.is_core]
        ctx.holdout_indices = [i for i, f in enumerate(source.faces) if not f.is_core]
        ctx.crop_manifest = _load_crop_manifest_as_paths(source_dir)

    def _load_source_remerge(self, ctx: _RunContext, source_dir: Path) -> None:
        """Load faces + cluster state from a snapshot for remerge mode.

        Converts the face-list-indexed cluster_result from the snapshot back
        into graph-local indices so the merge/exemplar stages work correctly.
        The export stage remaps back to face-list indices as usual.
        """
        import networkx as nx
        from face_cluster.loader import load_pipeline_result

        source = load_pipeline_result(source_dir)
        ctx.run_record["source_run"] = str(source_dir)
        ctx.write_run_record()

        ctx.faces = source.faces
        ctx.core_indices = [i for i, f in enumerate(source.faces) if f.is_core]
        ctx.holdout_indices = [i for i, f in enumerate(source.faces) if not f.is_core]
        ctx.crop_manifest = _load_crop_manifest_as_paths(source_dir)

        # Use merged result if available (it reflects manual merges), else base
        face_list_cr = source.merged_cluster_result or source.cluster_result

        # Convert face-list indices -> graph-local indices
        face_to_graph = {fi: k for k, fi in enumerate(ctx.core_indices)}

        graph_clusters: Dict[int, List[int]] = {}
        for cid, face_indices in face_list_cr.clusters.items():
            nodes = [face_to_graph[fi] for fi in face_indices if fi in face_to_graph]
            if nodes:
                graph_clusters[cid] = nodes

        graph_exemplars: Dict[int, List[int]] = {}
        for cid, face_indices in face_list_cr.exemplars.items():
            nodes = [face_to_graph[fi] for fi in face_indices if fi in face_to_graph]
            graph_exemplars[cid] = nodes if nodes else graph_clusters.get(cid, [])[:1]

        n_core = len(ctx.core_indices)
        graph_labels = np.full(n_core, -1, dtype=np.int32)
        for cid, nodes in graph_clusters.items():
            for node in nodes:
                graph_labels[node] = cid

        # Build distance matrix from core embeddings (n_core x n_core)
        # Must be computed before cluster_stats which needs pairwise distances.
        core_embs = np.array([
            ctx.faces[i].embedding_normalized for i in ctx.core_indices
        ], dtype=np.float32)
        dist_matrix = 1.0 - core_embs @ core_embs.T
        np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)

        # Compute cluster_stats (diameter etc.) so the merge stage can
        # evaluate merge candidates correctly (KeyError if empty).
        cluster_stats: Dict[int, Dict[str, float]] = {}
        for cid, nodes in graph_clusters.items():
            if len(nodes) < 2:
                cluster_stats[cid] = {'size': len(nodes), 'diameter': 0.0,
                                      'median_dist': 0.0, 'mean_dist': 0.0, 'p95_dist': 0.0}
            else:
                idx = np.array(nodes)
                pdists = dist_matrix[np.ix_(idx, idx)]
                upper = pdists[np.triu_indices_from(pdists, k=1)]
                cluster_stats[cid] = {
                    'size': len(nodes),
                    'diameter': float(upper.max()),
                    'median_dist': float(np.median(upper)),
                    'mean_dist': float(upper.mean()),
                    'p95_dist': float(np.percentile(upper, 95)),
                }

        from face_cluster.types import ClusterResult as CR
        ctx.cluster_result = CR(
            labels=graph_labels,
            clusters=graph_clusters,
            cluster_stats=cluster_stats,
            exemplars=graph_exemplars,
            n_clusters=len(graph_clusters),
            n_noise=int((graph_labels == -1).sum()),
        )

        from face_cluster.types import GraphResult as GR
        ctx.graph_result = GR(
            neighbors=[[] for _ in range(n_core)],
            neighbor_distances=[[] for _ in range(n_core)],
            edges=[],
            G=nx.Graph(),
            distance_matrix=dist_matrix,
        )

    # -- Embed cache helpers -------------------------------------------------

    def _try_load_embed_cache(self, ctx: _RunContext) -> bool:
        """Attempt to load faces from embed cache.  Returns True on a valid hit."""
        from face_cluster.cache import (
            compute_image_fingerprint,
            load_embed_cache,
            validate_cache,
        )

        logger.info(
            f"Computing image fingerprint for {len(ctx.image_paths)} images..."
        )
        fingerprint = compute_image_fingerprint(
            ctx.image_dir, self.SUPPORTED_EXTENSIONS
        )
        logger.info(f"Fingerprint: {fingerprint[:32]}...")

        cached = load_embed_cache(ctx.output_dir)
        if cached is None:
            logger.info("No embed cache found — will run full embed stage")
            return False

        faces, meta = cached
        valid, reason = validate_cache(meta, faces, fingerprint)
        if not valid:
            logger.info(f"Embed cache invalid ({reason}) — re-embedding")
            return False

        ctx.faces = faces
        ctx.progress(
            "embed", 1.0,
            f"Cache hit: {len(faces)} faces from {meta.get('created_at', '?')[:10]} "
            f"(saved {meta.get('embed_time_seconds', 0):.0f}s)"
        )
        logger.info(
            f"Embed cache valid — loaded {len(faces)} faces "
            f"(originally took {meta.get('embed_time_seconds', 0):.1f}s)"
        )
        return True

    def _write_embed_cache(self, ctx: _RunContext, embed_elapsed: float) -> None:
        """Write embed cache after a successful embed stage."""
        from face_cluster.cache import (
            compute_image_fingerprint,
            save_embed_cache,
        )

        try:
            fingerprint = compute_image_fingerprint(
                ctx.image_dir, self.SUPPORTED_EXTENSIONS
            )
            save_embed_cache(
                output_dir=ctx.output_dir,
                faces=ctx.faces,
                image_fingerprint=fingerprint,
                image_dir=ctx.image_dir,
                n_images=len(ctx.image_paths),
                embed_time_s=embed_elapsed,
            )
        except Exception as e:
            # Cache write failure is non-fatal — log and continue
            logger.warning(f"Failed to write embed cache (non-fatal): {e}")

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
        ctx.core_indices, ctx.holdout_indices, _ = gater.select_core_set(ctx.faces)

        if not ctx.core_indices:
            raise ValueError(
                f"Quality gate rejected all {len(ctx.faces)} faces. "
                f"blur_min={ctx.config.blur_min}, require_pose={ctx.config.require_pose}. "
                f"Try lowering blur_min or check face crops."
            )

        self._write_quality_config(ctx)
        self._update_quality_summary(ctx)

        ctx.progress("quality", 1.0,
                     f"{len(ctx.core_indices)} core, {len(ctx.holdout_indices)} holdout")
        return {"n_core": len(ctx.core_indices), "n_holdout": len(ctx.holdout_indices)}

    def _write_quality_config(self, ctx: _RunContext) -> None:
        """Write quality_config.json — exact thresholds used for this run."""
        cfg = ctx.config
        quality_cfg = {
            "blur_min":              cfg.blur_min,
            "yaw_max":               cfg.yaw_max,
            "pitch_max":             cfg.pitch_max,
            "area_min":              cfg.min_face_area,
            "require_pose":          cfg.require_pose,
            "top_k_per_image":       cfg.max_faces_per_image_core,
        }
        with open(ctx.output_dir / "quality_config.json", "w", encoding="utf-8") as f:
            json.dump(quality_cfg, f, indent=2)

    def _update_quality_summary(self, ctx: _RunContext) -> None:
        """Compute quality_summary dict and merge it into the run record."""
        gate_names = ("blur", "pose_yaw", "pose_pitch", "area", "top_k_per_image")
        rejected_by_gate: Dict[str, int] = {g: 0 for g in gate_names}
        near_threshold: Dict[str, int] = {g: 0 for g in gate_names[:-1]}

        for face in ctx.faces:
            verdict = face.quality_verdict
            if verdict is None or verdict.all_passed():
                if verdict is not None:
                    self._count_near_threshold(verdict, ctx.config, near_threshold)
                continue
            if verdict.rejection_reason:
                rejected_by_gate[verdict.rejection_reason] = \
                    rejected_by_gate.get(verdict.rejection_reason, 0) + 1

        quality_summary = {
            "total_detected":         len(ctx.faces),
            "rejected_by_gate":       rejected_by_gate,
            "survived":               len(ctx.core_indices),
            "passing_within_10_pct":  near_threshold,
        }
        ctx.run_record["quality_summary"] = quality_summary
        ctx.write_run_record()

    @staticmethod
    def _count_near_threshold(verdict, config, near_threshold: Dict[str, int]) -> None:
        """Increment near_threshold counts for core faces that barely passed."""
        thresholds = {
            "blur":       (verdict.gates.get("blur"),       config.blur_min,  "above"),
            "pose_yaw":   (verdict.gates.get("pose_yaw"),   config.yaw_max,   "below"),
            "pose_pitch": (verdict.gates.get("pose_pitch"), config.pitch_max, "below"),
            "area":       (verdict.gates.get("area"),       config.min_face_area or 0.0, "above"),
        }
        for gate_name, (gate, threshold, direction) in thresholds.items():
            if gate is None or threshold == 0.0:
                continue
            margin = abs(gate.value - threshold) / (threshold + 1e-9)
            if margin <= 0.10:
                near_threshold[gate_name] = near_threshold.get(gate_name, 0) + 1

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
        ctx.cluster_result, node_d10_map = selector.select_exemplars(
            ctx.cluster_result, ctx.graph_result
        )
        # Assign d10_score to core faces; holdout faces remain None
        for node, d10 in node_d10_map.items():
            face_idx = ctx.core_indices[node]
            ctx.faces[face_idx].d10_score = d10
        ctx.progress("exemplars", 1.0, "Exemplars selected")
        self._save_candidate_pairs(ctx)
        return {}

    def _save_candidate_pairs(self, ctx: _RunContext) -> None:
        """Compute and save top-300 candidate pair features (non-fatal).

        Uses graph-local indices (pre-remap) so the distance matrix aligns
        with node indices in cluster_result. cluster_a/cluster_b IDs are
        stable across the remap so the file remains consistent with faces.csv.
        """
        try:
            from face_cluster.features import FeatureComputer, MergeFeatureContext
            from face_cluster.export import save_candidate_pairs
            from datetime import datetime

            if ctx.cluster_result.n_clusters < 2 or ctx.graph_result is None:
                return

            # Distance matrix is graph-local (0..n_core-1); pass only core faces.
            core_faces = [ctx.faces[i] for i in ctx.core_indices]
            feat_ctx = MergeFeatureContext(
                cluster_result=ctx.cluster_result,
                faces=core_faces,
                distance_matrix=ctx.graph_result.distance_matrix,
            )
            fc = FeatureComputer()
            pairs = fc.compute_top_n_pairs(feat_ctx, max_dist=0.80, top_n=300)
            if not pairs:
                return

            df = fc.to_dataframe(pairs)
            df["feature_version"] = FeatureComputer.VERSION
            df["saved_at"] = datetime.now().isoformat()
            save_candidate_pairs(df, ctx.output_dir)
        except Exception as exc:
            # Non-fatal: candidate pairs are for analysis only, never block the run.
            logger.warning("Failed to save candidate pairs (non-fatal): %s", exc)

    def _merge(self, ctx: _RunContext) -> dict:
        if not ctx.config.merge_enabled:
            ctx.progress("merge", 1.0, "Merge disabled -- skipping")
            return {}
        from face_cluster.merge import ConservativeMerger
        ctx.progress("merge", 0.0, "Merging clusters...")
        merger = ConservativeMerger(ctx.config)
        merged, log, metadata = merger.merge_clusters_with_logging(ctx.cluster_result, ctx.graph_result)
        # Always copy: when no merges happen, merger returns the same object as
        # ctx.cluster_result.  _remap_to_face_indices would then remap it twice
        # (double-remap → IndexError).  A shallow copy breaks the aliasing.
        ctx.merged_cluster_result = _copy_cluster_result(merged)
        ctx.merge_log = log
        ctx.merge_metadata = metadata
        n_merges = sum(1 for e in log if e.get("actually_merged"))
        ctx.progress("merge", 1.0, f"{n_merges} merges -> {merged.n_clusters} clusters")
        return {"n_merges": n_merges, "n_clusters_merged": merged.n_clusters}

    def _export(self, ctx: _RunContext) -> dict:
        """Snapshot graph-local results, remap to face indices, write all output files."""
        from face_cluster.export import export_results, export_merged_results
        from face_cluster.run_exporter import RunExporter

        ctx.progress("export", 0.0, "Preparing export...")

        # Snapshot before remap: shallow copies still reference the graph-local dicts
        cluster_snap = _copy_cluster_result(ctx.cluster_result)
        merged_snap = _copy_cluster_result(ctx.merged_cluster_result) if ctx.merged_cluster_result else None

        self._remap_to_face_indices(ctx)

        ctx.progress("export", 0.3, "Writing base results...")
        export_results(
            faces=ctx.faces,
            cluster_result=cluster_snap,
            crop_manifest=ctx.crop_manifest,
            output_dir=ctx.output_dir,
            config=ctx.config,
            source_album=str(ctx.image_dir),
            core_indices=ctx.core_indices,
        )

        if merged_snap is not None:
            ctx.progress("export", 0.7, "Writing merged results...")
            export_merged_results(
                faces=ctx.faces,
                merged_cluster_result=merged_snap,
                merge_log=ctx.merge_log or [],
                output_dir=ctx.output_dir,
                core_indices=ctx.core_indices,
                merge_metadata=ctx.merge_metadata,
            )

        # spec-030 Phase 1 — dual-write the v4 layout to a parallel subdir.
        # Legacy artifacts above stay in place; the loader still uses them.
        # In Phase 4 the legacy writes go away and RunExporter takes the run root.
        ctx.progress("export", 0.9, "Writing v4 layout (dual-write)...")
        try:
            mode = ctx.run_record.get("mode", "pipeline_run")
            producer = "remerge" if mode == "remerge" else "fc_app"
            parent_run_id = None
            source_run = ctx.run_record.get("source_run")
            if source_run:
                parent_run_id = Path(source_run).name

            RunExporter(ctx.output_dir / "_v4").export(
                faces=ctx.faces,
                base_cluster_result=cluster_snap,
                merged_cluster_result=merged_snap,
                core_indices=ctx.core_indices,
                merge_log=ctx.merge_log or [],
                merge_metadata=ctx.merge_metadata,
                config=ctx.config,
                source_album=str(ctx.image_dir),
                producer=producer,
                run_id=ctx.run_record["run_id"],
                started_at=ctx.run_record["started_at"],
                finished_at=ctx.run_record.get("finished_at") or datetime.now().isoformat(),
                parent_run_id=parent_run_id,
                crop_source_dir=ctx.output_dir / "crops",
            )
        except Exception as e:
            # Phase 1 is additive — failure to write the parallel layout must not
            # break the legacy export.  Log and continue.
            logger.warning(f"v4 dual-write failed (non-fatal during Phase 1): {e}",
                           exc_info=True)

        ctx.progress("export", 1.0, "Export complete")
        return {}

    @staticmethod
    def _remap_to_face_indices(ctx: _RunContext):
        """Remap graph-local node indices (0..n_core-1) to face-list indices.

        Remaps both base and merged cluster results.
        Keeps live-run output consistent with loader.py (which rebuilds
        from CSV using face-list indices).
        """
        n = len(ctx.faces)
        FaceClusteringPipeline._remap_cluster(ctx.cluster_result, ctx.core_indices, n)
        if ctx.merged_cluster_result is not None:
            FaceClusteringPipeline._remap_cluster(ctx.merged_cluster_result, ctx.core_indices, n)

    @staticmethod
    def _remap_cluster(cr: ClusterResult, core_indices: List[int], n_faces: int):
        """Remap one ClusterResult from graph-local to face-list indices in-place."""
        cr.clusters = {
            cid: [core_indices[node] for node in nodes]
            for cid, nodes in cr.clusters.items()
        }
        cr.exemplars = {
            cid: [core_indices[node] for node in nodes]
            for cid, nodes in cr.exemplars.items()
        }
        labels = np.full(n_faces, -1, dtype=np.int32)
        for cid, face_indices in cr.clusters.items():
            for fi in face_indices:
                labels[fi] = cid
        cr.labels = labels
