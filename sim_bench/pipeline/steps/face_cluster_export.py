"""Export face clustering artifacts for the standalone Face Clustering App.

Generates: crops, CSVs, merge logs, SQLite DB, pipeline_run.json.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import dataclasses
import numpy as np
from PIL import Image, ImageOps

from face_cluster.export import export_results, export_merged_results
from face_cluster.result_db import write_results_db
from sim_bench.run_db.exporter import RunExporter
from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def export_for_analysis(face_records, base_cluster_result, merged_cluster_result,
                        core_indices, fc_cfg, merge_log, merge_metadata, context):
    """Export face clustering artifacts so the standalone app can load them."""
    cluster_result = merged_cluster_result

    # Determine output directory
    album_name = getattr(context, "album_name", None) or "album"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / album_name / f"face_clustering_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate face crops from source images using bboxes
    crop_manifest = _generate_crops_from_bboxes(face_records, output_dir)

    # Export BASE cluster results (pre-merge)
    export_results(
        faces=face_records,
        cluster_result=base_cluster_result,
        crop_manifest=crop_manifest,
        output_dir=output_dir,
        config=fc_cfg,
        source_album=album_name,
        core_indices=core_indices,
    )

    # Export merged results when merges occurred
    if merge_log is not None and any(e.get("actually_merged") for e in merge_log):
        try:
            export_merged_results(
                faces=face_records,
                merged_cluster_result=merged_cluster_result,
                merge_log=merge_log,
                output_dir=output_dir,
                core_indices=core_indices,
                merge_metadata=merge_metadata,
            )
            logger.info("Exported merged cluster results")
        except Exception as e:
            logger.warning(f"Failed to export merged results: {e}")
            _write_merge_log_fallback(merge_log, output_dir)
    elif merge_log is not None:
        _write_merge_log_fallback(merge_log, output_dir)

    if merge_metadata is not None:
        (output_dir / "merge_metadata.json").write_text(
            json.dumps(merge_metadata, indent=2, default=_numpy_safe), encoding="utf-8"
        )

    # Write face clustering results DB (SQLite)
    try:
        write_results_db(
            faces=face_records,
            base_cluster_result=base_cluster_result,
            merged_cluster_result=merged_cluster_result,
            core_indices=core_indices,
            merge_log=merge_log,
            merge_metadata=merge_metadata,
            config=fc_cfg,
            output_dir=output_dir,
            source_album=album_name,
            crop_manifest=crop_manifest,
        )
    except Exception as e:
        logger.warning(f"Failed to write results DB: {e}", exc_info=True)

    # Write pipeline_run.json
    run_info = {
        "run_id": timestamp,
        "source_album": album_name,
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "mode": "main_app_export",
        "status": "complete",
        "config": dataclasses.asdict(fc_cfg),
        "summary": {
            "n_faces": len(face_records),
            "n_core": len(core_indices),
            "n_clusters": base_cluster_result.n_clusters,
            "n_clusters_merged": merged_cluster_result.n_clusters,
            "n_noise": base_cluster_result.n_noise,
        },
        "stages": {
            "cluster": {"status": "done"},
            **({"merge": {"status": "done"}} if merge_log is not None else {}),
        },
    }
    (output_dir / "pipeline_run.json").write_text(
        json.dumps(run_info, indent=2, default=str), encoding="utf-8"
    )

    # spec-030 Phase 1 — dual-write the v4 layout to a parallel subdir alongside
    # the legacy artifacts above.  Albumify and the FC App go through the same
    # RunExporter so v4 layouts are byte-identical regardless of producer.
    # spec-033 P-C C-3: join image-level context onto each face row by image_path.
    # Albumify keys these dicts by str path; lookups elsewhere use the same key.
    image_scores = {}
    for path in set(getattr(context, "iqa_scores", {})) | set(getattr(context, "ava_scores", {})) \
            | set(getattr(context, "sharpness_scores", {})) | set(getattr(context, "scene_cluster_labels", {})):
        image_scores[path] = {
            "iqa": getattr(context, "iqa_scores", {}).get(path),
            "ava": getattr(context, "ava_scores", {}).get(path),
            "sharpness": getattr(context, "sharpness_scores", {}).get(path),
            "scene_cluster_id": getattr(context, "scene_cluster_labels", {}).get(path),
        }

    try:
        RunExporter(output_dir / "_v4").export(
            faces=face_records,
            base_cluster_result=base_cluster_result,
            merged_cluster_result=merged_cluster_result,
            core_indices=core_indices,
            merge_log=merge_log,
            merge_metadata=merge_metadata,
            config=fc_cfg,
            source_album=album_name,
            producer="albumify",
            run_id=timestamp,
            started_at=run_info["started_at"],
            finished_at=run_info["finished_at"],
            crop_source_dir=output_dir / "crops",
            # spec-033 P-C C-3 / spec-032 P1: forward the FilterContext so the
            # filter_decisions table is populated on Albumify runs (was empty).
            filters=getattr(context, "filters", None),
            # spec-033 P-C C-3: per-image scores joined onto each face row.
            image_scores=image_scores,
            # spec-040 Phase 4 (schema v5) — populate the images table.
            # Zero-face images get a row with n_faces=0.
            image_paths=[str(p) for p in getattr(context, "image_paths", []) or []],
        )
    except Exception as e:
        logger.warning(f"v4 dual-write failed (non-fatal during Phase 1): {e}",
                       exc_info=True)

    # Store export path on context for UI deep-link
    context.fc_export_dir = str(output_dir)
    logger.info(f"Exported face clustering artifacts to {output_dir}")


def _generate_crops_from_bboxes(face_records, output_dir) -> Dict:
    """Generate 112x112 face crop JPEGs from source images using bbox coordinates."""
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    saved = 0

    for face in face_records:
        img_path = face.image_path or getattr(face, 'image_id', None)
        if not img_path:
            continue

        bbox = face.bbox
        if not bbox or len(bbox) < 4:
            continue

        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)
                img_w, img_h = img.size

                x, y, w, h = bbox
                if all(0 <= v <= 1.0 for v in (x, y, w, h)):
                    x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h

                pad = 0.25 * min(w, h)
                left = max(0, int(x - pad))
                top = max(0, int(y - pad))
                right = min(img_w, int(x + w + pad))
                bottom = min(img_h, int(y + h + pad))

                if right <= left or bottom <= top:
                    continue

                crop = img.crop((left, top, right, bottom))
                crop = crop.resize((112, 112), Image.Resampling.LANCZOS)
                crop = crop.convert("RGB")

                filename = f"face_{face.face_id:04d}_aligned.jpg"
                crop_path = crops_dir / filename
                crop.save(crop_path, "JPEG", quality=85)

                manifest[face.face_id] = f"crops/{filename}"
                saved += 1
        except Exception as e:
            logger.debug(f"Failed to crop face {face.face_id} from {img_path}: {e}")

    # Write crop manifest
    manifest_path = output_dir / "crop_manifest.json"
    str_manifest = {str(k): v for k, v in manifest.items()}
    manifest_path.write_text(json.dumps(str_manifest, indent=2), encoding="utf-8")

    logger.info(f"Generated {saved} face crops in {crops_dir}")
    return manifest


def _write_merge_log_fallback(merge_log, output_dir):
    """Write merge_log.json when export_merged_results is skipped."""
    (output_dir / "merge_log.json").write_text(
        json.dumps(merge_log, indent=2, default=_numpy_safe), encoding="utf-8"
    )


def _numpy_safe(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)
