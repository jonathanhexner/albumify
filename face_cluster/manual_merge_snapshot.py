"""Save a manual-merge snapshot directory.

After the user approves merge pairs in the UI, this module writes a self-contained
output directory that can be used as the source for a subsequent remerge pipeline run.

The snapshot uses the SAME format as a regular pipeline run so that
``load_pipeline_result(snapshot_dir)`` and ``PipelineConfig.remerge(snapshot_dir, ...)``
work without any special-casing.

Schema written:
    faces.csv              -- cluster_id reflects the post-merge (approved pairs) state
    clusters.csv           -- merged cluster summary
    embeddings.npy         -- copied from parent (float32, n_faces x 512)
    embedding_face_ids.npy -- copied from parent (int32, n_faces)
    crop_manifest.json     -- {face_id: absolute_path_str} pointing to parent's crops
    pipeline_run.json      -- manual_merge metadata (source_type, parent_output_dir, ...)
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from face_cluster import run_history_db
from face_cluster.config import PipelineConfig
from face_cluster.types import ClusterResult, FaceRecord

logger = logging.getLogger(__name__)


def save_manual_merge_snapshot(
    faces: List[FaceRecord],
    merged_cluster_result: ClusterResult,
    approved_pairs: List[Tuple[int, int]],
    rejected_pairs: List[Tuple[int, int]],
    config: PipelineConfig,
    output_dir: Path,
    parent_output_dir: Optional[Path] = None,
    parent_run_id: Optional[str] = None,
    merge_round: int = 1,
) -> Path:
    """Write a manual-merge snapshot to ``output_dir``.

    The snapshot captures the current merged cluster state so it can serve
    as the starting point for ``pipeline.run(PipelineConfig.remerge(...))``.

    Args:
        faces:                All FaceRecord objects (core + holdout) from the
                              current pipeline result.
        merged_cluster_result: ClusterResult in **face-list index space** (as
                              stored in ``PipelineResult.merged_cluster_result``
                              or ``PipelineResult.cluster_result``).
        approved_pairs:       List of (cluster_id_a, cluster_id_b) pairs the
                              user approved for merging.
        rejected_pairs:       List of (cluster_id_a, cluster_id_b) pairs the
                              user rejected.
        config:               Pipeline config of the current run (saved for audit).
        output_dir:           Directory to write snapshot into (created if needed).
        parent_output_dir:    Output dir of the parent run (for lineage tracking).
        parent_run_id:        ``run_id`` of the parent pipeline_run.json (for lineage).
        merge_round:          Iteration counter (1 = first manual merge).

    Returns:
        ``output_dir`` as a resolved Path.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = _run_id(merge_round)

    action_id = run_history_db.start_action(
        "manual_merge",
        payload={
            "run_id":           run_id,
            "output_dir":       str(output_dir),
            "parent_output_dir": str(parent_output_dir) if parent_output_dir else None,
            "parent_run_id":    parent_run_id,
            "merge_round":      merge_round,
            "n_approved":       len(approved_pairs),
            "n_rejected":       len(rejected_pairs),
        },
    )

    try:
        # Apply user-approved pairs to the cluster state before writing.
        # merged_cluster_result reflects the ConservativeMerger output only;
        # user decisions are layered on top via union-find.
        user_clusters = _merge_cluster_assignments(
            merged_cluster_result.clusters, approved_pairs
        )
        user_exemplars = _merge_exemplars(
            user_clusters, merged_cluster_result.clusters, merged_cluster_result.exemplars
        )
        n_merged = len(merged_cluster_result.clusters) - len(user_clusters)
        logger.info(
            f"Snapshot: {len(approved_pairs)} approved pairs -> "
            f"{n_merged} clusters collapsed, {len(user_clusters)} remain"
        )

        # --- faces.csv -------------------------------------------------------
        face_list_idx_to_cluster: Dict[int, int] = {
            fi: cid
            for cid, face_indices in user_clusters.items()
            for fi in face_indices
        }

        rows = []
        for i, face in enumerate(faces):
            yaw = pitch = roll = None
            if face.pose is not None:
                yaw, pitch, roll = face.pose
            rows.append({
                "face_id":    face.face_id,
                "image_path": face.image_path,
                "image_id":   face.image_id,
                "crop_path":  "",
                "cluster_id": face_list_idx_to_cluster.get(i, -1),
                "is_core":    face.is_core,
                "blur_score": face.blur_score,
                "area":       face.area,
                "yaw":        yaw,
                "pitch":      pitch,
                "roll":       roll,
            })

        faces_df = pd.DataFrame(rows)
        faces_df.to_csv(output_dir / "faces.csv", index=False)
        logger.info(f"Snapshot: wrote {len(faces_df)} rows to faces.csv")

        # --- clusters.csv ----------------------------------------------------
        # Build parent map: canonical_id -> list of original cluster ids that merged into it
        canonical_of: Dict[int, int] = {}
        for new_cid, new_faces in user_clusters.items():
            face_set = set(new_faces)
            for old_cid, old_faces in merged_cluster_result.clusters.items():
                if set(old_faces) <= face_set:
                    canonical_of[old_cid] = new_cid

        parent_ids_map: Dict[int, List[int]] = {}
        for old_cid, canonical in canonical_of.items():
            parent_ids_map.setdefault(canonical, []).append(old_cid)

        cluster_rows = []
        for cid, face_indices in user_clusters.items():
            exemplar_face_ids = [faces[fi].face_id for fi in user_exemplars.get(cid, face_indices[:1])
                                 if fi < len(faces)]
            stats = merged_cluster_result.cluster_stats.get(cid, {})
            parents = parent_ids_map.get(cid, [])
            origin = "manual_merge" if len(parents) > 1 else "base"
            cluster_rows.append({
                "cluster_id":         cid,
                "size":               len(face_indices),
                "exemplar_face_ids":  json.dumps(exemplar_face_ids),
                "diameter":           stats.get("diameter"),
                "origin":             origin,
                "parent_cluster_ids": json.dumps(sorted(set(parents) - {cid})),
            })

        pd.DataFrame(cluster_rows).to_csv(output_dir / "clusters.csv", index=False)
        logger.info(f"Snapshot: wrote {len(cluster_rows)} clusters to clusters.csv")

        # --- embeddings.npy + embedding_face_ids.npy -------------------------
        EMB_DIM = 512
        emb_matrix = np.zeros((len(faces), EMB_DIM), dtype=np.float32)
        emb_face_ids = np.array([f.face_id for f in faces], dtype=np.int32)
        for i, face in enumerate(faces):
            if face.embedding_normalized is not None:
                emb_matrix[i] = face.embedding_normalized.astype(np.float32)
        np.save(output_dir / "embeddings.npy", emb_matrix)
        np.save(output_dir / "embedding_face_ids.npy", emb_face_ids)
        logger.info(f"Snapshot: wrote embeddings {emb_matrix.shape}")

        # --- crop_manifest.json (absolute paths) ----------------------------
        if parent_output_dir is not None:
            _write_absolute_manifest(Path(parent_output_dir), output_dir)
        else:
            logger.warning("Snapshot: no parent_output_dir; crop_manifest.json not written")

        # --- pipeline_run.json -----------------------------------------------
        config_dict = {
            k: v for k, v in vars(config).items()
            if not k.startswith("_") and not callable(v)
        }
        run_record = {
            "run_id":            run_id,
            "source_type":       "manual_merge",
            "parent_run_id":     parent_run_id,
            "parent_output_dir": str(parent_output_dir) if parent_output_dir else None,
            "approved_pairs":    [list(p) for p in approved_pairs],
            "rejected_pairs":    [list(p) for p in rejected_pairs],
            "merge_round":       merge_round,
            "config":            config_dict,
            "started_at":        datetime.now().isoformat(),
            "status":            "complete",
            "stages":            {},
            "summary": {
                "n_faces":    len(faces),
                "n_core":     sum(1 for f in faces if f.is_core),
                "n_clusters": len(user_clusters),
                "n_noise":    merged_cluster_result.n_noise,
                "n_manual_merges": n_merged,
            },
        }
        with open(output_dir / "pipeline_run.json", "w", encoding="utf-8") as fh:
            json.dump(run_record, fh, indent=2)
        logger.info(f"Snapshot: wrote pipeline_run.json (run_id={run_id})")

    except Exception as exc:
        run_history_db.fail_action(action_id, str(exc))
        raise

    run_history_db.complete_action(
        action_id,
        result_fields={
            "run_id":          run_id,
            "output_dir":      str(output_dir),
            "n_faces":         len(faces),
            "n_clusters":      len(user_clusters),
            "n_noise":         merged_cluster_result.n_noise,
            "n_manual_merges": n_merged,
        },
    )
    logger.info(
        f"Snapshot saved: {len(user_clusters)} clusters "
        f"({n_merged} manual merges applied), "
        f"{merged_cluster_result.n_noise} noise -> {output_dir}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_id(merge_round: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_merge_{merge_round}"


def _merge_cluster_assignments(
    clusters: Dict[int, List[int]],
    approved_pairs: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
    """Union-find merge of approved pairs. Returns {canonical_id: [face_indices]}.

    Canonical ID for each merged group is the minimum cluster ID in the group.
    Clusters not involved in any approved pair are returned unchanged.
    """
    known = set(clusters.keys())
    parent: Dict[int, int] = {c: c for c in known}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in approved_pairs:
        if a in known and b in known:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

    root_to_members: Dict[int, List[int]] = {}
    for cid in known:
        root_to_members.setdefault(_find(cid), []).append(cid)

    new: Dict[int, List[int]] = {}
    for members in root_to_members.values():
        canonical = min(members)
        merged_faces: List[int] = []
        for cid in members:
            merged_faces.extend(clusters[cid])
        new[canonical] = merged_faces

    return new


def _merge_exemplars(
    new_clusters: Dict[int, List[int]],
    old_clusters: Dict[int, List[int]],
    old_exemplars: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    """Build exemplars for merged clusters by combining old exemplar lists.

    For clusters unchanged by the merge, preserves original exemplars.
    For newly-merged clusters, concatenates all member exemplars (deduplicated).
    """
    result: Dict[int, List[int]] = {}
    # Build reverse: old_cid -> canonical
    canonical_of: Dict[int, int] = {}
    for cid, faces in new_clusters.items():
        face_set = set(faces)
        for old_cid, old_faces in old_clusters.items():
            if set(old_faces) <= face_set:
                canonical_of[old_cid] = cid

    for canonical in new_clusters:
        combined: List[int] = []
        seen: set = set()
        for old_cid, mapped in canonical_of.items():
            if mapped != canonical:
                continue
            for fi in old_exemplars.get(old_cid, old_clusters.get(old_cid, [])[:1]):
                if fi not in seen:
                    seen.add(fi)
                    combined.append(fi)
        result[canonical] = combined or new_clusters[canonical][:1]

    return result


def _write_absolute_manifest(parent_dir: Path, output_dir: Path) -> None:
    """Copy crop_manifest.json from parent_dir to output_dir with absolute paths."""
    src = parent_dir / "crop_manifest.json"
    if not src.exists():
        logger.warning(f"Snapshot: crop_manifest.json not found in {parent_dir}")
        return
    with open(src, encoding="utf-8") as fh:
        raw = json.load(fh)

    abs_manifest: Dict[str, str] = {}
    for fid, entry in raw.items():
        if Path(entry).is_absolute():
            # Already absolute (e.g. written by a previous recluster)
            abs_manifest[fid] = entry
        else:
            abs_manifest[fid] = str((parent_dir / entry).resolve())

    with open(output_dir / "crop_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(abs_manifest, fh, indent=2)
    logger.info(f"Snapshot: wrote crop_manifest.json ({len(abs_manifest)} entries)")
