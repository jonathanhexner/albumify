"""Export clustering results for manual labeling interface."""

from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.clustering_labels import NOISE_LABEL, is_noise
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


@register_step
class ExportForLabelingStep(BaseStep):
    """Export clustering results for labeling app.

    Generates:
    - faces.csv: Face metadata with cluster assignments
    - clusters.csv: Cluster statistics
    - face_crops/: Aligned face images
    - export_summary.json: Run metadata
    """

    _metadata = StepMetadata(
        name="export_for_labeling",
        display_name="Export for Labeling",
        description="Export faces.csv, clusters.csv, and crops for manual labeling",
        category="export",
        requires={
            "face_records",
            "initial_clusters",
            "debug_neighbors",
            "aligned_faces",
            "source_directory"
        },
        produces={
            "export_directory",
            "faces_csv_path",
            "clusters_csv_path"
        },
        depends_on=["compute_debug_distances"]
    )

    def process(self, context: PipelineContext, config: dict):
        """Export clustering data for labeling."""

        # Get output directory from config
        output_dir = Path(config.get('output_dir', 'results/clustering_export'))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting clustering data to: {output_dir}")

        # Get data from context
        face_records = context.face_records
        cluster_result = context.initial_clusters
        debug_neighbors = context.debug_neighbors

        # 1. Generate faces.csv
        logger.info("Generating faces.csv...")
        faces_df = self._generate_faces_dataframe(
            face_records,
            cluster_result
        )
        faces_csv = output_dir / 'faces.csv'
        faces_df.to_csv(faces_csv, index=False)
        logger.info(f"  Saved {len(faces_df)} faces to {faces_csv.name}")

        # 2. Generate clusters.csv
        logger.info("Generating clusters.csv...")
        clusters_df = self._generate_clusters_dataframe(
            cluster_result,
            face_records
        )
        clusters_csv = output_dir / 'clusters.csv'
        clusters_df.to_csv(clusters_csv, index=False)
        logger.info(f"  Saved {len(clusters_df)} clusters to {clusters_csv.name}")

        # 3. Save face crops
        logger.info("Saving face crops...")
        crops_dir = output_dir / 'face_crops'
        crops_dir.mkdir(exist_ok=True)

        saved_count = self._save_face_crops(face_records, crops_dir)
        logger.info(f"  Saved {saved_count} face crops to {crops_dir.name}/")

        # 4. Export debug_neighbors.json
        logger.info("Exporting debug_neighbors.json...")
        debug_json = output_dir / 'debug_neighbors.json'
        with open(debug_json, 'w') as f:
            json.dump(debug_neighbors, f, indent=2)
        logger.info(f"  Saved debug neighbors to {debug_json.name}")

        # 5. Generate export_summary.json
        # Count noise: faces with the NOISE_LABEL cluster_id in the dataframe.
        n_noise = int((faces_df['cluster_id'] == NOISE_LABEL).sum())

        # Add validation checks to summary
        validations = {
            'faces_count_matches_face_records': len(faces_df) == len(face_records),
            'crops_exist_for_all_faces': saved_count == len(face_records),
            'no_null_image_paths': all(f.image_path is not None for f in face_records),
            'face_ids_sequential': list(faces_df['face_id']) == list(range(len(face_records))),
            'debug_neighbors_complete': len(debug_neighbors.get('within_closest', {})) > 0
        }

        summary = {
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(context.source_directory),
            'embeddings_dir': str(crops_dir),
            'n_faces': len(face_records),
            'n_clusters': cluster_result.n_clusters,
            'n_noise': n_noise,
            'validations': validations,
            'config': {
                k: v for k, v in config.items()
                if k != 'output_dir'  # Don't include output_dir in config
            }
        }
        summary_json = output_dir / 'export_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Saved export summary to {summary_json.name}")

        # Verify validations passed
        failed_validations = [k for k, v in validations.items() if not v]
        if failed_validations:
            logger.warning(f"Failed validations: {failed_validations}")

        # Write to context
        context.export_directory = str(output_dir)
        context.faces_csv_path = str(faces_csv)
        context.clusters_csv_path = str(clusters_csv)

        logger.info("=" * 70)
        logger.info("EXPORT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Faces: {len(faces_df)}")
        logger.info(f"Clusters: {len(clusters_df)}")
        logger.info(f"Face crops: {saved_count}")
        logger.info(f"Validations: {sum(validations.values())}/{len(validations)} passed")
        logger.info("=" * 70)

        context.report_progress("export_for_labeling", 1.0,
                               f"Exported to {output_dir}")

    def _generate_faces_dataframe(self, face_records, cluster_result):
        """Generate faces.csv data from FaceRecord list and ClusterResult.

        Args:
            face_records: List of FaceRecord objects
            cluster_result: ClusterResult with cluster assignments

        Returns:
            DataFrame with face metadata
        """
        # Build node_idx -> cluster_id mapping
        # Note: cluster_result.labels uses node indices (0-based indices into core_indices)
        # We need to map from face_id to cluster_id

        # Get core_indices to map node_idx -> face_id
        # Actually, in our workflow: node_idx IS the index into face_records
        # cluster_result.labels[i] gives the cluster for face_records[core_indices[i]]

        # Build face_id -> cluster_id mapping
        face_id_to_cluster = {}
        for i, label in enumerate(cluster_result.labels):
            if not is_noise(label):
                face_id_to_cluster[i] = label

        faces_data = []
        for face_record in face_records:
            face_id = face_record.face_id
            cluster_id = face_id_to_cluster.get(face_id, NOISE_LABEL)

            faces_data.append({
                'face_id': face_id,
                'image_path': str(face_record.image_path),
                'cluster_id': cluster_id,
                'is_core': face_record.is_core,
                'bbox_x': face_record.bbox[0],
                'bbox_y': face_record.bbox[1],
            })

        return pd.DataFrame(faces_data)

    def _generate_clusters_dataframe(self, cluster_result, face_records):
        """Generate clusters.csv data from ClusterResult.

        Args:
            cluster_result: ClusterResult with clusters and stats
            face_records: List of FaceRecord objects (for exemplar lookup)

        Returns:
            DataFrame with cluster metadata
        """
        clusters_data = []
        for cluster_id in sorted(cluster_result.clusters.keys()):
            stats = cluster_result.cluster_stats[cluster_id]
            exemplar_indices = cluster_result.exemplars.get(cluster_id, [])

            # Get first exemplar's face_id (if exists)
            exemplar_face_id = exemplar_indices[0] if exemplar_indices else -1

            clusters_data.append({
                'cluster_id': cluster_id,
                'size': stats['size'],
                'diameter': stats['diameter'],
                'median_dist': stats['median_dist'],
                'exemplar_face_id': exemplar_face_id,
                'n_exemplars': len(exemplar_indices)
            })

        return pd.DataFrame(clusters_data)

    def _save_face_crops(self, face_records, crops_dir):
        """Save aligned face crops to directory from FaceRecord list.

        Args:
            face_records: List of FaceRecord objects
            crops_dir: Directory to save crops

        Returns:
            Number of crops saved
        """
        import cv2

        saved_count = 0
        for face_record in face_records:
            face_id = face_record.face_id
            aligned_face = face_record.aligned_face

            if aligned_face is None:
                logger.warning(f"No aligned face for face_id {face_id}")
                continue

            # Save crop
            crop_path = crops_dir / f"face_{face_id:04d}_aligned.jpg"

            # Convert RGB to BGR for OpenCV
            if len(aligned_face.shape) == 3 and aligned_face.shape[2] == 3:
                crop_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = aligned_face

            cv2.imwrite(str(crop_path), crop_bgr)
            saved_count += 1

        return saved_count

