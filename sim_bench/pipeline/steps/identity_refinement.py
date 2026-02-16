"""Identity Refinement step - post-clustering refinement for face assignments."""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.steps.attachment_strategies import (
    AttachmentStrategyFactory,
    AttachmentResult,
    ClusterInfo,
    cosine_distance,
)

logger = logging.getLogger(__name__)


@dataclass
class RefinementStats:
    """Statistics from identity refinement."""
    core_clusters: int = 0
    total_core_faces: int = 0
    noise_faces_input: int = 0
    auto_attached: int = 0
    user_overrides_applied: int = 0
    final_unassigned: int = 0


@register_step
class IdentityRefinementStep(BaseStep):
    """Refine face clustering with exemplar-based attachment.

    This step runs after cluster_people and:
    1. Separates core clusters from noise (cluster_id=-1)
    2. Selects high-quality exemplars per cluster
    3. Attempts to attach noise faces to clusters using configurable strategies
    4. Applies user overrides if available
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="identity_refinement",
            display_name="Identity Refinement",
            description="Refine face clustering with exemplar-based attachment.",
            category="people",
            requires={"people_clusters", "face_embeddings"},
            produces={"refined_people_clusters", "unassigned_faces"},
            depends_on=["cluster_people"],
            config_schema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable identity refinement"
                    },
                    "centroid_threshold": {
                        "type": "number",
                        "default": 0.38,
                        "description": "Max distance to centroid for attachment"
                    },
                    "exemplar_threshold": {
                        "type": "number",
                        "default": 0.40,
                        "description": "Max distance to exemplar for counting as match"
                    },
                    "reject_threshold": {
                        "type": "number",
                        "default": 0.45,
                        "description": "If best distance > this, never attach"
                    },
                    "exemplars_per_cluster": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of exemplars to select per cluster"
                    },
                    "exemplar_min_frontal_score": {
                        "type": "number",
                        "default": 0.6,
                        "description": "Minimum frontal score for exemplar selection"
                    },
                    "exemplar_selection_method": {
                        "type": "string",
                        "enum": ["quality", "diverse", "quality_diverse"],
                        "default": "quality_diverse",
                        "description": "Method for selecting exemplars"
                    },
                    "attachment_strategy": {
                        "type": "string",
                        "enum": ["centroid", "exemplar", "hybrid"],
                        "default": "hybrid",
                        "description": "Strategy for attachment decisions"
                    },
                    "small_cluster_threshold": {
                        "type": "integer",
                        "default": 3,
                        "description": "Clusters with <= N faces get relaxed exemplar requirements"
                    },
                    "small_cluster_min_matches": {
                        "type": "integer",
                        "default": 1,
                        "description": "Min exemplar matches for small clusters"
                    },
                    "preserve_original_clusters": {
                        "type": "boolean",
                        "default": True,
                        "description": "Keep original people_clusters unchanged"
                    },
                    "apply_user_overrides": {
                        "type": "boolean",
                        "default": True,
                        "description": "Apply stored user corrections"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Run identity refinement."""
        if not config.get("enabled", True):
            logger.info("Identity refinement disabled, passing through unchanged")
            context.refined_people_clusters = dict(context.people_clusters)
            return

        if not context.people_clusters:
            logger.warning("No people_clusters found, skipping refinement")
            return

        context.report_progress("identity_refinement", 0.1, "Separating core clusters from noise")

        # Step 1: Separate core clusters from noise
        core_clusters, noise_faces = self._separate_noise(context.people_clusters)

        stats = RefinementStats(
            core_clusters=len(core_clusters),
            total_core_faces=sum(len(faces) for faces in core_clusters.values()),
            noise_faces_input=len(noise_faces)
        )

        logger.info(f"Separated {stats.core_clusters} core clusters "
                   f"({stats.total_core_faces} faces) from {stats.noise_faces_input} noise faces")

        if not core_clusters:
            logger.warning("No core clusters found, all faces are noise")
            context.refined_people_clusters = {}
            context.unassigned_faces = noise_faces
            return

        context.report_progress("identity_refinement", 0.3, "Selecting exemplars")

        # Step 2: Select exemplars and compute centroids
        cluster_exemplars = {}
        cluster_centroids = {}

        for cluster_id, faces in core_clusters.items():
            exemplars = self._select_exemplars(faces, context.face_embeddings, config)
            cluster_exemplars[cluster_id] = exemplars

            centroid = self._compute_centroid(faces, context.face_embeddings)
            cluster_centroids[cluster_id] = centroid

        logger.info(f"Selected exemplars for {len(cluster_exemplars)} clusters")

        context.report_progress("identity_refinement", 0.5, "Attaching noise faces")

        # Step 3: Try to attach noise faces
        refined_clusters = {k: list(v) for k, v in core_clusters.items()}
        unassigned_faces = []
        attachment_decisions = {}

        strategy = AttachmentStrategyFactory.create(config.get("attachment_strategy", "hybrid"))

        for i, face in enumerate(noise_faces):
            face_key = self._face_key(face)
            embedding = self._get_face_embedding(face, context.face_embeddings)

            if embedding is None:
                attachment_decisions[face_key] = {
                    "attached": False,
                    "reason": "No embedding found"
                }
                unassigned_faces.append(face)
                continue

            best_result = self._try_attach_to_clusters(
                face, embedding, cluster_centroids, cluster_exemplars,
                context.face_embeddings, strategy, config
            )

            if best_result and best_result.attached:
                refined_clusters[best_result.cluster_id].append(face)
                face.cluster_id = best_result.cluster_id
                stats.auto_attached += 1
                attachment_decisions[face_key] = asdict(best_result)
            else:
                unassigned_faces.append(face)
                attachment_decisions[face_key] = asdict(best_result) if best_result else {
                    "attached": False,
                    "reason": "No clusters available"
                }

            if (i + 1) % 10 == 0:
                progress = 0.5 + (0.3 * (i + 1) / len(noise_faces))
                context.report_progress("identity_refinement", progress,
                                       f"Processed {i + 1}/{len(noise_faces)} noise faces")

        logger.info(f"Auto-attached {stats.auto_attached} faces, {len(unassigned_faces)} remain unassigned")

        context.report_progress("identity_refinement", 0.8, "Applying user overrides")

        # Step 4: Apply user overrides
        if config.get("apply_user_overrides", True) and context.user_overrides:
            refined_clusters, unassigned_faces, override_count = self._apply_user_overrides(
                refined_clusters, unassigned_faces, context.user_overrides,
                attachment_decisions, context
            )
            stats.user_overrides_applied = override_count
            logger.info(f"Applied {override_count} user overrides")

        stats.final_unassigned = len(unassigned_faces)

        # Store results
        if not config.get("preserve_original_clusters", True):
            context.people_clusters = refined_clusters

        context.refined_people_clusters = refined_clusters
        context.unassigned_faces = unassigned_faces
        context.cluster_exemplars = cluster_exemplars
        context.cluster_centroids = cluster_centroids
        context.attachment_decisions = attachment_decisions

        context.report_progress(
            "identity_refinement", 1.0,
            f"Refined {stats.core_clusters} clusters: "
            f"{stats.auto_attached} attached, {stats.final_unassigned} unassigned"
        )

        self._log_stats(stats)

    def _separate_noise(self, clusters: Dict[int, list]) -> tuple:
        """Separate core clusters from noise (cluster_id=-1)."""
        core_clusters = {}
        noise_faces = []

        for cluster_id, faces in clusters.items():
            if cluster_id == -1:
                noise_faces.extend(faces)
            else:
                core_clusters[cluster_id] = faces

        return core_clusters, noise_faces

    def _face_key(self, face) -> str:
        """Generate unique key for a face."""
        path_str = str(face.original_path).replace('\\', '/')
        return f"{path_str}:face_{face.face_index}"

    def _get_face_embedding(self, face, embeddings: dict) -> Optional[np.ndarray]:
        """Get embedding for a face."""
        face_key = self._face_key(face)
        embedding = embeddings.get(face_key)

        if embedding is not None:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm

        return None

    def _get_frontal_score(self, face, context: PipelineContext) -> float:
        """Get frontal score for a face."""
        if hasattr(face, 'frontal_score') and face.frontal_score is not None:
            return face.frontal_score

        face_key = self._face_key(face)
        path_str = str(face.original_path).replace('\\', '/')

        if hasattr(context, 'insightface_faces') and context.insightface_faces:
            face_data = context.insightface_faces.get(path_str, {})
            for f in face_data.get('faces', []):
                if f.get('face_index') == face.face_index:
                    return f.get('frontal_score', 0.5)

        return 0.5

    def _select_exemplars(
        self,
        faces: list,
        embeddings: dict,
        config: dict
    ) -> list:
        """Select exemplar faces for a cluster."""
        K = config.get("exemplars_per_cluster", 5)
        min_frontal = config.get("exemplar_min_frontal_score", 0.6)
        method = config.get("exemplar_selection_method", "quality_diverse")

        if len(faces) <= K:
            return list(faces)

        candidates = []
        for face in faces:
            frontal_score = getattr(face, 'frontal_score', None)
            if frontal_score is None:
                frontal_score = 0.5
            candidates.append((face, frontal_score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        high_quality = [(f, s) for f, s in candidates if s >= min_frontal]

        if len(high_quality) < K:
            high_quality = candidates[:K]

        if method == "quality":
            return [f for f, s in high_quality[:K]]
        elif method == "diverse":
            return self._diverse_selection(high_quality, embeddings, K)
        elif method == "quality_diverse":
            pool = high_quality[:K * 2]
            return self._diverse_selection(pool, embeddings, K)

        return [f for f, s in high_quality[:K]]

    def _diverse_selection(
        self,
        candidates: list,
        embeddings: dict,
        K: int
    ) -> list:
        """Select K diverse faces using farthest-point sampling."""
        if len(candidates) <= K:
            return [f for f, s in candidates]

        face_embeddings = []
        for face, score in candidates:
            emb = self._get_face_embedding(face, embeddings)
            if emb is not None:
                face_embeddings.append((face, emb))

        if len(face_embeddings) <= K:
            return [f for f, e in face_embeddings]

        selected = [face_embeddings[0]]

        while len(selected) < K:
            best_face = None
            best_min_dist = -1

            for face, emb in face_embeddings:
                if any(f is face for f, e in selected):
                    continue

                min_dist = min(
                    cosine_distance(emb, sel_emb)
                    for sel_face, sel_emb in selected
                )

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_face = (face, emb)

            if best_face:
                selected.append(best_face)
            else:
                break

        return [f for f, e in selected]

    def _compute_centroid(self, faces: list, embeddings: dict) -> np.ndarray:
        """Compute normalized centroid embedding for a cluster."""
        face_embeddings = []
        for face in faces:
            emb = self._get_face_embedding(face, embeddings)
            if emb is not None:
                face_embeddings.append(emb)

        if not face_embeddings:
            return np.zeros(512)

        centroid = np.mean(face_embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        return centroid

    def _try_attach_to_clusters(
        self,
        face,
        embedding: np.ndarray,
        centroids: dict,
        exemplars: dict,
        embeddings: dict,
        strategy,
        config: dict
    ) -> Optional[AttachmentResult]:
        """Try to attach a face to the best matching cluster."""
        best_result = None
        best_confidence = -1

        for cluster_id in centroids:
            centroid = centroids[cluster_id]
            cluster_exemplars = exemplars.get(cluster_id, [])

            exemplar_embeddings = []
            for ex in cluster_exemplars:
                ex_emb = self._get_face_embedding(ex, embeddings)
                if ex_emb is not None:
                    exemplar_embeddings.append(ex_emb)

            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                centroid=centroid,
                exemplar_embeddings=exemplar_embeddings
            )

            result = strategy.evaluate(embedding, cluster_info, config)

            if result.attached and result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result

        if best_result is None and centroids:
            first_cluster_id = next(iter(centroids))
            best_result = AttachmentResult(
                attached=False,
                cluster_id=first_cluster_id,
                reason="No cluster met attachment criteria"
            )

        return best_result

    def _apply_user_overrides(
        self,
        clusters: Dict[int, list],
        unassigned: list,
        overrides: list,
        decisions: dict,
        context: PipelineContext
    ) -> tuple:
        """Apply user corrections to clustering."""
        override_count = 0

        all_faces = {}
        for cluster_id, faces in clusters.items():
            for face in faces:
                all_faces[self._face_key(face)] = (face, cluster_id)
        for face in unassigned:
            all_faces[self._face_key(face)] = (face, None)

        next_cluster_id = max(clusters.keys(), default=-1) + 1

        for override in sorted(overrides, key=lambda o: o.get('created_at', '')):
            face_key = override.get('face_key')
            override_type = override.get('override_type')

            if face_key not in all_faces:
                continue

            face, current_cluster = all_faces[face_key]

            if override_type == "attach":
                target = override.get('to_cluster')
                if target is None or target not in clusters:
                    continue

                if current_cluster is not None:
                    clusters[current_cluster] = [
                        f for f in clusters[current_cluster]
                        if self._face_key(f) != face_key
                    ]
                else:
                    unassigned = [f for f in unassigned if self._face_key(f) != face_key]

                clusters[target].append(face)
                all_faces[face_key] = (face, target)
                override_count += 1

                decisions[face_key] = {
                    "attached": True,
                    "cluster_id": target,
                    "method": "user_assigned",
                    "confidence": 1.0,
                    "reason": "User manually assigned"
                }

            elif override_type == "split":
                if current_cluster is None:
                    continue

                clusters[current_cluster] = [
                    f for f in clusters[current_cluster]
                    if self._face_key(f) != face_key
                ]
                unassigned.append(face)
                all_faces[face_key] = (face, None)
                override_count += 1

                decisions[face_key] = {
                    "attached": False,
                    "method": "user_split",
                    "reason": "User marked as not belonging"
                }

            elif override_type == "reassign":
                target = override.get('to_cluster')
                if target is None or target not in clusters:
                    continue

                if current_cluster is not None:
                    clusters[current_cluster] = [
                        f for f in clusters[current_cluster]
                        if self._face_key(f) != face_key
                    ]
                else:
                    unassigned = [f for f in unassigned if self._face_key(f) != face_key]

                clusters[target].append(face)
                all_faces[face_key] = (face, target)
                override_count += 1

                decisions[face_key] = {
                    "attached": True,
                    "cluster_id": target,
                    "method": "user_reassigned",
                    "confidence": 1.0,
                    "reason": f"User moved from cluster {current_cluster} to {target}"
                }

            elif override_type == "create":
                new_id = override.get('new_cluster_id') or next_cluster_id
                next_cluster_id = max(next_cluster_id, new_id + 1)

                if new_id not in clusters:
                    clusters[new_id] = []

                if current_cluster is not None:
                    clusters[current_cluster] = [
                        f for f in clusters[current_cluster]
                        if self._face_key(f) != face_key
                    ]
                else:
                    unassigned = [f for f in unassigned if self._face_key(f) != face_key]

                clusters[new_id].append(face)
                all_faces[face_key] = (face, new_id)
                override_count += 1

                decisions[face_key] = {
                    "attached": True,
                    "cluster_id": new_id,
                    "method": "user_created_cluster",
                    "confidence": 1.0,
                    "reason": f"User created new cluster {new_id}"
                }

        clusters = {k: v for k, v in clusters.items() if v}

        return clusters, unassigned, override_count

    def _log_stats(self, stats: RefinementStats) -> None:
        """Log refinement statistics."""
        logger.info("=" * 60)
        logger.info("IDENTITY REFINEMENT STATS")
        logger.info("=" * 60)
        logger.info(f"Core clusters: {stats.core_clusters}")
        logger.info(f"Core faces: {stats.total_core_faces}")
        logger.info(f"Noise faces input: {stats.noise_faces_input}")
        logger.info(f"Auto-attached: {stats.auto_attached}")
        logger.info(f"User overrides: {stats.user_overrides_applied}")
        logger.info(f"Final unassigned: {stats.final_unassigned}")
        logger.info("=" * 60)
