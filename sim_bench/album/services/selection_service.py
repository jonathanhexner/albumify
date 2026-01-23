"""Best image selection service.

Pure business logic for selecting best images from clusters.
No UI dependencies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sim_bench.model_hub import ImageMetrics, ModelHub

logger = logging.getLogger(__name__)


class SelectionService:
    """Select best images from clusters based on weighted scoring."""

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        sel = config.get('album', {}).get('selection', {})
        self._images_per_cluster = sel.get('images_per_cluster', 1)
        self._ava_weight = sel.get('ava_weight', 0.5)
        self._iqa_weight = sel.get('iqa_weight', 0.2)
        self._portrait_weight = sel.get('portrait_weight', 0.3)
        self._use_siamese = sel.get('use_siamese_tiebreaker', True)

        logger.info(
            f"SelectionService: {self._images_per_cluster}/cluster, "
            f"weights(ava={self._ava_weight}, iqa={self._iqa_weight}, portrait={self._portrait_weight})"
        )

    def select_best(
        self,
        clusters: Dict[int, List[str]],
        metrics: Dict[str, ImageMetrics],
        hub: Optional[ModelHub] = None
    ) -> List[str]:
        """Select best images from each cluster."""
        selected = []
        for cluster_id, images in clusters.items():
            cluster_best = self._select_from_cluster(images, metrics, hub)
            selected.extend(cluster_best)

        logger.info(f"Selected {len(selected)} images from {len(clusters)} clusters")
        return selected

    def compute_score(self, metric: ImageMetrics) -> float:
        """Compute weighted composite score for an image."""
        return metric.get_composite_score(
            ava_weight=self._ava_weight,
            iqa_weight=self._iqa_weight,
            portrait_weight=self._portrait_weight
        )

    def _select_from_cluster(
        self,
        cluster_images: List[str],
        metrics: Dict[str, ImageMetrics],
        hub: Optional[ModelHub]
    ) -> List[str]:
        """Select best N images from a single cluster."""
        if len(cluster_images) <= self._images_per_cluster:
            return cluster_images

        scored = []
        for img_path in cluster_images:
            metric = metrics.get(img_path)
            if metric:
                scored.append((img_path, self.compute_score(metric)))

        scored.sort(key=lambda x: x[1], reverse=True)

        if self._use_siamese and hub and len(scored) >= 2:
            top_candidates = scored[:min(3, len(scored))]
            winner = self._select_with_siamese(top_candidates, hub)
            if winner:
                return [winner]

        return [path for path, _ in scored[:self._images_per_cluster]]

    def _select_with_siamese(
        self,
        candidates: List[tuple],
        hub: ModelHub
    ) -> Optional[str]:
        """Use Siamese model to select best image from top candidates."""
        if len(candidates) < 2:
            return candidates[0][0] if candidates else None

        paths = [path for path, _ in candidates]
        wins = {path: 0 for path in paths}

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                result = hub.compare_images(paths[i], paths[j])
                if result and 'winner' in result:
                    winner_path = paths[i] if result['winner'] == 1 else paths[j]
                    wins[winner_path] += 1
                    logger.debug(f"Siamese: {Path(paths[i]).name} vs {Path(paths[j]).name} -> {Path(winner_path).name}")

        best = max(wins, key=wins.get)
        logger.info(f"Siamese selection: {Path(best).name} (wins: {wins[best]}/{len(paths)-1})")
        return best

    def select_diverse(
        self,
        cluster_images: List[str],
        metrics: Dict[str, ImageMetrics],
        diversity_threshold: float = 0.1
    ) -> List[str]:
        """Select diverse images avoiding very similar ones."""
        embeddings, valid_paths = [], []

        for path in cluster_images:
            metric = metrics.get(path)
            if metric and metric.scene_embedding is not None:
                embeddings.append(metric.scene_embedding)
                valid_paths.append(path)

        if not embeddings:
            return cluster_images[:self._images_per_cluster]

        embeddings = np.array(embeddings)
        selected_indices = [0]

        for _ in range(self._images_per_cluster - 1):
            if len(selected_indices) >= len(valid_paths):
                break

            selected_embs = embeddings[selected_indices]
            max_min_dist, best_idx = -1, None

            for i in range(len(valid_paths)):
                if i in selected_indices:
                    continue
                distances = np.linalg.norm(selected_embs - embeddings[i], axis=1)
                min_dist = np.min(distances)
                if min_dist > max_min_dist:
                    max_min_dist, best_idx = min_dist, i

            if best_idx is not None and max_min_dist >= diversity_threshold:
                selected_indices.append(best_idx)
            else:
                break

        return [valid_paths[i] for i in selected_indices]
