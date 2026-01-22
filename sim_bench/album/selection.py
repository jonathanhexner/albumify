"""
Best image selection from clusters.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from sim_bench.model_hub import ImageMetrics, ModelHub

logger = logging.getLogger(__name__)


class BestImageSelector:
    """
    Selects best images from each cluster based on weighted scoring.
    
    Config-only constructor - reads from config['album']['selection'].
    
    Scoring combines:
    - AVA aesthetic score
    - IQA technical quality
    - Portrait metrics (eyes open, smiling)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize selector with configuration.
        
        Args:
            config: Full configuration dictionary
        """
        self._config = config
        sel = config.get('album', {}).get('selection', {})
        
        self._images_per_cluster = sel.get('images_per_cluster', 1)
        self._ava_weight = sel.get('ava_weight', 0.5)
        self._iqa_weight = sel.get('iqa_weight', 0.2)
        self._portrait_weight = sel.get('portrait_weight', 0.3)
        self._use_siamese = sel.get('use_siamese_tiebreaker', True)
        
        logger.info(
            f"BestImageSelector: {self._images_per_cluster} per cluster, "
            f"weights(ava={self._ava_weight}, iqa={self._iqa_weight}, "
            f"portrait={self._portrait_weight})"
        )
    
    def select_from_clusters(
        self,
        clusters: Dict[int, List[str]],
        metrics: Dict[str, ImageMetrics],
        hub: Optional[ModelHub] = None
    ) -> List[str]:
        """
        Select best images from each cluster.
        
        Args:
            clusters: Dictionary mapping cluster_id -> list of image paths
            metrics: Dictionary mapping image_path -> ImageMetrics
            hub: Optional ModelHub for Siamese tiebreaker
        
        Returns:
            List of selected image paths
        """
        selected = []
        
        for cluster_id, cluster_images in clusters.items():
            cluster_selected = self._select_from_cluster(
                cluster_images,
                metrics,
                hub
            )
            selected.extend(cluster_selected)
        
        logger.info(
            f"Selected {len(selected)} images from {len(clusters)} clusters"
        )
        return selected
    
    def _select_from_cluster(
        self,
        cluster_images: List[str],
        metrics: Dict[str, ImageMetrics],
        hub: Optional[ModelHub]
    ) -> List[str]:
        """
        Select best N images from a single cluster.

        Uses Siamese model to compare top candidates when available.
        """
        if len(cluster_images) <= self._images_per_cluster:
            return cluster_images

        scored = []
        for img_path in cluster_images:
            metric = metrics.get(img_path)
            if not metric:
                continue

            score = self.compute_score(metric)
            scored.append((img_path, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Use Siamese to compare top candidates (compare top 3, select best)
        if self._use_siamese and hub and len(scored) >= 2:
            top_candidates = scored[:min(3, len(scored))]
            winner = self._select_best_with_siamese(top_candidates, hub)
            if winner:
                return [winner]

        # Fallback to score-based selection
        top_n = scored[:self._images_per_cluster]
        return [path for path, _ in top_n]
    
    def compute_score(self, metrics: ImageMetrics) -> float:
        """
        Compute weighted composite score for an image.
        
        Args:
            metrics: Image metrics
        
        Returns:
            Composite score (0-1 scale)
        """
        return metrics.get_composite_score(
            ava_weight=self._ava_weight,
            iqa_weight=self._iqa_weight,
            portrait_weight=self._portrait_weight
        )
    
    def _select_best_with_siamese(
        self,
        candidates: List[tuple],
        hub: ModelHub
    ) -> Optional[str]:
        """
        Use Siamese model to select best image from candidates.

        Compares all pairs and selects image with most wins.
        Falls back to highest-scored if Siamese unavailable.

        Args:
            candidates: List of (path, score) tuples, sorted by score descending
            hub: ModelHub with Siamese model

        Returns:
            Path of best image, or None if comparison fails
        """
        if len(candidates) < 2:
            return candidates[0][0] if candidates else None

        paths = [path for path, score in candidates]

        # Track wins for each candidate
        wins = {path: 0 for path in paths}

        try:
            # Compare all pairs
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    result = hub.compare_images(paths[i], paths[j])
                    if result and 'winner' in result:
                        winner_path = paths[i] if result['winner'] == 1 else paths[j]
                        wins[winner_path] += 1
                        logger.debug(
                            f"Siamese: {Path(paths[i]).name} vs {Path(paths[j]).name} "
                            f"-> winner: {Path(winner_path).name} (conf: {result.get('confidence', 0):.2f})"
                        )

            # Select candidate with most wins
            best_path = max(wins, key=wins.get)
            logger.info(
                f"Siamese selection: {Path(best_path).name} "
                f"(wins: {wins[best_path]}/{len(paths)-1})"
            )
            return best_path

        except Exception as e:
            logger.warning(f"Siamese comparison failed: {e}, using score-based selection")
            return None
    
    def select_diverse(
        self,
        cluster_images: List[str],
        metrics: Dict[str, ImageMetrics],
        diversity_threshold: float = 0.1
    ) -> List[str]:
        """
        Select diverse images from cluster (avoid very similar images).
        
        Args:
            cluster_images: Images in cluster
            metrics: Image metrics
            diversity_threshold: Minimum embedding distance between selections
        
        Returns:
            List of diverse image paths
        """
        import numpy as np
        
        embeddings = []
        valid_paths = []
        
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
            
            max_min_dist = -1
            best_idx = None
            
            for i in range(len(valid_paths)):
                if i in selected_indices:
                    continue
                
                distances = np.linalg.norm(
                    selected_embs - embeddings[i],
                    axis=1
                )
                min_dist = np.min(distances)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx is not None and max_min_dist >= diversity_threshold:
                selected_indices.append(best_idx)
            else:
                break
        
        return [valid_paths[i] for i in selected_indices]
