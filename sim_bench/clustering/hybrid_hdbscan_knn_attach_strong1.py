"""
Hybrid HDBSCAN + Local Cohesion clustering - Strong single-exemplar attach variant.

Same as hybrid_hdbscan_knn but with additional attachment rule:
- Primary (unchanged): noise joins cluster if ≥2 exemplars within T
- Secondary (new): noise joins cluster if 1 exemplar within stricter threshold (0.8 * T)

Why: reduces leftover fragments/noise that would otherwise form tiny clusters later.

Distance metric: Cosine distance = 1 - cosine_similarity, clipped to [0, 2].
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field

import numpy as np

from sim_bench.clustering.hybrid_hdbscan_knn import (
    HybridHDBSCANKNN,
    ClusterState,
    AttachDecision,
)
from sim_bench.clustering.distance_utils import cosine_distance_to_set

logger = logging.getLogger(__name__)


class HybridHDBSCANKNNAttachStrong1(HybridHDBSCANKNN):
    """Hybrid HDBSCAN + Local Cohesion with strong single-exemplar attachment."""

    doc_explanation = """
Variant of hybrid_hdbscan_knn with additional single-exemplar attachment rule.

Primary Rule: Noise attaches if >=2 exemplars within cluster's T.
Secondary Rule: Noise attaches if 1 exemplar within stricter threshold (0.8 * T).

The secondary rule reduces leftover noise fragments that would otherwise form
tiny clusters. Requires stronger evidence (closer distance) when relying on
a single exemplar match.
"""

    decision_parameters = {
        **HybridHDBSCANKNN.decision_parameters,
        "attach_strong1_multiplier": {
            "description": "Multiplier for single-exemplar attach threshold",
            "default": 0.8,
            "decision_role": "Single exemplar attaches if dist <= T * this (0.8 = 20% stricter)"
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Multiplier for stricter single-exemplar threshold (0.8 = 80% of T)
        self.attach_strong1_multiplier = self.params.get('attach_strong1_multiplier', 0.8)

    def _attach_noise(
        self,
        labels: np.ndarray,
        cluster_states: Dict[int, ClusterState],
        features: np.ndarray,
        collect_decisions: bool = False
    ) -> Tuple[np.ndarray, int, List[AttachDecision]]:
        """Attach noise points with additional strong single-exemplar rule.

        Primary: attach if ≥2 exemplars within T
        Secondary: attach if 1 exemplar within (attach_strong1_multiplier * T)
        """
        final_labels = labels.copy()
        noise_indices = np.where(labels == -1)[0]
        attach_decisions: List[AttachDecision] = []

        if len(noise_indices) == 0 or len(cluster_states) == 0:
            return final_labels, 0, attach_decisions

        n_attached = 0

        for noise_idx in noise_indices:
            noise_embedding = features[noise_idx:noise_idx + 1]
            best_cluster: Optional[int] = None
            best_match_count = 0
            best_min_dist = float('inf')
            best_strong1 = False  # Track if best match is via strong1 rule
            candidates: List[Dict[str, Any]] = []

            for label, state in cluster_states.items():
                # Compute cosine distances to exemplars
                distances = cosine_distance_to_set(noise_embedding, state.exemplar_embeddings)

                # Count exemplars within threshold
                within_threshold = int(np.sum(distances <= state.threshold))
                min_dist = float(np.min(distances))

                # Stricter threshold for single-exemplar match
                strong1_threshold = self.attach_strong1_multiplier * state.threshold
                within_strong1 = int(np.sum(distances <= strong1_threshold))

                # Determine required matches
                n_exemplars = len(state.exemplar_indices)
                if n_exemplars < self.attach_min_exemplars:
                    required_matches = 1
                else:
                    required_matches = self.attach_min_exemplars

                # Primary rule: ≥required_matches within T
                qualifies_primary = within_threshold >= required_matches

                # Secondary rule: ≥1 within stricter threshold (strong1)
                qualifies_strong1 = within_strong1 >= 1

                qualifies = qualifies_primary or qualifies_strong1

                # Collect candidate info for debugging
                if collect_decisions:
                    candidates.append({
                        'cluster': int(label),
                        'threshold': float(state.threshold),
                        'strong1_threshold': float(strong1_threshold),
                        'matches': within_threshold,
                        'matches_strong1': within_strong1,
                        'required': required_matches,
                        'min_dist': min_dist,
                        'qualifies': qualifies,
                        'qualifies_primary': qualifies_primary,
                        'qualifies_strong1': qualifies_strong1,
                        'exemplar_distances': distances.tolist()
                    })

                # Check if this cluster qualifies
                if qualifies:
                    # Prefer primary matches over strong1 matches
                    # Then prefer more matches, then closer distance
                    is_primary = qualifies_primary
                    current_is_primary = not best_strong1

                    if is_primary and not current_is_primary:
                        # Primary match beats strong1 match
                        best_cluster = label
                        best_match_count = within_threshold
                        best_min_dist = min_dist
                        best_strong1 = False
                    elif is_primary == current_is_primary:
                        # Same rule type: prefer more matches, then closer
                        match_count = within_threshold if is_primary else within_strong1
                        if (match_count > best_match_count or
                            (match_count == best_match_count and min_dist < best_min_dist)):
                            best_cluster = label
                            best_match_count = match_count
                            best_min_dist = min_dist
                            best_strong1 = not is_primary
                    elif not is_primary and best_cluster is None:
                        # First strong1 match when no primary match yet
                        best_cluster = label
                        best_match_count = within_strong1
                        best_min_dist = min_dist
                        best_strong1 = True

            if best_cluster is not None:
                final_labels[noise_idx] = best_cluster
                n_attached += 1

            # Record decision
            if collect_decisions:
                attach_decisions.append(AttachDecision(
                    face_idx=int(noise_idx),
                    attached_to=int(best_cluster) if best_cluster is not None else None,
                    candidates=candidates
                ))

        return final_labels, n_attached, attach_decisions
