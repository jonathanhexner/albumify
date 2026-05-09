# Contract: Merge Evidence API

## evaluate_pair_evidence() — new module-level function in merge.py

```python
def evaluate_pair_evidence(
    cluster_id_a: int,
    cluster_id_b: int,
    cluster_result: ClusterResult,
    distance_matrix: np.ndarray,
    config: PipelineConfig,
) -> Dict:
    """Compute 4-gate merge evidence for an arbitrary cluster pair.

    This is a standalone wrapper around ConservativeMerger._evaluate_merge_evidence().
    Can be called for any pair — does NOT require the pair to be a candidate.

    Returns:
        Dict with keys:
            valid: bool              — all 4 gates passed
            exemplar_dist: float     — min exemplar-exemplar distance
            merge_threshold: float   — threshold used
            passes_exemplar: bool    — exemplar gate result
            support: int             — cross-cluster support count
            required_support: int    — required support
            passes_support: bool     — support gate result
            passes_margin: bool      — margin gate result
            margin_gap: float|None   — gap to competitor
            margin_competitor_id: int|None
            post_diameter: float     — diameter after hypothetical merge
            max_allowed_diameter: float
            passes_diameter: bool    — diameter gate result
    """
```

### Implementation

Internally instantiates a `ConservativeMerger(config)`, computes `cluster_thresholds` and `global_threshold` from `cluster_result` (same as done in `merge()` method), then delegates to `_evaluate_merge_evidence()`.

### Usage from app layer

```python
from face_cluster.merge import evaluate_pair_evidence

evidence = evaluate_pair_evidence(cid_a, cid_b, cluster_result, distance_matrix, config)
# evidence has all gate results for rendering in the manual merge preview
```

## harvest_negative_pairs() — new module-level function in merge.py

```python
def harvest_negative_pairs(
    cluster_result: ClusterResult,
    distance_matrix: np.ndarray,
    candidate_threshold: float = 0.45,
    min_dist: float = 0.5,
    max_count: int = 100,
    exclude_pairs: Optional[Set[Tuple[int, int]]] = None,
    random_state: int = 42,
) -> List[Tuple[int, int, float]]:
    """Sample non-candidate cluster pairs as negative training examples.

    Returns:
        List of (cluster_a, cluster_b, exemplar_dist) tuples,
        sorted by exemplar_dist ascending. At most max_count.
    """
```

### Behaviour

1. Enumerate all cluster pairs where `min_exemplar_dist > candidate_threshold AND >= min_dist`.
2. Exclude pairs in `exclude_pairs` (already have human labels).
3. Uniform random sample up to `max_count`.
4. Return with exemplar distances for transparency.
