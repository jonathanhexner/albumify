# Multi-Model Image Ranking Integration Proposal

## Overview

This document proposes a strategy to integrate three complementary image quality models into the **photo organization application** that clusters similar images and selects the best representative from each cluster.

### Application Context

The app workflow:
1. **User imports a photo library** (e.g., vacation photos, event photos)
2. **Clustering stage** - Groups similar images (duplicates, burst shots, same scene)
3. **Selection stage** - Picks the best image from each cluster
4. **Result** - Curated set of unique, high-quality images

This proposal focuses on **Stage 3: Selection** - using multiple models to choose the best image from each cluster.

### Available Models

| Model | Type | Speed | Strength | Output |
|-------|------|-------|----------|--------|
| **Rule-Based IQA** | Hand-crafted | Fast (~10ms) | Technical quality (sharpness, exposure, contrast) | Score 0-1 |
| **ResNet+MLP (AVA)** | Neural | Medium (~100ms) | Aesthetic quality (composition, appeal) | Score 1-10 |
| **Siamese (Photo-Triage)** | Neural | Slow (~150ms/pair) | Pairwise preference (which is better?) | Binary + confidence |

---

## Proposed Architecture: Cascaded Cluster Selection

### Design Principles

1. **Per-cluster evaluation** - Each cluster is ranked independently
2. **Speed-first filtering** - Use fast IQA to eliminate obviously poor images within cluster
3. **Progressive refinement** - Apply expensive models only to top candidates
4. **Configurable combination** - Weighted ensemble of all model signals

### End-to-End Application Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHOTO ORGANIZATION APP                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: IMPORT & CLUSTERING                                               │
│  ─────────────────────────────                                              │
│  • Load images from folder                                                  │
│  • Extract features (DINOv2, OpenCLIP, etc.)                               │
│  • Cluster similar images (DBSCAN, HDBSCAN, etc.)                          │
│  • Result: Dict[cluster_id, List[image_paths]]                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-MODEL SELECTION (per cluster)                               │
│  ────────────────────────────────────────────                               │
│                                                                             │
│   For each cluster:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step A: IQA Pre-Filter (Fast)                                      │  │
│   │  • Score all images in cluster for technical quality                │  │
│   │  • Remove obviously bad images (blur, bad exposure)                 │  │
│   │  • Keep top candidates or those above threshold                     │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step B: AVA Aesthetic Scoring (Medium)                             │  │
│   │  • Score remaining candidates for aesthetic appeal                  │  │
│   │  • Rank by aesthetic score                                          │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step C: Siamese Refinement (for top 2-3 only)                      │  │
│   │  • Pairwise comparisons among top candidates                        │  │
│   │  • Break ties, validate AVA ranking                                 │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Step D: Combine Scores                                             │  │
│   │  • Weighted combination: IQA + AVA + Siamese                        │  │
│   │  • Select best image for cluster                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: OUTPUT                                                            │
│  ───────────────                                                            │
│  • Display best image per cluster                                           │
│  • Option to see runner-ups                                                 │
│  • Export curated collection                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Architecture Overview (SOLID Principles)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEPENDENCY INJECTION & INTERFACES                                          │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ BaseQualityModel│  │ BaseQualityModel│  │ BaseQualityModel│             │
│  │   (Protocol)    │  │   (Protocol)    │  │   (Protocol)    │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐             │
│  │ RuleBasedIQA    │  │ AVAQualityModel │  │ SiameseQuality  │             │
│  │ (injected)      │  │ (injected)      │  │ (injected)      │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
              ┌─────────────────────────────────────────┐
              │         MultiModelRanker                │
              │  (receives models via constructor)      │
              └─────────────────────────────────────────┘
```

### Configuration (Data Classes)

```python
# sim_bench/ranking/config.py

from dataclasses import dataclass


@dataclass
class IQAFilterConfig:
    """Configuration for IQA pre-filtering stage."""

    min_threshold: float = 0.3
    skip_filter_below_n: int = 5
    min_candidates: int = 3


@dataclass
class SiameseConfig:
    """Configuration for Siamese refinement stage."""

    enabled: bool = True
    top_n: int = 3


@dataclass
class WeightConfig:
    """Weights for combining model scores."""

    iqa: float = 0.2
    ava: float = 0.5
    siamese: float = 0.3

    def __post_init__(self):
        total = self.iqa + self.ava + self.siamese
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"


@dataclass
class MultiModelConfig:
    """Complete configuration for multi-model ranking."""

    iqa_filter: IQAFilterConfig
    siamese: SiameseConfig
    weights: WeightConfig

    @classmethod
    def balanced(cls) -> "MultiModelConfig":
        return cls(
            iqa_filter=IQAFilterConfig(),
            siamese=SiameseConfig(),
            weights=WeightConfig(iqa=0.2, ava=0.5, siamese=0.3),
        )

    @classmethod
    def technical(cls) -> "MultiModelConfig":
        return cls(
            iqa_filter=IQAFilterConfig(),
            siamese=SiameseConfig(),
            weights=WeightConfig(iqa=0.4, ava=0.4, siamese=0.2),
        )

    @classmethod
    def aesthetic(cls) -> "MultiModelConfig":
        return cls(
            iqa_filter=IQAFilterConfig(),
            siamese=SiameseConfig(),
            weights=WeightConfig(iqa=0.1, ava=0.6, siamese=0.3),
        )
```

### Result Data Classes

```python
# sim_bench/ranking/results.py

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RankingScores:
    """Scores from each model for a single cluster."""

    iqa: Dict[str, float]
    ava: Dict[str, float]
    siamese_win_rates: Dict[str, float]
    combined: Dict[str, float]


@dataclass
class ClusterRankingResult:
    """Ranking result for a single cluster."""

    ranked_images: List[tuple[str, float]]
    filtered_out: List[str]
    scores: RankingScores


@dataclass
class ClusterSelectionResult:
    """Selection result for a single cluster."""

    best_images: List[str]
    runner_ups: List[str]
    scores: Optional[RankingScores]


@dataclass
class BatchSelectionResult:
    """Result of selecting best images from multiple clusters."""

    selections: Dict[str, ClusterSelectionResult]
    total_clusters: int
    total_selected: int
```

### Stage Components (Single Responsibility)

```python
# sim_bench/ranking/stages.py

from typing import Dict, List, Protocol, Tuple


class QualityScorer(Protocol):
    """Protocol for single-image quality scoring."""

    def score_image(self, image_path: str) -> float:
        ...


class PairwiseComparer(Protocol):
    """Protocol for pairwise image comparison."""

    def compare_images(self, img1: str, img2: str) -> Dict:
        ...


class IQAFilter:
    """Filters images by technical quality score."""

    def __init__(
        self,
        scorer: QualityScorer,
        config: IQAFilterConfig,
    ):
        self._scorer = scorer
        self._config = config

    def filter(
        self,
        image_paths: List[str],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Filter images, return candidates and all scores."""
        scores = {p: self._scorer.score_image(p) for p in image_paths}

        if len(image_paths) <= self._config.skip_filter_below_n:
            return image_paths, scores

        candidates = [
            (p, s) for p, s in scores.items()
            if s >= self._config.min_threshold
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if len(candidates) < self._config.min_candidates:
            sorted_all = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            candidates = sorted_all[:self._config.min_candidates]

        return [p for p, _ in candidates], scores


class AestheticRanker:
    """Ranks images by aesthetic quality score."""

    def __init__(self, scorer: QualityScorer):
        self._scorer = scorer

    def rank(
        self,
        image_paths: List[str],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Rank images, return ordered list and scores."""
        scores = {p: self._scorer.score_image(p) for p in image_paths}
        ranked = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
        return ranked, scores


class SiameseRefiner:
    """Refines ranking using pairwise comparisons."""

    def __init__(
        self,
        comparer: PairwiseComparer,
        config: SiameseConfig,
    ):
        self._comparer = comparer
        self._config = config

    def refine(
        self,
        ranked_candidates: List[str],
    ) -> Dict[str, float]:
        """Run tournament on top candidates, return win rates."""
        if not self._config.enabled:
            return {}

        top_n = min(self._config.top_n, len(ranked_candidates))
        if top_n < 2:
            return {}

        candidates = ranked_candidates[:top_n]
        win_counts = {p: 0 for p in candidates}

        for i, path1 in enumerate(candidates):
            for path2 in candidates[i + 1:]:
                result = self._comparer.compare_images(path1, path2)
                winner = path1 if result["prediction"] == 1 else path2
                win_counts[winner] += 1

        max_wins = top_n - 1
        return {p: wins / max_wins for p, wins in win_counts.items()}
```

### Score Combiner (Single Responsibility)

```python
# sim_bench/ranking/combiner.py

from typing import Dict, List, Tuple


class ScoreCombiner:
    """Combines scores from multiple models."""

    def __init__(self, weights: WeightConfig):
        self._weights = weights

    def combine(
        self,
        candidates: List[str],
        iqa_scores: Dict[str, float],
        ava_scores: Dict[str, float],
        siamese_win_rates: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """Combine all scores, return sorted (path, combined_score) list."""
        combined = []

        for path in candidates:
            iqa_norm = iqa_scores.get(path, 0.0)
            ava_norm = (ava_scores.get(path, 5.0) - 1.0) / 9.0

            if path in siamese_win_rates:
                score = (
                    self._weights.iqa * iqa_norm
                    + self._weights.ava * ava_norm
                    + self._weights.siamese * siamese_win_rates[path]
                )
            else:
                weight_sum = self._weights.iqa + self._weights.ava
                score = (
                    (self._weights.iqa / weight_sum) * iqa_norm
                    + (self._weights.ava / weight_sum) * ava_norm
                )

            combined.append((path, score))

        return sorted(combined, key=lambda x: x[1], reverse=True)
```

---

## Unified Multi-Model Ranker

```python
# sim_bench/ranking/ranker.py

from typing import List


class MultiModelRanker:
    """
    Cascaded multi-model image ranker.

    Composes IQAFilter, AestheticRanker, SiameseRefiner, and ScoreCombiner.
    All dependencies injected via constructor.
    """

    def __init__(
        self,
        iqa_filter: IQAFilter,
        aesthetic_ranker: AestheticRanker,
        siamese_refiner: SiameseRefiner,
        score_combiner: ScoreCombiner,
    ):
        self._iqa_filter = iqa_filter
        self._aesthetic_ranker = aesthetic_ranker
        self._siamese_refiner = siamese_refiner
        self._score_combiner = score_combiner

    def rank_cluster(self, image_paths: List[str]) -> ClusterRankingResult:
        """Rank images within a cluster using cascaded model evaluation."""
        # Stage 1: IQA filtering
        candidates, iqa_scores = self._iqa_filter.filter(image_paths)

        # Stage 2: AVA scoring
        ranked_candidates, ava_scores = self._aesthetic_ranker.rank(candidates)

        # Stage 3: Siamese refinement
        siamese_win_rates = self._siamese_refiner.refine(ranked_candidates)

        # Combine scores
        final_ranking = self._score_combiner.combine(
            candidates=candidates,
            iqa_scores=iqa_scores,
            ava_scores=ava_scores,
            siamese_win_rates=siamese_win_rates,
        )

        filtered_out = [p for p in image_paths if p not in candidates]

        return ClusterRankingResult(
            ranked_images=final_ranking,
            filtered_out=filtered_out,
            scores=RankingScores(
                iqa=iqa_scores,
                ava=ava_scores,
                siamese_win_rates=siamese_win_rates,
                combined=dict(final_ranking),
            ),
        )
```

### Factory for Easy Construction

```python
# sim_bench/ranking/factory.py

from typing import Protocol


class MultiModelRankerFactory:
    """Factory to construct MultiModelRanker with proper dependencies."""

    def __init__(
        self,
        iqa_scorer: QualityScorer,
        ava_scorer: QualityScorer,
        siamese_comparer: PairwiseComparer,
    ):
        self._iqa_scorer = iqa_scorer
        self._ava_scorer = ava_scorer
        self._siamese_comparer = siamese_comparer

    def create(self, config: MultiModelConfig) -> MultiModelRanker:
        """Create a configured MultiModelRanker instance."""
        iqa_filter = IQAFilter(
            scorer=self._iqa_scorer,
            config=config.iqa_filter,
        )

        aesthetic_ranker = AestheticRanker(
            scorer=self._ava_scorer,
        )

        siamese_refiner = SiameseRefiner(
            comparer=self._siamese_comparer,
            config=config.siamese,
        )

        score_combiner = ScoreCombiner(
            weights=config.weights,
        )

        return MultiModelRanker(
            iqa_filter=iqa_filter,
            aesthetic_ranker=aesthetic_ranker,
            siamese_refiner=siamese_refiner,
            score_combiner=score_combiner,
        )
```

---

## Configuration

```python
@dataclass
class MultiModelConfig:
    # Model paths
    ava_checkpoint: str
    siamese_checkpoint: str
    device: str = 'cpu'

    # Stage 1: IQA filtering
    iqa_min_threshold: float = 0.3        # Minimum IQA score to keep
    skip_filter_threshold: int = 5        # Skip filtering for clusters <= this size
    min_candidates: int = 3               # Always keep at least this many

    # Stage 3: Siamese refinement
    use_siamese_refinement: bool = True
    siamese_top_n: int = 5                # Number of top candidates to refine

    # Score combination weights (should sum to 1.0)
    weight_iqa: float = 0.2               # Technical quality weight
    weight_ava: float = 0.5               # Aesthetic quality weight
    weight_siamese: float = 0.3           # Pairwise preference weight
```

### Recommended Weight Presets

| Use Case | IQA | AVA | Siamese | Notes |
|----------|-----|-----|---------|-------|
| **Balanced** | 0.2 | 0.5 | 0.3 | Default - good for general use |
| **Technical Focus** | 0.4 | 0.4 | 0.2 | Prioritize sharpness/exposure |
| **Aesthetic Focus** | 0.1 | 0.6 | 0.3 | Prioritize composition/appeal |
| **Photo Triage** | 0.1 | 0.3 | 0.6 | Trust pairwise comparisons most |

---

## Integration with Existing App

### Primary Use Case: Select Best from Clusters

```python
# sim_bench/ranking/selector.py

from typing import Dict, List, Optional


class ClusterBestSelector:
    """
    Select the best image from each cluster using multi-model ranking.

    This is the primary interface for the photo organization app.
    Ranker is injected, not created internally.
    """

    def __init__(self, ranker: MultiModelRanker):
        self._ranker = ranker

    def select_best_per_cluster(
        self,
        clusters: Dict[str, List[str]],
        top_n: int = 1,
        include_scores: bool = False,
    ) -> BatchSelectionResult:
        """Process all clusters and select top N images from each."""
        selections: Dict[str, ClusterSelectionResult] = {}
        total_selected = 0

        for cluster_id, image_paths in clusters.items():
            result = self._select_from_cluster(
                image_paths=image_paths,
                top_n=top_n,
                include_scores=include_scores,
            )
            selections[cluster_id] = result
            total_selected += len(result.best_images)

        return BatchSelectionResult(
            selections=selections,
            total_clusters=len(clusters),
            total_selected=total_selected,
        )

    def _select_from_cluster(
        self,
        image_paths: List[str],
        top_n: int,
        include_scores: bool,
    ) -> ClusterSelectionResult:
        """Select best images from a single cluster."""
        if len(image_paths) == 1:
            return ClusterSelectionResult(
                best_images=image_paths,
                runner_ups=[],
                scores=None,
            )

        ranking = self._ranker.rank_cluster(image_paths)

        return ClusterSelectionResult(
            best_images=[p for p, _ in ranking.ranked_images[:top_n]],
            runner_ups=[p for p, _ in ranking.ranked_images[top_n:]],
            scores=ranking.scores if include_scores else None,
        )
```

### App Workflow Integration

```python
# app/core/services.py

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class OrganizationResult:
    """Result of organizing a photo library."""

    total_images: int
    num_clusters: int
    selected_images: List[str]
    cluster_details: BatchSelectionResult
    noise_images: List[str]


class PhotoOrganizationService:
    """
    Orchestrates the full photo organization pipeline.

    Dependencies injected via constructor.
    """

    def __init__(
        self,
        clusterer: ImageClusterer,
        selector: ClusterBestSelector,
    ):
        self._clusterer = clusterer
        self._selector = selector

    def organize(
        self,
        image_paths: List[str],
        top_n_per_cluster: int = 1,
    ) -> OrganizationResult:
        """Full pipeline: cluster images and select best from each."""
        # Step 1: Cluster similar images
        cluster_result = self._clusterer.cluster(image_paths)

        # Step 2: Select best from each cluster
        selection_result = self._selector.select_best_per_cluster(
            clusters=cluster_result.clusters,
            top_n=top_n_per_cluster,
            include_scores=True,
        )

        # Step 3: Compile results
        return OrganizationResult(
            total_images=len(image_paths),
            num_clusters=len(cluster_result.clusters),
            selected_images=[
                sel.best_images[0]
                for sel in selection_result.selections.values()
                if sel.best_images
            ],
            cluster_details=selection_result,
            noise_images=cluster_result.noise_images,
        )
```

### Wiring It All Together

```python
# app/bootstrap.py or app/dependencies.py

def create_photo_organization_service(
    ava_checkpoint: str,
    siamese_checkpoint: str,
    device: str = "cpu",
    config: MultiModelConfig = None,
) -> PhotoOrganizationService:
    """
    Bootstrap the photo organization service with all dependencies.

    This is the composition root - the only place where we wire dependencies.
    """
    config = config or MultiModelConfig.balanced()

    # Load models (these implement the Protocol interfaces)
    iqa_model = RuleBasedIQAModel()
    ava_model = AVAQualityModel.load(ava_checkpoint, device)
    siamese_model = SiameseQualityModel.load(siamese_checkpoint, device)

    # Create ranker via factory
    ranker_factory = MultiModelRankerFactory(
        iqa_scorer=iqa_model,
        ava_scorer=ava_model,
        siamese_comparer=siamese_model,
    )
    ranker = ranker_factory.create(config)

    # Create selector
    selector = ClusterBestSelector(ranker)

    # Create clusterer (existing)
    clusterer = ImageClusterer(
        feature_extractor=DINOv2Extractor(),
        clustering_method=DBSCANClusterer(),
    )

    return PhotoOrganizationService(
        clusterer=clusterer,
        selector=selector,
    )


# Usage in app
service = create_photo_organization_service(
    ava_checkpoint="models/ava_resnet50.pt",
    siamese_checkpoint="models/siamese_phototriage.pt",
    device="cuda",
    config=MultiModelConfig.balanced(),
)

result = service.organize(image_paths, top_n_per_cluster=1)
```

---

## Performance Considerations

### Typical Cluster Sizes

In photo organization, clusters are typically small (similar/duplicate photos):

| Scenario | Typical Cluster Size | Notes |
|----------|---------------------|-------|
| Burst shots | 3-10 images | Rapid fire capture |
| Same scene | 2-5 images | Slight variations |
| Group photo retakes | 3-8 images | "One more!" |
| Duplicates | 2-3 images | Copies/edits |

**Key insight:** Most clusters have < 10 images, making full Siamese comparison feasible.

### Complexity Analysis

| Stage | Time per Image | Total for N images, M candidates, T top |
|-------|---------------|----------------------------------------|
| IQA Filter | ~10ms | O(N) = 10ms * N |
| AVA Score | ~100ms | O(M) = 100ms * M |
| Siamese Refine | ~150ms per pair | O(T^2) = 150ms * T*(T-1)/2 |

**Example: Typical cluster of 5 images**
- IQA: 5 * 10ms = 50ms
- AVA: 5 * 100ms = 500ms (no filtering for small clusters)
- Siamese top 3: 3 pairs * 150ms = 450ms
- **Total per cluster: ~1 second**

**Example: Large burst of 20 images**
- IQA: 20 * 10ms = 200ms → filter to top 8
- AVA: 8 * 100ms = 800ms
- Siamese top 3: 3 pairs * 150ms = 450ms
- **Total: ~1.5 seconds** (vs 28.5 seconds for all-pairs Siamese on 20 images)

### Optimization Strategies

1. **Batch inference** - Process multiple images in batches for AVA
2. **Caching** - Cache IQA/AVA scores across sessions
3. **Early termination** - Stop Siamese when clear winner emerges
4. **GPU acceleration** - Move models to GPU for larger batches

---

## Future Extensions

### 1. Eye Open/Closed Detection

Add as a **hard filter** before IQA or as a **penalty factor**.

```python
class EyeStateDetector:
    """Detect closed eyes in portrait photos."""

    def detect(self, image_path: str) -> Dict:
        """
        Returns:
            {
                'has_faces': bool,
                'faces': [
                    {'bbox': [...], 'eyes_open': bool, 'confidence': float}
                ],
                'all_eyes_open': bool
            }
        """
        # Implementation options:
        # 1. MediaPipe Face Mesh (468 landmarks, includes eye aspect ratio)
        # 2. dlib + eye aspect ratio (EAR) calculation
        # 3. Train custom classifier on eye crops
        pass
```

**Integration approach:**

```python
# Option A: Hard filter - eliminate closed-eye photos
if not eye_detector.detect(path)['all_eyes_open']:
    filtered_out.append(path)
    continue

# Option B: Penalty factor - reduce score but don't eliminate
eye_result = eye_detector.detect(path)
if not eye_result['all_eyes_open']:
    penalty = 0.3  # Reduce combined score by 30%
    combined_score *= (1 - penalty)
```

### 2. Additional Feature Detectors

| Feature | Use Case | Integration |
|---------|----------|-------------|
| Smile detection | Group photos | Bonus for smiling faces |
| Blur detection (face-specific) | Portraits | Penalty for blurry faces |
| Red-eye detection | Flash photos | Hard filter or penalty |
| Blink detection | Burst photos | Hard filter |
| Expression quality | Portraits | Score modifier |

### 3. Content-Aware Weighting

Automatically adjust weights based on detected content:

```python
def get_adaptive_weights(image_path: str) -> Dict[str, float]:
    """Adjust model weights based on image content."""

    has_faces = face_detector.detect(image_path)
    is_landscape = aspect_ratio > 1.5

    if has_faces:
        # Portraits: prioritize Siamese (trained on portrait preferences)
        return {'iqa': 0.2, 'ava': 0.3, 'siamese': 0.5}
    elif is_landscape:
        # Landscapes: prioritize AVA (aesthetic composition)
        return {'iqa': 0.2, 'ava': 0.6, 'siamese': 0.2}
    else:
        # Default balanced
        return {'iqa': 0.2, 'ava': 0.5, 'siamese': 0.3}
```

---

## Implementation Roadmap

### Phase 1: Core Integration (MVP)
- [ ] Implement `MultiModelRanker` class
- [ ] Create `MultiModelConfig` dataclass
- [ ] Add `MultiModelRankTool` to agent tools
- [ ] Basic integration test

### Phase 2: App Integration
- [ ] Add configuration UI for weight presets
- [ ] Integrate with existing clustering workflow
- [ ] Add progress indicators for multi-stage processing
- [ ] Caching layer for scores

### Phase 3: Eye Detection
- [ ] Implement `EyeStateDetector` using MediaPipe
- [ ] Add as optional filter stage
- [ ] Train/fine-tune if needed for accuracy

### Phase 4: Advanced Features
- [ ] Content-aware adaptive weighting
- [ ] Batch processing optimization
- [ ] Score explanation/visualization
- [ ] A/B testing framework for weight tuning

---

## File Locations

New module structure (SOLID, single responsibility per file):

```
sim_bench/
└── ranking/
    ├── __init__.py                # Public exports
    ├── config.py                  # IQAFilterConfig, SiameseConfig, WeightConfig, MultiModelConfig
    ├── results.py                 # RankingScores, ClusterRankingResult, ClusterSelectionResult, BatchSelectionResult
    ├── protocols.py               # QualityScorer, PairwiseComparer (Protocol definitions)
    ├── stages.py                  # IQAFilter, AestheticRanker, SiameseRefiner
    ├── combiner.py                # ScoreCombiner
    ├── ranker.py                  # MultiModelRanker
    ├── selector.py                # ClusterBestSelector
    └── factory.py                 # MultiModelRankerFactory

app/
├── bootstrap.py                   # Composition root, dependency wiring
└── core/
    └── services.py                # PhotoOrganizationService (add to existing)

sim_bench/
└── detectors/                     # Future
    ├── __init__.py
    ├── protocols.py               # FaceDetector, EyeStateDetector protocols
    └── eye_state.py               # MediaPipe-based implementation
```

### Module Exports (`sim_bench/ranking/__init__.py`)

```python
from .config import (
    IQAFilterConfig,
    SiameseConfig,
    WeightConfig,
    MultiModelConfig,
)
from .results import (
    RankingScores,
    ClusterRankingResult,
    ClusterSelectionResult,
    BatchSelectionResult,
)
from .protocols import QualityScorer, PairwiseComparer
from .ranker import MultiModelRanker
from .selector import ClusterBestSelector
from .factory import MultiModelRankerFactory

__all__ = [
    "IQAFilterConfig",
    "SiameseConfig",
    "WeightConfig",
    "MultiModelConfig",
    "RankingScores",
    "ClusterRankingResult",
    "ClusterSelectionResult",
    "BatchSelectionResult",
    "QualityScorer",
    "PairwiseComparer",
    "MultiModelRanker",
    "ClusterBestSelector",
    "MultiModelRankerFactory",
]
```

---

## Summary

This proposal introduces a **cascaded multi-model ranking system** that:

1. **Optimizes compute** by using fast IQA filtering before expensive neural models
2. **Combines complementary signals** from technical (IQA), aesthetic (AVA), and preference (Siamese) models
3. **Provides flexibility** through configurable weights and thresholds
4. **Scales well** with O(N) + O(M) + O(T^2) complexity vs O(N^2) for pure pairwise
5. **Extends naturally** to future features like eye detection

The system integrates cleanly with the existing app architecture through the unified model interface and tool registry patterns already established in the codebase.
