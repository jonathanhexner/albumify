# PhotoTriage Pairwise Quality Assessment Benchmark - Implementation Plan

**Date:** 2025-11-20
**Status:** Ready for Implementation
**Estimated Time:** ~8-10 hours development + testing

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Module Specifications](#module-specifications)
5. [Output Format](#output-format)
6. [Configuration](#configuration)
7. [Testing Strategy](#testing-strategy)
8. [Timeline &amp; Milestones](#timeline--milestones)
9. [Dependencies](#dependencies)
10. [Documentation](#documentation)

---

## Overview

Create a pairwise image quality assessment benchmark using the PhotoTriage labeled pairs dataset. The benchmark compares multiple methods on their ability to predict which image in a pair users prefer, based on:

- Ground Truth: User votes aggregated with `MaxVote` (winning image ID)
- Filtering: `Agreement >= 0.7` and `num_reviewers >= 2`
- Methods: CLIP attributes (7), Rule-based (7), ViT-base (1) = 15 total

**Key Requirements:**

- ✅ Output format matches existing quality benchmark system
- ✅ CLIP uses separate attributes (NO aggregation)
- ✅ Modular SOLID architecture (<300 lines per file)
- ✅ Bradley-Terry model (lower priority, optional)
- ✅ Sample-size support for quick testing

---

## Problem Statement

### Previous Issues

1. **Incorrect Methodology:** Previous assessment picked "most liked photo from series" - not pairwise comparison
2. **Wrong Aggregation:** Treated each JSON row as separate experiment instead of aggregating votes per pair
3. **Missing Ground Truth:** Didn't properly use MaxVote as the winning image

### New Approach

1. **Correct Ground Truth:** Use MaxVote from aggregated votes (already computed in CSV)
2. **Pairwise Prediction:** Methods predict which of two images is better
3. **Quality Filtering:** Only use pairs with high agreement (>=70%) and multiple reviewers (>=2)
4. **Proper Evaluation:** Binary classification accuracy + Bradley-Terry log-likelihood

---

## Architecture

### Directory Structure

```
sim_bench/quality_assessment/pairwise/
├── __init__.py                    # Empty
├── data_loader.py                 # Load & filter pairs CSV
├── ground_truth.py                # Bradley-Terry model
├── predictor.py                   # Base class for all methods
├── clip_predictor.py              # CLIP with attribute prompts
├── rule_based_predictor.py        # Sharpness, contrast, etc.
├── dl_predictor.py                # ViT-base wrapper
├── evaluator.py                   # Compute metrics
└── benchmark_runner.py            # Orchestrate benchmark

scripts/
└── run_pairwise_benchmark.py      # CLI entry point

configs/
└── pairwise_benchmark.yaml        # Configuration

docs/quality_assessment/
└── PAIRWISE_BENCHMARK.md          # Full documentation
```

### Design Patterns

1. **Strategy Pattern:** `PairwisePredictor` abstract base class with method-specific implementations
2. **Factory Pattern:** `benchmark_runner.py` creates predictor instances from config
3. **Template Method:** `PairwisePredictor.predict_batch()` provides default loop implementation
4. **Dependency Injection:** Config passed to constructors, not hardcoded

### SOLID Principles

- **Single Responsibility:** Each module has one clear purpose
- **Open/Closed:** Easy to add new methods by extending `PairwisePredictor`
- **Liskov Substitution:** All predictors interchangeable via base class
- **Interface Segregation:** Minimal interface (predict_pair, get_method_name)
- **Dependency Inversion:** Depend on abstractions (base class), not concrete implementations

---

## Module Specifications

### 1. `pairwise/data_loader.py` (~150 lines)

**Purpose:** Load and filter pairs CSV

**Class:** `PairwiseDataLoader`

**Methods:**

```python
		def load_pairs(csv_path: Path) -> pd.DataFrame:
    """
    Load pairs CSV with columns:
    - series_id, compareID1, compareID2
    - compareFile1, compareFile2
    - Agreement, num_reviewers, MaxVote

    Returns: DataFrame with all pairs
    """

def filter_by_agreement(
    df: pd.DataFrame,
    min_agreement: float = 0.7,
    min_reviewers: int = 2
) -> pd.DataFrame:
    """
    Filter to high-quality pairs:
    - Agreement >= min_agreement
    - num_reviewers >= min_reviewers

    Returns: Filtered DataFrame
    """

def sample_pairs(
    df: pd.DataFrame,
    n: int = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample N random pairs for testing.

    Args:
        n: Number of pairs (None = use all)
        random_state: For reproducibility

    Returns: Sampled DataFrame
    """

def get_image_paths(
    df: pd.DataFrame,
    image_root: Path
) -> pd.DataFrame:
    """
    Construct full image paths from:
    - series_id (e.g., "000001")
    - compareFile1/compareFile2 (e.g., "1-1.JPG")

    Format: {image_root}/{series_id}-{compareID+1:02d}.JPG

    Adds columns:
    - image_path_1: Full path to first image
    - image_path_2: Full path to second image
    - ground_truth_winner: 1 or 2 (from MaxVote)

    Returns: DataFrame with paths
    """
```

**Error Handling:**

- Check CSV exists and has required columns
- Validate image paths exist
- Warn if many images missing (but continue)

---

### 2. `pairwise/ground_truth.py` (~200 lines)

**Purpose:** Bradley-Terry model for probabilistic ground truth

**Class:** `BradleyTerryModel`

**Background:**
Bradley-Terry model estimates "strength" parameters θ for each image such that:

```
P(image i beats image j) = exp(θ_i) / (exp(θ_i) + exp(θ_j))
```

These parameters are learned via maximum likelihood estimation from pairwise comparison data.

**Methods:**

```python
def __init__(self):
    self.image_strengths: Dict[str, float] = {}
    self.fitted: bool = False

def fit(
    self,
    pairs_df: pd.DataFrame,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> None:
    """
    Fit BT model using iterative maximum likelihood.

    Args:
        pairs_df: DataFrame with columns:
            - series_id, compareID1, compareID2
            - LEFT, RIGHT (vote counts)
        max_iterations: Max iterations for convergence
        tolerance: Convergence threshold

    Algorithm:
        1. Initialize all θ = 0
        2. Iterate until convergence:
            θ_i^(t+1) = log(W_i / Σ_j P_ij)
            where W_i = wins for image i
                  P_ij = P(i beats j | current θ)
    """

def compute_pair_probability(
    self,
    image1_id: str,
    image2_id: str
) -> float:
    """
    Compute P(image1 beats image2).

    Returns: Probability in [0, 1]
    """

def compute_log_likelihood(
    self,
    predictions: pd.DataFrame
) -> float:
    """
    Compute log-likelihood of predictions:
    LL = Σ log(P(predicted_winner | BT model))

    Args:
        predictions: DataFrame with:
            - image1_id, image2_id
            - predicted_winner (1 or 2)

    Returns: Log-likelihood (higher is better)
    """

def get_image_strength(self, image_id: str) -> float:
    """Get learned strength parameter for image."""
    return self.image_strengths.get(image_id, 0.0)
```

**Implementation Notes:**

- Use scipy.optimize for ML estimation if available
- Fallback to manual iterative updates
- Handle missing images gracefully (default θ=0)
- **Lower priority:** Implement after core benchmark works

---

### 3. `pairwise/predictor.py` (~250 lines)

**Purpose:** Abstract base class for all prediction methods

**Abstract Base Class:** `PairwisePredictor`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from pathlib import Path
import time

class PairwisePredictor(ABC):
    """
    Abstract base class for pairwise image quality predictors.

    All concrete predictors must implement:
    - predict_pair(): Compare two images
    - get_method_name(): Return unique identifier
    """

    def __init__(self):
        self.total_time = 0.0
        self.num_predictions = 0

    @abstractmethod
    def predict_pair(
        self,
        image1_path: Path,
        image2_path: Path
    ) -> Dict[str, Any]:
        """
        Predict which image is better.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            {
                'predicted_winner': 1 or 2,
                'score_image1': float,
                'score_image2': float
            }
        """
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return unique method identifier (e.g., 'clip_aesthetic_overall')."""
        pass

    def predict_batch(
        self,
        pairs: List[Tuple[Path, Path]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict for batch of pairs.

        Default implementation: Loop over predict_pair.
        Subclasses can override for batch optimization.

        Args:
            pairs: List of (image1_path, image2_path) tuples
            show_progress: Whether to show tqdm progress bar

        Returns:
            List of prediction dicts
        """
        results = []
        iterator = tqdm(pairs) if show_progress else pairs

        for img1, img2 in iterator:
            start = time.time()
            pred = self.predict_pair(img1, img2)
            self.total_time += time.time() - start
            self.num_predictions += 1
            results.append(pred)

        return results

    def get_avg_time_ms(self) -> float:
        """Get average prediction time per pair in milliseconds."""
        if self.num_predictions == 0:
            return 0.0
        return (self.total_time / self.num_predictions) * 1000

    def get_throughput(self) -> float:
        """Get pairs processed per second."""
        if self.total_time == 0:
            return 0.0
        return self.num_predictions / self.total_time
```

**Design Notes:**

- Template method pattern: `predict_batch` has default implementation
- Timing tracked automatically in base class
- Subclasses only implement `predict_pair` and `get_method_name`

---

### 4. `pairwise/clip_predictor.py` (~200 lines)

**Purpose:** CLIP-based prediction with attribute-specific prompts

**Class:** `CLIPPairwisePredictor(PairwisePredictor)`

**Configuration:**

- Model: `openai/clip-vit-base-patch32` (single model for speed)
- One predictor instance per attribute
- Each attribute = separate method in results

**Attribute Prompt Pairs:**

```python
ATTRIBUTE_PROMPTS = {
    'aesthetic_overall': (
        "a highly aesthetic, visually pleasing, beautiful photograph",
        "an unattractive, poorly composed, ugly photograph"
    ),
    'composition': (
        "a well-composed photograph with excellent visual balance",
        "a poorly-composed photograph with bad visual balance"
    ),
    'subject_placement': (
        "a photo with the subject well placed in the frame",
        "a photo with the subject not well placed in the frame"
    ),
    'cropping': (
        "a photo that is well cropped and shows the complete subject",
        "a photo that is poorly cropped or cuts off the subject"
    ),
    'sharpness': (
        "a sharp, in-focus photograph with clear details",
        "a blurry, out-of-focus photograph with unclear details"
    ),
    'exposure': (
        "a photo with good exposure and lighting",
        "a photo with poor exposure, too dark or too bright"
    ),
    'color': (
        "a photo with vibrant, natural colors",
        "a photo with dull, washed out colors"
    )
}
```

**Implementation:**

```python
class CLIPPairwisePredictor(PairwisePredictor):
    def __init__(
        self,
        attribute_name: str,
        model_name: str = "openai/clip-vit-base-patch32"
    ):
        super().__init__()
        self.attribute_name = attribute_name
        self.model_name = model_name

        # Get prompts for this attribute
        self.positive_prompt, self.negative_prompt = \
            ATTRIBUTE_PROMPTS[attribute_name]

        # Load CLIP model
        import clip
        self.model, self.preprocess = clip.load(model_name)
        self.model.eval()

        # Precompute text embeddings
        self.pos_emb = self._encode_text(self.positive_prompt)
        self.neg_emb = self._encode_text(self.negative_prompt)

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        tokens = clip.tokenize([text])
        with torch.no_grad():
            return self.model.encode_text(tokens)

    def _encode_image(self, image_path: Path) -> torch.Tensor:
        """Encode image to embedding."""
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            return self.model.encode_image(image_input)

    def _compute_attribute_score(
        self,
        image_path: Path
    ) -> float:
        """
        Compute attribute score for image.

        Score = similarity(image, positive) - similarity(image, negative)

        Returns: float in range approximately [-1, 1]
        """
        img_emb = self._encode_image(image_path)

        # Normalize embeddings
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        pos_emb_norm = self.pos_emb / self.pos_emb.norm(dim=-1, keepdim=True)
        neg_emb_norm = self.neg_emb / self.neg_emb.norm(dim=-1, keepdim=True)

        # Cosine similarities
        sim_pos = (img_emb @ pos_emb_norm.T).item()
        sim_neg = (img_emb @ neg_emb_norm.T).item()

        return sim_pos - sim_neg

    def predict_pair(
        self,
        image1_path: Path,
        image2_path: Path
    ) -> Dict[str, Any]:
        """Predict which image is better based on attribute."""
        score1 = self._compute_attribute_score(image1_path)
        score2 = self._compute_attribute_score(image2_path)

        return {
            'predicted_winner': 1 if score1 > score2 else 2,
            'score_image1': score1,
            'score_image2': score2
        }

    def get_method_name(self) -> str:
        """Return method name like 'clip_aesthetic_overall'."""
        return f"clip_{self.attribute_name}"
```

**Usage in Benchmark:**

```python
# Create 7 separate CLIP predictors (one per attribute)
clip_predictors = []
for attr_name in ATTRIBUTE_PROMPTS.keys():
    clip_predictors.append(
        CLIPPairwisePredictor(attr_name, model_name=config['clip_model'])
    )

# Each generates separate results - NO aggregation
```

---

### 5. `pairwise/rule_based_predictor.py` (~150 lines)

**Purpose:** Rule-based quality metrics (sharpness, contrast, etc.)

**Class:** `RuleBasedPredictor(PairwisePredictor)`

**Reuse Existing Code:**
Import methods from `sim_bench/quality_assessment/rule_based.py`:

- `assess_sharpness()`
- `assess_contrast()`
- `assess_exposure()`
- `assess_colorfulness()`

**Implementation:**

```python
from sim_bench.quality_assessment.rule_based import (
    assess_sharpness,
    assess_contrast,
    assess_exposure,
    assess_colorfulness
)

class RuleBasedPredictor(PairwisePredictor):
    def __init__(self, method_name: str):
        super().__init__()
        self.method_name = method_name
        self._setup_method()

    def _setup_method(self):
        """Configure scoring function and weights."""
        if self.method_name == 'sharpness_only':
            self.score_fn = assess_sharpness
        elif self.method_name == 'contrast_only':
            self.score_fn = assess_contrast
        elif self.method_name == 'exposure_only':
            self.score_fn = assess_exposure
        elif self.method_name == 'colorfulness_only':
            self.score_fn = assess_colorfulness
        elif self.method_name == 'composite_balanced':
            self.weights = {
                'sharpness': 0.3,
                'contrast': 0.2,
                'exposure': 0.3,
                'colorfulness': 0.2
            }
        elif self.method_name == 'composite_sharpness_focused':
            self.weights = {
                'sharpness': 0.5,
                'contrast': 0.15,
                'exposure': 0.25,
                'colorfulness': 0.1
            }
        elif self.method_name == 'composite_exposure_focused':
            self.weights = {
                'sharpness': 0.2,
                'contrast': 0.15,
                'exposure': 0.5,
                'colorfulness': 0.15
            }
        else:
            raise ValueError(f"Unknown method: {self.method_name}")

    def _compute_score(self, image_path: Path) -> float:
        """Compute quality score for image."""
        image = cv2.imread(str(image_path))

        if self.method_name.startswith('composite'):
            # Compute all metrics
            scores = {
                'sharpness': assess_sharpness(image),
                'contrast': assess_contrast(image),
                'exposure': assess_exposure(image),
                'colorfulness': assess_colorfulness(image)
            }
            # Weighted combination
            return sum(
                scores[k] * self.weights[k]
                for k in self.weights.keys()
            )
        else:
            # Single metric
            return self.score_fn(image)

    def predict_pair(
        self,
        image1_path: Path,
        image2_path: Path
    ) -> Dict[str, Any]:
        """Predict which image is better."""
        score1 = self._compute_score(image1_path)
        score2 = self._compute_score(image2_path)

        return {
            'predicted_winner': 1 if score1 > score2 else 2,
            'score_image1': score1,
            'score_image2': score2
        }

    def get_method_name(self) -> str:
        return self.method_name
```

**Methods Supported:**

- `sharpness_only`
- `contrast_only`
- `exposure_only`
- `colorfulness_only`
- `composite_balanced`
- `composite_sharpness_focused`
- `composite_exposure_focused`

---

### 6. `pairwise/dl_predictor.py` (~150 lines)

**Purpose:** Deep learning models (ViT-base only, skip NIMA)

**Class:** `ViTPredictor(PairwisePredictor)`

**Rationale for Skipping NIMA:**
Previous results showed:

- NIMA MobileNet: 0.424 accuracy
- NIMA ResNet50: 0.414 accuracy
- ViT-base: 0.412 accuracy
- **vs Sharpness-only: 0.649 accuracy**

NIMA provides minimal value and is slow to run.

**Implementation:**

```python
from sim_bench.quality_assessment.transformer_methods import (
    ViTQualityAssessor
)

class ViTPredictor(PairwisePredictor):
    def __init__(self):
        super().__init__()
        self.model = ViTQualityAssessor()

    def _compute_score(self, image_path: Path) -> float:
        """Compute quality score using ViT."""
        image = Image.open(image_path).convert('RGB')
        return self.model.assess_quality(image)

    def predict_pair(
        self,
        image1_path: Path,
        image2_path: Path
    ) -> Dict[str, Any]:
        """Predict which image is better."""
        score1 = self._compute_score(image1_path)
        score2 = self._compute_score(image2_path)

        return {
            'predicted_winner': 1 if score1 > score2 else 2,
            'score_image1': score1,
            'score_image2': score2
        }

    def get_method_name(self) -> str:
        return 'vit_base'
```

---

### 7. `pairwise/evaluator.py` (~250 lines)

**Purpose:** Compute evaluation metrics

**Class:** `PairwiseEvaluator`

**Methods:**

```python
class PairwiseEvaluator:
    def __init__(self):
        pass

    def compute_accuracy(
        self,
        predictions: pd.DataFrame
    ) -> float:
        """
        Compute binary classification accuracy.

        Args:
            predictions: DataFrame with:
                - ground_truth_winner (1 or 2)
                - predicted_winner (1 or 2)

        Returns: Accuracy in [0, 1]
        """
        correct = (
            predictions['predicted_winner'] ==
            predictions['ground_truth_winner']
        )
        return correct.mean()

    def compute_precision_recall_f1(
        self,
        predictions: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 score.

        Treat as binary: predicted winner 1 vs 2

        Returns: {'precision': float, 'recall': float, 'f1_score': float}
        """
        from sklearn.metrics import precision_recall_fscore_support

        y_true = predictions['ground_truth_winner'].values
        y_pred = predictions['predicted_winner'].values

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }

    def compute_confusion_matrix(
        self,
        predictions: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute 2x2 confusion matrix.

        Returns: np.ndarray shape (2, 2)
                 [[TN, FP],
                  [FN, TP]]
        """
        from sklearn.metrics import confusion_matrix

        y_true = predictions['ground_truth_winner'].values
        y_pred = predictions['predicted_winner'].values

        return confusion_matrix(y_true, y_pred, labels=[1, 2])

    def compute_metrics_by_agreement(
        self,
        predictions: pd.DataFrame,
        buckets: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Compute accuracy stratified by agreement level.

        Args:
            predictions: DataFrame with 'agreement' column
            buckets: List of (min, max) tuples for agreement ranges

        Returns: Dict mapping bucket name to accuracy

        Example:
            buckets = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
            Returns: {
                'accuracy_70_80': 0.589,
                'accuracy_80_90': 0.610,
                'accuracy_90_100': 0.678
            }
        """
        results = {}

        for min_agree, max_agree in buckets:
            mask = (
                (predictions['agreement'] >= min_agree) &
                (predictions['agreement'] < max_agree)
            )
            subset = predictions[mask]

            if len(subset) > 0:
                acc = self.compute_accuracy(subset)
                bucket_name = f"accuracy_{int(min_agree*100)}_{int(max_agree*100)}"
                results[bucket_name] = acc
            else:
                results[bucket_name] = 0.0

        return results

    def compute_log_likelihood(
        self,
        predictions: pd.DataFrame,
        bt_model: BradleyTerryModel
    ) -> float:
        """
        Compute log-likelihood under Bradley-Terry model.

        LL = Σ log(P(predicted_winner | BT probabilities))

        Args:
            predictions: DataFrame with:
                - image1_id, image2_id (constructed from series_id + compareID)
                - predicted_winner (1 or 2)
            bt_model: Fitted BradleyTerryModel

        Returns: Log-likelihood (higher is better)
        """
        if not bt_model.fitted:
            raise ValueError("BT model must be fitted first")

        log_likelihood = 0.0

        for _, row in predictions.iterrows():
            # Construct image IDs
            img1_id = f"{row['series_id']}_{row['compareID1']}"
            img2_id = f"{row['series_id']}_{row['compareID2']}"

            # Get BT probability for predicted winner
            if row['predicted_winner'] == 1:
                prob = bt_model.compute_pair_probability(img1_id, img2_id)
            else:
                prob = bt_model.compute_pair_probability(img2_id, img1_id)

            # Add log probability (with small epsilon to avoid log(0))
            log_likelihood += np.log(max(prob, 1e-10))

        return log_likelihood

    def evaluate_method(
        self,
        predictions: pd.DataFrame,
        bt_model: Optional[BradleyTerryModel] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a method.

        Args:
            predictions: DataFrame with predictions and ground truth
            bt_model: Optional BT model for log-likelihood

        Returns: Dict with all metrics
        """
        metrics = {}

        # Basic accuracy
        metrics['accuracy'] = self.compute_accuracy(predictions)

        # Precision, recall, F1
        prf = self.compute_precision_recall_f1(predictions)
        metrics.update(prf)

        # Stratified accuracy
        buckets = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        stratified = self.compute_metrics_by_agreement(predictions, buckets)
        metrics.update(stratified)

        # Bradley-Terry log-likelihood (if model provided)
        if bt_model is not None:
            metrics['log_likelihood'] = self.compute_log_likelihood(
                predictions, bt_model
            )

        return metrics
```

---

### 8. `pairwise/benchmark_runner.py` (~200 lines)

**Purpose:** Orchestrate full benchmark execution

**Class:** `PairwiseBenchmarkRunner`

**Main Workflow:**

```python
class PairwiseBenchmarkRunner:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_benchmark(self) -> None:
        """Run complete benchmark and save results."""

        # 1. Setup
        output_dir = self._create_output_directory()
        self._save_config(output_dir)

        # 2. Load data
        pairs_df = self._load_and_filter_data()
        self.logger.info(f"Loaded {len(pairs_df)} pairs for benchmarking")

        # 3. Initialize predictors
        predictors = self._initialize_predictors()
        self.logger.info(f"Initialized {len(predictors)} prediction methods")

        # 4. Run predictions
        all_results = {}
        for predictor in predictors:
            method_name = predictor.get_method_name()
            self.logger.info(f"Running method: {method_name}")

            results = self._run_method(predictor, pairs_df)
            all_results[method_name] = results

        # 5. Fit Bradley-Terry (optional)
        bt_model = None
        if self.config.get('use_bradley_terry', False):
            self.logger.info("Fitting Bradley-Terry model...")
            bt_model = BradleyTerryModel()
            bt_model.fit(pairs_df)

        # 6. Evaluate all methods
        evaluator = PairwiseEvaluator()
        summary_metrics = {}

        for method_name, results in all_results.items():
            self.logger.info(f"Evaluating method: {method_name}")
            metrics = evaluator.evaluate_method(results, bt_model)

            # Add timing info
            predictor = next(p for p in predictors if p.get_method_name() == method_name)
            metrics['avg_time_ms'] = predictor.get_avg_time_ms()
            metrics['throughput'] = predictor.get_throughput()

            summary_metrics[method_name] = metrics

        # 7. Save results in unified format
        self._save_unified_format(output_dir, summary_metrics, all_results)

        # 8. Generate visualizations
        if self.config.get('save_confusion_matrices', True):
            self._save_confusion_matrices(output_dir, all_results)

        self.logger.info(f"Benchmark complete! Results saved to: {output_dir}")

    def _load_and_filter_data(self) -> pd.DataFrame:
        """Load pairs CSV and apply filters."""
        loader = PairwiseDataLoader()

        # Load
        pairs_df = loader.load_pairs(self.config['pairs_csv'])

        # Filter by agreement and reviewers
        pairs_df = loader.filter_by_agreement(
            pairs_df,
            min_agreement=self.config['min_agreement'],
            min_reviewers=self.config['min_reviewers']
        )

        # Sample if requested
        if self.config.get('sample_size') is not None:
            pairs_df = loader.sample_pairs(pairs_df, self.config['sample_size'])

        # Get image paths
        pairs_df = loader.get_image_paths(pairs_df, self.config['image_root'])

        return pairs_df

    def _initialize_predictors(self) -> List[PairwisePredictor]:
        """Create predictor instances from config."""
        predictors = []

        # CLIP predictors (one per attribute)
        for attr_name in self.config['clip_attributes'].keys():
            predictors.append(
                CLIPPairwisePredictor(
                    attr_name,
                    model_name=self.config['clip_model']
                )
            )

        # Rule-based predictors
        for method_name in self.config['rule_based_methods']:
            predictors.append(RuleBasedPredictor(method_name))

        # Deep learning predictors
        for method_name in self.config['dl_methods']:
            if method_name == 'vit_base':
                predictors.append(ViTPredictor())

        return predictors

    def _run_method(
        self,
        predictor: PairwisePredictor,
        pairs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run predictions for one method."""

        # Prepare pairs list
        pairs = [
            (row['image_path_1'], row['image_path_2'])
            for _, row in pairs_df.iterrows()
        ]

        # Run predictions
        predictions = predictor.predict_batch(pairs, show_progress=True)

        # Combine with ground truth
        results = pairs_df.copy()
        results['predicted_winner'] = [p['predicted_winner'] for p in predictions]
        results['score_image1'] = [p['score_image1'] for p in predictions]
        results['score_image2'] = [p['score_image2'] for p in predictions]
        results['correct'] = (
            results['predicted_winner'] == results['ground_truth_winner']
        )

        return results

    def _save_unified_format(
        self,
        output_dir: Path,
        summary_metrics: Dict[str, Dict],
        all_results: Dict[str, pd.DataFrame]
    ) -> None:
        """Save results in format compatible with existing analysis tools."""

        dataset_name = self.config.get('dataset_name', 'phototriage')

        # 1. methods_summary.csv
        methods_summary_rows = []
        for method_name, metrics in summary_metrics.items():
            methods_summary_rows.append({
                'method': method_name,
                'avg_top1_accuracy': metrics['accuracy'],
                'avg_top2_accuracy': 0.0,
                'avg_mrr': 0.0,
                'avg_time_ms': metrics['avg_time_ms'],
                'datasets_tested': 1
            })

        pd.DataFrame(methods_summary_rows).to_csv(
            output_dir / 'methods_summary.csv',
            index=False
        )

        # 2. detailed_results.csv
        detailed_results_rows = []
        for method_name, metrics in summary_metrics.items():
            detailed_results_rows.append({
                'dataset': dataset_name,
                'method': method_name,
                'top1_accuracy': metrics['accuracy'],
                'top2_accuracy': 0.0,
                'mrr': 0.0,
                'avg_time_ms': metrics['avg_time_ms'],
                'throughput': metrics['throughput']
            })

        pd.DataFrame(detailed_results_rows).to_csv(
            output_dir / 'detailed_results.csv',
            index=False
        )

        # 3. Per-method directories with per_pair_results.csv
        for method_name, results in all_results.items():
            method_dir = output_dir / method_name
            method_dir.mkdir(exist_ok=True)

            results.to_csv(method_dir / 'per_pair_results.csv', index=False)

        # 4. Ranking CSVs
        self._save_ranking_csvs(output_dir, summary_metrics)

        # 5. summary.json
        self._save_summary_json(output_dir, summary_metrics)

    def _save_ranking_csvs(
        self,
        output_dir: Path,
        summary_metrics: Dict[str, Dict]
    ) -> None:
        """Save ranking CSVs for compatibility."""

        df = pd.DataFrame([
            {
                'method': method_name,
                'accuracy': metrics['accuracy'],
                'avg_time_ms': metrics['avg_time_ms'],
                'efficiency': metrics['accuracy'] / (metrics['avg_time_ms'] + 1)
            }
            for method_name, metrics in summary_metrics.items()
        ])

        # Accuracy ranking
        df.sort_values('accuracy', ascending=False).to_csv(
            output_dir / 'accuracy_ranking.csv',
            index=False
        )

        # Speed ranking
        df.sort_values('avg_time_ms', ascending=True).to_csv(
            output_dir / 'speed_ranking.csv',
            index=False
        )

        # Efficiency ranking
        df.sort_values('efficiency', ascending=False).to_csv(
            output_dir / 'efficiency_ranking.csv',
            index=False
        )
```

---

## Output Format

### Directory Structure

```
outputs/quality_benchmarks/pairwise_20251120_143022/
├── config.yaml                         # Copy of config
├── benchmark.log                       # Execution log
├── methods_summary.csv                 # Required for analysis
├── detailed_results.csv                # Required for analysis
├── accuracy_ranking.csv
├── speed_ranking.csv
├── efficiency_ranking.csv
├── summary.json
├── clip_aesthetic_overall/
│   └── per_pair_results.csv
├── clip_composition/
│   └── per_pair_results.csv
├── sharpness_only/
│   └── per_pair_results.csv
└── ...
```

### CSV Schemas

**`methods_summary.csv`:**

```csv
method,avg_top1_accuracy,avg_top2_accuracy,avg_mrr,avg_time_ms,datasets_tested
clip_aesthetic_overall,0.6234,0.0,0.0,45.2,1
sharpness_only,0.6495,0.0,0.0,12.3,1
```

**`detailed_results.csv`:**

```csv
dataset,method,top1_accuracy,top2_accuracy,mrr,avg_time_ms,throughput
phototriage,clip_aesthetic_overall,0.6234,0.0,0.0,45.2,22.1
phototriage,sharpness_only,0.6495,0.0,0.0,12.3,81.3
```

**`{method}/per_pair_results.csv`:**

```csv
pair_idx,series_id,compareID1,compareID2,compareFile1,compareFile2,ground_truth_winner,predicted_winner,score_image1,score_image2,correct,agreement,num_reviewers
0,1,0,1,1-1.JPG,1-2.JPG,2,2,0.234,0.567,True,0.85,5
```

---

## Configuration

**`configs/pairwise_benchmark.yaml`:**

```yaml
# Data Configuration
pairs_csv: "D:/Similar Images/automatic_triage_photo_series/photo_triage_pairs_embedding_labels.csv"
image_root: "D:/Similar Images/automatic_triage_photo_series/train_val"
dataset_name: "phototriage"

# Ground Truth Filtering
min_agreement: 0.7
min_reviewers: 2

# Sampling (for testing)
sample_size: null  # null = all pairs, or set integer

# CLIP Configuration
clip_model: "openai/clip-vit-base-patch32"

# CLIP Attribute Prompts (each = separate method)
clip_attributes:
  aesthetic_overall:
    positive: "a highly aesthetic, visually pleasing, beautiful photograph"
    negative: "an unattractive, poorly composed, ugly photograph"
  composition:
    positive: "a well-composed photograph with excellent visual balance"
    negative: "a poorly-composed photograph with bad visual balance"
  subject_placement:
    positive: "a photo with the subject well placed in the frame"
    negative: "a photo with the subject not well placed in the frame"
  cropping:
    positive: "a photo that is well cropped and shows the complete subject"
    negative: "a photo that is poorly cropped or cuts off the subject"
  sharpness:
    positive: "a sharp, in-focus photograph with clear details"
    negative: "a blurry, out-of-focus photograph with unclear details"
  exposure:
    positive: "a photo with good exposure and lighting"
    negative: "a photo with poor exposure, too dark or too bright"
  color:
    positive: "a photo with vibrant, natural colors"
    negative: "a photo with dull, washed out colors"

# Rule-Based Methods
rule_based_methods:
  - sharpness_only
  - contrast_only
  - exposure_only
  - colorfulness_only
  - composite_balanced
  - composite_sharpness_focused
  - composite_exposure_focused

# Deep Learning Methods
dl_methods:
  - vit_base

# Bradley-Terry Model (lower priority)
use_bradley_terry: false

# Output Configuration
output_dir: "outputs/quality_benchmarks"
save_per_pair_results: true
save_confusion_matrices: true
save_ranking_csvs: true
```

---

## Testing Strategy

### Phase 1: Quick Validation (~2 minutes)

```bash
python scripts/run_pairwise_benchmark.py --quick-test
```

**Expected:**

- Sample size: 100 pairs
- All 15 methods run successfully
- `methods_summary.csv` and `detailed_results.csv` generated
- No errors or crashes

**Verification:**

```python
import pandas as pd

# Check methods_summary.csv
summary = pd.read_csv('outputs/quality_benchmarks/pairwise_*/methods_summary.csv')
print(f"Methods tested: {len(summary)}")  # Should be 15
print(f"Columns: {list(summary.columns)}")  # Verify schema

# Check accuracy range
print(f"Accuracy range: {summary['avg_top1_accuracy'].min():.3f} - {summary['avg_top1_accuracy'].max():.3f}")
```

### Phase 2: Full Run (~35 minutes)

```bash
python scripts/run_pairwise_benchmark.py
```

**Expected:**

- All ~16,205 filtered pairs
- Complete results for all methods
- Total runtime: ~35 minutes

**Verification:**

- Check accuracy values are reasonable (0.4-0.7 range)
- Verify rule-based methods are fastest (10-15ms per pair)
- CLIP methods should be ~40-50ms per pair
- ViT-base should be ~90-100ms per pair

### Phase 3: With Bradley-Terry (~40 minutes)

Edit config:

```yaml
use_bradley_terry: true
```

Run:

```bash
	python scripts/run_pairwise_benchmark.py
```

**Expected:**

- Additional `log_likelihood` column in metrics
- BT model fitting should add ~5 minutes
- Log-likelihood values should be negative (higher is better)

---

## Timeline & Milestones

### Day 1: Core Infrastructure (4-5 hours)

- [ ] Create module structure
- [ ] Implement `data_loader.py` (1 hour)
- [ ] Implement `predictor.py` base class (1 hour)
- [ ] Implement `clip_predictor.py` (1.5 hours)
- [ ] Implement `rule_based_predictor.py` (1 hour)
- [ ] Implement `dl_predictor.py` (0.5 hours)
- [ ] Test: Verify each predictor works independently

### Day 2: Evaluation & Runner (3-4 hours)

- [ ] Implement `evaluator.py` (1.5 hours)
- [ ] Implement `benchmark_runner.py` (1.5 hours)
- [ ] Implement CLI script (0.5 hours)
- [ ] Create config file (0.5 hours)
- [ ] Test: Quick validation run (100 pairs)

### Day 3: Bradley-Terry & Polish (2-3 hours)

- [ ] Implement `ground_truth.py` (Bradley-Terry) (1.5 hours)
- [ ] Add confusion matrix visualization (0.5 hours)
- [ ] Fix bugs from full run (1 hour)
- [ ] Test: Full benchmark run

### Day 4: Documentation (1-2 hours)

- [ ] Write `PAIRWISE_BENCHMARK.md` (1 hour)
- [ ] Add docstrings and type hints (0.5 hours)
- [ ] Create usage examples (0.5 hours)

**Total Estimated Time: 10-14 hours**

---

## Dependencies

### Python Packages (already installed)

- `torch`, `torchvision` (for CLIP and ViT)
- `clip` (OpenAI CLIP)
- `transformers` (for ViT)
- `opencv-python` (for rule-based methods)
- `pillow` (image loading)
- `pandas`, `numpy` (data processing)
- `scikit-learn` (metrics)
- `tqdm` (progress bars)
- `pyyaml` (config loading)

### New Dependencies (if needed)

- `scipy` (for Bradley-Terry optimization) - may already be installed

---

## Documentation

### `docs/quality_assessment/PAIRWISE_BENCHMARK.md`

**Table of Contents:**

1. **Overview**

   - What is pairwise assessment?
   - Why use PhotoTriage dataset?
   - Key differences from series selection
2. **Methodology**

   - Ground truth definition (MaxVote)
   - Filtering criteria (agreement, reviewers)
   - Bradley-Terry model explanation
3. **How to Run**

   - Quick test
   - Full benchmark
   - Custom configuration
4. **Methods Tested**

   - CLIP with 7 attributes
   - Rule-based (7 methods)
   - ViT-base
   - How each method works
5. **Output Format**

   - CSV schemas
   - Directory structure
   - Integration with analysis tools
6. **Interpreting Results**

   - Accuracy metrics
   - Log-likelihood (Bradley-Terry)
   - Stratified by agreement
   - Speed vs accuracy tradeoffs
7. **Adding New Methods**

   - Extend `PairwisePredictor`
   - Example implementation
   - Register in config
8. **CLIP Prompt Design**

   - Guidelines for creating prompts
   - Positive vs negative phrasing
   - Attribute selection
9. **Comparison with Original Paper**

   - PhotoTriage paper methodology
   - Our implementation differences
   - Results interpretation
10. **Troubleshooting**

    - Common errors
    - Performance optimization
    - Memory management

---

## Success Criteria

### Implementation Complete When:

- [ ] All 15 methods run successfully
- [ ] Quick test (100 pairs) completes in ~2 minutes
- [ ] Full run completes in ~35-40 minutes
- [ ] Output format matches existing benchmark system
- [ ] `methods_summary.csv` and `detailed_results.csv` generated correctly
- [ ] Per-pair results saved for each method
- [ ] Analysis scripts can load results without modification

### Quality Criteria:

- [ ] All modules <300 lines of code
- [ ] SOLID principles followed
- [ ] Minimal if/else usage (strategy pattern)
- [ ] Empty `__init__.py` files
- [ ] Comprehensive docstrings
- [ ] Type hints on public methods
- [ ] No code duplication
- [ ] Clean separation of concerns

### Expected Results:

- CLIP aesthetic: ~0.60-0.65 accuracy
- Rule-based (sharpness): ~0.63-0.67 accuracy
- Rule-based (contrast): ~0.48-0.50 accuracy
- ViT-base: ~0.41-0.42 accuracy
- CLIP should be faster than previous DL methods
- Rule-based methods should be fastest (~10-15ms per pair)

---

## Notes for Future Sessions

### If Session Ends, Resume With:

1. **Check existing progress:**

   ```bash
   ls sim_bench/quality_assessment/pairwise/
   ```
2. **Start from last incomplete module** based on timeline above
3. **Test incrementally:**

   - After each module, write simple test
   - Don't wait until end to test
4. **Key reminders:**

   - Each CLIP attribute = separate method (NO aggregation)
   - Output format MUST match `methods_summary.csv` and `detailed_results.csv` schema
   - Use strategy pattern for predictors (avoid if/else chains)
   - Bradley-Terry is lower priority (implement last)
5. **Quick validation after implementation:**

   ```bash
   # Test data loading
   python -c "from sim_bench.quality_assessment.pairwise.data_loader import PairwiseDataLoader; ..."

   # Test quick run
   python scripts/run_pairwise_benchmark.py --quick-test
   ```

---

## End of Plan

**This document should contain everything needed to implement the pairwise benchmark in future sessions.**

**Questions before starting implementation?**
