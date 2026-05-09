# Implementation Plan: Labeling Review + ML Training Dashboard

**Feature**: 006-ml-training-dashboard
**Created**: 2026-04-17
**Status**: Phase 1 implemented

---

## Architecture

### New files

| File | Purpose |
|------|---------|
| `face_cluster/ml_trainer.py` | Training/evaluation/prediction — wraps sklearn, optional xgboost |
| `face_cluster/training_db.py` | Extend with `save_model_record`, `load_model_records`, `update_label` |
| `app/face_clustering.py` | Two new tabs: "Labeling Review", "ML Training" (tabs 9 and 10; rename current "Training Data" to "Labeling Review" and extend it) |
| `tests/face_clustering/test_ml_trainer.py` | Training round-trip on synthetic data |

### Dependencies

```
sklearn       — already in requirements (used by scripts/train_merge_classifier.py)
xgboost       — optional, gracefully disabled if missing
plotly         — already in requirements
pickle         — stdlib
```

### Model storage

```
~/.sim_bench/
    sim_bench.db          ← trained_models table (metadata)
    models/
        merge_lr_2026-04-17_143000.pkl
        merge_xgb_2026-04-18_091500.pkl
```

Each `.pkl` contains:
```python
{
    "model":         fitted sklearn/xgboost estimator,
    "scaler":        fitted StandardScaler,
    "feature_names": ["min_exemplar_dist", "size_a", ...],
    "metadata": {
        "model_type":       "logistic_regression",
        "feature_version":  3,
        "n_train":          180,
        "n_test":           45,
        "runs_used":        ["Germany_7", "Germany_8"],
        "split_strategy":   "random_stratified",
        "hyperparams":      {"C": 1.0, "class_weight": "balanced"},
        "metrics":          {"accuracy": 0.91, "f1": 0.88, "auc": 0.94},
        "created_at":       "2026-04-17T14:30:00",
    }
}
```

### DB table: `trained_models`

```sql
CREATE TABLE IF NOT EXISTS trained_models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,        -- e.g. "merge_lr_2026-04-17_143000"
    model_type    TEXT    NOT NULL,        -- logistic_regression | xgboost | mlp
    model_path    TEXT    NOT NULL,        -- full path to .pkl
    feature_version INTEGER NOT NULL,
    n_samples     INTEGER,
    n_train       INTEGER,
    n_test        INTEGER,
    accuracy      REAL,
    f1            REAL,
    auc_roc       REAL,
    created_at    TEXT    NOT NULL,
    metadata_json TEXT,                   -- full metrics + hyperparams + runs_used
    UNIQUE(name)
);
```

---

## Part A: Labeling Review Tab (replaces current "Training Data")

Extends the existing Training Data tab. The current content (summary metrics, per-run breakdown, pair inspector, export) is kept. New sections are added.

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Labeling Review                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Total    │ │ Approved │ │ Rejected │ │ Readiness      │ │
│  │   142    │ │    87    │ │    41    │ │ Needs work     │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
│                                                             │
│  ── Per-Run Breakdown ─── [chart as before]                 │
│                                                             │
│  ── Label Audit ─────────────────────────────────────────── │
│                                                             │
│  Filter: [All ▾]  [Disagreements only ☐]  Run: [All ▾]     │
│                                                             │
│  ┌──────┬──────┬────────┬───────┬──────┬───────┬──────────┐ │
│  │ Run  │ Pair │ Label  │ Gates │ Dist │ Agree │ Action   │ │
│  ├──────┼──────┼────────┼───────┼──────┼───────┼──────────┤ │
│  │ G_7  │ 0↔11 │ reject │  3/4  │ 0.38 │  ⚠   │ [Flip]   │ │
│  │ G_7  │ 2↔5  │ approve│  4/4  │ 0.21 │  ✓   │ [Flip]   │ │
│  │ G_8  │ 1↔3  │ approve│  1/4  │ 0.41 │  ⚠   │ [Flip]   │ │
│  └──────┴──────┴────────┴───────┴──────┴───────┴──────────┘ │
│                                                             │
│  Agreement: ✓ = label matches heuristic direction           │
│             ⚠ = disagreement (worth double-checking)        │
│                                                             │
│  ── Pair Inspector ─── [same as before: thumbnails + feats] │
│                                                             │
│  ── Suggested Next Labels ──────────────────────────────── │
│                                                             │
│  These unlabeled pairs are closest to the decision          │
│  boundary (most informative to label next):                 │
│                                                             │
│  ┌──────┬───────┬───────┬───────────┬──────────────────┐    │
│  │ Run  │ Pair  │ Gates │ Dist      │ [Label Approve]  │    │
│  │      │       │       │           │ [Label Reject ]  │    │
│  ├──────┼───────┼───────┼───────────┼──────────────────┤    │
│  │ G_9  │ 3↔7   │  2/4  │ 0.34      │ [✓] [✗]         │    │
│  │ G_9  │ 0↔4   │  2/4  │ 0.36      │ [✓] [✗]         │    │
│  └──────┴───────┴───────┴───────────┴──────────────────┘    │
│                                                             │
│  ── Export ─── [Download CSV] [Download Parquet]            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Agreement logic

```
gates_passed >= 3 → heuristic says "merge"
gates_passed <= 1 → heuristic says "don't merge"
gates_passed == 2 → ambiguous (no disagreement flagged)

Disagreement = (label==approve AND gates<=1) OR (label==reject AND gates>=3)
```

`n_gates_passed` is already stored in `merge_decisions.json` and can be carried into `features_json` or a new DB column.

### Suggested-next ranking

Score each unlabeled pair by `abs(n_gates_passed - 2)` (lower = more ambiguous = more informative). Break ties by `abs(exemplar_dist - threshold)`.

---

## Part B: ML Training Tab

### Layout — Dataset Configuration

```
┌─────────────────────────────────────────────────────────────┐
│  ML Training                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ── 1. Dataset ─────────────────────────────────────────    │
│                                                             │
│  Runs to include:  [☑ Germany_7] [☑ Germany_8] [☐ G_9]     │
│                    [Select All] [Deselect All]              │
│                                                             │
│  Labeled samples: 128 (87 approve / 41 reject)             │
│                                                             │
│  Split strategy:  ( ) Random stratified                     │
│                   ( ) By run (no album leakage)             │
│                                                             │
│  ┌─────────────────────────────────────────────┐            │
│  │  Train: [====70%====]  70%  (90 samples)    │            │
│  │  Val:   [==15%==]      15%  (19 samples)    │            │
│  │  Test:  [==15%==]      15%  (19 samples)    │            │
│  └─────────────────────────────────────────────┘            │
│                                                             │
│  Split by run assignment (when "By run" selected):          │
│  Train: Germany_7, Germany_8                                │
│  Val:   Germany_9                                           │
│  Test:  Germany_10                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layout — Model + Features

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ── 2. Model ───────────────────────────────────────────    │
│                                                             │
│  Model type:  [Logistic Regression ▾]                       │
│                                                             │
│  ┌─ Logistic Regression Parameters ──────────────────┐      │
│  │  C (regularization):  [1.0  ▾] or Auto (grid)    │      │
│  │  Class weight:        [balanced ▾]                │      │
│  │  Cross-validation:    [5 ▾] folds                 │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  ┌─ XGBoost Parameters ─────────────────────────────┐       │
│  │  n_estimators:   [100]    max_depth:  [3]        │       │
│  │  learning_rate:  [0.1]    subsample:  [0.8]      │       │
│  │  scale_pos_weight: [auto ▾]                      │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  ┌─ MLP Parameters ─────────────────────────────────┐       │
│  │  Hidden layers:   [(32, 16)]                     │       │
│  │  Activation:      [relu ▾]                       │       │
│  │  Learning rate:   [0.001]                        │       │
│  │  Max iterations:  [500]                          │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  ── 3. Features ─────────────────────────────────────────   │
│                                                             │
│  [☑] Group A: Distance (12 features)                        │
│  [☑] Groups B+C: Geometry (19 features)                     │
│  [☑] Group D: Source images (5 features)                    │
│  [☑] Group G: Quality/Pose (13 features)                    │
│                                                             │
│  Selected: 49 features                                      │
│                                                             │
│              [  Train Model  ]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layout — Results

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ── 4. Results ──────────────────────────────────────────   │
│                                                             │
│  Model: Logistic Regression  |  Trained: 2026-04-17 14:30  │
│  Data: 128 samples (90 train / 19 val / 19 test)           │
│  Best params: C=1.0, class_weight=balanced                  │
│                                                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ Accuracy   │ │ F1         │ │ AUC-ROC    │ │ Precision│ │
│  │   0.91     │ │   0.88     │ │   0.94     │ │   0.90   │ │
│  │ (test set) │ │ (test set) │ │ (test set) │ │ (test)   │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
│                                                             │
│  ┌─ Confusion Matrix (Test) ──┐  ┌─ ROC Curve ──────────┐  │
│  │          Pred:No  Pred:Yes │  │                       │  │
│  │  No:     [  15  ] [   2  ]│  │    .....------/       │  │
│  │  Yes:    [   1  ] [  12  ]│  │   /                   │  │
│  │                           │  │  / AUC = 0.94         │  │
│  └───────────────────────────┘  └───────────────────────┘  │
│                                                             │
│  ── Feature Importance ──────────────────────────────────   │
│                                                             │
│  min_exemplar_dist      ████████████████████  2.34          │
│  shared_source_images   ████████████████     -1.89          │
│  post_merge_diameter    ██████████           -1.12          │
│  support_fraction       ████████              0.98          │
│  size_ratio             ██████               -0.76          │
│  p50_cross_dist         █████                 0.62          │
│  ...                                                        │
│                                                             │
│  ── Per-Sample Predictions ──────────────────────────────   │
│  (click a row to see exemplar crops)                        │
│                                                             │
│  ┌──────┬──────┬───────┬──────┬────────┬──────────────────┐ │
│  │ Run  │ Pair │ True  │ Pred │ Prob   │ Correct?         │ │
│  ├──────┼──────┼───────┼──────┼────────┼──────────────────┤ │
│  │ G_7  │ 2↔5  │  1    │  1   │ 0.97   │ ✓               │ │
│  │ G_7  │ 0↔11 │  0    │  1   │ 0.62   │ ✗ (false pos)   │ │
│  │ G_8  │ 1↔3  │  1    │  0   │ 0.38   │ ✗ (false neg)   │ │
│  └──────┴──────┴───────┴──────┴────────┴──────────────────┘ │
│                                                             │
│           [Save Model]  [Download metrics JSON]             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layout — Model History + Apply

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ── 5. Saved Models ─────────────────────────────────────   │
│                                                             │
│  ┌─────────────────┬──────┬──────┬──────┬───────┬────────┐  │
│  │ Name            │ Type │  F1  │ AUC  │ N     │ Action │  │
│  ├─────────────────┼──────┼──────┼──────┼───────┼────────┤  │
│  │ lr_2026-04-17   │  LR  │ 0.88 │ 0.94 │  128  │ [Load] │  │
│  │ xgb_2026-04-18  │  XGB │ 0.91 │ 0.96 │  200  │ [Load] │  │
│  └─────────────────┴──────┴──────┴──────┴───────┴────────┘  │
│                                                             │
│  ── 6. Apply to Current Run ─────────────────────────────   │
│                                                             │
│  Model: [lr_2026-04-17 ▾]          [Predict]               │
│                                                             │
│  ┌──────┬──────┬────────┬───────────┬───────────────────┐   │
│  │ Pair │ Prob │ ML     │ Heuristic │ Agreement         │   │
│  ├──────┼──────┼────────┼───────────┼───────────────────┤   │
│  │ 0↔3  │ 0.92 │ merge  │  merge    │ ✓ both agree      │   │
│  │ 1↔7  │ 0.31 │ reject │  merge    │ ⚠ ML disagrees    │   │
│  │ 4↔9  │ 0.78 │ merge  │  reject   │ ⚠ heuristic tight │   │
│  └──────┴──────┴────────┴───────────┴───────────────────┘   │
│                                                             │
│  ML and heuristic agree on 8/10 pairs                       │
│  2 disagreements — review these manually                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## `face_cluster/ml_trainer.py` — Core API

```python
@dataclass
class TrainConfig:
    model_type: str = "logistic_regression"  # logistic_regression | xgboost | mlp
    test_fraction: float = 0.15
    val_fraction: float = 0.15
    split_strategy: str = "random_stratified"  # random_stratified | by_run
    cv_folds: int = 5
    feature_groups: List[str] = field(default_factory=lambda: ["A", "BC", "D", "G"])
    hyperparams: Dict = field(default_factory=dict)  # model-specific overrides
    random_state: int = 42

@dataclass
class TrainResult:
    model: Any                          # fitted estimator
    scaler: StandardScaler
    feature_names: List[str]
    metrics: Dict                       # {train: {...}, val: {...}, test: {...}}
    confusion_matrix_test: List[List[int]]
    roc_curve_test: Dict                # {fpr: [], tpr: [], auc: float}
    feature_importance: Dict[str, float]
    config: TrainConfig
    created_at: str

class MergeTrainer:
    """Train, evaluate, save, and predict with merge classifiers."""

    FEATURE_GROUPS = {
        "A":  ["min_exemplar_dist", "exemplar_dist_mean", ...],   # 12 features
        "BC": ["size_a", "size_b", "diameter_a", ...],            # 19 features
        "D":  ["n_images_a", "n_images_b", ...],                  # 5 features
        "G":  ["mean_blur_a", "mean_blur_b", ...],                # 13 features
    }

    def train(self, df: pd.DataFrame, config: TrainConfig) -> TrainResult:
        """Full training pipeline: split → scale → fit → evaluate."""

    def predict(self, model_path: Path, df: pd.DataFrame) -> pd.DataFrame:
        """Load model, predict probabilities for candidate pairs."""

    @staticmethod
    def save_model(result: TrainResult, model_dir: Path) -> Path:
        """Pickle model+scaler+metadata, return path."""

    @staticmethod
    def load_model(model_path: Path) -> Dict:
        """Load pickle, return {model, scaler, feature_names, metadata}."""
```

### Model-specific estimators

```python
def _make_estimator(config: TrainConfig):
    if config.model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            **config.hyperparams,  # C, class_weight
        )
    elif config.model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            **config.hyperparams,  # n_estimators, max_depth, etc.
        )
    elif config.model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            max_iter=500,
            **config.hyperparams,  # hidden_layer_sizes, activation, etc.
        )
```

### Feature importance extraction

```python
def _extract_importance(model, feature_names, model_type):
    if model_type == "logistic_regression":
        return dict(zip(feature_names, model.coef_[0]))
    elif model_type == "xgboost":
        return dict(zip(feature_names, model.feature_importances_))
    elif model_type == "mlp":
        # Use first-layer weights magnitude as proxy
        w = model.coefs_[0]
        return dict(zip(feature_names, np.abs(w).sum(axis=1)))
```

---

## Split strategy: "By Run"

When split_strategy == "by_run":
- Group all samples by `run_id`
- Assign entire runs to train/val/test (not individual samples)
- This prevents data leakage: faces from the same album won't appear in both train and test
- Run assignment: sort runs by size descending, greedily assign to the split that is furthest below its target percentage

---

## Tab Structure Change

Current tabs (9):
```
Run | Recluster | History | Clusters(Base) | Cluster Analysis |
Clusters(Merged) | Merge Analysis | Face Analysis | Training Data
```

New tabs (10):
```
Run | Recluster | History | Clusters(Base) | Cluster Analysis |
Clusters(Merged) | Merge Analysis | Face Analysis | Labeling Review | ML Training
```

"Training Data" renamed to "Labeling Review" and extended with audit + suggestions.
"ML Training" is a new 10th tab.

---

## Implementation Phases

### Phase 1 (MVP)
- `face_cluster/ml_trainer.py` with `MergeTrainer.train()`, `.save_model()`, `.load_model()`
- Logistic Regression only (grid search with balanced class weights)
- ML Training tab: dataset config + model config + train + results + save
- Labeling Review tab: audit table with disagreement flags + label flip

### Phase 2
- XGBoost + MLP model types
- Model comparison view
- Apply-to-current-run prediction
- Active labeling suggestions
- By-run split strategy

### Phase 3
- Automated threshold tuning (use model probability to set optimal merge threshold)
- Retrain triggers (when N new labels are added since last training)
- A/B comparison: heuristic vs. ML merger results side-by-side on a test run

---

## Files Changed Summary

| File | Change |
|------|--------|
| `face_cluster/ml_trainer.py` | **New** — MergeTrainer class |
| `face_cluster/training_db.py` | Add `save_model_record`, `load_model_records`, `update_label`, `trained_models` table |
| `app/face_clustering.py` | Rename Training Data → Labeling Review; add audit section + label flip; add ML Training tab |
| `tests/face_clustering/test_ml_trainer.py` | **New** — train on synthetic data, save/load round-trip |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Overfitting on small datasets | Show train vs. test metrics side-by-side; warn if gap > 10% |
| XGBoost not installed | `try: import xgboost` with clear install hint in UI |
| Feature version mismatch | Stored in model metadata; check before prediction |
| Model pickle security | Only load models from `~/.sim_bench/models/` (trusted local dir) |
