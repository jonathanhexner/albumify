# Data Model: ML Model Expansion & Merge Interpretability

## Entities

### TrainConfig (modified)

```
model_type: str
  # Existing: "logistic_regression" | "xgboost" | "mlp"
  # New:      "decision_tree" | "random_forest" | "catboost"
test_fraction: float = 0.15
val_fraction: float = 0.15
split_strategy: str = "random_stratified"
cv_folds: int = 5
feature_groups: List[str] = ["A", "BC", "D", "G"]
hyperparams: Dict[str, Any] = {}
random_state: int = 42
```

### FEATURE_CONFIG_MAP (new static dict)

```
{
    "min_exemplar_dist":             {"config_field": "merge_exemplar_threshold",         "default": 0.35, "direction": "lower = more merges"},
    "support_fraction":              {"config_field": "merge_support_frac",               "default": 0.3,  "direction": "lower = more merges"},
    "n_cross_pairs_below_threshold": {"config_field": "merge_support_min",                "default": 2,    "direction": "lower = more merges"},
    "dist_to_threshold_ratio":       {"config_field": "merge_exemplar_threshold",         "default": 0.35, "direction": "lower ratio = more merges"},
    "post_merge_diameter":           {"config_field": "merge_diameter_expansion_factor",   "default": 1.5,  "direction": "higher = more merges"},
    "blur_min_a":                    {"config_field": "blur_min",                          "default": 50.0, "direction": "lower = more core faces"},
    "blur_min_b":                    {"config_field": "blur_min",                          "default": 50.0, "direction": "lower = more core faces"},
    "frontal_frac_a":                {"config_field": "yaw_max",                           "default": 30.0, "direction": "higher = more frontal faces"},
    "frontal_frac_b":                {"config_field": "yaw_max",                           "default": 30.0, "direction": "higher = more frontal faces"},
    # Features with no direct config knob:
    "exemplar_dist_mean":            null,
    "exemplar_dist_std":             null,
    "diameter_a":                    null,
    "diameter_b":                    null,
    "diameter_ratio":                null,
    "diameter_expansion":            null,
    "cross_dist_iqr":                null,
    "size_ratio":                    null,
    "area_ratio":                    null,
    "pose_diff":                     null,
    ...
}
```

### merge_training_data table (modified)

```sql
-- Existing columns:
id              INTEGER PRIMARY KEY AUTOINCREMENT
run_id          TEXT    NOT NULL
album_path      TEXT
cluster_a       INTEGER NOT NULL
cluster_b       INTEGER NOT NULL
label           INTEGER
feature_version INTEGER NOT NULL
features_json   TEXT    NOT NULL
output_dir      TEXT
exemplar_ids    TEXT
saved_at        TEXT    NOT NULL
UNIQUE(run_id, cluster_a, cluster_b)

-- New column:
source          TEXT    DEFAULT 'human'   -- 'human' | 'auto_negative'
```

### PairEvidence (return type of evaluate_pair_evidence)

Identical to the dict returned by `ConservativeMerger._evaluate_merge_evidence()`:

```
{
    valid:                 bool,
    exemplar_dist:         float,
    merge_threshold:       float,
    passes_exemplar:       bool,
    support:               int,
    required_support:      int,
    passes_support:        bool,
    passes_margin:         bool,
    margin_gap:            float | None,
    margin_dist_to_b:      float | None,
    margin_competitor_dist: float | None,
    margin_competitor_id:  int | None,
    post_diameter:         float,
    max_allowed_diameter:  float,
    passes_diameter:       bool,
}
```

## Relationships

```
TrainConfig --[selects]--> _make_estimator() --> fitted model
fitted model --[has]--> feature_importances_ / coef_ --> FEATURE_CONFIG_MAP lookup
FEATURE_CONFIG_MAP --[maps]--> PipelineConfig / ConservativeMergerConfig fields

merge_training_data --[source='human']--> from user merge decisions
merge_training_data --[source='auto_negative']--> from harvested non-candidate pairs

evaluate_pair_evidence() --[computes]--> PairEvidence for any (cid_a, cid_b) pair
save_manual_merge_snapshot() --[applies]--> approved_pairs via union-find
```

## State Transitions

### Training Sample Lifecycle

```
Non-candidate pair (dist > threshold)
  --> [Harvest Negatives] --> source='auto_negative', label=0
  --> [stored in merge_training_data]

Manual merge pair (any distance)
  --> [User confirms] --> source='human', label=1
  --> [stored in merge_training_data]
```
