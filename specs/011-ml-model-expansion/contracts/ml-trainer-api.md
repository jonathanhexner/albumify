# Contract: ML Trainer API Extensions

## _make_estimator(config: TrainConfig) -> Any

### New model_type branches

```
"decision_tree" -> DecisionTreeClassifier(
    max_depth     = hp.get("max_depth", 5),
    class_weight  = hp.get("class_weight", "balanced"),
    random_state  = config.random_state,
)

"random_forest" -> RandomForestClassifier(
    n_estimators  = hp.get("n_estimators", 100),
    max_depth     = hp.get("max_depth", 5),
    class_weight  = hp.get("class_weight", "balanced"),
    n_jobs        = -1,
    random_state  = config.random_state,
)

"catboost" -> CatBoostClassifier(
    iterations    = hp.get("iterations", 100),
    depth         = hp.get("depth", 4),
    learning_rate = hp.get("learning_rate", 0.1),
    verbose       = 0,
    random_state  = config.random_state,
    auto_class_weights = "Balanced",
)
```

CatBoost import is guarded with try/except ImportError (same pattern as XGBoost).

## _extract_importance(model, feature_names, model_type) -> Dict[str, float]

### New branches

```
"decision_tree"  -> dict(zip(feature_names, model.feature_importances_.tolist()))
"random_forest"  -> dict(zip(feature_names, model.feature_importances_.tolist()))
"catboost"       -> dict(zip(feature_names, model.feature_importances_.tolist()))
```

All three use the same `feature_importances_` attribute (unsigned Gini/gain).

## compute_pair_feature_contributions() — no changes needed

The existing proxy path (`feature_importances_` branch) handles all tree-based models. No model-type-specific logic needed — `hasattr(model, 'feature_importances_')` is the check.

## FEATURE_CONFIG_MAP — new constant

```python
FEATURE_CONFIG_MAP: Dict[str, Optional[Dict]] = {
    "min_exemplar_dist": {"field": "merge_exemplar_threshold", "default": 0.35, "hint": "lower = more merges"},
    "support_fraction": {"field": "merge_support_frac", "default": 0.3, "hint": "lower = more merges"},
    ...
}
```

Exposed as `MergeTrainer.FEATURE_CONFIG_MAP` for the UI to consume.
