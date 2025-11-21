"""
Train a regression model to learn optimal CLIP prompt aggregation.

Instead of using hardcoded aggregation (weighted, mean), this learns
the optimal weights for combining CLIP prompt scores based on ground
truth labels from the dataset.

The model learns which aesthetic dimensions (focus, composition, exposure, etc.)
are most predictive of image quality in your specific dataset.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_benchmark_results(results_file: Path) -> Dict:
    """Load benchmark results JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_features_and_labels(
    results: Dict,
    feature_type: str = 'contrast'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix and labels from benchmark results.

    Args:
        results: Benchmark results dictionary
        feature_type: Type of features to extract:
            - 'contrast': Contrastive pair scores (pos - neg)
            - 'positive': Only positive prompt scores
            - 'negative': Only negative prompt scores
            - 'all': All detailed scores
            - 'contrast_and_attrs': Contrastive + positive/negative attributes

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,) - 1.0 for best image, 0.0 for others
        feature_names: List of feature names
    """
    X_list = []
    y_list = []
    feature_names = None

    # Iterate through series
    series_results = results.get('series_results', [])

    for series in series_results:
        detailed_scores = series.get('detailed_scores')
        if not detailed_scores:
            continue  # Skip series without detailed scores

        ground_truth_idx = series.get('ground_truth_idx', 0)
        num_images = series['num_images']

        # Extract features for each image in series
        for img_idx, img_detailed in enumerate(detailed_scores):
            if not img_detailed:
                continue

            # Extract features based on type
            features = []
            names = []

            for key in sorted(img_detailed.keys()):
                value = img_detailed[key]

                # Filter by feature type
                if feature_type == 'contrast' and key.startswith('contrast_'):
                    features.append(value)
                    names.append(key)
                elif feature_type == 'positive' and key.startswith('pos_'):
                    features.append(value)
                    names.append(key)
                elif feature_type == 'negative' and key.startswith('neg_'):
                    features.append(value)
                    names.append(key)
                elif feature_type == 'all':
                    features.append(value)
                    names.append(key)
                elif feature_type == 'contrast_and_attrs':
                    if key.startswith('contrast_') or key.startswith('positive_') or key.startswith('negative_'):
                        features.append(value)
                        names.append(key)

            if not features:
                continue

            # Set feature names on first iteration
            if feature_names is None:
                feature_names = names

            # Binary label: 1.0 if this is the best image, 0.0 otherwise
            label = 1.0 if img_idx == ground_truth_idx else 0.0

            X_list.append(features)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(f"Extracted {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Positive samples (best images): {np.sum(y)}")
    logger.info(f"Negative samples (not best): {len(y) - np.sum(y)}")

    return X, y, feature_names


def train_regression_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> Dict[str, any]:
    """
    Train multiple regression models and compare performance.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Feature names

    Returns:
        Dictionary with trained models and performance metrics
    """
    logger.info("="*80)
    logger.info("Training Regression Models")
    logger.info("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.astype(int)
    )

    logger.info(f"\nTrain set: {X_train.shape[0]} samples")
    logger.info(f"Test set:  {X_test.shape[0]} samples")

    # Define models to try
    models = {
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}

    logger.info("\n" + "="*80)
    logger.info("Model Performance Comparison")
    logger.info("="*80)

    for name, model in models.items():
        logger.info(f"\n{name}:")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        logger.info(f"  Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        logger.info(f"  Test MSE:  {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
        logger.info(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logger.info(f"\n  Top 5 Important Features:")
            top_indices = np.argsort(importances)[::-1][:5]
            for idx in top_indices:
                logger.info(f"    {feature_names[idx]}: {importances[idx]:.4f}")

        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            logger.info(f"\n  Top 5 Positive Coefficients:")
            top_indices = np.argsort(coefs)[::-1][:5]
            for idx in top_indices:
                logger.info(f"    {feature_names[idx]}: {coefs[idx]:+.4f}")

            logger.info(f"\n  Top 5 Negative Coefficients:")
            bottom_indices = np.argsort(coefs)[:5]
            for idx in bottom_indices:
                logger.info(f"    {feature_names[idx]}: {coefs[idx]:+.4f}")

        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    logger.info("\n" + "="*80)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    logger.info("="*80)

    return {
        'models': results,
        'best_model_name': best_model_name,
        'best_model': results[best_model_name]['model'],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }


def save_model(model, feature_names: List[str], output_file: Path):
    """Save trained model and feature names."""
    model_data = {
        'model': model,
        'feature_names': feature_names
    }

    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"\nSaved model to: {output_file}")


def main():
    """Train CLIP aggregation regression model."""
    logger.info("="*80)
    logger.info("CLIP Prompt Aggregation - Regression Model Training")
    logger.info("="*80)

    # Find most recent benchmark results with detailed scores
    benchmark_dir = Path("outputs/quality_benchmarks")

    if not benchmark_dir.exists():
        logger.error(f"Benchmark directory not found: {benchmark_dir}")
        logger.info("\nPlease run a quality benchmark first:")
        logger.info("  python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml")
        return

    # Find all benchmark runs
    benchmark_runs = sorted(benchmark_dir.glob("benchmark_*"))
    if not benchmark_runs:
        logger.error("No benchmark runs found")
        return

    # Try to find a run with CLIP results
    results_file = None
    for run_dir in reversed(benchmark_runs):
        # Look for results.json
        potential_file = run_dir / "results.json"
        if potential_file.exists():
            # Check if it has CLIP detailed scores
            with open(potential_file, 'r') as f:
                data = json.load(f)
                # Check if any method has detailed_scores
                for dataset_name, dataset_results in data.items():
                    method_results = dataset_results.get('method_results', {})
                    for method_name, method_data in method_results.items():
                        series_results = method_data.get('series_results', [])
                        if series_results and series_results[0].get('detailed_scores'):
                            results_file = potential_file
                            logger.info(f"\nUsing benchmark results: {results_file}")
                            break
                    if results_file:
                        break
            if results_file:
                break

    if not results_file:
        logger.error("\nNo benchmark results with detailed CLIP scores found!")
        logger.info("\nPlease run a CLIP benchmark first:")
        logger.info("  1. Make sure configs/quality_benchmark.deep_learning.yaml includes clip_aesthetic or clip_learned")
        logger.info("  2. Run: python run_quality_benchmark.py configs/quality_benchmark.deep_learning.yaml")
        return

    # Load results
    logger.info("\nLoading benchmark results...")
    with open(results_file, 'r') as f:
        all_results = json.load(f)

    # Find the dataset and method with detailed scores
    selected_results = None
    for dataset_name, dataset_results in all_results.items():
        method_results = dataset_results.get('method_results', {})
        for method_name, method_data in method_results.items():
            if method_data.get('series_results', [{}])[0].get('detailed_scores'):
                selected_results = method_data
                logger.info(f"Dataset: {dataset_name}")
                logger.info(f"Method: {method_name}")
                break
        if selected_results:
            break

    if not selected_results:
        logger.error("Could not find results with detailed scores!")
        return

    # Extract features and labels
    logger.info("\nExtracting features and labels...")
    X, y, feature_names = extract_features_and_labels(
        selected_results,
        feature_type='contrast'  # Use contrastive scores
    )

    if X.shape[0] == 0:
        logger.error("No features extracted! Make sure benchmark has detailed_scores.")
        return

    # Train models
    training_results = train_regression_models(X, y, feature_names)

    # Save best model
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_file = output_dir / "clip_aggregation_model.pkl"

    save_model(
        training_results['best_model'],
        feature_names,
        model_file
    )

    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"\nBest model: {training_results['best_model_name']}")
    logger.info(f"Saved to: {model_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Use the trained model to predict quality scores")
    logger.info("  2. Compare learned aggregation vs hardcoded (weighted, mean)")
    logger.info("  3. Analyze which aesthetic dimensions are most predictive")
    logger.info("="*80)


if __name__ == "__main__":
    main()
