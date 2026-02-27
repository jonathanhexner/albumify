"""
Train logistic regression classifier for cluster merge prediction.

Takes candidate pairs and corrected labels, generates training data,
trains model, and evaluates performance.

Usage:
    python scripts/train_merge_classifier.py \
        --candidate-pairs results/training/dataset1/candidate_pairs.csv \
        --corrected-labels results/training/dataset1/corrected_labels.csv \
        --output results/training/dataset1/model \
        --test-split 0.2
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Configure logging to both console and file."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f'training_{timestamp}.log'

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")
    return log_file


def generate_training_labels(
    pairs_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate binary merge labels from corrected identities.

    Args:
        pairs_df: Candidate pairs with features
        labels_df: Corrected labels (cluster_id -> corrected_identity)

    Returns:
        DataFrame with features + binary 'label' column
    """
    logger.info("Generating training labels...")

    # Create cluster_id -> identity mapping
    cluster_to_identity = dict(zip(labels_df['cluster_id'], labels_df['corrected_identity']))

    # Generate labels for each pair
    labels = []
    for _, row in pairs_df.iterrows():
        cluster_1 = row['cluster_id_1']
        cluster_2 = row['cluster_id_2']

        identity_1 = cluster_to_identity.get(cluster_1, None)
        identity_2 = cluster_to_identity.get(cluster_2, None)

        if identity_1 is None or identity_2 is None:
            logger.warning(f"Missing identity for cluster pair ({cluster_1}, {cluster_2})")
            continue

        # Label = 1 if same identity (should merge), 0 otherwise
        label = 1 if identity_1 == identity_2 else 0
        labels.append(label)

    if len(labels) != len(pairs_df):
        logger.warning(f"Generated {len(labels)} labels for {len(pairs_df)} pairs")

    # Add labels to dataframe
    training_df = pairs_df.copy()
    training_df['label'] = labels

    # Log class distribution
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    logger.info(f"Class distribution:")
    logger.info(f"  Should merge (1): {n_positive} ({100*n_positive/len(labels):.1f}%)")
    logger.info(f"  Should not merge (0): {n_negative} ({100*n_negative/len(labels):.1f}%)")

    if n_positive == 0 or n_negative == 0:
        logger.error("Training data has only one class! Need both positive and negative examples.")
        sys.exit(1)

    return training_df


def prepare_features(
    training_df: pd.DataFrame,
    feature_names: list = None
) -> tuple:
    """
    Prepare features for training.

    Args:
        training_df: DataFrame with features + label
        feature_names: List of feature column names (if None, auto-detect)

    Returns:
        (X, y, feature_names, scaler)
    """
    logger.info("Preparing features...")

    # Auto-detect feature columns (exclude cluster IDs and label)
    if feature_names is None:
        exclude_cols = ['cluster_id_1', 'cluster_id_2', 'label']
        feature_names = [col for col in training_df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_names)} features:")
    for i, name in enumerate(feature_names, 1):
        logger.info(f"  {i}. {name}")

    # Extract features and labels
    X = training_df[feature_names].values
    y = training_df['label'].values

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Feature matrix shape: {X_scaled.shape}")
    logger.info(f"Labels shape: {y.shape}")

    # Check for NaN or inf
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        logger.error("Features contain NaN or inf values!")
        sys.exit(1)

    return X_scaled, y, feature_names, scaler


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Train logistic regression with hyperparameter tuning.

    Args:
        X: Feature matrix (scaled)
        y: Labels
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        (model, X_train, X_test, y_train, y_test, best_params)
    """
    logger.info("Training logistic regression...")

    # Train-test split (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(y_train)} samples ({sum(y_train)} positive)")
    logger.info(f"Test set: {len(y_test)} samples ({sum(y_test)} positive)")

    # Hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],  # Regularization strength
        'class_weight': ['balanced', None],  # Handle class imbalance
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }

    # Grid search with cross-validation
    logger.info("Running grid search with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        LogisticRegression(random_state=random_state),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.3f}")

    model = grid_search.best_estimator_

    return model, X_train, X_test, y_train, y_test, grid_search.best_params_


def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> dict:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        feature_names: Feature names

    Returns:
        Dict of metrics
    """
    logger.info("Evaluating model...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Classification reports
    report_train = classification_report(y_train, y_train_pred, output_dict=True)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)

    # ROC AUC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    # Feature importance (coefficients)
    feature_importance = dict(zip(feature_names, model.coef_[0]))
    feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

    logger.info(f"\nTrain Performance:")
    logger.info(f"  Accuracy: {report_train['accuracy']:.3f}")
    logger.info(f"  Precision (merge): {report_train['1']['precision']:.3f}")
    logger.info(f"  Recall (merge): {report_train['1']['recall']:.3f}")
    logger.info(f"  F1 (merge): {report_train['1']['f1-score']:.3f}")
    logger.info(f"  ROC AUC: {auc_train:.3f}")

    logger.info(f"\nTest Performance:")
    logger.info(f"  Accuracy: {report_test['accuracy']:.3f}")
    logger.info(f"  Precision (merge): {report_test['1']['precision']:.3f}")
    logger.info(f"  Recall (merge): {report_test['1']['recall']:.3f}")
    logger.info(f"  F1 (merge): {report_test['1']['f1-score']:.3f}")
    logger.info(f"  ROC AUC: {auc_test:.3f}")

    logger.info(f"\nTop 10 Most Important Features:")
    for i, (name, coef) in enumerate(feature_importance_sorted[:10], 1):
        logger.info(f"  {i}. {name}: {coef:.4f}")

    metrics = {
        'train': {
            'confusion_matrix': cm_train.tolist(),
            'classification_report': report_train,
            'roc_auc': float(auc_train),
            'fpr': fpr_train.tolist(),
            'tpr': tpr_train.tolist(),
        },
        'test': {
            'confusion_matrix': cm_test.tolist(),
            'classification_report': report_test,
            'roc_auc': float(auc_test),
            'fpr': fpr_test.tolist(),
            'tpr': tpr_test.tolist(),
        },
        'feature_importance': {name: float(coef) for name, coef in feature_importance.items()},
        'feature_importance_sorted': [(name, float(coef)) for name, coef in feature_importance_sorted],
    }

    return metrics


def plot_diagnostics(
    metrics: dict,
    output_dir: Path,
    feature_names: list
):
    """
    Generate diagnostic plots.

    Args:
        metrics: Metrics dict from evaluate_model()
        output_dir: Output directory for plots
        feature_names: Feature names
    """
    logger.info("Generating diagnostic plots...")

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_train = np.array(metrics['train']['confusion_matrix'])
    cm_test = np.array(metrics['test']['confusion_matrix'])

    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (Train)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix (Test)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))

    fpr_train = metrics['train']['fpr']
    tpr_train = metrics['train']['tpr']
    auc_train = metrics['train']['roc_auc']

    fpr_test = metrics['test']['fpr']
    tpr_test = metrics['test']['tpr']
    auc_test = metrics['test']['roc_auc']

    ax.plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_train:.3f})', linewidth=2)
    ax.plot(fpr_test, tpr_test, label=f'Test (AUC = {auc_test:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Feature importance
    feature_importance_sorted = metrics['feature_importance_sorted']
    names = [x[0] for x in feature_importance_sorted]
    values = [x[1] for x in feature_importance_sorted]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))

    colors = ['green' if v > 0 else 'red' for v in values]
    ax.barh(names, values, color=colors, alpha=0.7)
    ax.set_xlabel('Coefficient (Log Odds Ratio)')
    ax.set_title('Feature Importance (Logistic Regression Coefficients)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plots to: {plots_dir}")


def save_artifacts(
    model,
    scaler,
    metrics: dict,
    feature_names: list,
    best_params: dict,
    output_dir: Path
):
    """
    Save trained model and artifacts.

    Args:
        model: Trained model
        scaler: Feature scaler
        metrics: Evaluation metrics
        feature_names: Feature names
        best_params: Best hyperparameters
        output_dir: Output directory
    """
    logger.info("Saving artifacts...")

    # Save model + scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
    }

    model_path = output_dir / 'merge_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Saved model to: {model_path}")

    # Save metrics as JSON
    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Save feature names
    features_path = output_dir / 'feature_names.txt'
    with open(features_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    logger.info(f"Saved feature names to: {features_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train logistic regression classifier for cluster merging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--candidate-pairs',
        type=Path,
        required=True,
        help='Path to candidate_pairs.csv (from export script)'
    )
    parser.add_argument(
        '--corrected-labels',
        type=Path,
        required=True,
        help='Path to corrected_labels.csv (from labeling app)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for model and metrics'
    )

    # Training parameters
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.candidate_pairs.exists():
        print(f"ERROR: candidate_pairs.csv not found: {args.candidate_pairs}")
        sys.exit(1)

    if not args.corrected_labels.exists():
        print(f"ERROR: corrected_labels.csv not found: {args.corrected_labels}")
        sys.exit(1)

    # Setup logging
    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output)

    logger.info("="*60)
    logger.info("Merge Classifier Training")
    logger.info("="*60)
    logger.info(f"Candidate pairs: {args.candidate_pairs}")
    logger.info(f"Corrected labels: {args.corrected_labels}")
    logger.info(f"Output: {args.output}")

    try:
        # Load data
        logger.info("Loading data...")
        pairs_df = pd.read_csv(args.candidate_pairs)
        labels_df = pd.read_csv(args.corrected_labels)

        logger.info(f"Loaded {len(pairs_df)} candidate pairs")
        logger.info(f"Loaded {len(labels_df)} cluster labels")

        # Generate training labels
        training_df = generate_training_labels(pairs_df, labels_df)

        # Save training data
        training_data_path = args.output / 'merge_training_data.csv'
        training_df.to_csv(training_data_path, index=False)
        logger.info(f"Saved training data to: {training_data_path}")

        # Prepare features
        X, y, feature_names, scaler = prepare_features(training_df)

        # Train model
        model, X_train, X_test, y_train, y_test, best_params = train_model(
            X, y,
            test_size=args.test_split,
            random_state=args.random_seed
        )

        # Evaluate model
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)

        # Generate plots
        plot_diagnostics(metrics, args.output, feature_names)

        # Save artifacts
        save_artifacts(model, scaler, metrics, feature_names, best_params, args.output)

        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        logger.info(f"Test F1 Score: {metrics['test']['classification_report']['1']['f1-score']:.3f}")
        logger.info(f"Test ROC AUC: {metrics['test']['roc_auc']:.3f}")
        logger.info(f"\nOutput files:")
        logger.info(f"  - merge_classifier.pkl (model + scaler)")
        logger.info(f"  - training_metrics.json")
        logger.info(f"  - merge_training_data.csv")
        logger.info(f"  - plots/ (confusion matrices, ROC, feature importance)")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
