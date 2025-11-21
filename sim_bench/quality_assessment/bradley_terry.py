"""
Bradley-Terry model for pairwise comparison evaluation.

The Bradley-Terry (BT) model is a probabilistic model for pairwise comparisons.
Given quality scores q_i and q_j for images i and j, the probability that i is
preferred over j is:

    P(i > j) = exp(q_i) / (exp(q_i) + exp(q_j))
              = 1 / (1 + exp(q_j - q_i))

This module provides:
1. Parameter estimation: Fit BT quality parameters from pairwise comparisons
2. Log-likelihood evaluation: Measure how well predicted scores match observations
3. Probabilistic metrics: Beyond binary accuracy, assess calibration quality

Reference:
Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs:
I. The method of paired comparisons. Biometrika, 39(3/4), 324-345.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
import logging

logger = logging.getLogger(__name__)


class BradleyTerryModel:
    """
    Bradley-Terry model for pairwise comparison analysis.

    Estimates quality parameters from pairwise comparison data using
    maximum likelihood estimation (MLE).
    """

    def __init__(self, regularization: float = 0.01):
        """
        Initialize Bradley-Terry model.

        Args:
            regularization: L2 regularization strength to prevent overfitting
        """
        self.regularization = regularization
        self.quality_params = {}
        self.is_fitted = False

    def fit(
        self,
        pairs: List[Tuple[str, str, int]],
        initial_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Fit Bradley-Terry parameters using maximum likelihood estimation.

        Args:
            pairs: List of (image_a_id, image_b_id, winner) tuples
                   where winner is 0 (A won) or 1 (B won)
            initial_params: Optional initial quality parameters

        Returns:
            Dictionary mapping image_id -> quality parameter
        """
        # Extract unique images
        all_images = set()
        for img_a, img_b, _ in pairs:
            all_images.add(img_a)
            all_images.add(img_b)

        image_list = sorted(all_images)
        image_to_idx = {img: idx for idx, img in enumerate(image_list)}
        n_images = len(image_list)

        # Initialize parameters
        if initial_params:
            x0 = np.array([initial_params.get(img, 0.0) for img in image_list])
        else:
            x0 = np.zeros(n_images)

        # Define negative log-likelihood + regularization
        def neg_log_likelihood(params):
            nll = 0.0

            for img_a, img_b, winner in pairs:
                idx_a = image_to_idx[img_a]
                idx_b = image_to_idx[img_b]

                q_a = params[idx_a]
                q_b = params[idx_b]

                # P(A > B) = 1 / (1 + exp(q_b - q_a))
                # Use log-sum-exp trick for numerical stability
                diff = q_b - q_a

                if winner == 0:  # A won
                    # log P(A > B) = -log(1 + exp(q_b - q_a))
                    nll -= -np.logaddexp(0, diff)
                else:  # B won
                    # log P(B > A) = -log(1 + exp(q_a - q_b))
                    nll -= -np.logaddexp(0, -diff)

            # Add L2 regularization
            nll += 0.5 * self.regularization * np.sum(params ** 2)

            return nll

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0,
            method='BFGS',
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"BT optimization did not converge: {result.message}")

        # Store fitted parameters
        self.quality_params = {img: result.x[idx] for img, idx in image_to_idx.items()}
        self.is_fitted = True

        logger.info(f"Fitted BT model for {n_images} images from {len(pairs)} pairs")

        return self.quality_params

    def predict_proba(self, image_a: str, image_b: str) -> float:
        """
        Predict probability that image_a is preferred over image_b.

        Args:
            image_a: First image ID
            image_b: Second image ID

        Returns:
            P(image_a > image_b) in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        q_a = self.quality_params.get(image_a, 0.0)
        q_b = self.quality_params.get(image_b, 0.0)

        # P(A > B) = 1 / (1 + exp(q_b - q_a))
        return expit(q_a - q_b)

    def evaluate_log_likelihood(
        self,
        pairs: List[Tuple[str, str, int]],
        quality_scores: Dict[str, float]
    ) -> float:
        """
        Evaluate log-likelihood of observed comparisons given quality scores.

        This is the key metric for comparing different quality assessment methods:
        which method's scores best explain the observed user preferences?

        Args:
            pairs: List of (image_a_id, image_b_id, winner) tuples
            quality_scores: Predicted quality scores from a method

        Returns:
            Log-likelihood (higher is better)
        """
        log_likelihood = 0.0

        for img_a, img_b, winner in pairs:
            q_a = quality_scores.get(img_a, 0.0)
            q_b = quality_scores.get(img_b, 0.0)

            # P(A > B) = sigmoid(q_a - q_b)
            diff = q_a - q_b

            if winner == 0:  # A won
                # log P(A > B) = -log(1 + exp(q_b - q_a))
                log_likelihood += -np.logaddexp(0, -diff)
            else:  # B won
                # log P(B > A) = -log(1 + exp(q_a - q_b))
                log_likelihood += -np.logaddexp(0, diff)

        return log_likelihood

    def compute_metrics(
        self,
        pairs: List[Tuple[str, str, int]],
        quality_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute comprehensive Bradley-Terry evaluation metrics.

        Args:
            pairs: List of (image_a_id, image_b_id, winner) tuples
            quality_scores: Predicted quality scores from a method

        Returns:
            Dictionary with metrics:
            - log_likelihood: Total log-likelihood
            - avg_log_likelihood: Average per comparison
            - accuracy: Binary classification accuracy
            - calibration_error: Expected calibration error
        """
        log_likelihood = 0.0
        correct = 0
        total = len(pairs)

        # For calibration: bin predictions and check match with outcomes
        prob_bins = np.linspace(0, 1, 11)  # 10 bins
        bin_counts = np.zeros(len(prob_bins) - 1)
        bin_correct = np.zeros(len(prob_bins) - 1)

        for img_a, img_b, winner in pairs:
            q_a = quality_scores.get(img_a, 0.0)
            q_b = quality_scores.get(img_b, 0.0)

            # Predicted probability that A wins
            diff = q_a - q_b
            prob_a_wins = expit(diff)

            # Log-likelihood
            if winner == 0:  # A won
                log_likelihood += -np.logaddexp(0, -diff)
                actual = 1
            else:  # B won
                log_likelihood += -np.logaddexp(0, diff)
                actual = 0

            # Accuracy
            predicted = 0 if prob_a_wins > 0.5 else 1
            if predicted == winner:
                correct += 1

            # Calibration
            bin_idx = np.digitize(prob_a_wins, prob_bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(bin_counts) - 1)
            bin_counts[bin_idx] += 1
            bin_correct[bin_idx] += actual

        # Expected calibration error
        ece = 0.0
        for i in range(len(bin_counts)):
            if bin_counts[i] > 0:
                bin_accuracy = bin_correct[i] / bin_counts[i]
                bin_confidence = (prob_bins[i] + prob_bins[i+1]) / 2
                ece += (bin_counts[i] / total) * abs(bin_accuracy - bin_confidence)

        return {
            'log_likelihood': log_likelihood,
            'avg_log_likelihood': log_likelihood / total if total > 0 else 0.0,
            'accuracy': correct / total if total > 0 else 0.0,
            'calibration_error': ece,
            'num_pairs': total
        }


def fit_bradley_terry_from_results(
    per_pair_results: List[Dict],
    regularization: float = 0.01
) -> Tuple[BradleyTerryModel, Dict[str, float]]:
    """
    Fit Bradley-Terry model from pairwise evaluation results.

    Args:
        per_pair_results: List of pair result dicts with keys:
            - pair_id, image_a_path, image_b_path, actual (winner)
        regularization: L2 regularization strength

    Returns:
        Tuple of (fitted BT model, quality parameters dict)
    """
    # Convert results to BT format
    pairs = []
    for result in per_pair_results:
        img_a = result['pair_id'].split('_')[0] + '_' + result['pair_id'].split('_')[1]
        img_b = result['pair_id'].split('_')[0] + '_' + result['pair_id'].split('_')[2]

        # Winner: 0 if A won, 1 if B won
        winner = 0 if result['actual'] == 'A' else 1
        pairs.append((img_a, img_b, winner))

    # Fit model
    bt_model = BradleyTerryModel(regularization=regularization)
    quality_params = bt_model.fit(pairs)

    return bt_model, quality_params


def evaluate_method_with_bt(
    per_pair_results: List[Dict],
    regularization: float = 0.01
) -> Dict[str, float]:
    """
    Evaluate a quality assessment method using Bradley-Terry metrics.

    Args:
        per_pair_results: List of pair result dicts with keys:
            - pair_id, score_a, score_b, actual (winner)
        regularization: L2 regularization strength

    Returns:
        Dictionary with BT metrics
    """
    # Extract quality scores from method predictions
    quality_scores = {}
    pairs = []

    for result in per_pair_results:
        pair_id = result['pair_id']
        img_a = pair_id.split('_')[0] + '_' + pair_id.split('_')[1]
        img_b = pair_id.split('_')[0] + '_' + pair_id.split('_')[2]

        quality_scores[img_a] = result['score_a']
        quality_scores[img_b] = result['score_b']

        winner = 0 if result['actual'] == 'A' else 1
        pairs.append((img_a, img_b, winner))

    # Create BT model for evaluation
    bt_model = BradleyTerryModel(regularization=regularization)

    # Compute metrics
    metrics = bt_model.compute_metrics(pairs, quality_scores)

    return metrics


def fit_bt_from_pairs_data(
    pairs: List[Dict],
    image_id_key_a: str = 'image_a_id',
    image_id_key_b: str = 'image_b_id',
    winner_key: str = 'chosen_image',
    regularization: float = 0.01
) -> Tuple[BradleyTerryModel, Dict[str, float]]:
    """
    Fit Bradley-Terry model from pairs data with ground truth winners.
    
    This fits BT from the actual user preferences (ground truth), which can then
    be used to evaluate how well method predictions match the probabilistic
    ground truth model.
    
    Args:
        pairs: List of pair dictionaries with:
            - image_a_id/image_b_id: Unique identifiers for images
            - chosen_image: 'A' or 'B' indicating which image was preferred
        image_id_key_a: Key for first image ID in pair dict
        image_id_key_b: Key for second image ID in pair dict
        winner_key: Key for winner ('A' or 'B')
        regularization: L2 regularization strength
        
    Returns:
        Tuple of (fitted BT model, quality parameters dict)
    """
    # Convert pairs to BT format: (image_a_id, image_b_id, winner)
    # winner: 0 if A won, 1 if B won
    bt_pairs = []
    
    for pair in pairs:
        img_a = pair.get(image_id_key_a)
        img_b = pair.get(image_id_key_b)
        winner_str = pair.get(winner_key, '').upper()
        
        if not img_a or not img_b:
            continue
            
        # Convert winner to 0/1
        if winner_str == 'A':
            winner = 0
        elif winner_str == 'B':
            winner = 1
        else:
            # Try to infer from other fields
            if 'MaxVote' in pair:
                max_vote = pair['MaxVote']
                if max_vote == 1:
                    winner = 0  # Image 1 (A) won
                elif max_vote == 2:
                    winner = 1  # Image 2 (B) won
                else:
                    continue
            else:
                continue
        
        bt_pairs.append((str(img_a), str(img_b), winner))
    
    if not bt_pairs:
        raise ValueError("No valid pairs found for BT fitting")
    
    # Fit model
    bt_model = BradleyTerryModel(regularization=regularization)
    quality_params = bt_model.fit(bt_pairs)
    
    logger.info(f"Fitted BT model from {len(bt_pairs)} ground truth pairs")
    
    return bt_model, quality_params


def compute_bt_log_likelihood_for_predictions(
    bt_model: BradleyTerryModel,
    predictions: List[Dict],
    image_id_key_a: str = 'image_a_id',
    image_id_key_b: str = 'image_b_id',
    predicted_winner_key: str = 'predicted'
) -> float:
    """
    Compute log-likelihood of predictions under fitted BT model.
    
    This evaluates how well method predictions match the probabilistic
    ground truth model learned from user preferences.
    
    Args:
        bt_model: Fitted BradleyTerryModel from ground truth
        predictions: List of prediction dicts with:
            - image_a_id/image_b_id: Image identifiers
            - predicted: 'A' or 'B' (predicted winner)
        image_id_key_a: Key for first image ID
        image_id_key_b: Key for second image ID
        predicted_winner_key: Key for predicted winner
        
    Returns:
        Log-likelihood (higher is better)
    """
    if not bt_model.is_fitted:
        raise ValueError("BT model must be fitted before computing log-likelihood")
    
    log_likelihood = 0.0
    
    for pred in predictions:
        img_a = str(pred.get(image_id_key_a, ''))
        img_b = str(pred.get(image_id_key_b, ''))
        predicted_str = pred.get(predicted_winner_key, '').upper()
        
        if not img_a or not img_b:
            continue
        
        # Get BT probability for predicted winner
        if predicted_str == 'A':
            prob = bt_model.predict_proba(img_a, img_b)
        elif predicted_str == 'B':
            prob = bt_model.predict_proba(img_b, img_a)
        else:
            # Try numeric format (1 or 2)
            if predicted_str == '1' or pred.get('predicted_winner') == 1:
                prob = bt_model.predict_proba(img_a, img_b)
            elif predicted_str == '2' or pred.get('predicted_winner') == 2:
                prob = bt_model.predict_proba(img_b, img_a)
            else:
                continue
        
        # Add log probability (with small epsilon to avoid log(0))
        log_likelihood += np.log(max(prob, 1e-10))
    
    return log_likelihood


__all__ = [
    'BradleyTerryModel',
    'fit_bradley_terry_from_results',
    'evaluate_method_with_bt',
    'fit_bt_from_pairs_data',
    'compute_bt_log_likelihood_for_predictions'
]
