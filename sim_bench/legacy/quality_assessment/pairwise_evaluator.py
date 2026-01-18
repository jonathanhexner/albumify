"""
Pairwise evaluator for quality assessment methods.

Evaluates methods on pairwise comparison tasks: given two images,
predict which one was preferred by users.

This is the correct evaluation for PhotoTriage, which provides
pairwise preferences rather than absolute series rankings.
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.bradley_terry import (
    BradleyTerryModel,
    fit_bt_from_pairs_data,
    compute_bt_log_likelihood_for_predictions
)

logger = logging.getLogger(__name__)


class PairwiseEvaluator:
    """Evaluate quality assessment methods on pairwise classification."""

    def __init__(
        self,
        pairs_file: Path,
        method: QualityAssessor,
        logger: Optional[logging.Logger] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
        bt_model: Optional[BradleyTerryModel] = None
    ):
        """
        Initialize pairwise evaluator.

        Args:
            pairs_file: Path to pairs JSONL file (from PhotoTriage dataset)
            method: Quality assessment method to evaluate
            logger: Optional logger instance
            sample_size: If provided, randomly sample this many pairs
            random_seed: Random seed for sampling
            bt_model: Optional pre-fitted Bradley-Terry model from ground truth
        """
        self.pairs_file = Path(pairs_file)
        self.method = method
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.bt_model = bt_model

        # Load pairs
        self.pairs = self._load_pairs()

        # Sample if requested
        if sample_size is not None and sample_size < len(self.pairs):
            random.seed(random_seed)
            self.pairs = random.sample(self.pairs, sample_size)
            self.logger.info(f"Sampled {len(self.pairs)} pairs (seed={random_seed})")

        self.results = None

    def _load_pairs(self) -> List[Dict]:
        """Load pairs from JSONL file."""
        pairs = []

        self.logger.info(f"Loading pairs from {self.pairs_file}")

        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)
                pairs.append(pair)

        self.logger.info(f"Loaded {len(pairs)} pairs")

        return pairs

    def _compute_bradley_terry_metrics(self, pair_results: List[Dict]) -> Dict[str, float]:
        """
        Compute Bradley-Terry metrics from pair results.

        Args:
            pair_results: List of pair result dicts with score_a, score_b, actual

        Returns:
            Dictionary with BT metrics
        """
        try:
            from sim_bench.quality_assessment.bradley_terry import evaluate_method_with_bt

            bt_metrics = evaluate_method_with_bt(pair_results, regularization=0.01)
            return bt_metrics

        except Exception as e:
            self.logger.warning(f"Could not compute Bradley-Terry metrics: {e}")
            return {
                'log_likelihood': 0.0,
                'avg_log_likelihood': 0.0,
                'accuracy': 0.0,
                'calibration_error': 0.0,
                'num_pairs': 0
            }

    def evaluate(
        self,
        verbose: bool = True,
        use_attributes: bool = False,
        attribute_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate method on pairwise classification.

        Args:
            verbose: Whether to show progress bar
            use_attributes: If True, also evaluate per-attribute
            attribute_filter: If provided, only evaluate pairs with this attribute

        Returns:
            Dictionary with evaluation results:
            {
                'global': {
                    'accuracy': float,
                    'num_pairs': int,
                    'num_correct': int
                },
                'per_attribute': {  # if use_attributes=True
                    'sharpness': {'accuracy': ..., 'num_pairs': ...},
                    ...
                },
                'per_pair_results': [...]
            }
        """
        self.logger.info(f"Evaluating {self.method.name} on pairwise task...")

        # Filter pairs if needed
        eval_pairs = self.pairs
        if attribute_filter:
            eval_pairs = [
                p for p in self.pairs
                if any(attr['name'] == attribute_filter for attr in p.get('attributes', []))
            ]
            self.logger.info(f"Filtered to {len(eval_pairs)} pairs with attribute '{attribute_filter}'")

        # Evaluate each pair
        correct_global = 0
        per_attribute_correct = defaultdict(int)
        per_attribute_total = defaultdict(int)
        pair_results = []

        iterator = tqdm(eval_pairs, desc="Evaluating pairs") if verbose else eval_pairs

        for pair in iterator:
            try:
                # Load image paths
                image_a_path = pair['image_a_path']
                image_b_path = pair['image_b_path']

                # Assess both images
                score_a = self.method.assess_image(image_a_path)
                score_b = self.method.assess_image(image_b_path)

                # Predict winner (0=A, 1=B)
                predicted = 0 if score_a > score_b else 1

                # Ground truth (0=A, 1=B)
                chosen = 0 if pair['chosen_image'] == 'A' else 1

                # Check if correct
                is_correct = (predicted == chosen)
                correct_global += int(is_correct)

                # Per-attribute evaluation
                attribute_predictions = {}
                if use_attributes:
                    for attr in pair.get('attributes', []):
                        attr_name = attr['name']
                        attr_winner = 0 if attr['winner'] == 'A' else 1

                        # Same prediction for all attributes (single quality score)
                        attr_correct = (predicted == attr_winner)

                        per_attribute_correct[attr_name] += int(attr_correct)
                        per_attribute_total[attr_name] += 1

                        attribute_predictions[attr_name] = {
                            'predicted': 'A' if predicted == 0 else 'B',
                            'actual': attr['winner'],
                            'correct': attr_correct
                        }

                # Store result
                pair_results.append({
                    'pair_id': pair['pair_id'],
                    'series_id': pair['series_id'],
                    'score_a': float(score_a),
                    'score_b': float(score_b),
                    'predicted': 'A' if predicted == 0 else 'B',
                    'actual': pair['chosen_image'],
                    'correct': is_correct,
                    'preference_strength': pair.get('preference_strength', 0.5),
                    'attribute_predictions': attribute_predictions if use_attributes else {}
                })

            except Exception as e:
                self.logger.error(f"Error processing pair {pair.get('pair_id', 'unknown')}: {e}")
                continue

        # Compute global accuracy
        num_pairs = len(pair_results)
        global_accuracy = correct_global / num_pairs if num_pairs > 0 else 0.0

        # Compute per-attribute accuracies
        per_attribute_metrics = {}
        if use_attributes:
            for attr_name, correct_count in per_attribute_correct.items():
                total_count = per_attribute_total[attr_name]
                accuracy = correct_count / total_count if total_count > 0 else 0.0

                per_attribute_metrics[attr_name] = {
                    'accuracy': accuracy,
                    'num_pairs': total_count,
                    'num_correct': correct_count
                }

        # Additional metrics
        # Agreement by preference strength (strong vs weak preferences)
        strong_threshold = 0.75
        weak_threshold = 0.55

        strong_pairs = [r for r in pair_results if r['preference_strength'] >= strong_threshold]
        weak_pairs = [r for r in pair_results if r['preference_strength'] <= weak_threshold]

        strong_accuracy = (
            sum(r['correct'] for r in strong_pairs) / len(strong_pairs)
            if strong_pairs else 0.0
        )
        weak_accuracy = (
            sum(r['correct'] for r in weak_pairs) / len(weak_pairs)
            if weak_pairs else 0.0
        )

        # Cache statistics
        cache_stats = self.method.get_cache_stats()

        # Bradley-Terry metrics (from method predictions)
        bt_metrics = self._compute_bradley_terry_metrics(pair_results)
        
        # If BT model provided (fitted from ground truth), compute log-likelihood under ground truth model
        if self.bt_model is not None:
            try:
                # Convert pair_results format for BT
                bt_predictions = []
                for result in pair_results:
                    # Extract image IDs from pair_id (format: series_id_compareID1_compareID2)
                    # or use series_id directly if available
                    if 'series_id' in result:
                        series_id = result['series_id']
                        # Try to get compareID from pair_id or other fields
                        if 'pair_id' in result:
                            pair_id_parts = result['pair_id'].split('_')
                            if len(pair_id_parts) >= 3:
                                img_a_id = f"{series_id}_{pair_id_parts[1]}"
                                img_b_id = f"{series_id}_{pair_id_parts[2]}"
                            else:
                                # Fallback: use indices
                                img_a_id = f"{series_id}_0"
                                img_b_id = f"{series_id}_1"
                        else:
                            # Use default IDs
                            img_a_id = f"{series_id}_0"
                            img_b_id = f"{series_id}_1"
                    else:
                        # Try to extract from pair_id alone
                        if 'pair_id' in result:
                            pair_id_parts = result['pair_id'].split('_')
                            if len(pair_id_parts) >= 3:
                                series_id = pair_id_parts[0]
                                img_a_id = f"{series_id}_{pair_id_parts[1]}"
                                img_b_id = f"{series_id}_{pair_id_parts[2]}"
                            else:
                                continue
                        else:
                            continue
                    
                    bt_predictions.append({
                        'image_a_id': img_a_id,
                        'image_b_id': img_b_id,
                        'predicted': result['predicted']
                    })
                
                if bt_predictions:
                    bt_log_likelihood = compute_bt_log_likelihood_for_predictions(
                        self.bt_model,
                        bt_predictions
                    )
                    global_metrics['bt_ground_truth_log_likelihood'] = bt_log_likelihood
                    global_metrics['bt_ground_truth_avg_log_likelihood'] = bt_log_likelihood / len(bt_predictions)
                else:
                    global_metrics['bt_ground_truth_log_likelihood'] = 0.0
                    global_metrics['bt_ground_truth_avg_log_likelihood'] = 0.0
            except Exception as e:
                self.logger.warning(f"Could not compute BT log-likelihood from ground truth model: {e}")
                global_metrics['bt_ground_truth_log_likelihood'] = 0.0
                global_metrics['bt_ground_truth_avg_log_likelihood'] = 0.0

        results = {
            'global': {
                'accuracy': global_accuracy,
                'num_pairs': num_pairs,
                'num_correct': correct_global,
                'strong_preference_accuracy': strong_accuracy,
                'weak_preference_accuracy': weak_accuracy,
                'num_strong_pairs': len(strong_pairs),
                'num_weak_pairs': len(weak_pairs),
                # Bradley-Terry metrics
                'bt_log_likelihood': bt_metrics['log_likelihood'],
                'bt_avg_log_likelihood': bt_metrics['avg_log_likelihood'],
                'bt_calibration_error': bt_metrics['calibration_error']
            },
            'per_attribute': per_attribute_metrics if use_attributes else {},
            'per_pair_results': pair_results,
            'method_info': {
                'name': self.method.name,
                'config': self.method.get_config()
            },
            'cache_stats': cache_stats,
            'bradley_terry': bt_metrics
        }

        self.results = results

        return results

    def print_results(self):
        """Print evaluation results."""
        if self.results is None:
            print("No results available. Run evaluate() first.")
            return

        global_metrics = self.results['global']

        print(f"\n{'='*70}")
        print(f"Pairwise Quality Assessment Evaluation")
        print(f"{'='*70}")
        print(f"Method: {self.results['method_info']['name']}")
        print(f"Dataset: {self.pairs_file.stem}")
        print(f"Number of pairs: {global_metrics['num_pairs']}")

        print(f"\nGlobal Performance:")
        print(f"  Pairwise Accuracy: {global_metrics['accuracy']:.4f} ({global_metrics['accuracy']*100:.2f}%)")
        print(f"  Correct predictions: {global_metrics['num_correct']} / {global_metrics['num_pairs']}")

        print(f"\nBy Preference Strength:")
        print(f"  Strong preferences (>=0.75): {global_metrics['strong_preference_accuracy']:.4f} "
              f"({global_metrics['num_strong_pairs']} pairs)")
        print(f"  Weak preferences (<=0.55): {global_metrics['weak_preference_accuracy']:.4f} "
              f"({global_metrics['num_weak_pairs']} pairs)")

        # Print Bradley-Terry metrics
        print(f"\nBradley-Terry Metrics:")
        print(f"  Log-Likelihood: {global_metrics['bt_log_likelihood']:.2f}")
        print(f"  Avg Log-Likelihood: {global_metrics['bt_avg_log_likelihood']:.4f}")
        print(f"  Calibration Error: {global_metrics['bt_calibration_error']:.4f}")

        # Per-attribute results
        if self.results['per_attribute']:
            print(f"\nPer-Attribute Performance:")
            print(f"{'-'*70}")
            print(f"{'Attribute':<30} {'Accuracy':<12} {'Pairs':<10}")
            print(f"{'-'*70}")

            # Sort by accuracy
            sorted_attrs = sorted(
                self.results['per_attribute'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )

            for attr_name, metrics in sorted_attrs:
                print(f"{attr_name:<30} "
                      f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)  "
                      f"{metrics['num_pairs']}")

        print(f"{'='*70}\n")

    def save_results(self, output_path: Path):
        """Save results to JSON file."""
        if self.results is None:
            print("No results to save.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def save_results_csv(self, output_path: Path):
        """Save per-pair results as CSV."""
        import pandas as pd

        if self.results is None:
            print("No results to save.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(self.results['per_pair_results'])

        # Flatten attribute predictions if present
        if 'attribute_predictions' in df.columns and self.results['per_attribute']:
            # This will be complex, so just convert to string for CSV
            df['attribute_predictions'] = df['attribute_predictions'].apply(
                lambda x: json.dumps(x) if x else '{}'
            )

        df.to_csv(output_path, index=False)

        self.logger.info(f"CSV results saved to {output_path}")


def compare_methods_pairwise(
    pairs_file: Path,
    methods: List[Tuple[str, QualityAssessor]],
    use_attributes: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare multiple methods on pairwise task.

    Args:
        pairs_file: Path to pairs JSONL
        methods: List of (name, method) tuples
        use_attributes: Whether to evaluate per-attribute
        verbose: Show progress

    Returns:
        Comparison results
    """
    all_results = {}

    for name, method in methods:
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")

        evaluator = PairwiseEvaluator(pairs_file, method)
        results = evaluator.evaluate(verbose=verbose, use_attributes=use_attributes)
        evaluator.print_results()

        all_results[name] = results

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("Method Comparison - Pairwise Classification")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Pairwise Accuracy':<20} {'Strong Pref':<15} {'Weak Pref':<15}")
    print(f"{'-'*80}")

    for name, results in all_results.items():
        global_metrics = results['global']
        print(f"{name:<30} "
              f"{global_metrics['accuracy']:.4f} ({global_metrics['accuracy']*100:.2f}%)  "
              f"{global_metrics['strong_preference_accuracy']:.4f}        "
              f"{global_metrics['weak_preference_accuracy']:.4f}")

    print(f"{'='*80}\n")

    return all_results
