"""
Evaluator for image quality assessment methods on series selection tasks.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.datasets import load_dataset

# Set up logger
logger = logging.getLogger(__name__)


class QualityEvaluator:
    """Evaluate quality assessment methods on image series selection."""
    
    def __init__(self, dataset, method: QualityAssessor, logger: Optional[logging.Logger] = None):
        """
        Initialize evaluator.
        
        Args:
            dataset: Dataset object with series information
            method: Quality assessment method to evaluate
            logger: Optional logger instance (uses module logger if None)
        """
        self.dataset = dataset
        self.method = method
        self.results = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate method on dataset.
        
        Args:
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with evaluation results
        """
        # Get dataset info
        images = self.dataset.get_images()
        evaluation_data = self.dataset.get_evaluation_data()
        groups = evaluation_data.get('groups', [])
        
        if not groups:
            raise ValueError("Dataset must have group/series information")
        
        # Group images by series
        series_dict = defaultdict(list)
        for idx, (img_path, group_id) in enumerate(zip(images, groups)):
            series_dict[group_id].append({
                'index': idx,
                'path': img_path
            })
        
        # Remove single-image series
        series_dict = {k: v for k, v in series_dict.items() if len(v) > 1}
        
        num_series = len(series_dict)
        self.logger.info(f"Evaluating {self.method.name} on {num_series} series...")
        if verbose:
            print(f"Evaluating on {num_series} series...")
        
        # Evaluate each series
        correct_top1 = 0
        correct_top2 = 0
        reciprocal_ranks = []
        kendall_taus = []
        series_results = []
        
        iterator = tqdm(series_dict.items()) if verbose else series_dict.items()
        
        for group_id, series_images in iterator:
            # Get paths and ground truth
            paths = [img['path'] for img in series_images]
            indices = [img['index'] for img in series_images]
            
            # For PhotoTriage: first image in series is typically best
            # For other datasets: may need different ground truth
            ground_truth_best_idx = 0  # Assumes first is best
            
            # Assess quality
            try:
                scores = self.method.assess_batch(paths)
                predicted_best_idx = int(np.argmax(scores))
                
                # Compute full ranking (best to worst)
                ranked_indices = np.argsort(scores)[::-1].tolist()  # Descending order
                # ranked_indices[i] = index of image ranked at position i (0 = best)
                
                # Compute metrics
                is_top1_correct = (predicted_best_idx == ground_truth_best_idx)
                correct_top1 += int(is_top1_correct)
                
                # Top-2 accuracy
                top2_indices = ranked_indices[:2]
                is_top2_correct = ground_truth_best_idx in top2_indices
                correct_top2 += int(is_top2_correct)
                
                # Mean Reciprocal Rank
                rank_of_best = ranked_indices.index(ground_truth_best_idx) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank_of_best)
                
                # Store per-image ranking information
                # ranked_by_position[i] = index of image at rank i+1 (1-indexed)
                # image_rank[i] = rank (1-indexed) of image at position i
                image_ranks = [ranked_indices.index(i) + 1 for i in range(len(paths))]
                
                series_results.append({
                    'group_id': group_id,
                    'num_images': len(paths),
                    'predicted_idx': predicted_best_idx,
                    'ground_truth_idx': ground_truth_best_idx,
                    'correct': is_top1_correct,
                    'scores': scores.tolist(),
                    'ranking': ranked_indices,  # Full ranking: [best_idx, 2nd_best_idx, ..., worst_idx]
                    'image_ranks': image_ranks,  # Rank for each image: [rank_of_img0, rank_of_img1, ...]
                    'image_paths': paths  # Store paths for reference
                })
                
            except Exception as e:
                error_msg = f"Error processing series {group_id}: {e}"
                self.logger.error(error_msg, exc_info=True)
                if verbose:
                    print(error_msg)
                continue
        
        # Compute overall metrics
        num_series = len(series_results)
        
        # Log cache statistics
        cache_stats = self.method.get_cache_stats()
        self.logger.info(f"Cache statistics: {cache_stats}")
        
        metrics = {
            'top1_accuracy': correct_top1 / num_series if num_series > 0 else 0.0,
            'top2_accuracy': correct_top2 / num_series if num_series > 0 else 0.0,
            'mean_reciprocal_rank': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            'num_series': num_series,
            'method': self.method.name,
            'method_config': self.method.get_config(),
            'cache_stats': cache_stats
        }
        
        # Accuracy by series size
        size_to_correct = defaultdict(list)
        for result in series_results:
            size_to_correct[result['num_images']].append(result['correct'])
        
        accuracy_by_size = {
            size: np.mean(correct_list)
            for size, correct_list in size_to_correct.items()
        }
        metrics['accuracy_by_series_size'] = accuracy_by_size
        
        self.results = {
            'metrics': metrics,
            'per_series_results': series_results
        }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results."""
        if self.results is None:
            print("No results available. Run evaluate() first.")
            return
        
        metrics = self.results['metrics']
        
        print(f"\n{'='*60}")
        print(f"Quality Assessment Evaluation Results")
        print(f"{'='*60}")
        print(f"Method: {metrics['method']}")
        print(f"Dataset: {self.dataset.name}")
        print(f"Number of series: {metrics['num_series']}")
        print(f"\nPerformance:")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
        print(f"  Top-2 Accuracy: {metrics['top2_accuracy']:.4f} ({metrics['top2_accuracy']*100:.2f}%)")
        print(f"  Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.4f}")
        
        print(f"\nAccuracy by Series Size:")
        for size, acc in sorted(metrics['accuracy_by_series_size'].items()):
            print(f"  {size} images: {acc:.4f} ({acc*100:.2f}%)")
        
        print(f"{'='*60}\n")
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        import json
        
        if self.results is None:
            print("No results to save.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    @staticmethod
    def compare_methods(
        dataset,
        methods: List[Tuple[str, QualityAssessor]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple quality assessment methods.
        
        Args:
            dataset: Dataset to evaluate on
            methods: List of (name, method) tuples
            verbose: Whether to show progress
            
        Returns:
            Comparison results
        """
        all_results = {}
        
        for name, method in methods:
            print(f"\nEvaluating {name}...")
            evaluator = QualityEvaluator(dataset, method)
            results = evaluator.evaluate(verbose=verbose)
            evaluator.print_results()
            all_results[name] = results
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("Method Comparison")
        print(f"{'='*80}")
        print(f"{'Method':<25} {'Top-1 Acc':<12} {'Top-2 Acc':<12} {'MRR':<12}")
        print(f"{'-'*80}")
        
        for name, results in all_results.items():
            metrics = results['metrics']
            print(f"{name:<25} {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)  "
                  f"{metrics['top2_accuracy']:.4f} ({metrics['top2_accuracy']*100:.2f}%)  "
                  f"{metrics['mean_reciprocal_rank']:.4f}")
        
        print(f"{'='*80}\n")
        
        return all_results


def evaluate_on_phototriage(
    method: QualityAssessor,
    dataset_config_path: str = "configs/dataset.phototriage.yaml",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate on PhotoTriage dataset.
    
    Args:
        method: Quality assessment method
        dataset_config_path: Path to PhotoTriage config
        verbose: Whether to show progress
        
    Returns:
        Evaluation results
    """
    import yaml
    
    # Load dataset
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)
    
    dataset = load_dataset('phototriage', dataset_config)
    dataset.load_data()
    
    # Evaluate
    evaluator = QualityEvaluator(dataset, method)
    results = evaluator.evaluate(verbose=verbose)
    evaluator.print_results()
    
    return results


