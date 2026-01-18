"""
Pairwise quality assessment benchmark.

Evaluates multiple quality assessment methods on pairwise comparison tasks
using the PhotoTriage dataset.
"""

import json
import yaml
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.pairwise_evaluator import PairwiseEvaluator, compare_methods_pairwise
from sim_bench.quality_assessment.registry import create_quality_assessor
from sim_bench.quality_assessment.bradley_terry import (
    BradleyTerryModel,
    fit_bt_from_pairs_data
)

logger = logging.getLogger(__name__)


class PairwiseBenchmark:
    """
    Benchmark framework for pairwise quality assessment.

    Evaluates quality methods on pairwise classification:
    Given two images, predict which was preferred by users.
    """

    def __init__(
        self,
        pairs_file: Path,
        methods_config: List[Dict],
        output_dir: Path,
        use_attributes: bool = False,
        verbose: bool = True,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
        use_bradley_terry: bool = False
    ):
        """
        Initialize pairwise benchmark.

        Args:
            pairs_file: Path to pairs JSONL file
            methods_config: List of method configurations
            output_dir: Directory to save results
            use_attributes: Whether to evaluate per-attribute
            verbose: Show progress
            sample_size: If provided, sample this many pairs for faster testing
            random_seed: Random seed for sampling
        """
        self.pairs_file = Path(pairs_file)
        self.methods_config = methods_config
        self.output_dir = Path(output_dir)
        self.use_attributes = use_attributes
        self.verbose = verbose
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.use_bradley_terry = use_bradley_terry

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # BT model (will be fitted from ground truth if enabled)
        self.bt_model: Optional[BradleyTerryModel] = None

        # Setup file logging
        self.log_file = self.output_dir / 'benchmark.log'
        self._setup_file_logging()

        # Results storage
        self.results = {}
        
        # Cache pairs for BT fitting
        self._pairs_cache: Optional[List[Dict]] = None

    def _setup_file_logging(self):
        """Setup file logging for benchmark."""
        # Get root logger
        root_logger = logging.getLogger()

        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add to root logger
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to file: {self.log_file}")

    def run(self) -> Dict[str, Any]:
        """
        Run pairwise benchmark for all configured methods.

        Returns:
            Dictionary with all results
        """
        logger.info("="*80)
        logger.info("Starting Pairwise Quality Assessment Benchmark")
        logger.info("="*80)
        logger.info(f"Pairs file: {self.pairs_file}")
        logger.info(f"Number of methods: {len(self.methods_config)}")
        method_names = [m.get('name', 'unknown') for m in self.methods_config]
        logger.info(f"Methods to evaluate: {method_names}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use attributes: {self.use_attributes}")
        logger.info(f"Use Bradley-Terry: {self.use_bradley_terry}")

        start_time = time.time()
        
        # Fit Bradley-Terry model from ground truth if enabled
        if self.use_bradley_terry:
            logger.info("\n" + "="*80)
            logger.info("Fitting Bradley-Terry Model from Ground Truth")
            logger.info("="*80)
            try:
                # Load pairs for BT fitting
                pairs_for_bt = self._load_pairs_for_bt()
                if pairs_for_bt:
                    self.bt_model, bt_params = fit_bt_from_pairs_data(
                        pairs_for_bt,
                        image_id_key_a='image_a_id',
                        image_id_key_b='image_b_id',
                        winner_key='chosen_image',
                        regularization=0.01
                    )
                    logger.info(f"Fitted BT model for {len(bt_params)} images")
                    logger.info(f"BT model will be used to evaluate predictions")
                else:
                    logger.warning("No pairs available for BT fitting, skipping")
            except Exception as e:
                logger.error(f"Failed to fit BT model: {e}", exc_info=True)
                logger.warning("Continuing without BT model")
                self.bt_model = None

        # Evaluate each method
        for method_config in self.methods_config:
            method_name = method_config.get('name', 'unknown')

            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating: {method_name}")
            logger.info(f"{'='*80}")

            try:
                # Create method
                method = create_quality_assessor(method_config)

                # Create evaluator (pass BT model if available)
                evaluator = PairwiseEvaluator(
                    pairs_file=self.pairs_file,
                    method=method,
                    sample_size=self.sample_size,
                    random_seed=self.random_seed,
                    bt_model=self.bt_model
                )

                # Run evaluation
                method_start = time.time()
                results = evaluator.evaluate(
                    verbose=self.verbose,
                    use_attributes=self.use_attributes
                )
                method_time = time.time() - method_start

                # Add timing info
                results['runtime_seconds'] = method_time

                # Store results
                self.results[method_name] = results

                # Print results
                evaluator.print_results()

                # Save individual method results
                self._save_method_results(method_name, results)

            except Exception as e:
                logger.error(f"Error evaluating {method_name}: {e}", exc_info=True)

                # Save error log for this method
                error_log_path = self.output_dir / f'{method_name.replace("/", "_")}_error.log'
                try:
                    with open(error_log_path, 'w', encoding='utf-8') as f:
                        f.write(f"Error evaluating {method_name}\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(f"Exception: {e}\n\n")
                        f.write("Full traceback:\n")
                        f.write(traceback.format_exc())
                    logger.info(f"Error details saved to: {error_log_path}")
                except Exception as log_err:
                    logger.error(f"Failed to save error log: {log_err}")

                continue

        total_time = time.time() - start_time

        logger.info(f"\n{'='*80}")
        logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        logger.info(f"{'='*80}")

        # Generate comparison reports
        self._generate_comparison_reports()

        return self.results

    def _load_pairs_for_bt(self) -> List[Dict]:
        """
        Load pairs data for Bradley-Terry fitting.
        
        Returns:
            List of pair dictionaries with image IDs and winners
        """
        if self._pairs_cache is not None:
            return self._pairs_cache
        
        pairs = []
        
        try:
            # Load pairs from JSONL file
            with open(self.pairs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    pair = json.loads(line)
                    
                    # Extract image IDs
                    # Try different possible formats
                    image_a_id = None
                    image_b_id = None
                    
                    # Format 1: Direct image_a_id/image_b_id
                    if 'image_a_id' in pair and 'image_b_id' in pair:
                        image_a_id = pair['image_a_id']
                        image_b_id = pair['image_b_id']
                    # Format 2: From pair_id (series_id_compareID1_compareID2)
                    elif 'pair_id' in pair:
                        pair_id_parts = pair['pair_id'].split('_')
                        if len(pair_id_parts) >= 3:
                            series_id = pair_id_parts[0]
                            image_a_id = f"{series_id}_{pair_id_parts[1]}"
                            image_b_id = f"{series_id}_{pair_id_parts[2]}"
                    # Format 3: From series_id and compareID fields
                    elif 'series_id' in pair and 'compareID1' in pair and 'compareID2' in pair:
                        series_id = pair['series_id']
                        image_a_id = f"{series_id}_{pair['compareID1']}"
                        image_b_id = f"{series_id}_{pair['compareID2']}"
                    
                    if image_a_id and image_b_id:
                        pair_for_bt = {
                            'image_a_id': str(image_a_id),
                            'image_b_id': str(image_b_id),
                            'chosen_image': pair.get('chosen_image', pair.get('MaxVote', ''))
                        }
                        pairs.append(pair_for_bt)
            
            self._pairs_cache = pairs
            logger.info(f"Loaded {len(pairs)} pairs for BT fitting")
            
        except Exception as e:
            logger.error(f"Error loading pairs for BT: {e}", exc_info=True)
            return []
        
        return pairs

    def _save_method_results(self, method_name: str, results: Dict[str, Any]):
        """Save results for individual method."""
        # Create method subdirectory
        method_dir = self.output_dir / method_name.replace('/', '_')
        method_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = method_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save CSV with per-pair results
        csv_path = method_dir / 'per_pair_results.csv'
        df = pd.DataFrame(results['per_pair_results'])

        # Flatten attribute predictions if present
        if 'attribute_predictions' in df.columns and results.get('per_attribute'):
            df['attribute_predictions'] = df['attribute_predictions'].apply(
                lambda x: json.dumps(x) if x else '{}'
            )

        df.to_csv(csv_path, index=False)

        logger.info(f"Results saved to {method_dir}")

    def _generate_comparison_reports(self):
        """Generate comparison reports across all methods."""
        if not self.results:
            logger.warning("No results to compare")
            return

        logger.info("\n" + "="*80)
        logger.info("Generating Comparison Reports")
        logger.info("="*80)

        # 1. Overall comparison table
        self._save_overall_comparison()

        # 2. Per-attribute comparison (if applicable)
        if self.use_attributes:
            self._save_attribute_comparison()

        # 3. Preference strength analysis
        self._save_preference_strength_analysis()

        # 4. Summary statistics
        self._save_summary_statistics()

        # 5. Unified format (compatible with quality benchmark analysis)
        self._save_unified_format()

    def _save_overall_comparison(self):
        """Save overall pairwise accuracy comparison."""
        data = []

        for method_name, results in self.results.items():
            global_metrics = results['global']

            data.append({
                'Method': method_name,
                'Pairwise Accuracy': global_metrics['accuracy'],
                'BT Log-Likelihood': global_metrics.get('bt_log_likelihood', 0.0),
                'BT Avg LL': global_metrics.get('bt_avg_log_likelihood', 0.0),
                'BT Calibration Error': global_metrics.get('bt_calibration_error', 0.0),
                'Num Pairs': global_metrics['num_pairs'],
                'Num Correct': global_metrics['num_correct'],
                'Runtime (s)': results.get('runtime_seconds', 0)
            })

        df = pd.DataFrame(data)
        df = df.sort_values('BT Log-Likelihood', ascending=False)  # Sort by BT metric

        # Save CSV
        csv_path = self.output_dir / 'overall_comparison.csv'
        df.to_csv(csv_path, index=False)

        # Print table
        print("\n" + "="*80)
        print("Overall Pairwise Accuracy Comparison")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        logger.info(f"Overall comparison saved to {csv_path}")

    def _save_attribute_comparison(self):
        """Save per-attribute accuracy comparison."""
        # Collect all attributes
        all_attributes = set()
        for results in self.results.values():
            all_attributes.update(results.get('per_attribute', {}).keys())

        if not all_attributes:
            return

        # Build comparison table
        data = []

        for attr in sorted(all_attributes):
            row = {'Attribute': attr}

            for method_name, results in self.results.items():
                attr_metrics = results.get('per_attribute', {}).get(attr, {})
                accuracy = attr_metrics.get('accuracy', 0.0)
                row[method_name] = accuracy

            data.append(row)

        df = pd.DataFrame(data)

        # Save CSV
        csv_path = self.output_dir / 'attribute_comparison.csv'
        df.to_csv(csv_path, index=False)

        # Print table
        print("\n" + "="*80)
        print("Per-Attribute Accuracy Comparison")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        logger.info(f"Attribute comparison saved to {csv_path}")

    def _save_preference_strength_analysis(self):
        """Save accuracy by preference strength."""
        data = []

        for method_name, results in self.results.items():
            global_metrics = results['global']

            data.append({
                'Method': method_name,
                'Overall Accuracy': global_metrics['accuracy'],
                'Strong Preference Accuracy': global_metrics['strong_preference_accuracy'],
                'Weak Preference Accuracy': global_metrics['weak_preference_accuracy'],
                'Num Strong Pairs': global_metrics['num_strong_pairs'],
                'Num Weak Pairs': global_metrics['num_weak_pairs']
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Overall Accuracy', ascending=False)

        # Save CSV
        csv_path = self.output_dir / 'preference_strength_analysis.csv'
        df.to_csv(csv_path, index=False)

        # Print table
        print("\n" + "="*80)
        print("Accuracy by Preference Strength")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        logger.info(f"Preference strength analysis saved to {csv_path}")

    def _save_summary_statistics(self):
        """Save summary statistics."""
        summary = {
            'benchmark_info': {
                'pairs_file': str(self.pairs_file),
                'num_methods': len(self.results),
                'use_attributes': self.use_attributes,
                'timestamp': datetime.now().isoformat()
            },
            'methods': {}
        }

        for method_name, results in self.results.items():
            method_config = results['method_info']['config']
            summary['methods'][method_name] = {
                'global_accuracy': results['global']['accuracy'],
                'num_pairs': results['global']['num_pairs'],
                'runtime_seconds': results.get('runtime_seconds', 0),
                'config': method_config
            }
            
            # Add prompt information for CLIP methods
            if 'prompt_texts' in method_config:
                summary['methods'][method_name]['prompts'] = method_config['prompt_texts']

        # Save JSON
        json_path = self.output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary statistics saved to {json_path}")

    def _save_unified_format(self):
        """
        Save results in unified format compatible with quality benchmark analysis.
        Generates methods_summary.csv and detailed_results.csv.
        """
        # Extract dataset name from pairs file (e.g., "phototriage" from "pairs_train.jsonl")
        dataset_name = Path(self.pairs_file).stem.replace('pairs_', '').replace('_train', '').replace('_test', '')
        if not dataset_name:
            dataset_name = "pairwise"

        # Build methods_summary.csv (overall method performance)
        methods_summary_rows = []
        for method_name, results in self.results.items():
            global_metrics = results['global']
            runtime_seconds = results.get('runtime_seconds', 0)
            
            methods_summary_rows.append({
                'method': method_name,
                'avg_top1_accuracy': global_metrics['accuracy'],  # Map pairwise accuracy to top1
                'avg_top2_accuracy': 0.0,  # Not applicable for pairwise
                'avg_mrr': 0.0,  # Not applicable for pairwise
                'avg_time_ms': runtime_seconds * 1000,  # Convert to milliseconds
                'datasets_tested': 1
            })

        if methods_summary_rows:
            methods_summary_df = pd.DataFrame(methods_summary_rows)
            methods_summary_df = methods_summary_df.sort_values('avg_top1_accuracy', ascending=False)
            methods_summary_path = self.output_dir / 'methods_summary.csv'
            methods_summary_df.to_csv(methods_summary_path, index=False)
            logger.info(f"Methods summary saved to {methods_summary_path}")

        # Build detailed_results.csv (per-dataset, per-method results)
        detailed_results_rows = []
        for method_name, results in self.results.items():
            global_metrics = results['global']
            runtime_seconds = results.get('runtime_seconds', 0)
            num_pairs = global_metrics.get('num_pairs', 0)
            
            # Calculate throughput (pairs per second)
            throughput = num_pairs / runtime_seconds if runtime_seconds > 0 else 0
            
            detailed_results_rows.append({
                'dataset': dataset_name,
                'method': method_name,
                'top1_accuracy': global_metrics['accuracy'],
                'top2_accuracy': 0.0,  # Not applicable for pairwise
                'mrr': 0.0,  # Not applicable for pairwise
                'avg_time_ms': runtime_seconds * 1000,
                'throughput': throughput
            })

        if detailed_results_rows:
            detailed_results_df = pd.DataFrame(detailed_results_rows)
            detailed_results_path = self.output_dir / 'detailed_results.csv'
            detailed_results_df.to_csv(detailed_results_path, index=False)
            logger.info(f"Detailed results saved to {detailed_results_path}")


def run_pairwise_benchmark_from_config(
    config_path: Path,
    method_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run pairwise benchmark from YAML configuration.

    Args:
        config_path: Path to benchmark configuration YAML
        method_filter: Optional list of method names to run (filters methods from config)

    Returns:
        Benchmark results

    Example config:
        dataset:
          pairs_file: "data/phototriage/pairs_train.jsonl"
          use_attributes: true

        methods:
          - name: "Sharpness"
            type: "rule_based"
            method: "sharpness"

          - name: "CLIP-Aesthetic"
            type: "clip_aesthetic"
            variant: "laion"

        output:
          base_dir: "results/pairwise_benchmark"
          create_timestamp_dir: true

        execution:
          verbose: true
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration
    dataset_config = config['dataset']
    methods_config = config['methods']
    output_config = config['output']
    execution_config = config.get('execution', {})
    
    # Filter methods if method_filter is provided
    if method_filter:
        logger.info(f"Applying method filter: {method_filter}")
        original_count = len(methods_config)
        filtered_methods = []
        method_names_in_config = {m.get('name') for m in methods_config if 'name' in m}
        
        for method_name in method_filter:
            if method_name not in method_names_in_config:
                logger.warning(f"Method '{method_name}' not found in config. Available methods: {sorted(method_names_in_config)}")
            else:
                # Find matching method config
                for method_config in methods_config:
                    if method_config.get('name') == method_name:
                        filtered_methods.append(method_config)
                        break
        
        if not filtered_methods:
            raise ValueError(
                f"No valid methods found in filter: {method_filter}. "
                f"Available methods: {sorted(method_names_in_config)}"
            )
        
        methods_config = filtered_methods
        logger.info(f"Filtered from {original_count} to {len(methods_config)} method(s): {[m.get('name') for m in methods_config]}")
    else:
        logger.info(f"No method filter provided, running all {len(methods_config)} methods from config")

    # Determine output directory
    base_output_dir = Path(output_config['base_dir'])

    if output_config.get('create_timestamp_dir', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_output_dir / f"pairwise_{timestamp}"
    else:
        output_dir = base_output_dir

    # Extract sampling config if present
    sampling_config = dataset_config.get('sampling', {})
    sample_size = None
    random_seed = 42
    if sampling_config.get('enabled', False):
        sample_size = sampling_config.get('num_pairs')
        random_seed = sampling_config.get('random_seed', 42)

    # Create benchmark
    benchmark = PairwiseBenchmark(
        pairs_file=Path(dataset_config['pairs_file']),
        methods_config=methods_config,
        output_dir=output_dir,
        use_attributes=dataset_config.get('use_attributes', False),
        verbose=execution_config.get('verbose', True),
        sample_size=sample_size,
        random_seed=random_seed,
        use_bradley_terry=execution_config.get('use_bradley_terry', False)
    )

    # Save config to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path = output_dir / 'config.yaml'
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run benchmark
    results = benchmark.run()

    return results


def compare_pairwise_vs_series_selection(
    pairwise_results_dir: Path,
    series_results_dir: Path,
    output_path: Path
):
    """
    Compare pairwise classification vs series selection results.

    Args:
        pairwise_results_dir: Directory with pairwise benchmark results
        series_results_dir: Directory with series selection benchmark results
        output_path: Path to save comparison
    """
    # Load pairwise results
    pairwise_summary_path = pairwise_results_dir / 'summary.json'
    with open(pairwise_summary_path, 'r') as f:
        pairwise_summary = json.load(f)

    # Load series selection results
    series_summary_path = series_results_dir / 'summary.json'
    with open(series_summary_path, 'r') as f:
        series_summary = json.load(f)

    # Build comparison table
    data = []

    for method_name in pairwise_summary['methods'].keys():
        row = {'Method': method_name}

        # Pairwise accuracy
        if method_name in pairwise_summary['methods']:
            row['Pairwise Accuracy'] = pairwise_summary['methods'][method_name]['global_accuracy']
        else:
            row['Pairwise Accuracy'] = None

        # Series selection accuracy
        if method_name in series_summary.get('methods', {}):
            row['Series Top-1 Accuracy'] = series_summary['methods'][method_name].get('top1_accuracy', None)
        else:
            row['Series Top-1 Accuracy'] = None

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("Pairwise vs Series Selection Comparison")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    logger.info(f"Comparison saved to {output_path}")
