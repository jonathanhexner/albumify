"""
Benchmark framework for quality assessment methods.
Flexible system to evaluate multiple methods on multiple datasets.
"""

import yaml
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging

from sim_bench.datasets import load_dataset
from sim_bench.quality_assessment.evaluator import QualityEvaluator
from sim_bench.logging_config import setup_logger


class QualityBenchmark:
    """Benchmark framework for quality assessment methods."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "outputs/quality_benchmarks"):
        """
        Initialize benchmark.
        
        Args:
            config: Benchmark configuration dictionary
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.output_dir / f"benchmark_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        log_file = self.run_dir / "benchmark.log"
        self.logger = setup_logger(
            "sim_bench.quality_benchmark",
            log_file,
            level="INFO",
            console=False
        )
        self.logger.info(f"Quality benchmark initialized. Output directory: {self.run_dir}")
        
        # Save configuration
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.results = {}
        
    def run(self) -> Dict[str, Any]:
        """
        Run complete benchmark.
        
        Returns:
            Comprehensive benchmark results
        """
        self.logger.info("=" * 80)
        self.logger.info("QUALITY BENCHMARK START")
        self.logger.info("=" * 80)
        self.logger.info(f"Datasets: {[d['name'] for d in self.config['datasets']]}")
        self.logger.info(f"Methods: {[m['name'] for m in self.config['methods']]}")
        self.logger.info("=" * 80)
        print(f"\n{'='*80}")
        print("Quality Assessment Benchmark")
        print(f"{'='*80}")
        print(f"Output directory: {self.run_dir}")
        print(f"Datasets: {len(self.config['datasets'])}")
        print(f"Methods: {len(self.config['methods'])}")
        print(f"{'='*80}\n")
        
        # Load all datasets
        datasets = self._load_datasets()
        
        # Run benchmarks
        for dataset_name, dataset in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")
            
            dataset_results = self._benchmark_dataset(dataset_name, dataset)
            self.results[dataset_name] = dataset_results
            
            # Save intermediate results
            self._save_dataset_results(dataset_name, dataset_results)
        
        # Create comprehensive summary
        summary = self._create_summary()
        self._save_summary(summary)
        
        # Print final results
        self._print_summary(summary)
        
        self.logger.info("=" * 80)
        self.logger.info("QUALITY BENCHMARK COMPLETE")
        self.logger.info("=" * 80)
        
        return {
            'summary': summary,
            'detailed_results': self.results,
            'config': self.config,
            'run_dir': str(self.run_dir)
        }
    
    def _load_datasets(self) -> Dict[str, Any]:
        """Load all configured datasets."""
        datasets = {}
        
        for dataset_config in self.config['datasets']:
            name = dataset_config['name']
            config_path = dataset_config.get('config', f"configs/dataset.{name}.yaml")
            
            print(f"Loading dataset: {name}")
            
            # Load dataset config
            with open(config_path) as f:
                ds_config = yaml.safe_load(f)
            
            # Apply any overrides from benchmark config
            if 'sampling' in dataset_config:
                ds_config['sampling'] = dataset_config['sampling']
            
            # Load dataset
            dataset = load_dataset(name, ds_config)
            dataset.load_data()
            
            # Apply sampling if configured
            if 'sampling' in dataset_config:
                dataset.apply_sampling(dataset_config['sampling'])
            
            datasets[name] = dataset
            
            print(f"  Loaded {len(dataset.get_images())} images")
        
        return datasets
    
    def _benchmark_dataset(self, dataset_name: str, dataset) -> Dict[str, Any]:
        """Benchmark all methods on a single dataset."""
        results = {}
        
        for method_config in self.config['methods']:
            method_name = method_config['name']
            
            print(f"\n{'-'*80}")
            print(f"Method: {method_name}")
            print(f"{'-'*80}")
            
            try:
                method_results = self._benchmark_method(
                    method_name,
                    method_config,
                    dataset_name,
                    dataset
                )
                results[method_name] = method_results
                
                # Print quick summary
                metrics = method_results['metrics']
                print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
                print(f"  Avg Runtime: {method_results['timing']['avg_per_image_ms']:.2f}ms per image")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                
                results[method_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def _benchmark_method(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        dataset_name: str,
        dataset
    ) -> Dict[str, Any]:
        """Benchmark a single method on a dataset."""
        
        # Import here to avoid circular dependency
        from sim_bench.quality_assessment import load_quality_method
        
        # Create method
        config = method_config.get('config', {})
        method = load_quality_method(method_config['type'], config)
        
        # Time method creation
        creation_time = time.time()
        
        # Warm-up run (for GPU methods)
        if config.get('device') == 'cuda':
            print("  Warming up GPU...")
            test_img = dataset.get_images()[0]
            _ = method.assess_image(test_img)
        
        warmup_time = time.time() - creation_time
        
        # Run evaluation
        print("  Running evaluation...")
        self.logger.info(f"Starting evaluation: {method_name} on {dataset_name}")
        evaluator = QualityEvaluator(dataset, method, logger=self.logger)
        
        eval_start = time.time()
        eval_results = evaluator.evaluate(verbose=True)
        eval_time = time.time() - eval_start
        
        self.logger.info(f"Evaluation complete: {method_name} on {dataset_name} ({eval_time:.2f}s)")
        
        # Compute timing statistics
        num_series = eval_results['metrics']['num_series']
        num_images = len(dataset.get_images())
        
        timing = {
            'total_eval_time_s': eval_time,
            'avg_per_series_ms': (eval_time / num_series * 1000) if num_series > 0 else 0,
            'avg_per_image_ms': (eval_time / num_images * 1000) if num_images > 0 else 0,
            'warmup_time_s': warmup_time,
            'throughput_images_per_sec': num_images / eval_time if eval_time > 0 else 0
        }
        
        # Save detailed per-series results as CSV
        series_results_path = self.run_dir / f"{dataset_name}_{method_name}_series.csv"
        series_df = pd.DataFrame(eval_results['per_series_results'])
        
        # Convert list columns to strings for CSV
        list_columns = ['scores', 'ranking', 'image_ranks', 'image_paths']
        for col in list_columns:
            if col in series_df.columns:
                series_df[col] = series_df[col].apply(
                    lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x)
                )
        
        series_df.to_csv(series_results_path, index=False)
        
        return {
            'method_name': method_name,
            'method_type': method_config['type'],
            'method_config': method.get_config(),
            'metrics': eval_results['metrics'],
            'timing': timing,
            'status': 'success',
            'series_results_file': str(series_results_path.relative_to(self.run_dir))
        }
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create comprehensive summary of all results."""
        summary = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'num_datasets': len(self.config['datasets']),
                'num_methods': len(self.config['methods']),
                'run_dir': str(self.run_dir)
            },
            'datasets': {},
            'methods': {},
            'comparison': {}
        }
        
        # Per-dataset summaries
        for dataset_name, dataset_results in self.results.items():
            summary['datasets'][dataset_name] = {
                'num_methods': len(dataset_results),
                'methods': {}
            }
            
            for method_name, method_results in dataset_results.items():
                if method_results.get('status') == 'success':
                    summary['datasets'][dataset_name]['methods'][method_name] = {
                        'top1_accuracy': method_results['metrics']['top1_accuracy'],
                        'top2_accuracy': method_results['metrics']['top2_accuracy'],
                        'mrr': method_results['metrics']['mean_reciprocal_rank'],
                        'avg_time_ms': method_results['timing']['avg_per_image_ms'],
                        'throughput': method_results['timing']['throughput_images_per_sec']
                    }
        
        # Per-method summaries (average across datasets)
        method_stats = defaultdict(lambda: defaultdict(list))
        
        for dataset_name, dataset_results in self.results.items():
            for method_name, method_results in dataset_results.items():
                if method_results.get('status') == 'success':
                    method_stats[method_name]['top1_accuracy'].append(
                        method_results['metrics']['top1_accuracy']
                    )
                    method_stats[method_name]['top2_accuracy'].append(
                        method_results['metrics']['top2_accuracy']
                    )
                    method_stats[method_name]['mrr'].append(
                        method_results['metrics']['mean_reciprocal_rank']
                    )
                    method_stats[method_name]['avg_time_ms'].append(
                        method_results['timing']['avg_per_image_ms']
                    )
        
        for method_name, stats in method_stats.items():
            summary['methods'][method_name] = {
                'avg_top1_accuracy': np.mean(stats['top1_accuracy']),
                'avg_top2_accuracy': np.mean(stats['top2_accuracy']),
                'avg_mrr': np.mean(stats['mrr']),
                'avg_time_ms': np.mean(stats['avg_time_ms']),
                'datasets_tested': len(stats['top1_accuracy'])
            }
        
        # Comparison rankings
        if method_stats:
            # Rank by accuracy
            accuracy_ranking = sorted(
                summary['methods'].items(),
                key=lambda x: x[1]['avg_top1_accuracy'],
                reverse=True
            )
            summary['comparison']['accuracy_ranking'] = [
                {'method': name, 'accuracy': data['avg_top1_accuracy']}
                for name, data in accuracy_ranking
            ]
            
            # Rank by speed
            speed_ranking = sorted(
                summary['methods'].items(),
                key=lambda x: x[1]['avg_time_ms']
            )
            summary['comparison']['speed_ranking'] = [
                {'method': name, 'time_ms': data['avg_time_ms']}
                for name, data in speed_ranking
            ]
            
            # Efficiency score (accuracy / time)
            efficiency_ranking = sorted(
                summary['methods'].items(),
                key=lambda x: x[1]['avg_top1_accuracy'] / (x[1]['avg_time_ms'] + 1e-6),
                reverse=True
            )
            summary['comparison']['efficiency_ranking'] = [
                {
                    'method': name,
                    'efficiency': data['avg_top1_accuracy'] / (data['avg_time_ms'] + 1e-6),
                    'accuracy': data['avg_top1_accuracy'],
                    'time_ms': data['avg_time_ms']
                }
                for name, data in efficiency_ranking
            ]
        
        return summary
    
    def _save_dataset_results(self, dataset_name: str, results: Dict[str, Any]):
        """Save results for a single dataset as CSV."""
        # Save main metrics as CSV
        metrics_rows = []
        for method_name, method_results in results.items():
            if method_results.get('status') == 'success':
                row = {
                    'method': method_name,
                    'method_type': method_results.get('method_type', ''),
                    'top1_accuracy': method_results['metrics']['top1_accuracy'],
                    'top2_accuracy': method_results['metrics']['top2_accuracy'],
                    'top3_accuracy': method_results['metrics'].get('top3_accuracy', 0),
                    'mrr': method_results['metrics']['mean_reciprocal_rank'],
                    'mean_rank': method_results['metrics'].get('mean_rank', 0),
                    'num_series': method_results['metrics'].get('num_series', 0),
                    'avg_time_ms': method_results['timing']['avg_per_image_ms'],
                    'throughput': method_results['timing']['throughput_images_per_sec'],
                    'status': 'success'
                }
                metrics_rows.append(row)
            else:
                metrics_rows.append({
                    'method': method_name,
                    'status': 'failed',
                    'error': method_results.get('error', 'Unknown error')
                })
        
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df.to_csv(self.run_dir / f"{dataset_name}_results.csv", index=False)
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save summary results as CSV."""
        # CSV (primary format)
        self._save_summary_csv(summary)
        
        # Also save rankings as separate CSV files
        if summary.get('comparison'):
            # Accuracy ranking
            if summary['comparison'].get('accuracy_ranking'):
                acc_df = pd.DataFrame(summary['comparison']['accuracy_ranking'])
                acc_df.to_csv(self.run_dir / "accuracy_ranking.csv", index=False)
            
            # Speed ranking
            if summary['comparison'].get('speed_ranking'):
                speed_df = pd.DataFrame(summary['comparison']['speed_ranking'])
                speed_df.to_csv(self.run_dir / "speed_ranking.csv", index=False)
            
            # Efficiency ranking
            if summary['comparison'].get('efficiency_ranking'):
                eff_df = pd.DataFrame(summary['comparison']['efficiency_ranking'])
                eff_df.to_csv(self.run_dir / "efficiency_ranking.csv", index=False)
    
    def _save_summary_csv(self, summary: Dict[str, Any]):
        """Save summary as CSV tables."""
        # Overall method performance
        if summary['methods']:
            method_df = pd.DataFrame([
                {
                    'method': name,
                    'avg_top1_accuracy': data['avg_top1_accuracy'],
                    'avg_top2_accuracy': data['avg_top2_accuracy'],
                    'avg_mrr': data['avg_mrr'],
                    'avg_time_ms': data['avg_time_ms'],
                    'datasets_tested': data['datasets_tested']
                }
                for name, data in summary['methods'].items()
            ])
            method_df = method_df.sort_values('avg_top1_accuracy', ascending=False)
            method_df.to_csv(self.run_dir / "methods_summary.csv", index=False)
        
        # Per-dataset results
        rows = []
        for dataset_name, dataset_data in summary['datasets'].items():
            for method_name, method_data in dataset_data['methods'].items():
                rows.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'top1_accuracy': method_data['top1_accuracy'],
                    'top2_accuracy': method_data['top2_accuracy'],
                    'mrr': method_data['mrr'],
                    'avg_time_ms': method_data['avg_time_ms'],
                    'throughput': method_data['throughput']
                })
        
        if rows:
            detail_df = pd.DataFrame(rows)
            detail_df.to_csv(self.run_dir / "detailed_results.csv", index=False)
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary to console."""
        print(f"\n\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}\n")
        
        # Method rankings
        if summary['comparison'].get('accuracy_ranking'):
            print("Method Rankings by Accuracy:")
            print(f"{'-'*80}")
            for i, item in enumerate(summary['comparison']['accuracy_ranking'], 1):
                print(f"  {i}. {item['method']:<30} {item['accuracy']:.4f} ({item['accuracy']*100:.2f}%)")
            
            print(f"\nMethod Rankings by Speed:")
            print(f"{'-'*80}")
            for i, item in enumerate(summary['comparison']['speed_ranking'], 1):
                print(f"  {i}. {item['method']:<30} {item['time_ms']:.2f} ms/image")
            
            print(f"\nMethod Rankings by Efficiency (Accuracy/Time):")
            print(f"{'-'*80}")
            for i, item in enumerate(summary['comparison']['efficiency_ranking'], 1):
                print(f"  {i}. {item['method']:<30} "
                      f"Acc: {item['accuracy']*100:.2f}%, "
                      f"Time: {item['time_ms']:.2f}ms, "
                      f"Eff: {item['efficiency']:.4f}")
        
        # Per-dataset results
        print(f"\n{'='*80}")
        print("Results by Dataset:")
        print(f"{'='*80}\n")
        
        for dataset_name, dataset_data in summary['datasets'].items():
            print(f"\n{dataset_name}:")
            print(f"{'-'*80}")
            print(f"{'Method':<30} {'Top-1 Acc':<12} {'Top-2 Acc':<12} {'MRR':<10} {'Time (ms)':<12}")
            print(f"{'-'*80}")
            
            # Sort by accuracy
            methods_sorted = sorted(
                dataset_data['methods'].items(),
                key=lambda x: x[1]['top1_accuracy'],
                reverse=True
            )
            
            for method_name, method_data in methods_sorted:
                print(f"{method_name:<30} "
                      f"{method_data['top1_accuracy']:.4f}      "
                      f"{method_data['top2_accuracy']:.4f}      "
                      f"{method_data['mrr']:.4f}    "
                      f"{method_data['avg_time_ms']:.2f}")
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {self.run_dir}")
        print(f"{'='*80}\n")


def load_benchmark_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def run_benchmark_from_config(config_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Run benchmark from configuration file.
    
    Args:
        config_path: Path to benchmark YAML config
        output_dir: Output directory (optional)
        
    Returns:
        Benchmark results
    """
    config = load_benchmark_config(config_path)
    
    if output_dir:
        benchmark = QualityBenchmark(config, output_dir)
    else:
        benchmark = QualityBenchmark(config)
    
    return benchmark.run()


