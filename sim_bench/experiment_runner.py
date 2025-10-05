"""
Experiment runner for managing single and batch evaluations.
Handles the orchestration of methods, datasets, and result collection.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from sim_bench.datasets import load_dataset
from sim_bench.feature_extraction import load_method
from sim_bench import metrics_api as metrics
from sim_bench.result_manager import ResultManager


class ExperimentRunner:
    """Orchestrates experiment execution and result collection."""
    
    def __init__(self, run_config: Dict[str, Any], dataset_config: Dict[str, Any]):
        """
        Initialize experiment runner.
        
        Args:
            run_config: Run configuration dictionary
            dataset_config: Dataset configuration dictionary
        """
        self.run_config = run_config
        self.dataset_config = dataset_config
        self.result_manager = ResultManager(run_config)
        
        # Load dataset once for all experiments
        self.dataset = load_dataset(dataset_config['name'], dataset_config)
        self.dataset.load_data()
        self.dataset.apply_sampling(run_config.get('sampling', {}))
        
        print(f"Dataset: {dataset_config['name']}")
        print(f"Total images: {len(self.dataset.get_images())}")
        print(f"Query images: {len(self.dataset.get_queries())}")
    
    def run_single_method(self, method_name: str) -> Dict[str, Any]:
        """
        Run evaluation for a single method.
        
        Args:
            method_name: Name of the method to run
            
        Returns:
            Dictionary containing method results
        """
        print(f"Method: {method_name}")
        
        # Load method configuration
        method_config_path = Path(f"configs/methods/{method_name}.yaml")
        if not method_config_path.exists():
            raise FileNotFoundError(f"Method config not found: {method_config_path}")
        
        method_config = yaml.safe_load(method_config_path.read_text())
        
        # Load and initialize method
        method = load_method(method_name, method_config)
        
        # Extract features
        image_paths = self.dataset.get_images()
        feature_matrix = method.extract_features(image_paths)
        
        # Compute distance matrix
        distance_matrix = method.compute_distances(feature_matrix)
        
        # Get rankings (indices sorted by distance)
        ranking_indices = np.argsort(distance_matrix, axis=1)
        
        # Compute metrics
        evaluation_data = self.dataset.get_evaluation_data()
        computed_metrics = metrics.compute_metrics(
            ranking_indices, 
            evaluation_data, 
            self.run_config
        )
        
        # Save results
        self.result_manager.save_method_results(
            method_name=method_name,
            method_config=method_config,
            ranking_indices=ranking_indices,
            distance_matrix=distance_matrix,
            computed_metrics=computed_metrics,
            dataset=self.dataset
        )
        
        # Print primary metric
        self._print_primary_metric(computed_metrics)
        
        return {
            'method': method_name,
            'metrics': computed_metrics,
            'config': method_config
        }
    
    def run_multiple_methods(self, method_names: List[str]) -> List[Dict[str, Any]]:
        """
        Run evaluation for multiple methods.
        
        Args:
            method_names: List of method names to run
            
        Returns:
            List of result dictionaries for each method
        """
        all_results = []
        
        for method_name in method_names:
            print(f"\n=== Running method: {method_name} ===")
            
            try:
                method_result = self.run_single_method(method_name)
                all_results.append(method_result)
                
            except Exception as error:
                print(f"❌ ERROR running {method_name}: {error}")
                continue
        
        # Create summary if multiple methods were run
        if len(all_results) > 1:
            self.result_manager.create_summary(all_results)
        
        return all_results
    
    def _print_primary_metric(self, computed_metrics: Dict[str, float]) -> None:
        """Print the primary metric for the dataset."""
        if 'ns_score' in computed_metrics:
            print(f"N-S score: {computed_metrics['ns_score']:.3f}")
        elif 'map_full' in computed_metrics:
            print(f"mAP (full): {computed_metrics['map_full']:.3f}")
    
    def get_output_directory(self) -> Path:
        """Get the output directory for this experiment."""
        return self.result_manager.run_directory


class BenchmarkRunner:
    """Runs comprehensive benchmarks across multiple datasets and methods."""
    
    def __init__(self, benchmark_config: Dict[str, Any]):
        """
        Initialize benchmark runner.
        
        Args:
            benchmark_config: Benchmark configuration dictionary
        """
        self.benchmark_config = benchmark_config
        self.master_result_manager = ResultManager(benchmark_config)
    
    def run_comprehensive_benchmark(self) -> List[Dict[str, Any]]:
        """
        Run comprehensive benchmark across all configured datasets and methods.
        
        Returns:
            List of all results across datasets and methods
        """
        print("=" * 60)
        print("🚀 COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        
        all_results = []
        
        # Process each dataset
        for dataset_config in self.benchmark_config['datasets']:
            dataset_name = dataset_config['name']
            dataset_config_path = dataset_config['config']
            
            print(f"\n📊 Dataset: {dataset_name}")
            print("-" * 40)
            
            try:
                dataset_results = self._run_dataset_benchmark(
                    dataset_name, 
                    dataset_config_path, 
                    dataset_config
                )
                
                # Add dataset information to results
                for result in dataset_results:
                    result['dataset'] = dataset_name
                    all_results.append(result)
                    
            except Exception as error:
                print(f"❌ Error running {dataset_name}: {error}")
                continue
        
        # Create comprehensive summary
        if all_results:
            self._create_comprehensive_summary(all_results)
        
        return all_results
    
    def _run_dataset_benchmark(
        self, 
        dataset_name: str, 
        dataset_config_path: str, 
        dataset_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run benchmark for a single dataset."""
        
        # Load dataset configuration
        with open(dataset_config_path, 'r') as file:
            dataset_config_data = yaml.safe_load(file)
        
        # Create run configuration for this dataset
        run_config = self._create_dataset_run_config(dataset_name, dataset_config)
        
        # Create experiment runner
        experiment_runner = ExperimentRunner(run_config, dataset_config_data)
        
        # Run all methods
        method_names = self.benchmark_config['methods']
        return experiment_runner.run_multiple_methods(method_names)
    
    def _create_dataset_run_config(
        self, 
        dataset_name: str, 
        dataset_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create run configuration for a specific dataset."""
        
        run_config = {
            'dataset': dataset_name,
            'output_dir': str(self.master_result_manager.run_directory),
            'run_name': f"{dataset_name}_results",
            'logging': self.benchmark_config.get('logging', {}),
            'save': self.benchmark_config.get('save', {}),
            'methods': self.benchmark_config['methods'],
            'sampling': dataset_config.get('sampling', {}),
            'random_seed': self.benchmark_config.get('random_seed', 42)
        }
        
        # Add dataset-specific settings
        if dataset_name in self.benchmark_config:
            run_config.update(self.benchmark_config[dataset_name])
        
        return run_config
    
    def _create_comprehensive_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Create and save comprehensive summary across all datasets."""
        
        print(f"\n📈 Creating comprehensive summary...")
        
        try:
            import pandas as pd
            
            # Flatten results for DataFrame
            flattened_results = []
            for result in all_results:
                flat_result = {
                    'dataset': result['dataset'],
                    'method': result['method']
                }
                flat_result.update(result['metrics'])
                flattened_results.append(flat_result)
            
            # Create DataFrame and save
            results_dataframe = pd.DataFrame(flattened_results)
            summary_path = self.master_result_manager.run_directory / "comprehensive_summary.csv"
            results_dataframe.to_csv(summary_path, index=False)
            
            # Print summary table
            self._print_comprehensive_summary(results_dataframe)
            
            print(f"\n✅ Comprehensive results saved to: {summary_path}")
            print(f"📁 All detailed results in: {self.master_result_manager.run_directory}")
            
        except ImportError:
            print("❌ pandas not available for comprehensive summary")
        except Exception as error:
            print(f"❌ Error creating comprehensive summary: {error}")
    
    def _print_comprehensive_summary(self, results_dataframe) -> None:
        """Print formatted summary table."""
        
        print(f"\n🎯 BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        for dataset in results_dataframe['dataset'].unique():
            dataset_results = results_dataframe[results_dataframe['dataset'] == dataset]
            print(f"\n📊 {dataset.upper()}:")
            
            for _, row in dataset_results.iterrows():
                method = row['method']
                if dataset == 'ukbench':
                    ns_score = row.get('ns_score', 'N/A')
                    recall_1 = row.get('recall@1', 'N/A')
                    recall_4 = row.get('recall@4', 'N/A')
                    print(f"  {method:12} | N-S: {ns_score:5.3f} | R@1: {recall_1:5.3f} | R@4: {recall_4:5.3f}")
                elif dataset == 'holidays':
                    map_full = row.get('map_full', 'N/A')
                    map_10 = row.get('map@10', 'N/A')
                    recall_1 = row.get('recall@1', 'N/A')
                    print(f"  {method:12} | mAP: {map_full:5.3f} | mAP@10: {map_10:5.3f} | R@1: {recall_1:5.3f}")
