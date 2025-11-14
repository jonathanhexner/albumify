"""
Experiment runner for managing single and batch evaluations.
Handles the orchestration of methods, datasets, and result collection.
"""

import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

from sim_bench.datasets import load_dataset
from sim_bench.feature_extraction import load_method
from sim_bench import metrics_api as metrics
from sim_bench.result_manager import ResultManager
from sim_bench.feature_cache import FeatureCache
from sim_bench.logging_config import setup_logger, log_experiment_start, log_method_start, log_results, log_experiment_end
from sim_bench.detailed_logging import (
    setup_detailed_logger, log_sampling_details, log_feature_extraction_details,
    log_distance_computation_details, log_ranking_details, log_cache_operation
)
from sim_bench.clustering import load_clustering_method


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
        
        # Initialize logger
        log_file = self.result_manager.run_directory / "experiment.log"
        log_level = run_config.get('logging', {}).get('level', 'INFO')
        self.logger = setup_logger("sim_bench", log_file, log_level, console=False)
        
        # Initialize detailed logger (separate file for verbose details)
        detailed_log_file = self.result_manager.run_directory / "detailed.log"
        detailed_enabled = run_config.get('logging', {}).get('detailed', False)
        if detailed_enabled:
            self.detailed_logger = setup_detailed_logger(detailed_log_file, level='DEBUG')
        else:
            self.detailed_logger = None
        
        # Log experiment start
        log_experiment_start(self.logger, {**run_config, **dataset_config})
        
        # Initialize feature cache
        cache_enabled = run_config.get('cache_features', True)
        self.feature_cache = FeatureCache() if cache_enabled else None
        
        # Load dataset once for all experiments
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_config['name']}")
        print(f"{'='*60}")
        self.logger.info(f"Loading dataset: {dataset_config['name']}")
        
        self.dataset = load_dataset(dataset_config['name'], dataset_config)
        self.dataset.load_data()
        
        # Apply sampling if configured
        sampling_config = run_config.get('sampling', {})
        if sampling_config:
            print(f"Applying sampling: {sampling_config}")
            self.logger.info(f"Applying sampling: {sampling_config}")
        self.dataset.apply_sampling(sampling_config)
        
        print(f"[OK] Total images: {len(self.dataset.get_images())}")
        print(f"[OK] Query images: {len(self.dataset.get_queries())}")
        self.logger.info(f"Total images: {len(self.dataset.get_images())}")
        self.logger.info(f"Query images: {len(self.dataset.get_queries())}")
        
        # Detailed logging: sampling details
        if self.detailed_logger:
            log_sampling_details(
                self.detailed_logger,
                sampling_config,
                self.dataset.get_evaluation_data().get('groups', []),
                self.dataset.get_images()
            )
    
    def run_single_method(self, method_name: str) -> Dict[str, Any]:
        """
        Run evaluation for a single method.

        Args:
            method_name: Name of the method to run

        Returns:
            Dictionary containing method results
        """
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        # Load method
        method_config = self._load_method_config(method_name)
        method = load_method(method_name, method_config)

        # Extract features (with caching)
        image_paths = self.dataset.get_images()
        feature_matrix = self._extract_features_with_cache(method_name, method_config, method, image_paths)

        # Compute distances and rankings
        distance_matrix = self._compute_distances(method, feature_matrix, image_paths)
        ranking_indices = self._compute_rankings(distance_matrix)

        # Evaluate metrics
        computed_metrics = self._evaluate_metrics(ranking_indices)

        # Save and return results
        self._save_results(method_name, method_config, ranking_indices, distance_matrix, computed_metrics)

        return {
            'method': method_name,
            'metrics': computed_metrics,
            'config': method_config
        }

    def _load_method_config(self, method_name: str) -> Dict[str, Any]:
        """Load method configuration from file."""
        method_config_path = Path(f"configs/methods/{method_name}.yaml")
        if not method_config_path.exists():
            raise FileNotFoundError(f"Method config not found: {method_config_path}")

        method_config = yaml.safe_load(method_config_path.read_text())
        print(f"[OK] Loaded config: {method_config_path}")
        log_method_start(self.logger, method_name, method_config)
        return method_config

    def _extract_features_with_cache(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        method: Any,
        image_paths: List[Path]
    ) -> np.ndarray:
        """Extract features with caching support."""
        print(f"\n[1/4] Feature Extraction")
        print(f"-" * 60)

        # Try cache first
        feature_matrix = self._try_load_from_cache(method_name, method_config, image_paths)

        # Extract if not cached
        if feature_matrix is None:
            feature_matrix = self._extract_and_cache_features(method_name, method_config, method, image_paths)

        print(f"[OK] Feature matrix shape: {feature_matrix.shape}")

        if self.detailed_logger:
            log_feature_extraction_details(self.detailed_logger, method_name, image_paths, feature_matrix)

        return feature_matrix

    def _try_load_from_cache(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        image_paths: List[Path]
    ) -> Optional[np.ndarray]:
        """Try to load features from cache."""
        if not self.feature_cache:
            return None

        cache_path = self.feature_cache.get_cache_path(method_name, method_config, image_paths)
        feature_matrix = self.feature_cache.load(method_name, method_config, image_paths)

        if feature_matrix is not None:
            print(f"[CACHE] Loaded features from cache")
            self.logger.info(f"Features loaded from cache: {cache_path}")
            if self.detailed_logger:
                log_cache_operation(self.detailed_logger, 'hit', method_name, cache_path, True)

        return feature_matrix

    def _extract_and_cache_features(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        method: Any,
        image_paths: List[Path]
    ) -> np.ndarray:
        """Extract features and save to cache."""
        print(f"Extracting features for {len(image_paths)} images...")

        if self.detailed_logger and self.feature_cache:
            cache_path = self.feature_cache.get_cache_path(method_name, method_config, image_paths)
            log_cache_operation(self.detailed_logger, 'miss', method_name, cache_path, False,
                               "Features not in cache, extracting...")

        feature_matrix = method.extract_features(image_paths)

        # Save to cache
        if self.feature_cache:
            cache_path = self.feature_cache.get_cache_path(method_name, method_config, image_paths)
            self.feature_cache.save(method_name, method_config, image_paths, feature_matrix)
            print(f"[CACHE] Saved features to cache")
            self.logger.info(f"Features saved to cache: {cache_path}")
            if self.detailed_logger:
                log_cache_operation(self.detailed_logger, 'save', method_name, cache_path, True)

        return feature_matrix

    def _compute_distances(
        self,
        method: Any,
        feature_matrix: np.ndarray,
        image_paths: List[Path]
    ) -> np.ndarray:
        """Compute distance matrix."""
        print(f"\n[2/4] Distance Computation")
        print(f"-" * 60)
        print(f"Computing {len(image_paths)} x {len(image_paths)} distance matrix...")

        distance_matrix = method.compute_distances(feature_matrix)
        print(f"[OK] Distance matrix computed: {distance_matrix.shape}")

        if self.detailed_logger:
            log_distance_computation_details(self.detailed_logger, method.name if hasattr(method, 'name') else 'unknown', distance_matrix)

        return distance_matrix

    def _compute_rankings(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Compute ranking indices from distance matrix."""
        print(f"\n[3/4] Ranking Computation")
        print(f"-" * 60)

        ranking_indices = np.argsort(distance_matrix, axis=1)
        print(f"[OK] Rankings computed for {len(ranking_indices)} queries")

        return ranking_indices

    def _evaluate_metrics(self, ranking_indices: np.ndarray) -> Dict[str, float]:
        """Evaluate all configured metrics."""
        print(f"\n[4/4] Metric Evaluation")
        print(f"-" * 60)

        evaluation_data = self.dataset.get_evaluation_data()

        # Detailed logging: rankings
        if self.detailed_logger:
            k = self.run_config.get('k', 10)
            groups = evaluation_data.get('groups', [])
            log_ranking_details(self.detailed_logger, ranking_indices, groups, k)

        computed_metrics = metrics.compute_metrics(
            ranking_indices,
            evaluation_data,
            self.run_config
        )

        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        for metric_name, metric_value in computed_metrics.items():
            if metric_name not in ['num_queries', 'num_images']:
                print(f"  {metric_name:20s}: {metric_value:.4f}")
        print(f"{'='*60}")

        return computed_metrics

    def _save_results(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        ranking_indices: np.ndarray,
        distance_matrix: np.ndarray,
        computed_metrics: Dict[str, float]
    ) -> None:
        """Save method results to disk."""
        log_results(self.logger, method_name, computed_metrics)

        print(f"\nSaving results...")
        self.result_manager.save_method_results(
            method_name=method_name,
            method_config=method_config,
            ranking_indices=ranking_indices,
            distance_matrix=distance_matrix,
            computed_metrics=computed_metrics,
            dataset=self.dataset
        )
        print(f"[OK] Results saved to: {self.result_manager.run_directory / method_name}")

    def run_clustering(self, method_name: str, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run feature extraction and clustering for a single method.

        Args:
            method_name: Name of the method to run
            cluster_config: Clustering configuration dictionary

        Returns:
            Dictionary containing clustering results
        """
        print(f"\n{'='*60}")
        print(f"CLUSTERING EXPERIMENT")
        print(f"{'='*60}")
        print(f"Method: {method_name}")
        print(f"Algorithm: {cluster_config.get('algorithm', 'dbscan')}")
        print(f"{'='*60}")

        # Load method
        method_config = self._load_method_config(method_name)
        method = load_method(method_name, method_config)

        # Extract features (with caching)
        image_paths = self.dataset.get_images()
        feature_matrix = self._extract_features_with_cache(method_name, method_config, method, image_paths)

        # Load clustering method using factory
        print(f"\n[2/3] Clustering")
        print(f"-" * 60)
        clusterer = load_clustering_method(cluster_config)
        labels, stats = clusterer.cluster(feature_matrix)

        # Save results
        print(f"\n[3/3] Saving Results")
        print(f"-" * 60)
        output_dir = self.result_manager.run_directory
        experiment_name = self.run_config.get('experiment', {}).get('name', 'Clustering Experiment')
        clusterer.save_results(output_dir, image_paths, labels, stats, experiment_name)

        return {
            'method': method_name,
            'labels': labels,
            'stats': stats,
            'config': cluster_config
        }
    
    def run_multiple_methods(self, method_names: List[str]) -> Dict[str, Any]:
        """
        Run evaluation for multiple methods.
        
        Args:
            method_names: List of method names to run
            
        Returns:
            Comprehensive results dictionary for EDA
        """
        comprehensive_results = {
            'dataset_name': self.dataset.name,
            'methods': [],
            'method_performance': [],
            'per_query_details': [],
            'feature_statistics': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(self.dataset.get_images()),
                'total_queries': len(self.dataset.get_queries())
            }
        }
        
        for method_name in method_names:
            print(f"\n=== Running method: {method_name} ===")
            
            try:
                # Run single method
                method_result = self.run_single_method(method_name)
                
                # Compute feature statistics
                method_config = method_result['config']
                method = load_method(method_name, method_config)
                image_paths = self.dataset.get_images()
                features = method.extract_features(image_paths)
                
                method_stats = {
                    'method_name': method_name,
                    'feature_shape': features.shape,
                    'feature_mean': np.mean(features, axis=0),
                    'feature_std': np.std(features, axis=0),
                    'feature_min': np.min(features, axis=0),
                    'feature_max': np.max(features, axis=0)
                }
                
                comprehensive_results['feature_statistics'][method_name] = method_stats
                comprehensive_results['methods'].append(method_name)
                comprehensive_results['method_performance'].append({
                    'method': method_name,
                    'metrics': method_result['metrics']
                })
                
                # Per-query details
                evaluation_data = self.dataset.get_evaluation_data()
                groups = evaluation_data.get('groups', [])
                
                for query_idx, query_path in enumerate(self.dataset.get_queries()):
                    query_details = {
                        'method_name': method_name,
                        'query_idx': query_idx,
                        'query_path': query_path,
                        'ground_truth_group': groups[query_idx] if groups else None,
                        'top_k_indices': method_result.get('ranking_indices', [])[query_idx] if 'ranking_indices' in method_result else None
                    }
                    comprehensive_results['per_query_details'].append(query_details)
                
            except Exception as error:
                self.logger.error(f"ERROR running {method_name}: {error}", exc_info=True)
                print(f"ERROR running {method_name}: {error}")
                continue
        
        # Create summary if multiple methods were run
        if len(comprehensive_results['methods']) > 1:
            self.result_manager.create_summary(comprehensive_results['method_performance'])
        
        return comprehensive_results
    
    def _print_primary_metric(self, computed_metrics: Dict[str, float]) -> None:
        """Print the primary metric for the dataset."""
        if 'ns_score' in computed_metrics:
            print(f"N-S score: {computed_metrics['ns_score']:.3f}")
        elif 'map' in computed_metrics:
            print(f"mAP (full): {computed_metrics['map']:.3f}")
    
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
        
        # Initialize logger
        log_file = self.master_result_manager.run_directory / "benchmark.log"
        log_level = benchmark_config.get('logging', {}).get('level', 'INFO')
        self.logger = setup_logger("sim_bench.benchmark", log_file, log_level, console=False)
    
    def run_comprehensive_benchmark(self) -> List[Dict[str, Any]]:
        """
        Run comprehensive benchmark across all configured datasets and methods.
        
        Returns:
            List of all results across datasets and methods
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE BENCHMARK")
        print(f"{'='*60}")
        
        all_results = []
        
        # Process each dataset
        for dataset_config in self.benchmark_config['datasets']:
            dataset_name = dataset_config['name']
            dataset_config_path = dataset_config['config']
            
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
            
            try:
                dataset_results = self._run_dataset_benchmark(
                    dataset_name, 
                    dataset_config_path, 
                    dataset_config
                )
                
                # Add dataset information to results
                # dataset_results is a dict with 'method_performance' list
                if dataset_results and 'method_performance' in dataset_results:
                    for result in dataset_results['method_performance']:
                        result['dataset'] = dataset_name
                        all_results.append(result)
                    
            except Exception as error:
                self.logger.error(f"Error running {dataset_name}: {error}", exc_info=True)
                print(f"Error running {dataset_name}: {error}")
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
            'metrics': self.benchmark_config.get('metrics', []),  # FIX: Pass metrics list
            'k': self.benchmark_config.get('k', 4),                # FIX: Pass k parameter
            'sampling': dataset_config.get('sampling', {}),
            'random_seed': self.benchmark_config.get('random_seed', 42),
            'cache_features': self.benchmark_config.get('cache_features', True)  # Pass cache setting too
        }
        
        # Add dataset-specific settings
        if dataset_name in self.benchmark_config:
            run_config.update(self.benchmark_config[dataset_name])
        
        return run_config
    
    def _create_comprehensive_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Create and save comprehensive summary across all datasets."""
        
        print(f"\nCreating comprehensive summary...")
        
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
            
            print(f"\nComprehensive results saved to: {summary_path}")
            print(f"All detailed results in: {self.master_result_manager.run_directory}")
            
        except ImportError:
            self.logger.error("ERROR: pandas not available for comprehensive summary")
            print("ERROR: pandas not available for comprehensive summary")
        except Exception as error:
            self.logger.error(f"ERROR creating comprehensive summary: {error}", exc_info=True)
            print(f"ERROR creating comprehensive summary: {error}")
    
    def _print_comprehensive_summary(self, results_dataframe) -> None:
        """Print formatted summary table."""
        
        print(f"\nBENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        for dataset in results_dataframe['dataset'].unique():
            dataset_results = results_dataframe[results_dataframe['dataset'] == dataset]
            print(f"\n{dataset.upper()}:")
            
            for _, row in dataset_results.iterrows():
                method = row['method']
                if dataset == 'ukbench':
                    ns_score = row.get('ns_score', 0.0)
                    recall_1 = row.get('recall@1', 0.0)
                    recall_4 = row.get('recall@4', 0.0)
                    # Convert to float if string
                    ns_score = float(ns_score) if ns_score != 'N/A' else 0.0
                    recall_1 = float(recall_1) if recall_1 != 'N/A' else 0.0
                    recall_4 = float(recall_4) if recall_4 != 'N/A' else 0.0
                    print(f"  {method:12} | N-S: {ns_score:5.3f} | R@1: {recall_1:5.3f} | R@4: {recall_4:5.3f}")
                elif dataset == 'holidays':
                    map_full = row.get('map', 0.0)
                    map_10 = row.get('map@10', 0.0)
                    recall_1 = row.get('recall@1', 0.0)
                    # Convert to float if string
                    map_full = float(map_full) if map_full != 'N/A' else 0.0
                    map_10 = float(map_10) if map_10 != 'N/A' else 0.0
                    recall_1 = float(recall_1) if recall_1 != 'N/A' else 0.0
                    print(f"  {method:12} | mAP: {map_full:5.3f} | mAP@10: {map_10:5.3f} | R@1: {recall_1:5.3f}")
