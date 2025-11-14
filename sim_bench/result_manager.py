"""
Result management for saving experiment outputs.
Handles CSV generation, file organization, and summary creation.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from sim_bench.datasets.base import BaseDataset


class ResultManager:
    """Manages experiment result saving and organization."""
    
    def __init__(self, run_config: Dict[str, Any]):
        """
        Initialize result manager.
        
        Args:
            run_config: Run configuration dictionary
        """
        self.run_config = run_config
        self.run_directory = self._create_run_directory()
    
    def _create_run_directory(self) -> Path:
        """Create timestamped run directory."""
        output_root = Path(self.run_config['output_dir'])
        run_name = self.run_config.get('run_name')
        
        if run_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = output_root / f"{run_name}_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = output_root / timestamp
        
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory
    
    def save_method_results(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        ranking_indices: np.ndarray,
        distance_matrix: np.ndarray,
        computed_metrics: Dict[str, float],
        dataset: BaseDataset
    ) -> None:
        """
        Save complete results for a single method.
        
        Args:
            method_name: Name of the method
            method_config: Method configuration
            ranking_indices: Ranking indices matrix
            distance_matrix: Distance matrix
            computed_metrics: Computed metrics dictionary
            dataset: Dataset instance
        """
        method_directory = self.run_directory / method_name
        method_directory.mkdir(parents=True, exist_ok=True)
        
        # Save different result files
        self._save_metrics_csv(method_directory, method_name, computed_metrics, dataset)
        self._save_per_query_csv(method_directory, ranking_indices, computed_metrics, dataset)
        self._save_rankings_csv(method_directory, ranking_indices, distance_matrix, dataset)
        self._save_manifest_json(method_directory, method_name, method_config, dataset)
        
        print(f"Results written to: {method_directory}")
    
    def _save_metrics_csv(
        self,
        method_directory: Path,
        method_name: str,
        computed_metrics: Dict[str, float],
        dataset: BaseDataset
    ) -> None:
        """Save overall metrics CSV."""
        
        if not self.run_config.get('save', {}).get('metrics_csv', True):
            return
        
        metrics_file = method_directory / "metrics.csv"

        with open(metrics_file, "w", newline="") as file:
            writer = csv.writer(file)

            # Extract all metric columns (excluding metadata)
            metadata_keys = {'num_queries', 'num_images'}
            metric_names = [key for key in computed_metrics.keys() if key not in metadata_keys]

            # Build header: method, all metrics (sorted for consistency), metadata
            header = ["method"] + sorted(metric_names) + ["num_queries", "num_gallery", "created_at"]

            # Build row with corresponding values
            row = [method_name]
            for metric_name in sorted(metric_names):
                value = computed_metrics.get(metric_name, 0)
                row.append(f"{value:.6f}")

            # Add metadata
            row.extend([
                len(dataset.get_queries()),
                len(dataset.get_images()),
                datetime.now().isoformat()
            ])

            writer.writerow(header)
            writer.writerow(row)
    
    def _save_per_query_csv(
        self,
        method_directory: Path,
        ranking_indices: np.ndarray,
        computed_metrics: Dict[str, float],
        dataset: BaseDataset
    ) -> None:
        """Save per-query results CSV using unified approach."""
        
        if not self.run_config.get('save', {}).get('per_query_csv', True):
            return
        
        per_query_file = method_directory / "per_query.csv"
        
        with open(per_query_file, "w", newline="") as file:
            writer = csv.writer(file)
            
            # Write unified per-query CSV
            self._write_unified_per_query(writer, ranking_indices, dataset)
    
    def _write_unified_per_query(
        self,
        writer: csv.writer,
        ranking_indices: np.ndarray,
        dataset: BaseDataset
    ) -> None:
        """Write per-query results using dataset abstraction (works for all datasets)."""
        
        # Write header
        writer.writerow(["query_idx", "query_path", "group_id", "num_relevant", "ns_hitcount@4", "ap@10"])
        
        images = dataset.get_images()
        queries = dataset.get_queries()
        evaluation_data = dataset.get_evaluation_data()
        groups = evaluation_data.get('groups', [])
        k_value = self.run_config.get('k', 4)
        
        for query_idx in queries:
            num_relevant = dataset.get_num_relevant(query_idx)
            query_group = groups[query_idx]
            
            # Count hits in top-k (excluding self at rank 0)
            hits = sum(1 for result_idx in ranking_indices[query_idx][1:k_value+1] 
                      if dataset.is_relevant(query_idx, result_idx))
            
            # Compute AP@10 using correct normalization
            relevant_count = 0
            precision_sum = 0.0
            
            for rank, result_idx in enumerate(ranking_indices[query_idx][1:11], start=1):
                if dataset.is_relevant(query_idx, result_idx):
                    relevant_count += 1
                    precision_sum += relevant_count / rank
            
            # CORRECT: Divide by actual number of relevant images for this query
            ap_10 = precision_sum / num_relevant if num_relevant > 0 else 0.0
            
            writer.writerow([
                query_idx,
                images[query_idx],
                query_group,
                num_relevant,
                hits,
                f"{ap_10:.6f}"
            ])
    
    def _save_rankings_csv(
        self,
        method_directory: Path,
        ranking_indices: np.ndarray,
        distance_matrix: np.ndarray,
        dataset: BaseDataset
    ) -> None:
        """Save rankings CSV with image identifiers instead of array positions."""
        
        if not self.run_config.get('save', {}).get('rankings_csv', True):
            return
        
        rankings_file = method_directory / "rankings.csv"
        topk_limit = int(self.run_config.get('save', {}).get('topk_rankings_k', 10))
        
        with open(rankings_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["query_filename", "rank", "result_filename", "distance"])
            
            images = dataset.get_images()
            queries = dataset.get_queries()
            
            for query_idx in queries:
                query_filename = Path(images[query_idx]).name
                for rank in range(min(topk_limit, len(images))):
                    result_idx = ranking_indices[query_idx][rank]
                    result_filename = Path(images[result_idx]).name
                    distance = distance_matrix[query_idx][result_idx]
                    writer.writerow([query_filename, rank, result_filename, f"{distance:.6f}"])
    
    def _save_manifest_json(
        self,
        method_directory: Path,
        method_name: str,
        method_config: Dict[str, Any],
        dataset: BaseDataset = None
    ) -> None:
        """Save method manifest JSON with dataset metadata."""
        
        manifest_file = method_directory / "manifest.json"
        manifest_data = {
            "method": method_name,
            "config": method_config,
            "created_at": datetime.now().isoformat()
        }
        
        # Add dataset metadata for future analysis
        if dataset:
            manifest_data["dataset"] = {
                "name": dataset.dataset_config.get('name', 'unknown'),
                "total_images": len(dataset.get_images()),
                "total_queries": len(dataset.get_queries()),
                "num_groups": len(set(dataset.data.get('groups', [])))
            }
        
        with open(manifest_file, 'w') as file:
            json.dump(manifest_data, file, indent=2)
    
    def create_summary(self, method_results: List[Dict[str, Any]]) -> None:
        """
        Create summary CSV comparing multiple methods.
        
        Args:
            method_results: List of method result dictionaries
        """
        if not self.run_config.get('save', {}).get('summary_csv', True):
            return
        
        if not method_results:
            return
        
        summary_file = self.run_directory / "summary.csv"
        
        # Flatten results for CSV
        flattened_results = []
        for result in method_results:
            flat_result = {'method': result['method']}
            flat_result.update(result['metrics'])
            flattened_results.append(flat_result)
        
        # Write CSV
        with open(summary_file, 'w', newline='') as file:
            if flattened_results:
                fieldnames = flattened_results[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_results)
        
        print(f"\nSummary saved to: {summary_file}")
