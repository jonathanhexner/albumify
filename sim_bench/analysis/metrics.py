"""
Utility functions for computing metrics from raw ranking data.
These functions allow on-demand computation of additional metrics beyond what's saved in per_query.csv.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def compute_recall_at_k(
    rankings_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    k: int,
    dataset_type: str = "ukbench"
) -> pd.DataFrame:
    """
    Compute recall@k for each query from rankings data.
    
    Args:
        rankings_df: DataFrame with columns [query_idx, rank, result_idx, distance]
        per_query_df: DataFrame with query metadata (must have group_id for ukbench)
        k: Number of top results to consider
        dataset_type: "ukbench" or "holidays"
    
    Returns:
        DataFrame with columns [query_idx, recall@k]
    """
    results = []
    
    if dataset_type == "ukbench":
        # For UKBench: count how many of top-k results are from the same group
        # Each group has 4 images, so max relevant is 3 (excluding query itself)
        group_map = per_query_df.set_index('query_idx')['group_id'].to_dict()
        
        for query_idx in per_query_df['query_idx']:
            query_group = group_map[query_idx]
            
            # Get top-k results (excluding rank 0 which is the query itself)
            query_rankings = rankings_df[
                (rankings_df['query_idx'] == query_idx) & 
                (rankings_df['rank'] > 0) & 
                (rankings_df['rank'] <= k)
            ]
            
            # Count relevant results (same group as query)
            relevant_retrieved = sum(
                group_map.get(int(row['result_idx']), -1) == query_group 
                for _, row in query_rankings.iterrows()
            )
            
            # For UKBench: max relevant is 3 (4 images per group - 1 query)
            # Standard recall definition: relevant_retrieved / total_relevant
            # This gives values like 0, 1/3, 2/3, 3/3 for recall@1, @2, @3
            max_relevant = 3
            recall = relevant_retrieved / max_relevant
            
            results.append({
                'query_idx': query_idx,
                f'recall@{k}': recall
            })
    
    else:  # holidays
        # For Holidays: use relevance information if available
        # This requires the relevance_map which isn't in the CSV
        # For now, just compute based on group_id if available
        if 'group_id' in per_query_df.columns:
            group_map = per_query_df.set_index('query_idx')['group_id'].to_dict()
            
            for query_idx in per_query_df['query_idx']:
                query_group = group_map[query_idx]
                
                query_rankings = rankings_df[
                    (rankings_df['query_idx'] == query_idx) & 
                    (rankings_df['rank'] > 0) & 
                    (rankings_df['rank'] <= k)
                ]
                
                relevant_retrieved = sum(
                    group_map.get(int(row['result_idx']), -1) == query_group 
                    for _, row in query_rankings.iterrows()
                )
                
                # For holidays, need to know total relevant images
                # This is a limitation without the full dataset info
                # Use num_relevant from per_query if available
                if 'num_relevant' in per_query_df.columns:
                    num_relevant = per_query_df[per_query_df['query_idx'] == query_idx]['num_relevant'].iloc[0]
                    recall = relevant_retrieved / num_relevant if num_relevant > 0 else 0.0
                else:
                    recall = 0.0  # Cannot compute without relevance info
                
                results.append({
                    'query_idx': query_idx,
                    f'recall@{k}': recall
                })
        else:
            raise ValueError("Cannot compute recall for Holidays without group_id or relevance information")
    
    return pd.DataFrame(results)


def compute_multiple_recalls(
    rankings_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    k_values: List[int],
    dataset_type: str = "ukbench"
) -> pd.DataFrame:
    """
    Compute recall@k for multiple k values.
    
    Args:
        rankings_df: DataFrame with columns [query_idx, rank, result_idx, distance]
        per_query_df: DataFrame with query metadata
        k_values: List of k values to compute (e.g., [1, 2, 3, 4])
        dataset_type: "ukbench" or "holidays"
    
    Returns:
        DataFrame with columns [query_idx, recall@1, recall@2, ...]
    """
    # Start with query_idx
    result_df = per_query_df[['query_idx']].copy()
    
    # Compute recall for each k
    for k in k_values:
        recall_df = compute_recall_at_k(rankings_df, per_query_df, k, dataset_type)
        result_df = result_df.merge(recall_df, on='query_idx', how='left')
    
    return result_df


def compute_precision_at_k(
    rankings_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    k: int,
    dataset_type: str = "ukbench"
) -> pd.DataFrame:
    """
    Compute precision@k for each query.
    
    Args:
        rankings_df: DataFrame with rankings
        per_query_df: DataFrame with query metadata
        k: Number of top results to consider
        dataset_type: "ukbench" or "holidays"
    
    Returns:
        DataFrame with columns [query_idx, precision@k]
    """
    results = []
    
    if dataset_type == "ukbench":
        group_map = per_query_df.set_index('query_idx')['group_id'].to_dict()
        
        for query_idx in per_query_df['query_idx']:
            query_group = group_map[query_idx]
            
            query_rankings = rankings_df[
                (rankings_df['query_idx'] == query_idx) & 
                (rankings_df['rank'] > 0) & 
                (rankings_df['rank'] <= k)
            ]
            
            relevant_retrieved = sum(
                group_map.get(int(row['result_idx']), -1) == query_group 
                for _, row in query_rankings.iterrows()
            )
            
            precision = relevant_retrieved / k if k > 0 else 0.0
            
            results.append({
                'query_idx': query_idx,
                f'precision@{k}': precision
            })
    
    else:  # holidays
        if 'group_id' in per_query_df.columns:
            group_map = per_query_df.set_index('query_idx')['group_id'].to_dict()
            
            for query_idx in per_query_df['query_idx']:
                query_group = group_map[query_idx]
                
                query_rankings = rankings_df[
                    (rankings_df['query_idx'] == query_idx) & 
                    (rankings_df['rank'] > 0) & 
                    (rankings_df['rank'] <= k)
                ]
                
                relevant_retrieved = sum(
                    group_map.get(int(row['result_idx']), -1) == query_group 
                    for _, row in query_rankings.iterrows()
                )
                
                precision = relevant_retrieved / k if k > 0 else 0.0
                
                results.append({
                    'query_idx': query_idx,
                    f'precision@{k}': precision
                })
        else:
            raise ValueError("Cannot compute precision for Holidays without group_id information")
    
    return pd.DataFrame(results)


def get_top_k_results(
    rankings_df: pd.DataFrame,
    query_idx: int,
    k: int,
    include_distances: bool = True
) -> pd.DataFrame:
    """
    Get the top-k results for a specific query.
    
    Args:
        rankings_df: DataFrame with rankings
        query_idx: Query index
        k: Number of results to return
        include_distances: Whether to include distance values
    
    Returns:
        DataFrame with top-k results
    """
    query_results = rankings_df[
        (rankings_df['query_idx'] == query_idx) & 
        (rankings_df['rank'] <= k)
    ].copy()
    
    query_results = query_results.sort_values('rank')
    
    if not include_distances:
        query_results = query_results.drop(columns=['distance'])
    
    return query_results


def compute_enriched_per_query(
    method: str,
    k_values: Optional[List[int]] = None,
    experiment_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute an enriched per-query DataFrame with additional metrics.
    
    This function loads the raw per_query.csv and rankings.csv, then computes
    additional metrics like recall@1, recall@2, etc.
    
    Args:
        method: Method name
        k_values: List of k values for recall/precision (default: [1, 2, 3, 4])
        experiment_dir: Experiment directory (uses global config if None)
    
    Returns:
        Enriched DataFrame with original columns plus computed metrics
    """
    from .io import load_per_query, load_rankings
    
    if k_values is None:
        k_values = [1, 2, 3, 4]
    
    # Load data
    per_query_df = load_per_query(method, experiment_dir)
    rankings_df = load_rankings(method, experiment_dir)
    
    # Detect dataset type (simple heuristic based on columns)
    dataset_type = "ukbench" if "ns_hitcount@4" in per_query_df.columns else "holidays"
    
    # Start with original per_query data
    enriched_df = per_query_df.copy()
    
    # Compute recalls
    recalls_df = compute_multiple_recalls(rankings_df, per_query_df, k_values, dataset_type)
    
    # Merge
    enriched_df = enriched_df.merge(recalls_df.drop(columns=['query_idx']), left_index=True, right_index=True)
    
    return enriched_df

