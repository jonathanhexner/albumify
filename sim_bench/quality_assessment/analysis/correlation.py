"""
Correlation analysis for quality assessment methods.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def compute_correlation_matrix(
    merged_df: pd.DataFrame,
    methods: Optional[List[str]] = None,
    metric: str = 'correct'
) -> pd.DataFrame:
    """
    Compute correlation matrix between methods based on per-series performance.
    
    Args:
        merged_df: Merged DataFrame from merge_per_series_metrics()
        methods: List of methods to include (None = all methods in merged_df)
        metric: Metric to correlate ('correct' for binary accuracy)
        
    Returns:
        Correlation matrix DataFrame
    """
    if methods is None:
        # Extract methods from column names
        if metric == 'correct':
            method_cols = [col for col in merged_df.columns if col.endswith('_correct')]
            methods = [col.replace('_correct', '') for col in method_cols]
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Build correlation data
    corr_data = {}
    for method in methods:
        col_name = f'{method}_correct' if metric == 'correct' else f'{method}_{metric}'
        if col_name in merged_df.columns:
            corr_data[method] = merged_df[col_name].values
    
    if not corr_data:
        raise ValueError("No method data found for correlation")
    
    corr_df = pd.DataFrame(corr_data)
    corr_matrix = corr_df.corr()
    
    return corr_matrix




