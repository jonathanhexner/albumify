"""
Merge per_pair_results.csv from multiple methods into a single DataFrame for analysis.

This script combines pairwise comparison results from different quality assessment methods,
allowing for correlation analysis, agreement studies, and comparative evaluation.
"""

from pathlib import Path
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sim_bench.analysis.utils import get_project_root


def load_pairwise_results(benchmark_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load per_pair_results.csv from all method subdirectories.

    Args:
        benchmark_dir: Path to benchmark directory (e.g., pairwise_20251120_100520)

    Returns:
        Dict mapping method_name -> DataFrame with columns:
        [pair_id, series_id, score_a, score_b, predicted, actual, correct,
         preference_strength, attribute_predictions]
    """
    results = {}

    for method_dir in benchmark_dir.iterdir():
        if not method_dir.is_dir():
            continue

        results_file = method_dir / "per_pair_results.csv"
        if not results_file.exists():
            continue

        method_name = method_dir.name
        df = pd.read_csv(results_file)
        results[method_name] = df
        print(f"Loaded {method_name}: {len(df)} pairs")

    return results


def merge_pairwise_results(
    results_dict: dict[str, pd.DataFrame],
    include_scores: bool = True,
    include_predictions: bool = True
) -> pd.DataFrame:
    """
    Merge pairwise results from multiple methods into a single DataFrame.

    Args:
        results_dict: Dict from load_pairwise_results()
        include_scores: Include score_a and score_b for each method
        include_predictions: Include predicted outcome for each method

    Returns:
        Merged DataFrame with columns:
        - pair_id, series_id, actual (ground truth)
        - For each method:
            - {method}_correct: whether method was correct
            - {method}_score_a: score for image A (if include_scores)
            - {method}_score_b: score for image B (if include_scores)
            - {method}_predicted: predicted winner A or B (if include_predictions)
            - {method}_preference_strength: confidence of prediction
    """
    if not results_dict:
        raise ValueError("results_dict is empty")

    # Start with pair_id, series_id, and actual from first method
    first_method = list(results_dict.keys())[0]
    merged_df = results_dict[first_method][['pair_id', 'series_id', 'actual']].copy()

    # Add columns for each method
    for method_name, df in results_dict.items():
        # Verify same pairs
        if not df['pair_id'].equals(merged_df['pair_id']):
            raise ValueError(f"Method {method_name} has different pairs!")

        # Add correct/incorrect
        merged_df[f'{method_name}_correct'] = df['correct'].astype(int)

        # Add scores
        if include_scores and 'score_a' in df.columns:
            merged_df[f'{method_name}_score_a'] = df['score_a']
            merged_df[f'{method_name}_score_b'] = df['score_b']

        # Add predictions
        if include_predictions and 'predicted' in df.columns:
            merged_df[f'{method_name}_predicted'] = df['predicted']

        # Add preference strength (confidence)
        if 'preference_strength' in df.columns:
            merged_df[f'{method_name}_preference_strength'] = df['preference_strength']

    return merged_df


def compute_method_agreement(merged_df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """
    Compute pairwise agreement between methods.

    Args:
        merged_df: DataFrame from merge_pairwise_results()
        methods: List of method names

    Returns:
        Agreement matrix (methods x methods) with values 0-1
    """
    agreement_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for method1 in methods:
        for method2 in methods:
            col1 = f'{method1}_correct'
            col2 = f'{method2}_correct'

            if col1 not in merged_df.columns or col2 not in merged_df.columns:
                agreement_matrix.loc[method1, method2] = None
                continue

            # Agreement = fraction of pairs where both methods agree
            agreement = (merged_df[col1] == merged_df[col2]).mean()
            agreement_matrix.loc[method1, method2] = agreement

    return agreement_matrix


def analyze_disagreements(
    merged_df: pd.DataFrame,
    method1: str,
    method2: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Find pairs where two methods disagree.

    Args:
        merged_df: DataFrame from merge_pairwise_results()
        method1: First method name
        method2: Second method name
        top_n: Number of top disagreement cases to return

    Returns:
        DataFrame with disagreement cases, sorted by preference strength difference
    """
    col1_correct = f'{method1}_correct'
    col2_correct = f'{method2}_correct'
    col1_strength = f'{method1}_preference_strength'
    col2_strength = f'{method2}_preference_strength'

    # Find disagreements
    disagreements = merged_df[merged_df[col1_correct] != merged_df[col2_correct]].copy()

    # Calculate confidence difference
    if col1_strength in disagreements.columns and col2_strength in disagreements.columns:
        disagreements['strength_diff'] = abs(
            disagreements[col1_strength] - disagreements[col2_strength]
        )
        disagreements = disagreements.sort_values('strength_diff', ascending=False)

    return disagreements.head(top_n)


def compute_accuracy_by_series(merged_df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """
    Compute per-series accuracy for each method.

    Args:
        merged_df: DataFrame from merge_pairwise_results()
        methods: List of method names

    Returns:
        DataFrame with columns: series_id, num_pairs, {method}_accuracy for each method
    """
    series_stats = []

    for series_id, group in merged_df.groupby('series_id'):
        stats = {
            'series_id': series_id,
            'num_pairs': len(group)
        }

        for method in methods:
            col = f'{method}_correct'
            if col in group.columns:
                stats[f'{method}_accuracy'] = group[col].mean()

        series_stats.append(stats)

    return pd.DataFrame(series_stats)


if __name__ == '__main__':
    # Example usage
    PROJECT_ROOT = get_project_root()
    BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "pairwise_benchmark_3hour" / "pairwise_20251120_100520"

    print(f"Loading results from: {BENCHMARK_DIR.name}\n")

    # Load results
    results_dict = load_pairwise_results(BENCHMARK_DIR)
    methods = list(results_dict.keys())

    print(f"\n{'='*80}")
    print(f"Found {len(methods)} methods")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'='*80}\n")

    # Merge results
    print("Merging results...")
    merged_df = merge_pairwise_results(results_dict, include_scores=True, include_predictions=True)
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}\n")

    # Compute agreement
    print("Computing method agreement...")
    agreement_matrix = compute_method_agreement(merged_df, methods)
    print("\nMethod Agreement Matrix:")
    print(agreement_matrix.round(3))

    # Overall accuracy
    print(f"\n{'='*80}")
    print("Overall Accuracy:")
    print(f"{'='*80}")
    for method in methods:
        col = f'{method}_correct'
        accuracy = merged_df[col].mean()
        print(f"{method:25s}: {accuracy:.3f}")

    # Save merged results
    output_path = BENCHMARK_DIR / "merged_pairwise_results.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Merged results to: {output_path.name}")

    # Save agreement matrix
    agreement_path = BENCHMARK_DIR / "method_agreement_matrix.csv"
    agreement_matrix.to_csv(agreement_path)
    print(f"[SAVED] Agreement matrix to: {agreement_path.name}")

    # Compute per-series accuracy
    print("\nComputing per-series accuracy...")
    series_accuracy = compute_accuracy_by_series(merged_df, methods)
    series_output = BENCHMARK_DIR / "per_series_accuracy.csv"
    series_accuracy.to_csv(series_output, index=False)
    print(f"[SAVED] Per-series accuracy to: {series_output.name}")
    print(f"\nSeries accuracy summary:")
    print(series_accuracy.describe())
