#!/usr/bin/env python3
"""Compare face clustering benchmark results across different runs.

Usage:
    python scripts/compare_benchmarks.py results/run_a results/run_b
    python scripts/compare_benchmarks.py results/run_a results/run_b --method hybrid_hdbscan_knn
    python scripts/compare_benchmarks.py results/run_a results/run_b --output comparison_report.md

This tool helps you understand:
1. Did my changes improve or hurt clustering quality?
2. Which faces changed cluster assignments?
3. How do thresholds and parameters differ?
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Parsed benchmark result for comparison."""
    path: Path
    timestamp: str
    album_path: str
    total_faces: int
    methods: Dict[str, Dict[str, Any]]
    face_metadata: List[Dict[str, Any]]

    @classmethod
    def load(cls, results_dir: Path) -> "BenchmarkResult":
        """Load benchmark result from directory."""
        json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
        if not json_files:
            raise FileNotFoundError(f"No benchmark JSON found in {results_dir}")

        with open(json_files[0], encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            path=json_files[0],
            timestamp=data.get("timestamp", "unknown"),
            album_path=data.get("album_path", ""),
            total_faces=data.get("total_faces", 0),
            methods=data.get("methods", {}),
            face_metadata=data.get("face_metadata", []),
        )


@dataclass
class MethodComparison:
    """Comparison between two method results."""
    method_name: str

    # Basic metrics
    n_clusters_a: int
    n_clusters_b: int
    n_noise_a: int
    n_noise_b: int

    # Label differences
    n_changed: int
    changed_indices: List[int]
    pct_changed: float

    # Cluster size stats
    sizes_a: List[int]
    sizes_b: List[int]

    # Parameter differences
    params_a: Dict[str, Any]
    params_b: Dict[str, Any]
    param_diffs: Dict[str, Tuple[Any, Any]]


def compare_methods(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    method: str,
) -> Optional[MethodComparison]:
    """Compare a specific method across two benchmark runs."""
    if method not in result_a.methods or method not in result_b.methods:
        return None

    data_a = result_a.methods[method]
    data_b = result_b.methods[method]

    labels_a = np.array(data_a.get("labels", []))
    labels_b = np.array(data_b.get("labels", []))

    stats_a = data_a.get("stats", {})
    stats_b = data_b.get("stats", {})

    # Find changed assignments
    if len(labels_a) == len(labels_b):
        changed = np.where(labels_a != labels_b)[0].tolist()
    else:
        changed = []

    # Get cluster sizes
    def get_sizes(labels):
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        return sorted(counts.tolist(), reverse=True)

    # Get params
    params_a = stats_a.get("params", {})
    params_b = stats_b.get("params", {})

    # Find param differences
    all_keys = set(params_a.keys()) | set(params_b.keys())
    param_diffs = {}
    for key in all_keys:
        val_a = params_a.get(key)
        val_b = params_b.get(key)
        if val_a != val_b:
            param_diffs[key] = (val_a, val_b)

    return MethodComparison(
        method_name=method,
        n_clusters_a=stats_a.get("n_clusters", 0),
        n_clusters_b=stats_b.get("n_clusters", 0),
        n_noise_a=stats_a.get("n_noise", 0),
        n_noise_b=stats_b.get("n_noise", 0),
        n_changed=len(changed),
        changed_indices=changed[:100],  # Limit to first 100
        pct_changed=100 * len(changed) / max(len(labels_a), 1),
        sizes_a=get_sizes(labels_a),
        sizes_b=get_sizes(labels_b),
        params_a=params_a,
        params_b=params_b,
        param_diffs=param_diffs,
    )


def format_comparison_report(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    comparisons: List[MethodComparison],
) -> str:
    """Format comparison as markdown report."""
    lines = []
    lines.append("# Benchmark Comparison Report")
    lines.append("")
    lines.append("## Run Information")
    lines.append("")
    lines.append("| | Run A | Run B |")
    lines.append("|---|---|---|")
    lines.append(f"| **File** | `{result_a.path.name}` | `{result_b.path.name}` |")
    lines.append(f"| **Timestamp** | {result_a.timestamp} | {result_b.timestamp} |")
    lines.append(f"| **Album** | {Path(result_a.album_path).name} | {Path(result_b.album_path).name} |")
    lines.append(f"| **Total Faces** | {result_a.total_faces} | {result_b.total_faces} |")
    lines.append("")

    for cmp in comparisons:
        lines.append(f"## Method: `{cmp.method_name}`")
        lines.append("")

        # Metrics summary
        lines.append("### Metrics")
        lines.append("")
        lines.append("| Metric | Run A | Run B | Delta |")
        lines.append("|--------|-------|-------|-------|")

        delta_c = cmp.n_clusters_b - cmp.n_clusters_a
        delta_n = cmp.n_noise_b - cmp.n_noise_a
        delta_c_str = f"+{delta_c}" if delta_c > 0 else str(delta_c)
        delta_n_str = f"+{delta_n}" if delta_n > 0 else str(delta_n)

        lines.append(f"| Clusters | {cmp.n_clusters_a} | {cmp.n_clusters_b} | {delta_c_str} |")
        lines.append(f"| Noise | {cmp.n_noise_a} | {cmp.n_noise_b} | {delta_n_str} |")
        lines.append(f"| Changed | — | — | {cmp.n_changed} ({cmp.pct_changed:.1f}%) |")
        lines.append("")

        # Parameter differences
        if cmp.param_diffs:
            lines.append("### Parameter Differences")
            lines.append("")
            lines.append("| Parameter | Run A | Run B |")
            lines.append("|-----------|-------|-------|")
            for param, (val_a, val_b) in cmp.param_diffs.items():
                lines.append(f"| `{param}` | {val_a} | {val_b} |")
            lines.append("")

        # Cluster size distribution
        lines.append("### Cluster Sizes (top 10)")
        lines.append("")
        lines.append("| Rank | Run A | Run B |")
        lines.append("|------|-------|-------|")
        for i in range(min(10, max(len(cmp.sizes_a), len(cmp.sizes_b)))):
            size_a = cmp.sizes_a[i] if i < len(cmp.sizes_a) else "—"
            size_b = cmp.sizes_b[i] if i < len(cmp.sizes_b) else "—"
            lines.append(f"| {i+1} | {size_a} | {size_b} |")
        lines.append("")

        # Changed faces
        if cmp.changed_indices:
            lines.append("### Changed Face Assignments")
            lines.append("")
            indices_str = ", ".join(map(str, cmp.changed_indices[:50]))
            if len(cmp.changed_indices) > 50:
                indices_str += f" ... (+{len(cmp.changed_indices) - 50} more)"
            lines.append(f"Indices: {indices_str}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def print_summary(comparisons: List[MethodComparison]) -> None:
    """Print quick summary to console."""
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    for cmp in comparisons:
        delta_c = cmp.n_clusters_b - cmp.n_clusters_a
        delta_n = cmp.n_noise_b - cmp.n_noise_a

        delta_c_str = f"+{delta_c}" if delta_c > 0 else str(delta_c)
        delta_n_str = f"+{delta_n}" if delta_n > 0 else str(delta_n)

        logger.info(f"\n{cmp.method_name}:")
        logger.info(f"  Clusters: {cmp.n_clusters_a} → {cmp.n_clusters_b} ({delta_c_str})")
        logger.info(f"  Noise:    {cmp.n_noise_a} → {cmp.n_noise_b} ({delta_n_str})")
        logger.info(f"  Changed:  {cmp.n_changed} faces ({cmp.pct_changed:.1f}%)")

        if cmp.param_diffs:
            logger.info(f"  Params changed: {list(cmp.param_diffs.keys())}")

    logger.info("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare face clustering benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_a", type=Path, help="Path to first benchmark results directory")
    parser.add_argument("run_b", type=Path, help="Path to second benchmark results directory")
    parser.add_argument("--method", "-m", help="Compare specific method only")
    parser.add_argument("--output", "-o", type=Path, help="Save markdown report to file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output report, no console summary")

    args = parser.parse_args()

    # Load results
    try:
        result_a = BenchmarkResult.load(args.run_a)
        result_b = BenchmarkResult.load(args.run_b)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    # Determine methods to compare
    if args.method:
        methods = [args.method]
    else:
        methods = sorted(set(result_a.methods.keys()) & set(result_b.methods.keys()))

    if not methods:
        logger.error("No common methods found between runs")
        sys.exit(1)

    # Compare
    comparisons = []
    for method in methods:
        cmp = compare_methods(result_a, result_b, method)
        if cmp:
            comparisons.append(cmp)

    if not comparisons:
        logger.error("Could not compare any methods")
        sys.exit(1)

    # Output
    if not args.quiet:
        print_summary(comparisons)

    report = format_comparison_report(result_a, result_b, comparisons)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        logger.info(f"\nReport saved to: {args.output}")
    else:
        if args.quiet:
            print(report)


if __name__ == "__main__":
    main()
