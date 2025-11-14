"""
Script to re-run all baseline experiments with fixed metrics.

This will regenerate all CSV files with:
- Correct metric values (no more zeros from name mismatches)
- Consistent schema across all datasets
- Complete comprehensive metrics

Run this to replace the old buggy experiments.
"""

import subprocess
import sys
from pathlib import Path

# Methods to evaluate
METHODS = ["deep", "emd", "sift_bovw", "dinov2", "openclip"]

# Datasets
DATASETS = ["ukbench", "holidays"]

# Output directory
OUTPUT_DIR = "outputs/baseline_runs/comprehensive_baseline_fixed"

def run_experiment(methods: list, datasets: list):
    """Run experiment with specified methods and datasets."""

    methods_str = ",".join(methods)
    datasets_str = ",".join(datasets)

    print(f"\n{'='*80}")
    print(f"Running: {methods_str} on {datasets_str}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable, "-m", "sim_bench.cli",
        "--methods", methods_str,
        "--datasets", datasets_str,
        "--run-config", "configs/run.yaml"
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERROR] Failed to run {methods_str} on {datasets_str}")
        return False

    return True

def main():
    """Run all baseline experiments."""

    print("="*80)
    print("BASELINE EXPERIMENTS - COMPLETE RE-RUN WITH FIXED METRICS")
    print("="*80)
    print(f"\nMethods: {', '.join(METHODS)}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Output: {OUTPUT_DIR}")
    print("\nThis will:")
    print("  1. Compute ALL comprehensive metrics for each experiment")
    print("  2. Generate consistent CSV schema across all datasets")
    print("  3. Fix all name mismatch bugs (map_full, prec@10, etc.)")
    print("\nEstimated time: ~30-60 minutes depending on your hardware")
    print("\n" + "="*80)

    response = input("\nProceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return

    # Option 1: Run all methods together (faster, one run per dataset)
    print("\n[OPTION] Run all methods together (faster)")
    print("  - All 5 methods on UKBench in one run")
    print("  - All 5 methods on Holidays in one run")

    # Option 2: Run each method separately (easier to debug)
    print("\n[OPTION] Run each method separately (easier to debug)")
    print("  - One run per method-dataset combination (10 total runs)")

    option = input("\nChoose option (1=together, 2=separate): ").strip()

    if option == "1":
        # Run all methods together per dataset
        for dataset in DATASETS:
            success = run_experiment(METHODS, [dataset])
            if not success:
                print(f"\n[FAILED] Stopping due to error")
                return

    elif option == "2":
        # Run each method separately
        for method in METHODS:
            for dataset in DATASETS:
                success = run_experiment([method], [dataset])
                if not success:
                    print(f"\n[FAILED] Stopping due to error")
                    return

    else:
        print("Invalid option. Aborted.")
        return

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults location: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Check the CSV files have consistent columns")
    print("  2. Run multi-experiment analysis notebook")
    print("  3. Verify all metrics have reasonable values (no zeros)")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
