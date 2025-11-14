# Scripts Directory

This directory contains utility scripts and tools for the sim-bench project.

## Structure

### utilities/

One-off utility scripts for data processing, metrics recalculation, and fixing experiment results:

- **convert_heic_to_jpeg.py** - Convert HEIC images to JPEG format for datasets
- **add_missing_ukbench_metrics.py** - Add missing recall@10 and mAP metrics to UKBench results
- **fix_all_dinov2.py** - Fix DINOv2 metrics.csv files
- **fix_all_metrics.py** - General metrics fix utility
- **recalculate_metrics.py** - Recalculate metrics from existing rankings.csv files
- **rerun_baseline_experiments.py** - Re-run baseline experiments with fixed metrics

## Usage

These utilities are typically run once to fix or process specific datasets or experiment results:

```bash
# Example: Convert HEIC images to JPEG
python scripts/utilities/convert_heic_to_jpeg.py

# Example: Recalculate metrics from existing rankings
python scripts/utilities/recalculate_metrics.py

# Example: Re-run baseline experiments
python scripts/utilities/rerun_baseline_experiments.py
```

## Note

Most of these scripts contain hardcoded paths to specific datasets or output directories. Review and modify the paths in each script before running.
