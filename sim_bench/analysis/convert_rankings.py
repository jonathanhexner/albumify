"""
Utility to convert old rankings.csv (with array indices) to new format (with filenames).

This allows analyzing old experiments without rerunning them.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from sim_bench.datasets import load_dataset


def convert_rankings_to_filenames(
    rankings_file: Path,
    dataset_name: str,
    dataset_config: Optional[Dict[str, Any]] = None,
    backup: bool = True
) -> Path:
    """
    Convert rankings.csv from array indices to filenames.
    
    Args:
        rankings_file: Path to rankings.csv
        dataset_name: Dataset name ('ukbench' or 'holidays')
        dataset_config: Dataset configuration (if None, loads from configs/)
        backup: If True, create backup before converting
    
    Returns:
        Path to the converted file
    """
    if not rankings_file.exists():
        raise FileNotFoundError(f"File not found: {rankings_file}")
    
    # Load dataset to get image mappings
    if dataset_config is None:
        config_file = Path(__file__).parent.parent.parent / "configs" / f"dataset.{dataset_name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_file}")
        with open(config_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
    
    dataset = load_dataset(dataset_name, dataset_config)
    images = dataset.get_images()
    
    # Load existing rankings
    df = pd.read_csv(rankings_file)
    
    # Check if already converted
    if 'query_filename' in df.columns:
        print(f"[OK] {rankings_file.name} already uses filenames")
        return rankings_file
    
    print(f"Converting {rankings_file} from indices to filenames...")
    
    # Backup if requested
    if backup:
        backup_file = rankings_file.with_suffix('.csv.backup')
        import shutil
        shutil.copy2(rankings_file, backup_file)
        print(f"  Created backup: {backup_file.name}")
    
    # Convert indices to filenames
    df['query_filename'] = df['query_idx'].apply(lambda idx: Path(images[idx]).name)
    df['result_filename'] = df['result_idx'].apply(lambda idx: Path(images[idx]).name)
    
    # Drop old columns and reorder
    df = df.drop(columns=['query_idx', 'result_idx'])
    df = df[['query_filename', 'rank', 'result_filename', 'distance']]
    
    # Save converted file
    df.to_csv(rankings_file, index=False)
    print(f"[OK] Converted file saved: {rankings_file}")
    print(f"  Converted {len(df)} rankings")
    
    return rankings_file


def convert_experiment_rankings(
    experiment_dir: Path,
    dataset_name: str,
    methods: Optional[list] = None,
    backup: bool = True
) -> None:
    """
    Convert all rankings.csv files in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        dataset_name: Dataset name
        methods: List of method names to convert (None = all subdirectories)
        backup: If True, create backups before converting
    
    Example:
        >>> convert_experiment_rankings(
        ...     Path("outputs/.../2025-10-24_01-10-45"),
        ...     dataset_name="holidays"
        ... )
    """
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    print(f"Converting experiment: {experiment_dir.name}")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    
    # Find all method directories
    if methods is None:
        method_dirs = [d for d in experiment_dir.iterdir() 
                      if d.is_dir() and (d / 'rankings.csv').exists()]
    else:
        method_dirs = [experiment_dir / method for method in methods 
                      if (experiment_dir / method / 'rankings.csv').exists()]
    
    if not method_dirs:
        print("[WARNING] No rankings.csv files found to convert")
        return
    
    for method_dir in method_dirs:
        rankings_file = method_dir / 'rankings.csv'
        print(f"\n{method_dir.name}:")
        try:
            convert_rankings_to_filenames(
                rankings_file,
                dataset_name=dataset_name,
                backup=backup
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Conversion complete!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python convert_rankings.py <experiment_dir> <dataset_name>")
        print("Example: python convert_rankings.py outputs/.../2025-10-24_01-10-45 holidays")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    dataset_name = sys.argv[2]
    
    convert_experiment_rankings(exp_dir, dataset_name)

