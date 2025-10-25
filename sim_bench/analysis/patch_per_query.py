"""
Utility to patch existing per_query.csv files with missing columns.

This allows analyzing old experiments without rerunning them.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import shutil


def load_dataset_config(dataset_name: str, config_dir: Path = None) -> Dict[str, Any]:
    """Load dataset configuration."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
    
    config_file = config_dir / f"dataset.{dataset_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_group_id_from_path(image_path: str, dataset_type: str) -> int:
    """
    Extract group ID from image path.
    
    Args:
        image_path: Full path to image
        dataset_type: 'ukbench' or 'holidays'
    
    Returns:
        Group ID (integer)
    """
    filename = Path(image_path).stem
    
    if dataset_type == 'ukbench':
        # ukbench00123 -> group 30 (123 // 4)
        img_id = int(filename.replace('ukbench', ''))
        return img_id // 4
    elif dataset_type == 'holidays':
        # 100000 -> group 1000 (100000 // 100)
        img_id = int(filename)
        return img_id // 100
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def patch_per_query_with_group_id(
    per_query_file: Path,
    dataset_type: str,
    backup: bool = True,
    in_place: bool = True
) -> Path:
    """
    Add group_id column to existing per_query.csv.
    
    Args:
        per_query_file: Path to per_query.csv
        dataset_type: 'ukbench' or 'holidays'
        backup: If True, create backup before modifying
        in_place: If True, modify original file; if False, create new file
    
    Returns:
        Path to the patched file
    
    Example:
        >>> patch_per_query_with_group_id(
        ...     Path("outputs/.../deep/per_query.csv"),
        ...     dataset_type="holidays"
        ... )
    """
    if not per_query_file.exists():
        raise FileNotFoundError(f"File not found: {per_query_file}")
    
    # Load existing data
    df = pd.read_csv(per_query_file)
    
    # Check if already has group_id
    if 'group_id' in df.columns:
        print(f"[OK] {per_query_file.name} already has group_id column")
        return per_query_file
    
    print(f"Patching {per_query_file} with group_id column...")
    
    # Backup if requested
    if backup:
        backup_file = per_query_file.with_suffix('.csv.backup')
        shutil.copy2(per_query_file, backup_file)
        print(f"  Created backup: {backup_file.name}")
    
    # Add group_id column
    df['group_id'] = df['query_path'].apply(
        lambda path: get_group_id_from_path(path, dataset_type)
    )
    
    # Reorder columns to put group_id after query_path
    cols = df.columns.tolist()
    # Remove group_id from its current position
    cols.remove('group_id')
    # Insert after query_path
    query_path_idx = cols.index('query_path')
    cols.insert(query_path_idx + 1, 'group_id')
    df = df[cols]
    
    # Save
    if in_place:
        output_file = per_query_file
    else:
        output_file = per_query_file.with_name('per_query_patched.csv')
    
    df.to_csv(output_file, index=False)
    print(f"[OK] Patched file saved: {output_file}")
    print(f"  Added group_id for {len(df)} queries")
    
    return output_file


def patch_experiment_directory(
    experiment_dir: Path,
    dataset_type: str,
    methods: Optional[list] = None,
    backup: bool = True
) -> None:
    """
    Patch all per_query.csv files in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        dataset_type: 'ukbench' or 'holidays'
        methods: List of method names to patch (None = all subdirectories)
        backup: If True, create backups before modifying
    
    Example:
        >>> patch_experiment_directory(
        ...     Path("outputs/baseline_runs/comprehensive_baseline/2025-10-24_01-10-45"),
        ...     dataset_type="holidays"
        ... )
    """
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    print(f"Patching experiment: {experiment_dir.name}")
    print(f"Dataset type: {dataset_type}")
    print("=" * 60)
    
    # Find all method directories
    if methods is None:
        method_dirs = [d for d in experiment_dir.iterdir() 
                      if d.is_dir() and (d / 'per_query.csv').exists()]
    else:
        method_dirs = [experiment_dir / method for method in methods 
                      if (experiment_dir / method / 'per_query.csv').exists()]
    
    if not method_dirs:
        print("[WARNING] No per_query.csv files found to patch")
        return
    
    for method_dir in method_dirs:
        per_query_file = method_dir / 'per_query.csv'
        print(f"\n{method_dir.name}:")
        try:
            patch_per_query_with_group_id(
                per_query_file,
                dataset_type=dataset_type,
                backup=backup,
                in_place=True
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Patching complete!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python patch_per_query.py <experiment_dir> <dataset_type>")
        print("Example: python patch_per_query.py outputs/.../2025-10-24_01-10-45 holidays")
        sys.exit(1)
    
    exp_dir = Path(sys.argv[1])
    dataset_type = sys.argv[2]
    
    patch_experiment_directory(exp_dir, dataset_type)

