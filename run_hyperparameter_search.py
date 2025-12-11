"""
Hyperparameter search for PhotoTriage Multi-Feature Ranker.

This script runs multiple training experiments with different hyperparameters
defined in a CSV file and tracks results in a detailed CSV output.

Usage:
    # Run all experiments from CSV
    python run_hyperparameter_search.py

    # Run specific experiments
    python run_hyperparameter_search.py --experiments simple_mlp clip_only

    # Use custom CSV file
    python run_hyperparameter_search.py --config configs/my_experiments.csv

    # Resume from previous run
    python run_hyperparameter_search.py --resume
"""

import argparse
import subprocess
import json
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


def load_experiments_from_csv(csv_path: str) -> pd.DataFrame:
    """Load experiment configurations from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Experiment config CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['experiment_name', 'mlp_hidden_dims', 'batch_size', 'learning_rate', 'max_epochs']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    print(f"Loaded {len(df)} experiments from {csv_path}")
    return df


def parse_mlp_dims(dims_str: str) -> List[str]:
    """Parse MLP dimensions from CSV string (e.g., '256,128' -> ['256', '128'])."""
    if pd.isna(dims_str) or dims_str == '':
        return []
    return [str(int(float(d))) for d in str(dims_str).split(',')]


def build_command(exp_row: pd.Series, output_dir: Path) -> List[str]:
    """Build training command from experiment row."""
    cmd = [sys.executable, "train_multifeature_ranker.py"]

    # MLP hidden dims
    mlp_dims = parse_mlp_dims(exp_row['mlp_hidden_dims'])
    if mlp_dims:
        cmd.append("--mlp_hidden_dims")
        cmd.extend(mlp_dims)

    # Batch size
    cmd.extend(["--batch_size", str(int(exp_row['batch_size']))])

    # Learning rate
    cmd.extend(["--learning_rate", str(float(exp_row['learning_rate']))])

    # Max epochs
    cmd.extend(["--max_epochs", str(int(exp_row['max_epochs']))])

    # Optional: Dropout
    if 'dropout' in exp_row and not pd.isna(exp_row['dropout']):
        cmd.extend(["--dropout", str(float(exp_row['dropout']))])

    # Optional: Feature flags
    if 'use_clip' in exp_row and not pd.isna(exp_row['use_clip']):
        cmd.extend(["--use_clip", str(exp_row['use_clip']).lower()])

    if 'use_cnn_features' in exp_row and not pd.isna(exp_row['use_cnn_features']):
        cmd.extend(["--use_cnn_features", str(exp_row['use_cnn_features']).lower()])

    if 'use_iqa_features' in exp_row and not pd.isna(exp_row['use_iqa_features']):
        cmd.extend(["--use_iqa_features", str(exp_row['use_iqa_features']).lower()])

    # Comparison mode (diff_only or full)
    if 'comparison_mode' in exp_row and not pd.isna(exp_row['comparison_mode']):
        cmd.extend(["--comparison_mode", str(exp_row['comparison_mode'])])

    # Optional: CNN backbone and layer
    if 'cnn_backbone' in exp_row and not pd.isna(exp_row['cnn_backbone']):
        cmd.extend(["--cnn_backbone", str(exp_row['cnn_backbone'])])

    if 'cnn_layer' in exp_row and not pd.isna(exp_row['cnn_layer']):
        cmd.extend(["--cnn_layer", str(exp_row['cnn_layer'])])

    # Optional: Activation function
    if 'activation' in exp_row and not pd.isna(exp_row['activation']):
        cmd.extend(["--activation", str(exp_row['activation'])])

    # Optional: Optimizer settings
    if 'optimizer' in exp_row and not pd.isna(exp_row['optimizer']):
        cmd.extend(["--optimizer", str(exp_row['optimizer'])])

    if 'momentum' in exp_row and not pd.isna(exp_row['momentum']):
        cmd.extend(["--momentum", str(float(exp_row['momentum']))])

    if 'weight_decay' in exp_row and not pd.isna(exp_row['weight_decay']):
        cmd.extend(["--weight_decay", str(float(exp_row['weight_decay']))])

    # Optional: Use visual tower
    if 'use_visual_tower' in exp_row and not pd.isna(exp_row['use_visual_tower']):
        cmd.extend(["--use_visual_tower", str(exp_row['use_visual_tower']).lower()])

    # Optional: Use LayerNorm
    if 'use_layernorm' in exp_row and not pd.isna(exp_row['use_layernorm']):
        cmd.extend(["--use_layernorm", str(exp_row['use_layernorm']).lower()])

    # Optional: Use feature normalization
    if 'use_feature_normalization' in exp_row and not pd.isna(exp_row['use_feature_normalization']):
        cmd.extend(["--use_feature_normalization", str(exp_row['use_feature_normalization']).lower()])

    # Random seed for train/val/test split
    if 'seed' in exp_row and not pd.isna(exp_row['seed']):
        cmd.extend(["--seed", str(int(exp_row['seed']))])

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_output_dir = output_dir / f"{exp_row['experiment_name']}_{timestamp}"
    cmd.extend(["--output_dir", str(exp_output_dir)])

    return cmd


def extract_training_metrics(output_dir: Path) -> Dict[str, Any]:
    """
    Extract training metrics from experiment output directory.

    Reads from structured data (checkpoint, test_results.json) instead of parsing logs.
    Logs are only used as fallback if structured data is missing.
    """
    metrics = {
        'best_train_acc': None,
        'best_train_loss': None,
        'best_val_acc': None,
        'best_val_loss': None,
        'final_epoch': None,
        'test_acc': None,
        'test_loss': None,
        # Architecture metadata
        'input_dim': None,
        'clip_dim': None,
        'cnn_dim': None,
        'iqa_dim': None,
        'total_parameters': None,
        'architecture_summary': None,
    }

    # Read best model checkpoint (PRIMARY SOURCE)
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            metrics['best_val_acc'] = checkpoint.get('val_accuracy')
            metrics['best_val_loss'] = checkpoint.get('val_loss')
            metrics['final_epoch'] = checkpoint.get('epoch')

            # Extract architecture metadata if available
            arch = checkpoint.get('architecture', {})
            if arch:
                metrics['input_dim'] = arch.get('input_dim')
                metrics['clip_dim'] = arch.get('clip_dim')
                metrics['cnn_dim'] = arch.get('cnn_dim')
                metrics['iqa_dim'] = arch.get('iqa_dim')
                metrics['total_parameters'] = arch.get('total_parameters')
                metrics['architecture_summary'] = arch.get('architecture_summary')
        except Exception as e:
            logger.warning(f"Failed to read checkpoint: {e}")

    # Read test results (SECONDARY SOURCE)
    test_results_path = output_dir / 'test_results.json'
    if test_results_path.exists():
        try:
            with open(test_results_path) as f:
                test_results = json.load(f)
            metrics['test_acc'] = test_results.get('accuracy')
            metrics['test_loss'] = test_results.get('loss')

            # Architecture metadata might also be in test results
            if 'architecture' in test_results and not metrics['input_dim']:
                arch = test_results['architecture']
                metrics['input_dim'] = arch.get('input_dim')
                metrics['clip_dim'] = arch.get('clip_dim')
                metrics['cnn_dim'] = arch.get('cnn_dim')
                metrics['iqa_dim'] = arch.get('iqa_dim')
                metrics['total_parameters'] = arch.get('total_parameters')
                metrics['architecture_summary'] = arch.get('architecture_summary')
        except Exception as e:
            logger.warning(f"Failed to read test results: {e}")

    # FALLBACK: Parse training log only if checkpoint didn't have the data
    if metrics['best_val_acc'] is None or metrics['final_epoch'] is None:
        training_log_path = output_dir / 'training.log'
        if training_log_path.exists():
            try:
                with open(training_log_path, encoding='utf-8') as f:
                    log_content = f.read()

                # Parse log for best model save
                for line in log_content.split('\n'):
                    if 'New best model saved' in line and 'val_acc:' in line:
                        try:
                            val_acc_str = line.split('val_acc:')[1].strip().rstrip(')')
                            if metrics['best_val_acc'] is None:
                                metrics['best_val_acc'] = float(val_acc_str)
                        except:
                            pass

                    if 'Early stopping triggered after' in line:
                        try:
                            epoch_str = line.split('after')[1].strip().split()[0]
                            if metrics['final_epoch'] is None:
                                metrics['final_epoch'] = int(epoch_str)
                        except:
                            pass
            except Exception as e:
                logger.warning(f"Failed to parse training log: {e}")

    return metrics


def run_experiment(exp_row: pd.Series, output_base: Path) -> Dict[str, Any]:
    """Run a single training experiment."""
    exp_name = exp_row['experiment_name']

    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    if 'notes' in exp_row and not pd.isna(exp_row['notes']):
        print(f"Notes: {exp_row['notes']}")
    print(f"{'='*80}\n")

    # Build command
    cmd = build_command(exp_row, output_base)

    print(f"Command: {' '.join(cmd)}\n")

    # Run training
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        success = True
        error = None
    except subprocess.CalledProcessError as e:
        print(f"\nExperiment FAILED: {e}")
        success = False
        error = str(e)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        raise

    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()

    # Extract output directory from command
    output_dir_idx = cmd.index("--output_dir") + 1
    output_dir = Path(cmd[output_dir_idx])

    # Extract metrics
    metrics = extract_training_metrics(output_dir) if success else {}

    # Build result record
    result_record = {
        'experiment_name': exp_name,
        'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'success': success,
        'error': error,
        'duration_seconds': duration_seconds,
        'duration_minutes': duration_seconds / 60,
        'output_dir': str(output_dir),

        # Configuration
        'mlp_hidden_dims': exp_row['mlp_hidden_dims'],
        'batch_size': int(exp_row['batch_size']),
        'learning_rate': float(exp_row['learning_rate']),
        'max_epochs': int(exp_row['max_epochs']),
        'dropout': float(exp_row['dropout']) if 'dropout' in exp_row and not pd.isna(exp_row['dropout']) else None,

        # Feature flags (from CSV)
        'use_clip': exp_row.get('use_clip'),
        'use_cnn_features': exp_row.get('use_cnn_features'),
        'use_iqa_features': exp_row.get('use_iqa_features'),
        'comparison_mode': exp_row.get('comparison_mode', 'diff_only'),
        'seed': exp_row.get('seed', 42),

        # Architecture metadata (extracted from checkpoint/results)
        'input_dim': metrics.get('input_dim'),
        'clip_dim': metrics.get('clip_dim'),
        'cnn_dim': metrics.get('cnn_dim'),
        'iqa_dim': metrics.get('iqa_dim'),
        'total_parameters': metrics.get('total_parameters'),
        'architecture_summary': metrics.get('architecture_summary'),

        # Training metrics
        'final_epoch': metrics.get('final_epoch'),
        'best_val_acc': metrics.get('best_val_acc'),
        'best_val_loss': metrics.get('best_val_loss'),

        # Test metrics
        'test_acc': metrics.get('test_acc'),
        'test_loss': metrics.get('test_loss'),

        # Notes
        'notes': exp_row.get('notes', ''),
    }

    return result_record


def save_results(results: List[Dict], output_file: Path):
    """Save results to CSV file."""
    if not results:
        print("No results to save")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by test accuracy (best first)
    if 'test_acc' in df.columns:
        df = df.sort_values('test_acc', ascending=False, na_position='last')

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Select columns for display
    display_cols = [
        'experiment_name', 'test_acc', 'best_val_acc', 'final_epoch',
        'duration_minutes', 'success'
    ]
    display_df = df[[col for col in display_cols if col in df.columns]].copy()

    # Format display
    if 'test_acc' in display_df.columns:
        display_df['test_acc'] = display_df['test_acc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    if 'best_val_acc' in display_df.columns:
        display_df['best_val_acc'] = display_df['best_val_acc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    if 'duration_minutes' in display_df.columns:
        display_df['duration_minutes'] = display_df['duration_minutes'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))
    print()

    # Print best configuration
    if 'test_acc' in df.columns and len(df) > 0:
        best_idx = df['test_acc'].idxmax()
        if pd.notna(df.loc[best_idx, 'test_acc']):
            best = df.loc[best_idx]
            print(f"{'='*80}")
            print("BEST CONFIGURATION")
            print(f"{'='*80}")
            print(f"Experiment: {best['experiment_name']}")
            print(f"Test Accuracy: {best['test_acc']:.4f}")
            print(f"Test Loss: {best['test_loss']:.4f}")
            print(f"Best Val Accuracy: {best['best_val_acc']:.4f}")
            print(f"Duration: {best['duration_minutes']:.1f} min")
            print(f"Output: {best['output_dir']}")
            print(f"Config: MLP={best['mlp_hidden_dims']}, BS={best['batch_size']}, LR={best['learning_rate']}")
            print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter search from CSV config')

    parser.add_argument('--config', type=str,
                        default='configs/hyperparameter_experiments.csv',
                        help='Path to CSV file with experiment configurations')

    parser.add_argument('--experiments', type=str, nargs='+',
                        help='Specific experiments to run (experiment_name from CSV)')

    parser.add_argument('--output_base', type=str,
                        default='outputs/phototriage_multifeature/hyperparameter_search',
                        help='Base directory for experiment outputs')

    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run (skip completed experiments)')

    args = parser.parse_args()

    # Load experiments from CSV
    try:
        experiments_df = load_experiments_from_csv(args.config)
    except Exception as e:
        print(f"Error loading experiments CSV: {e}")
        sys.exit(1)

    # Filter experiments if specified
    if args.experiments:
        experiments_df = experiments_df[experiments_df['experiment_name'].isin(args.experiments)]
        if len(experiments_df) == 0:
            print(f"No experiments found matching: {args.experiments}")
            sys.exit(1)

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH")
    print(f"{'='*80}")
    print(f"Config file: {args.config}")
    print(f"Experiments to run: {len(experiments_df)}")
    print(f"Output base: {args.output_base}")
    print(f"{'='*80}\n")

    # Create output directory
    output_base = Path(args.output_base)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_dir = output_base / f"search_{timestamp}"
    search_dir.mkdir(parents=True, exist_ok=True)

    # Results file
    results_file = search_dir / "results.csv"

    # Load previous results if resuming
    completed_experiments = set()
    all_results = []

    if args.resume and results_file.exists():
        previous_df = pd.read_csv(results_file)
        completed_experiments = set(previous_df[previous_df['success'] == True]['experiment_name'])
        all_results = previous_df.to_dict('records')
        print(f"Resuming: {len(completed_experiments)} experiments already completed\n")

    # Run experiments
    for idx, (_, exp_row) in enumerate(experiments_df.iterrows(), 1):
        exp_name = exp_row['experiment_name']

        if exp_name in completed_experiments:
            print(f"[{idx}/{len(experiments_df)}] Skipping {exp_name} (already completed)")
            continue

        print(f"\n[{idx}/{len(experiments_df)}] Running experiment: {exp_name}")

        try:
            result = run_experiment(exp_row, search_dir)
            all_results.append(result)

            # Save intermediate results after each experiment
            save_results(all_results, results_file)

        except KeyboardInterrupt:
            print("\n\nSearch interrupted by user. Saving partial results...")
            save_results(all_results, results_file)
            break

    # Final summary
    if all_results:
        print(f"\n{'='*80}")
        print("HYPERPARAMETER SEARCH COMPLETE")
        print(f"{'='*80}")
        print(f"Total experiments: {len(all_results)}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}\n")
    else:
        print("\nNo experiments completed.")


if __name__ == "__main__":
    # Import torch only if needed (for loading checkpoints)
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not available, some metrics extraction may fail")
        torch = None

    main()
