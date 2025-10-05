"""
Unified command-line interface for sim-bench.
Single command with comma-separated lists for methods and datasets.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

from sim_bench.experiment_runner import ExperimentRunner, BenchmarkRunner


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def get_available_methods() -> List[str]:
    """Get list of available methods by scanning config directory."""
    methods_dir = Path("configs/methods")
    if not methods_dir.exists():
        return ['chi_square', 'emd', 'deep', 'sift_bovw']  # fallback
    
    methods = []
    for config_file in methods_dir.glob("*.yaml"):
        methods.append(config_file.stem)
    return sorted(methods)


def get_available_datasets() -> List[str]:
    """Get list of available datasets by scanning config directory."""
    configs_dir = Path("configs")
    if not configs_dir.exists():
        return ['ukbench', 'holidays']  # fallback
    
    datasets = []
    for config_file in configs_dir.glob("dataset.*.yaml"):
        # Extract dataset name from "dataset.NAME.yaml"
        dataset_name = config_file.stem.replace("dataset.", "")
        datasets.append(dataset_name)
    return sorted(datasets)


def parse_comma_list(value: str) -> List[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def run_unified_benchmark(args) -> None:
    """
    Run unified benchmark with flexible method and dataset selection.
    
    Args:
        args: Command line arguments
    """
    try:
        # Parse methods
        if args.methods:
            methods = parse_comma_list(args.methods)
        else:
            methods = get_available_methods()
            print(f"No methods specified, running all available: {', '.join(methods)}")
        
        # Parse datasets  
        if args.datasets:
            datasets = parse_comma_list(args.datasets)
        else:
            datasets = get_available_datasets()
            print(f"No datasets specified, running all available: {', '.join(datasets)}")
        
        # Load run configuration
        run_config = load_yaml_config(args.run_config)
        
        print("=" * 60)
        print("SIM-BENCH EVALUATION")
        print("=" * 60)
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Methods: {', '.join(methods)}")
        print("=" * 60)
        
        # Single dataset, single method - use simple runner
        if len(datasets) == 1 and len(methods) == 1:
            dataset_name = datasets[0]
            method_name = methods[0]
            
            dataset_config_path = f"configs/dataset.{dataset_name}.yaml"
            dataset_config = load_yaml_config(dataset_config_path)
            
            experiment_runner = ExperimentRunner(run_config, dataset_config)
            experiment_runner.run_single_method(method_name)
            
            print(f"\nResults saved to: {experiment_runner.get_output_directory()}")
        
        # Single dataset, multiple methods - use method runner
        elif len(datasets) == 1 and len(methods) > 1:
            dataset_name = datasets[0]
            
            dataset_config_path = f"configs/dataset.{dataset_name}.yaml"
            dataset_config = load_yaml_config(dataset_config_path)
            
            # Update run config with methods
            run_config['methods'] = methods
            
            experiment_runner = ExperimentRunner(run_config, dataset_config)
            experiment_runner.run_multiple_methods(methods)
            
            print(f"\nResults saved to: {experiment_runner.get_output_directory()}")
        
        # Multiple datasets - use comprehensive benchmark
        else:
            # Create benchmark configuration
            benchmark_config = {
                'output_dir': run_config.get('output_dir', 'outputs'),
                'run_name': 'unified_benchmark',
                'methods': methods,
                'datasets': [],
                'logging': run_config.get('logging', {}),
                'save': run_config.get('save', {}),
                'random_seed': run_config.get('random_seed', 42)
            }
            
            # Add dataset configurations
            for dataset_name in datasets:
                dataset_config_path = f"configs/dataset.{dataset_name}.yaml"
                dataset_config = {
                    'name': dataset_name,
                    'config': dataset_config_path,
                    'sampling': run_config.get('sampling', {})
                }
                benchmark_config['datasets'].append(dataset_config)
                
                # Add dataset-specific settings if they exist in run config
                if dataset_name in run_config:
                    benchmark_config[dataset_name] = run_config[dataset_name]
            
            benchmark_runner = BenchmarkRunner(benchmark_config)
            results = benchmark_runner.run_comprehensive_benchmark()
            
            if not results:
                print("No results generated!")
        
    except Exception as error:
        print(f"Error running benchmark: {error}")
        raise


def create_argument_parser() -> argparse.ArgumentParser:
    """Create unified argument parser."""
    
    available_methods = get_available_methods()
    available_datasets = get_available_datasets()
    
    parser = argparse.ArgumentParser(
        description="Simple image similarity benchmark - unified interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Methods: {', '.join(available_methods)}
Available Datasets: {', '.join(available_datasets)}

Examples:
  # Single method, single dataset
  python -m sim_bench.cli --methods chi_square --datasets ukbench
  
  # Multiple methods, single dataset  
  python -m sim_bench.cli --methods chi_square,emd,deep --datasets ukbench
  
  # Single method, multiple datasets
  python -m sim_bench.cli --methods chi_square --datasets ukbench,holidays
  
  # Multiple methods, multiple datasets
  python -m sim_bench.cli --methods chi_square,emd --datasets ukbench,holidays
  
  # All methods, all datasets (default)
  python -m sim_bench.cli
  
  # All methods, specific dataset
  python -m sim_bench.cli --datasets ukbench
  
  # Specific methods, all datasets  
  python -m sim_bench.cli --methods chi_square,emd
        """
    )
    
    parser.add_argument(
        '--methods', '-m',
        type=str,
        help=f"Comma-separated list of methods to run. Available: {', '.join(available_methods)}. Default: all methods"
    )
    
    parser.add_argument(
        '--datasets', '-d', 
        type=str,
        help=f"Comma-separated list of datasets to run. Available: {', '.join(available_datasets)}. Default: all datasets"
    )
    
    parser.add_argument(
        '--run-config', '-c',
        type=str,
        default='configs/run.yaml',
        help="Path to run configuration YAML file (default: configs/run.yaml)"
    )
    
    return parser


def main() -> None:
    """Main entry point for the unified CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    run_unified_benchmark(args)


if __name__ == '__main__':
    main()
