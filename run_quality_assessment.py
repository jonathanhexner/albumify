#!/usr/bin/env python
"""
Run image quality assessment experiments on PhotoTriage or other datasets.

Examples:
    # Rule-based method
    python run_quality_assessment.py --method rule_based --dataset phototriage
    
    # CNN method (NIMA)
    python run_quality_assessment.py --method nima --dataset phototriage --device cuda
    
    # Transformer method (ViT)
    python run_quality_assessment.py --method vit --dataset phototriage --device cuda
    
    # Compare all methods
    python run_quality_assessment.py --compare-all --dataset phototriage
"""

import argparse
import yaml
from pathlib import Path
import torch

from sim_bench.datasets import load_dataset
from sim_bench.quality_assessment.registry import create_quality_assessor
from sim_bench.quality_assessment.evaluator import QualityEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate image quality assessment methods"
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['rule_based', 'nima', 'vit'],
        help='Quality assessment method to use'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='phototriage',
        help='Dataset name'
    )
    
    parser.add_argument(
        '--dataset-config',
        type=str,
        help='Path to dataset config (default: configs/dataset.{dataset}.yaml)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='mobilenet_v2',
        help='CNN/ViT backbone (for nima/vit methods)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to fine-tuned model weights'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all methods'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results JSON'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show progress bars'
    )
    
    return parser.parse_args()


def load_dataset_from_config(dataset_name: str, config_path: str = None):
    """Load dataset from configuration."""
    if config_path is None:
        config_path = f"configs/dataset.{dataset_name}.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, dataset_config)
    dataset.load_data()
    
    return dataset


def create_method(args):
    """Create quality assessment method from arguments."""
    config = {
        'device': args.device,
        'batch_size': args.batch_size
    }
    
    if args.method in ['nima', 'vit']:
        config['backbone'] = args.backbone if args.method == 'nima' else args.backbone.replace('mobilenet_v2', 'google/vit-base-patch16-224')
        config['weights_path'] = args.weights

    # Create using registry
    config['type'] = args.method
    method = create_quality_assessor(config)
    
    return method


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load dataset
    dataset = load_dataset_from_config(args.dataset, args.dataset_config)
    
    if args.compare_all:
        # Compare all methods
        print("\n" + "="*80)
        print("Comparing All Quality Assessment Methods")
        print("="*80 + "\n")
        
        methods = []
        
        # Rule-based
        print("Loading rule-based method...")
        methods.append(('Rule-Based', create_quality_assessor({'type': 'rule_based'})))

        # NIMA
        try:
            print("Loading NIMA (MobileNetV2)...")
            methods.append((
                'NIMA (MobileNetV2)',
                create_quality_assessor({
                    'type': 'nima',
                    'backbone': 'mobilenet_v2',
                    'device': args.device,
                    'batch_size': args.batch_size
                })
            ))
        except Exception as e:
            print(f"Could not load NIMA: {e}")

        # ViT
        try:
            print("Loading ViT...")
            methods.append((
                'ViT (Base)',
                create_quality_assessor({
                    'type': 'vit',
                    'model_name': 'google/vit-base-patch16-224',
                    'device': args.device,
                    'batch_size': max(1, args.batch_size // 2)  # ViT uses more memory
                })
            ))
        except Exception as e:
            print(f"Could not load ViT: {e}")
        
        # Run comparison
        results = QualityEvaluator.compare_methods(
            dataset,
            methods,
            verbose=args.verbose
        )
        
        # Save results if requested
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    'dataset': args.dataset,
                    'results': {name: res['metrics'] for name, res in results.items()}
                }, f, indent=2)
            
            print(f"\nResults saved to {output_path}")
    
    else:
        # Single method evaluation
        if args.method is None:
            print("Error: Either --method or --compare-all must be specified")
            return
        
        print(f"\nCreating {args.method} method...")
        method = create_method(args)
        
        print(f"\nEvaluating {args.method} on {args.dataset}...")
        evaluator = QualityEvaluator(dataset, method)
        results = evaluator.evaluate(verbose=args.verbose)
        evaluator.print_results()
        
        # Save results if requested
        if args.output:
            evaluator.save_results(args.output)


if __name__ == '__main__':
    main()





