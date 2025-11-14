#!/usr/bin/env python
"""
Test script for quality assessment benchmark framework.
"""

import yaml
from pathlib import Path
import tempfile

from sim_bench.quality_assessment.benchmark import QualityBenchmark


def test_quick_benchmark():
    """Test benchmark with minimal config."""
    
    # Create minimal test config
    config = {
        'datasets': [
            {
                'name': 'phototriage',
                'config': 'configs/dataset.phototriage.yaml',
                'sampling': {
                    'strategy': 'random',
                    'num_series': 5,  # Very small for testing
                    'seed': 42
                }
            }
        ],
        'methods': [
            {
                'name': 'sharpness_only',
                'type': 'rule_based',
                'config': {
                    'weights': {
                        'sharpness': 1.0,
                        'exposure': 0.0,
                        'colorfulness': 0.0,
                        'contrast': 0.0,
                        'noise': 0.0
                    }
                }
            },
            {
                'name': 'composite',
                'type': 'rule_based',
                'config': {
                    'weights': {
                        'sharpness': 0.3,
                        'exposure': 0.2,
                        'colorfulness': 0.2,
                        'contrast': 0.15,
                        'noise': 0.15
                    }
                }
            }
        ],
        'settings': {
            'verbose': True
        }
    }
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nRunning test benchmark...")
        print(f"Output: {tmpdir}\n")
        
        # Run benchmark
        benchmark = QualityBenchmark(config, output_dir=tmpdir)
        results = benchmark.run()
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        # Verify results structure
        assert 'summary' in results
        assert 'detailed_results' in results
        assert 'config' in results
        assert 'run_dir' in results
        
        summary = results['summary']
        
        # Verify summary structure
        assert 'benchmark_info' in summary
        assert 'datasets' in summary
        assert 'methods' in summary
        assert 'comparison' in summary
        
        # Verify methods were run
        assert 'phototriage' in summary['datasets']
        assert 'sharpness_only' in summary['datasets']['phototriage']['methods']
        assert 'composite' in summary['datasets']['phototriage']['methods']
        
        # Print key results
        print("\nMethod Performance:")
        for method_name, method_data in summary['methods'].items():
            print(f"\n{method_name}:")
            print(f"  Top-1 Accuracy: {method_data['avg_top1_accuracy']:.4f}")
            print(f"  Top-2 Accuracy: {method_data['avg_top2_accuracy']:.4f}")
            print(f"  Avg Time: {method_data['avg_time_ms']:.2f}ms")
        
        # Print rankings
        print("\n\nAccuracy Ranking:")
        for i, item in enumerate(summary['comparison']['accuracy_ranking'], 1):
            print(f"  {i}. {item['method']}: {item['accuracy']:.4f}")
        
        print("\nSpeed Ranking:")
        for i, item in enumerate(summary['comparison']['speed_ranking'], 1):
            print(f"  {i}. {item['method']}: {item['time_ms']:.2f}ms")
        
        print("\nEfficiency Ranking:")
        for i, item in enumerate(summary['comparison']['efficiency_ranking'], 1):
            print(f"  {i}. {item['method']}: {item['efficiency']:.4f}")
        
        print("\n" + "="*80)
        print("TEST PASSED ✓")
        print("="*80)


if __name__ == '__main__':
    test_quick_benchmark()


