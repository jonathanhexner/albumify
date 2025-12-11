"""
Visualization tools for quality assessment benchmarks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import json


class BenchmarkVisualizer:
    """Create visualizations from benchmark results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Path to benchmark results directory
        """
        self.results_dir = Path(results_dir)
        
        # Load data
        self.summary = self._load_json('summary.json')
        self.detailed_df = pd.read_csv(self.results_dir / 'detailed_results.csv')
        self.methods_df = pd.read_csv(self.results_dir / 'methods_summary.csv')
        
        # Setup style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(self.results_dir / filename) as f:
            return json.load(f)
    
    def plot_accuracy_comparison(self, save_path: str = None):
        """Plot accuracy comparison across methods."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top-1 accuracy
        methods_sorted = self.methods_df.sort_values('avg_top1_accuracy', ascending=True)
        ax1.barh(methods_sorted['method'], methods_sorted['avg_top1_accuracy'])
        ax1.set_xlabel('Top-1 Accuracy')
        ax1.set_title('Method Comparison: Top-1 Accuracy')
        ax1.set_xlim(0, 1.0)
        
        # Add value labels
        for i, v in enumerate(methods_sorted['avg_top1_accuracy']):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Top-2 accuracy
        methods_sorted = self.methods_df.sort_values('avg_top2_accuracy', ascending=True)
        ax2.barh(methods_sorted['method'], methods_sorted['avg_top2_accuracy'])
        ax2.set_xlabel('Top-2 Accuracy')
        ax2.set_title('Method Comparison: Top-2 Accuracy')
        ax2.set_xlim(0, 1.0)
        
        for i, v in enumerate(methods_sorted['avg_top2_accuracy']):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_speed_comparison(self, save_path: str = None):
        """Plot speed comparison across methods."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods_sorted = self.methods_df.sort_values('avg_time_ms', ascending=True)
        bars = ax.barh(methods_sorted['method'], methods_sorted['avg_time_ms'])
        
        # Color bars by speed
        colors = ['green' if x < 10 else 'orange' if x < 50 else 'red' 
                  for x in methods_sorted['avg_time_ms']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Average Time (ms per image)')
        ax.set_title('Method Comparison: Processing Speed')
        ax.set_xscale('log')
        
        # Add value labels
        for i, v in enumerate(methods_sorted['avg_time_ms']):
            ax.text(v * 1.1, i, f'{v:.2f}ms', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'speed_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_accuracy_vs_speed(self, save_path: str = None):
        """Plot accuracy vs speed tradeoff."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            self.methods_df['avg_time_ms'],
            self.methods_df['avg_top1_accuracy'],
            s=200,
            alpha=0.6,
            c=range(len(self.methods_df)),
            cmap='viridis'
        )
        
        # Add method labels
        for idx, row in self.methods_df.iterrows():
            ax.annotate(
                row['method'],
                (row['avg_time_ms'], row['avg_top1_accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        
        ax.set_xlabel('Average Time (ms per image)')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_title('Accuracy vs Speed Tradeoff')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'accuracy_vs_speed.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_per_dataset_comparison(self, save_path: str = None):
        """Plot method performance per dataset."""
        datasets = self.detailed_df['dataset'].unique()
        
        fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4 * len(datasets)))
        
        if len(datasets) == 1:
            axes = [axes]
        
        for ax, dataset in zip(axes, datasets):
            dataset_data = self.detailed_df[self.detailed_df['dataset'] == dataset]
            dataset_data = dataset_data.sort_values('top1_accuracy', ascending=True)
            
            ax.barh(dataset_data['method'], dataset_data['top1_accuracy'])
            ax.set_xlabel('Top-1 Accuracy')
            ax.set_title(f'Dataset: {dataset}')
            ax.set_xlim(0, 1.0)
            
            # Add value labels
            for i, v in enumerate(dataset_data['top1_accuracy']):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.results_dir / 'per_dataset_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_all(self):
        """Generate all visualizations."""
        print(f"Generating visualizations in {self.results_dir}...")
        
        self.plot_accuracy_comparison()
        print("  - accuracy_comparison.png")
        
        self.plot_speed_comparison()
        print("  - speed_comparison.png")
        
        self.plot_accuracy_vs_speed()
        print("  - accuracy_vs_speed.png")
        
        self.plot_per_dataset_comparison()
        print("  - per_dataset_comparison.png")
        
        print("Done!")
    
    def generate_report(self, output_path: str = None):
        """Generate markdown report."""
        if output_path is None:
            output_path = self.results_dir / 'REPORT.md'
        
        with open(output_path, 'w') as f:
            f.write("# Quality Assessment Benchmark Report\n\n")
            
            # Overview
            info = self.summary['benchmark_info']
            f.write(f"**Date:** {info['timestamp']}\n\n")
            f.write(f"**Datasets:** {info['num_datasets']}\n\n")
            f.write(f"**Methods:** {info['num_methods']}\n\n")
            
            # Method rankings
            f.write("## Method Rankings\n\n")
            
            f.write("### By Accuracy\n\n")
            f.write("| Rank | Method | Top-1 Accuracy |\n")
            f.write("|------|--------|----------------|\n")
            for i, item in enumerate(self.summary['comparison']['accuracy_ranking'], 1):
                f.write(f"| {i} | {item['method']} | {item['accuracy']:.4f} ({item['accuracy']*100:.2f}%) |\n")
            
            f.write("\n### By Speed\n\n")
            f.write("| Rank | Method | Time (ms/image) |\n")
            f.write("|------|--------|----------------|\n")
            for i, item in enumerate(self.summary['comparison']['speed_ranking'], 1):
                f.write(f"| {i} | {item['method']} | {item['time_ms']:.2f} |\n")
            
            f.write("\n### By Efficiency\n\n")
            f.write("| Rank | Method | Accuracy | Time (ms) | Efficiency |\n")
            f.write("|------|--------|----------|-----------|------------|\n")
            for i, item in enumerate(self.summary['comparison']['efficiency_ranking'], 1):
                f.write(f"| {i} | {item['method']} | {item['accuracy']*100:.2f}% | "
                       f"{item['time_ms']:.2f} | {item['efficiency']:.4f} |\n")
            
            # Per-dataset results
            f.write("\n## Results by Dataset\n\n")
            for dataset_name, dataset_data in self.summary['datasets'].items():
                f.write(f"### {dataset_name}\n\n")
                f.write("| Method | Top-1 Acc | Top-2 Acc | MRR | Time (ms) |\n")
                f.write("|--------|-----------|-----------|-----|------------|\n")
                
                methods_sorted = sorted(
                    dataset_data['methods'].items(),
                    key=lambda x: x[1]['top1_accuracy'],
                    reverse=True
                )
                
                for method_name, method_data in methods_sorted:
                    f.write(f"| {method_name} | {method_data['top1_accuracy']:.4f} | "
                           f"{method_data['top2_accuracy']:.4f} | {method_data['mrr']:.4f} | "
                           f"{method_data['avg_time_ms']:.2f} |\n")
                
                f.write("\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("![Accuracy Comparison](accuracy_comparison.png)\n\n")
            f.write("![Speed Comparison](speed_comparison.png)\n\n")
            f.write("![Accuracy vs Speed](accuracy_vs_speed.png)\n\n")
            f.write("![Per-Dataset Comparison](per_dataset_comparison.png)\n\n")
        
        print(f"Report saved to: {output_path}")


def visualize_benchmark(results_dir: str):
    """
    Generate all visualizations and report for a benchmark.
    
    Args:
        results_dir: Path to benchmark results directory
    """
    viz = BenchmarkVisualizer(results_dir)
    viz.plot_all()
    viz.generate_report()







