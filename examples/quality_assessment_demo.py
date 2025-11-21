"""
Demo script showing how to use the quality assessment framework.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.quality_assessment import RuleBasedQuality, NIMAQuality, ViTQuality
from sim_bench.quality_assessment.evaluator import evaluate_on_phototriage


def demo_single_image():
    """Demo: Assess quality of a single image."""
    print("\n" + "="*60)
    print("Demo 1: Single Image Quality Assessment")
    print("="*60)
    
    # Example image path (replace with your own)
    image_path = "path/to/your/image.jpg"
    
    # Rule-based method
    print("\n1. Rule-Based Method:")
    rule_based = RuleBasedQuality()
    score = rule_based.assess_image(image_path)
    print(f"   Quality Score: {score:.4f}")
    
    # Get detailed breakdown
    detailed = rule_based.get_detailed_scores(image_path)
    print(f"   Sharpness: {detailed['sharpness_normalized']:.4f}")
    print(f"   Exposure: {detailed['exposure']:.4f}")
    print(f"   Colorfulness: {detailed['colorfulness_normalized']:.4f}")
    print(f"   Contrast: {detailed['contrast']:.4f}")


def demo_image_series():
    """Demo: Select best image from a series."""
    print("\n" + "="*60)
    print("Demo 2: Best Image Selection from Series")
    print("="*60)
    
    # Example series (replace with your own)
    series_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]
    
    # Rule-based selection
    print("\n1. Rule-Based Selection:")
    rule_based = RuleBasedQuality()
    result = rule_based.select_best_from_series(series_paths)
    print(f"   Best image: {result['best_path']}")
    print(f"   Score: {result['best_score']:.4f}")
    print(f"   All scores: {result['scores']}")


def demo_method_comparison():
    """Demo: Compare different methods on a series."""
    print("\n" + "="*60)
    print("Demo 3: Method Comparison")
    print("="*60)
    
    series_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]
    
    methods = {
        'Rule-Based': RuleBasedQuality(),
        'NIMA (MobileNet)': NIMAQuality(backbone='mobilenet_v2', device='cpu'),
    }
    
    print("\nComparing methods on image series:")
    for name, method in methods.items():
        result = method.select_best_from_series(series_paths)
        print(f"\n{name}:")
        print(f"  Selected: {Path(result['best_path']).name}")
        print(f"  Score: {result['best_score']:.4f}")


def demo_phototriage_evaluation():
    """Demo: Evaluate on PhotoTriage dataset."""
    print("\n" + "="*60)
    print("Demo 4: PhotoTriage Evaluation")
    print("="*60)
    
    # Rule-based evaluation
    print("\nEvaluating Rule-Based method on PhotoTriage:")
    method = RuleBasedQuality()
    results = evaluate_on_phototriage(method)
    
    print(f"\nTop-1 Accuracy: {results['metrics']['top1_accuracy']:.2%}")
    print(f"Mean Reciprocal Rank: {results['metrics']['mean_reciprocal_rank']:.4f}")


def demo_custom_weights():
    """Demo: Custom weighting of rule-based metrics."""
    print("\n" + "="*60)
    print("Demo 5: Custom Metric Weights")
    print("="*60)
    
    # Default weights
    default_method = RuleBasedQuality()
    
    # Custom weights (e.g., focus more on sharpness)
    custom_weights = {
        'sharpness': 0.60,      # Prioritize sharpness
        'exposure': 0.20,
        'colorfulness': 0.15,
        'contrast': 0.05
    }
    custom_method = RuleBasedQuality(weights=custom_weights)
    
    image_path = "path/to/your/image.jpg"
    
    print(f"\nDefault weights: {default_method.weights}")
    print(f"Score: {default_method.assess_image(image_path):.4f}")
    
    print(f"\nCustom weights: {custom_method.weights}")
    print(f"Score: {custom_method.assess_image(image_path):.4f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Image Quality Assessment Framework Demo")
    print("="*60)
    
    print("\nNote: Replace example paths with your actual image paths")
    print("Some demos require the PhotoTriage dataset to be configured")
    
    # Run demos (comment out those requiring specific data)
    # demo_single_image()
    # demo_image_series()
    # demo_method_comparison()
    # demo_phototriage_evaluation()
    # demo_custom_weights()
    
    print("\nTo run full evaluation on PhotoTriage:")
    print("  python run_quality_assessment.py --method rule_based --dataset phototriage")
    print("  python run_quality_assessment.py --compare-all --dataset phototriage")





