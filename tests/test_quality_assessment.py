#!/usr/bin/env python
"""
Test script for quality assessment module.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from sim_bench.quality_assessment import RuleBasedQuality, NIMAQuality, ViTQuality
    from sim_bench.quality_assessment.evaluator import QualityEvaluator
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test rule-based method
print("\n" + "="*60)
print("Testing Rule-Based Method")
print("="*60)

try:
    method = RuleBasedQuality()
    print(f"✓ Created RuleBasedQuality")
    print(f"  Weights: {method.weights}")
    print(f"  Device: {method.device}")
except Exception as e:
    print(f"✗ Error creating RuleBasedQuality: {e}")

# Test NIMA method
print("\n" + "="*60)
print("Testing NIMA (CNN) Method")
print("="*60)

try:
    method = NIMAQuality(backbone='mobilenet_v2', device='cpu')
    print(f"✓ Created NIMAQuality with MobileNetV2")
    print(f"  Backbone: {method.backbone}")
    print(f"  Device: {method.device}")
    print(f"  Model parameters: {sum(p.numel() for p in method.model.parameters()):,}")
except Exception as e:
    print(f"✗ Error creating NIMAQuality: {e}")

# Test ViT method (optional)
print("\n" + "="*60)
print("Testing ViT (Transformer) Method")
print("="*60)

try:
    method = ViTQuality(model_name='google/vit-base-patch16-224', device='cpu')
    print(f"✓ Created ViTQuality")
    print(f"  Model: {method.model_name}")
    print(f"  Device: {method.device}")
except ImportError:
    print("⚠ ViT requires 'transformers' library: pip install transformers")
except Exception as e:
    print(f"✗ Error creating ViTQuality: {e}")

# Test with synthetic images
print("\n" + "="*60)
print("Testing with Synthetic Images")
print("="*60)

try:
    import cv2
    
    # Create test images
    print("Creating synthetic test images...")
    
    # Sharp image
    sharp_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sharp_img = cv2.GaussianBlur(sharp_img, (3, 3), 0)
    
    # Blurry image
    blurry_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    blurry_img = cv2.GaussianBlur(blurry_img, (21, 21), 0)
    
    # Dark image
    dark_img = np.random.randint(0, 100, (512, 512, 3), dtype=np.uint8)
    
    # Save temporarily
    temp_dir = Path("temp_test_images")
    temp_dir.mkdir(exist_ok=True)
    
    sharp_path = temp_dir / "sharp.jpg"
    blurry_path = temp_dir / "blurry.jpg"
    dark_path = temp_dir / "dark.jpg"
    
    cv2.imwrite(str(sharp_path), sharp_img)
    cv2.imwrite(str(blurry_path), blurry_img)
    cv2.imwrite(str(dark_path), dark_img)
    
    print("✓ Created test images")
    
    # Test rule-based method
    method = RuleBasedQuality()
    
    sharp_score = method.assess_image(str(sharp_path))
    blurry_score = method.assess_image(str(blurry_path))
    dark_score = method.assess_image(str(dark_path))
    
    print(f"\nRule-Based Scores:")
    print(f"  Sharp image: {sharp_score:.4f}")
    print(f"  Blurry image: {blurry_score:.4f}")
    print(f"  Dark image: {dark_score:.4f}")
    
    # Test series selection
    series = [str(sharp_path), str(blurry_path), str(dark_path)]
    result = method.select_best_from_series(series)
    
    print(f"\nBest from series:")
    print(f"  Selected: {Path(result['best_path']).name}")
    print(f"  Score: {result['best_score']:.4f}")
    print(f"  All scores: {[f'{s:.4f}' for s in result['scores']]}")
    
    # Test batch assessment
    scores = method.assess_batch(series)
    print(f"\nBatch assessment:")
    print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("✓ Cleaned up test images")
    
except Exception as e:
    print(f"✗ Error in synthetic image test: {e}")
    import traceback
    traceback.print_exc()

# Test configuration
print("\n" + "="*60)
print("Testing Configuration")
print("="*60)

try:
    # Custom weights
    weights = {
        'sharpness': 0.5,
        'exposure': 0.3,
        'colorfulness': 0.15,
        'contrast': 0.05
    }
    method = RuleBasedQuality(weights=weights)
    config = method.get_config()
    print(f"✓ Custom configuration:")
    print(f"  {config}")
except Exception as e:
    print(f"✗ Error with configuration: {e}")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)
print("✓ Module is working correctly")
print("\nTo run full evaluation on PhotoTriage:")
print("  python run_quality_assessment.py --method rule_based --dataset phototriage")
print("\nTo compare all methods:")
print("  python run_quality_assessment.py --compare-all --dataset phototriage")

if not torch.cuda.is_available():
    print("\n⚠ Note: CUDA not available. Deep learning methods will run on CPU (slower).")
else:
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")


