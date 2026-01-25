"""
Sanity check for Phase 2 multitask pretraining implementation.

Tests:
1. Model instantiation and forward pass
2. Uncertainty weighting loss computation
3. Dataset loading (with synthetic data)
4. Full training step simulation

Usage:
    python -m sim_bench.training.phase2_pretraining.sanity_check
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def test_model():
    """Test model instantiation and forward pass."""
    print("=" * 60)
    print("Test 1: Model instantiation and forward pass")
    print("=" * 60)

    from sim_bench.training.phase2_pretraining.multitask_model import MultitaskFaceModel

    config = {
        'model': {
            'backbone': 'resnet18',  # Use smaller backbone for faster testing
            'embedding_dim': 512,
            'num_expression_classes': 8,
            'num_landmarks': 5,
            'dropout': 0.0,
            'use_uncertainty_weighting': True
        }
    }

    model = MultitaskFaceModel(config)
    print(f"  Model created: {type(model).__name__}")
    print(f"  Backbone: {model.backbone_name}")
    print(f"  Embedding dim: {model.embedding_dim}")
    print(f"  Num landmarks: {model.num_landmarks}")
    print(f"  Uncertainty weighting: {model.use_uncertainty_weighting}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"\n  Forward pass successful!")
    print(f"  Expression logits shape: {outputs['expression_logits'].shape}")  # Should be [4, 8]
    print(f"  Landmarks shape: {outputs['landmarks'].shape}")  # Should be [4, 5, 2]
    print(f"  Features shape: {outputs['features'].shape}")  # Should be [4, 512]

    assert outputs['expression_logits'].shape == (batch_size, 8), "Expression logits shape mismatch"
    assert outputs['landmarks'].shape == (batch_size, 5, 2), "Landmarks shape mismatch"
    assert outputs['features'].shape == (batch_size, 512), "Features shape mismatch"

    print("\n  [PASS] Model test passed!")
    return True


def test_uncertainty_weighting():
    """Test uncertainty weighting loss computation."""
    print("\n" + "=" * 60)
    print("Test 2: Uncertainty weighting")
    print("=" * 60)

    from sim_bench.training.phase2_pretraining.multitask_model import UncertaintyWeighting

    uw = UncertaintyWeighting(num_tasks=2)
    print(f"  Initial log_vars: {uw.log_vars.data}")

    # Simulate losses
    expr_loss = torch.tensor(2.0)
    lm_loss = torch.tensor(0.01)

    total_loss, weighted_losses = uw([expr_loss, lm_loss])

    print(f"\n  Input losses: expression={expr_loss.item():.4f}, landmark={lm_loss.item():.4f}")
    print(f"  Weighted losses: {[w.item() for w in weighted_losses]}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Verify total loss is positive
    assert total_loss.item() > 0, f"Total loss should be positive, got {total_loss.item()}"
    print(f"  Total loss is positive: {total_loss.item() > 0}")

    # Test get_weights method
    weights = uw.get_weights()
    print(f"  Uncertainty weights: {weights}")
    assert 'expression_weight' in weights, "get_weights should return expression_weight"
    assert 'landmark_weight' in weights, "get_weights should return landmark_weight"

    # Verify gradients flow through
    total_loss.backward()
    print(f"  log_vars gradients: {uw.log_vars.grad}")

    assert uw.log_vars.grad is not None, "Gradients should flow through uncertainty params"

    # Test that loss stays positive even with extreme log_var values
    print("\n  Testing loss stays positive with extreme values...")
    uw2 = UncertaintyWeighting(num_tasks=2)
    uw2.log_vars.data = torch.tensor([-10.0, -10.0])  # Very negative (would cause old bug)

    total_loss2, _ = uw2([expr_loss, lm_loss])
    print(f"  With log_vars=[-10, -10]: total_loss = {total_loss2.item():.4f}")
    assert total_loss2.item() > 0, f"Loss should stay positive even with extreme log_vars, got {total_loss2.item()}"

    print("\n  [PASS] Uncertainty weighting test passed!")
    return True


def test_loss_computation():
    """Test full loss computation with model."""
    print("\n" + "=" * 60)
    print("Test 3: Loss computation")
    print("=" * 60)

    from sim_bench.training.phase2_pretraining.multitask_model import MultitaskFaceModel
    from sim_bench.training.phase2_pretraining.train_multitask import compute_losses

    config = {
        'model': {
            'backbone': 'resnet18',
            'embedding_dim': 512,
            'num_expression_classes': 8,
            'num_landmarks': 5,
            'dropout': 0.0,
            'use_uncertainty_weighting': True
        },
        'training': {
            'expression_weight': 1.0,
            'landmark_weight': 1.0
        }
    }

    model = MultitaskFaceModel(config)
    batch_size = 4

    # Create dummy data
    images = torch.randn(batch_size, 3, 224, 224)
    expressions = torch.randint(0, 8, (batch_size,))
    landmarks = torch.rand(batch_size, 5, 2)  # Random landmarks in [0, 1]
    has_landmarks = torch.tensor([True, True, False, True])  # Some missing

    model.train()
    outputs = model(images)

    targets = {
        'expression': expressions,
        'landmarks': landmarks,
        'has_landmarks': has_landmarks
    }

    total_loss, expr_loss, lm_loss, weighted_losses = compute_losses(
        model, outputs, targets, 'cpu', config
    )

    print(f"  Expression loss: {expr_loss.item():.4f}")
    print(f"  Landmark loss: {lm_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Test backward pass
    total_loss.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"  Gradients computed: {has_grads}")

    assert has_grads, "Model should have gradients after backward"

    print("\n  [PASS] Loss computation test passed!")
    return True


def test_landmark_flip():
    """Test that horizontal flip correctly transforms landmarks."""
    print("\n" + "=" * 60)
    print("Test 4: Landmark horizontal flip")
    print("=" * 60)

    from sim_bench.training.phase2_pretraining.affectnet_dataset import FLIP_SWAP_5

    # Simulate 5-point landmarks: [left_eye, right_eye, nose, mouth_left, mouth_right]
    landmarks = np.array([
        [0.3, 0.3],  # left_eye at x=0.3
        [0.7, 0.3],  # right_eye at x=0.7
        [0.5, 0.5],  # nose at center
        [0.4, 0.7],  # mouth_left
        [0.6, 0.7],  # mouth_right
    ], dtype=np.float32)

    print(f"  Original landmarks:")
    for i, (x, y) in enumerate(landmarks):
        print(f"    Point {i}: ({x:.2f}, {y:.2f})")

    # Apply flip
    flipped = landmarks.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]  # Mirror x
    flipped = flipped[FLIP_SWAP_5]  # Swap left/right

    print(f"\n  Flipped landmarks:")
    for i, (x, y) in enumerate(flipped):
        print(f"    Point {i}: ({x:.2f}, {y:.2f})")

    # Verify: after flip, what was left_eye (0.3) should now be at right position (0.7 -> 0.3)
    # and what was right_eye (0.7) should be at left position (0.7 -> 0.3)
    # Actually: left_eye was at 0.3, after mirror becomes 0.7, then swaps to position 0
    # right_eye was at 0.7, after mirror becomes 0.3, then swaps to position 1

    # After flip: new left_eye = old right_eye mirrored = 1-0.7 = 0.3
    # After flip: new right_eye = old left_eye mirrored = 1-0.3 = 0.7
    expected_left_eye_x = 1.0 - 0.7  # Original right_eye mirrored
    expected_right_eye_x = 1.0 - 0.3  # Original left_eye mirrored

    print(f"\n  Verification:")
    print(f"    New left_eye x: {flipped[0, 0]:.2f} (expected: {expected_left_eye_x:.2f})")
    print(f"    New right_eye x: {flipped[1, 0]:.2f} (expected: {expected_right_eye_x:.2f})")

    assert abs(flipped[0, 0] - expected_left_eye_x) < 0.01, "Left eye flip incorrect"
    assert abs(flipped[1, 0] - expected_right_eye_x) < 0.01, "Right eye flip incorrect"

    print("\n  [PASS] Landmark flip test passed!")
    return True


def test_training_step():
    """Test a full training step simulation."""
    print("\n" + "=" * 60)
    print("Test 5: Full training step")
    print("=" * 60)

    from sim_bench.training.phase2_pretraining.multitask_model import MultitaskFaceModel
    from sim_bench.training.phase2_pretraining.train_multitask import compute_losses

    config = {
        'model': {
            'backbone': 'resnet18',
            'embedding_dim': 512,
            'num_expression_classes': 8,
            'num_landmarks': 5,
            'dropout': 0.0,
            'use_uncertainty_weighting': True
        },
        'training': {
            'expression_weight': 1.0,
            'landmark_weight': 1.0
        }
    }

    model = MultitaskFaceModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    batch_size = 4

    # Simulate 3 training steps
    print("  Simulating 3 training steps...")
    losses = []

    for step in range(3):
        images = torch.randn(batch_size, 3, 224, 224)
        expressions = torch.randint(0, 8, (batch_size,))
        landmarks = torch.rand(batch_size, 5, 2)
        has_landmarks = torch.ones(batch_size, dtype=torch.bool)

        optimizer.zero_grad()
        outputs = model(images)

        targets = {
            'expression': expressions,
            'landmarks': landmarks,
            'has_landmarks': has_landmarks
        }

        total_loss, expr_loss, lm_loss, _ = compute_losses(
            model, outputs, targets, 'cpu', config
        )

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        print(f"    Step {step + 1}: loss = {total_loss.item():.4f}")

    # Check uncertainty parameters changed
    log_vars = model.uncertainty.log_vars.data
    print(f"\n  Final log_vars: {log_vars.tolist()}")

    print("\n  [PASS] Training step test passed!")
    return True


def test_overfit():
    """Test that model can overfit to a small fixed batch."""
    print("\n" + "=" * 60)
    print("Test 6: Overfit test (verify model can learn)")
    print("=" * 60)

    from sim_bench.training.phase2_pretraining.multitask_model import MultitaskFaceModel
    from sim_bench.training.phase2_pretraining.train_multitask import compute_losses

    config = {
        'model': {
            'backbone': 'resnet18',
            'embedding_dim': 512,
            'num_expression_classes': 8,
            'num_landmarks': 5,
            'dropout': 0.0,
            'use_uncertainty_weighting': False  # Simpler for overfit test
        },
        'training': {
            'expression_weight': 1.0,
            'landmark_weight': 10.0  # Boost landmark loss for faster convergence
        }
    }

    model = MultitaskFaceModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 4
    num_steps = 200

    # Create FIXED batch (same data every iteration)
    torch.manual_seed(42)
    fixed_images = torch.randn(batch_size, 3, 224, 224)
    fixed_expressions = torch.tensor([0, 1, 2, 3])  # Different classes
    fixed_landmarks = torch.tensor([
        [[0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.4, 0.7], [0.6, 0.7]],
        [[0.35, 0.35], [0.65, 0.35], [0.5, 0.55], [0.45, 0.75], [0.55, 0.75]],
        [[0.25, 0.25], [0.75, 0.25], [0.5, 0.45], [0.35, 0.65], [0.65, 0.65]],
        [[0.32, 0.32], [0.68, 0.32], [0.5, 0.52], [0.42, 0.72], [0.58, 0.72]],
    ], dtype=torch.float32)
    fixed_has_landmarks = torch.ones(batch_size, dtype=torch.bool)

    targets = {
        'expression': fixed_expressions,
        'landmarks': fixed_landmarks,
        'has_landmarks': fixed_has_landmarks
    }

    print(f"  Training on fixed batch for {num_steps} steps...")
    print(f"  Target expressions: {fixed_expressions.tolist()}")

    model.train()
    initial_loss = None
    final_loss = None
    final_acc = None

    for step in range(num_steps):
        optimizer.zero_grad()
        outputs = model(fixed_images)

        total_loss, expr_loss, lm_loss, _ = compute_losses(
            model, outputs, targets, 'cpu', config
        )

        total_loss.backward()
        optimizer.step()

        # Calculate expression accuracy
        with torch.no_grad():
            _, preds = outputs['expression_logits'].max(1)
            acc = (preds == fixed_expressions).float().mean().item() * 100

        if step == 0:
            initial_loss = total_loss.item()
            print(f"    Step {step + 1:3d}: loss={total_loss.item():.4f}, expr_loss={expr_loss.item():.4f}, lm_loss={lm_loss.item():.4f}, acc={acc:.0f}%")

        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1:3d}: loss={total_loss.item():.4f}, expr_loss={expr_loss.item():.4f}, lm_loss={lm_loss.item():.4f}, acc={acc:.0f}%")

        final_loss = total_loss.item()
        final_acc = acc

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {(1 - final_loss/initial_loss) * 100:.1f}%")
    print(f"  Final expression accuracy: {final_acc:.0f}%")

    # Verify overfit success
    loss_reduced = final_loss < initial_loss * 0.5  # Loss should reduce by at least 50%
    accuracy_high = final_acc >= 75  # Should get at least 75% accuracy on 4 samples

    if loss_reduced and accuracy_high:
        print("\n  [PASS] Overfit test passed! Model can learn.")
        return True
    else:
        print(f"\n  [FAIL] Overfit test failed!")
        print(f"    Loss reduced enough: {loss_reduced} (need 50% reduction)")
        print(f"    Accuracy high enough: {accuracy_high} (need >= 75%)")
        return False


def main():
    print("\n" + "#" * 60)
    print("# Phase 2 Multitask Pretraining - Sanity Check")
    print("#" * 60)

    tests = [
        ("Model instantiation", test_model),
        ("Uncertainty weighting", test_uncertainty_weighting),
        ("Loss computation", test_loss_computation),
        ("Landmark flip", test_landmark_flip),
        ("Training step", test_training_step),
        ("Overfit test", test_overfit),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [FAIL] {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All sanity checks passed!")
        return 0
    else:
        print("Some sanity checks failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
