"""
Quick inline check you can add to your train_siamese_e2e.py

Add this right before your inspect_model_output call to diagnose differences.
"""
import torch
import numpy as np
import hashlib


def quick_determinism_snapshot(model, loader, device, name="Run"):
    """
    Quick snapshot of determinism-critical state.

    Run this in both your runs and compare outputs.
    """
    print(f"\n{'='*70}")
    print(f"DETERMINISM SNAPSHOT: {name}")
    print(f"{'='*70}")

    # 1. Random state
    print("\n[1] Random State:")
    print(f"  PyTorch RNG (first 10): {torch.get_rng_state()[:10].tolist()}")
    print(f"  NumPy RNG (first 3): {np.random.get_state()[1][:3].tolist()}")

    # 2. Model initialization
    print("\n[2] Model First Layer:")
    first_param = next(model.parameters())
    print(f"  Shape: {list(first_param.shape)}")
    print(f"  Mean: {first_param.mean().item():.10f}")
    print(f"  Std: {first_param.std().item():.10f}")
    print(f"  First 5 values: {first_param.flatten()[:5].tolist()}")

    # 3. First batch
    print("\n[3] First Batch:")
    batch = next(iter(loader))

    # Files
    if 'image1' in batch:
        files1 = batch['image1'][:5]  # First 5
        files2 = batch['image2'][:5]
        print(f"  Batch size: {len(batch['image1'])}")
        print(f"  First file1: {files1[0]}")
        print(f"  First file2: {files2[0]}")

        # File order hash
        all_files = "|".join(batch['image1'])
        file_hash = hashlib.md5(all_files.encode()).hexdigest()
        print(f"  File order hash: {file_hash}")

    # Tensor values
    img1 = batch['img1']
    winners = batch['winner']
    print(f"  IMG1 shape: {list(img1.shape)}")
    print(f"  IMG1 mean: {img1.mean().item():.10f}")
    print(f"  IMG1 std: {img1.std().item():.10f}")
    print(f"  IMG1 first pixel: {img1[0, :, 0, 0].tolist()}")
    print(f"  Winners: {winners.tolist()}")

    # Tensor hash
    img_hash = hashlib.md5(img1.cpu().numpy().tobytes()).hexdigest()
    print(f"  IMG1 tensor hash: {img_hash}")

    # 4. Model output on first batch
    print("\n[4] Model Output:")
    model.eval()
    with torch.no_grad():
        logits = model(img1.to(device), batch['img2'].to(device))
        preds = logits.argmax(dim=-1)

    print(f"  Logits shape: {list(logits.shape)}")
    print(f"  First 3 logits: {logits[:3].tolist()}")
    print(f"  First 3 preds: {preds[:3].tolist()}")
    print(f"  Accuracy: {(preds.cpu() == winners).float().mean().item():.4f}")

    logit_hash = hashlib.md5(logits.cpu().numpy().tobytes()).hexdigest()
    print(f"  Logits hash: {logit_hash}")

    # 5. DataLoader config
    print("\n[5] DataLoader Config:")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Num workers: {loader.num_workers}")
    print(f"  Dataset size: {len(loader.dataset)}")

    print(f"\n{'='*70}")
    print("COPY THE ABOVE OUTPUT AND COMPARE WITH OTHER RUN")
    print(f"{'='*70}\n")


# === USAGE IN train_siamese_e2e.py ===
# Add around line 586, right before inspect_model_output:
#
#     from quick_determinism_check import quick_determinism_snapshot
#     quick_determinism_snapshot(model, train_loader, config['device'], name="MyRun")
#
# Then run twice and compare outputs!
