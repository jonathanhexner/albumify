"""
Comprehensive determinism checker for debugging training discrepancies.

Use this to identify why two training runs with the same seed/data/model differ.
"""
import logging
import hashlib
import torch
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def hash_tensor(tensor):
    """Create hash of tensor values for comparison."""
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()


def hash_string_list(strings):
    """Create hash of string list for comparison."""
    combined = "|".join(sorted(strings))
    return hashlib.md5(combined.encode()).hexdigest()


def check_batch_determinism(loader, num_batches=3, save_path=None):
    """
    Check if dataloader produces deterministic batches.

    Args:
        loader: DataLoader to test
        num_batches: Number of batches to check
        save_path: Optional path to save batch hashes

    Returns:
        dict: Batch information including hashes and file lists
    """
    logger.info("=== Checking DataLoader Determinism ===")

    batch_info = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        img1 = batch['img1']
        img2 = batch['img2']
        winners = batch['winner']

        # Hash the tensors
        img1_hash = hash_tensor(img1)
        img2_hash = hash_tensor(img2)
        winner_hash = hash_tensor(winners)

        # Get filenames if available
        files1 = batch.get('image1', [])
        files2 = batch.get('image2', [])

        # Hash the filename order
        files1_hash = hash_string_list(files1) if files1 else None
        files2_hash = hash_string_list(files2) if files2 else None

        info = {
            'batch_idx': batch_idx,
            'batch_size': len(winners),
            'img1_hash': img1_hash,
            'img2_hash': img2_hash,
            'winner_hash': winner_hash,
            'files1_hash': files1_hash,
            'files2_hash': files2_hash,
            'files1': files1[:5] if files1 else None,  # First 5 for inspection
            'files2': files2[:5] if files2 else None,
            'img1_shape': list(img1.shape),
            'img2_shape': list(img2.shape),
            'img1_mean': img1.mean().item(),
            'img1_std': img1.std().item(),
        }

        batch_info.append(info)

        logger.info(f"\nBatch {batch_idx}:")
        logger.info(f"  Size: {info['batch_size']}")
        logger.info(f"  IMG1 hash: {img1_hash}")
        logger.info(f"  IMG2 hash: {img2_hash}")
        logger.info(f"  Winner hash: {winner_hash}")
        logger.info(f"  Files1 hash: {files1_hash}")
        logger.info(f"  Files2 hash: {files2_hash}")
        if files1:
            logger.info(f"  First file1: {files1[0]}")
            logger.info(f"  First file2: {files2[0]}")
        logger.info(f"  IMG1 stats: mean={info['img1_mean']:.6f}, std={info['img1_std']:.6f}")

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(batch_info, f, indent=2)
        logger.info(f"\nBatch info saved to: {save_path}")

    return batch_info


def check_model_determinism(model, save_path=None):
    """
    Check model weight initialization state.

    Args:
        model: Model to inspect
        save_path: Optional path to save weight hashes

    Returns:
        dict: Weight hashes for each parameter
    """
    logger.info("\n=== Checking Model Determinism ===")

    weight_info = {}

    for name, param in model.named_parameters():
        param_hash = hash_tensor(param.data)
        stats = {
            'hash': param_hash,
            'shape': list(param.shape),
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item(),
        }
        weight_info[name] = stats

        logger.info(f"\n{name}:")
        logger.info(f"  Hash: {param_hash}")
        logger.info(f"  Shape: {stats['shape']}")
        logger.info(f"  Stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(weight_info, f, indent=2)
        logger.info(f"\nModel weight info saved to: {save_path}")

    return weight_info


def check_random_state(save_path=None):
    """
    Check current random number generator states.

    Args:
        save_path: Optional path to save state info

    Returns:
        dict: Random state information
    """
    logger.info("\n=== Checking Random State ===")

    # Get Python random state
    import random
    py_state = random.getstate()

    # Get numpy state
    np_state = np.random.get_state()

    # Get PyTorch state
    torch_state = torch.get_rng_state()
    torch_cuda_available = torch.cuda.is_available()

    state_info = {
        'python_random_version': py_state[0],
        'numpy_random_type': np_state[0],
        'torch_rng_hash': hashlib.md5(torch_state.numpy().tobytes()).hexdigest(),
        'torch_cuda_available': torch_cuda_available,
    }

    if torch_cuda_available:
        cuda_state = torch.cuda.get_rng_state()
        state_info['torch_cuda_rng_hash'] = hashlib.md5(cuda_state.cpu().numpy().tobytes()).hexdigest()
        logger.info(f"CUDA RNG hash: {state_info['torch_cuda_rng_hash']}")

    logger.info(f"PyTorch RNG hash: {state_info['torch_rng_hash']}")

    # Test if random generation is deterministic
    logger.info("\nTesting random generation:")
    test_tensor = torch.randn(10)
    test_hash = hash_tensor(test_tensor)
    logger.info(f"Random tensor hash: {test_hash}")
    state_info['test_random_hash'] = test_hash

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(state_info, f, indent=2)
        logger.info(f"\nRandom state info saved to: {save_path}")

    return state_info


def check_transform_determinism(transform, sample_input, num_runs=5):
    """
    Check if transforms are deterministic.

    Args:
        transform: Transform to test
        sample_input: Sample input to transform
        num_runs: Number of times to run transform

    Returns:
        bool: True if deterministic, False otherwise
    """
    logger.info("\n=== Checking Transform Determinism ===")

    if transform is None:
        logger.info("No transform provided")
        return True

    hashes = []
    for i in range(num_runs):
        output = transform(sample_input)
        if isinstance(output, torch.Tensor):
            h = hash_tensor(output)
        else:
            # Convert to tensor if needed
            h = hash_tensor(torch.tensor(np.array(output)))
        hashes.append(h)
        logger.info(f"Run {i+1}: {h}")

    is_deterministic = len(set(hashes)) == 1

    if is_deterministic:
        logger.info("✓ Transform is DETERMINISTIC")
    else:
        logger.info("✗ Transform is NON-DETERMINISTIC")
        logger.info(f"  Got {len(set(hashes))} unique hashes out of {num_runs} runs")

    return is_deterministic


def comprehensive_determinism_check(model, loader, output_dir, transform=None):
    """
    Run all determinism checks and save results.

    Args:
        model: Model to check
        loader: DataLoader to check
        output_dir: Directory to save results
        transform: Optional transform to check

    Returns:
        dict: All check results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE DETERMINISM CHECK")
    logger.info("="*70)

    results = {}

    # Check random state
    results['random_state'] = check_random_state(
        save_path=output_dir / 'random_state.json'
    )

    # Check model weights
    results['model_weights'] = check_model_determinism(
        model, save_path=output_dir / 'model_weights.json'
    )

    # Check dataloader batches
    results['batches'] = check_batch_determinism(
        loader, num_batches=3, save_path=output_dir / 'batch_info.json'
    )

    # Check transforms if provided
    if transform is not None:
        # Get a sample image from the first batch
        batch = next(iter(loader))
        sample = batch['img1'][0]  # First image from batch
        results['transform_deterministic'] = check_transform_determinism(
            transform, sample
        )

    logger.info("\n" + "="*70)
    logger.info("CHECK COMPLETE")
    logger.info("="*70)

    return results


def compare_determinism_outputs(output_dir1, output_dir2):
    """
    Compare determinism check outputs from two different runs.

    Args:
        output_dir1: First output directory
        output_dir2: Second output directory

    Returns:
        dict: Comparison results
    """
    logger.info("\n" + "="*70)
    logger.info("COMPARING DETERMINISM CHECKS")
    logger.info("="*70)

    output_dir1 = Path(output_dir1)
    output_dir2 = Path(output_dir2)

    comparison = {}

    # Compare batch info
    logger.info("\n=== Comparing Batches ===")
    batch_file1 = output_dir1 / 'batch_info.json'
    batch_file2 = output_dir2 / 'batch_info.json'

    if batch_file1.exists() and batch_file2.exists():
        with open(batch_file1) as f:
            batches1 = json.load(f)
        with open(batch_file2) as f:
            batches2 = json.load(f)

        batch_matches = []
        for b1, b2 in zip(batches1, batches2):
            match = {
                'batch_idx': b1['batch_idx'],
                'img1_match': b1['img1_hash'] == b2['img1_hash'],
                'img2_match': b1['img2_hash'] == b2['img2_hash'],
                'winner_match': b1['winner_hash'] == b2['winner_hash'],
                'files1_match': b1['files1_hash'] == b2['files1_hash'],
                'files2_match': b1['files2_hash'] == b2['files2_hash'],
            }
            batch_matches.append(match)

            logger.info(f"\nBatch {match['batch_idx']}:")
            logger.info(f"  IMG1 tensors match: {match['img1_match']}")
            logger.info(f"  IMG2 tensors match: {match['img2_match']}")
            logger.info(f"  Winners match: {match['winner_match']}")
            logger.info(f"  Files1 match: {match['files1_match']}")
            logger.info(f"  Files2 match: {match['files2_match']}")

            if not match['files1_match']:
                logger.info(f"  Run1 first file: {b1['files1'][0] if b1['files1'] else 'N/A'}")
                logger.info(f"  Run2 first file: {b2['files1'][0] if b2['files1'] else 'N/A'}")

        comparison['batches'] = batch_matches

    # Compare model weights
    logger.info("\n=== Comparing Model Weights ===")
    weights_file1 = output_dir1 / 'model_weights.json'
    weights_file2 = output_dir2 / 'model_weights.json'

    if weights_file1.exists() and weights_file2.exists():
        with open(weights_file1) as f:
            weights1 = json.load(f)
        with open(weights_file2) as f:
            weights2 = json.load(f)

        weight_matches = {}
        all_match = True

        for name in weights1.keys():
            if name in weights2:
                matches = weights1[name]['hash'] == weights2[name]['hash']
                weight_matches[name] = matches
                if not matches:
                    all_match = False
                    logger.info(f"\n{name}: MISMATCH")
                    logger.info(f"  Run1 hash: {weights1[name]['hash']}")
                    logger.info(f"  Run2 hash: {weights2[name]['hash']}")

        if all_match:
            logger.info("✓ All model weights match!")
        else:
            logger.info(f"✗ {sum(not m for m in weight_matches.values())} weights differ")

        comparison['model_weights'] = weight_matches

    logger.info("\n" + "="*70)

    return comparison
