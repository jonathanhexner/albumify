"""
Compare the embeddings produced by extract_embeddings_clean.py to the correct embeddings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Load the embeddings saved by extract_embeddings_clean.py
clean_output_dir = Path("results/embedding_verification_test/test_clean_output")
clean_files = sorted(clean_output_dir.glob("embeddings_CLEAN_*.npy"))

if not clean_files:
    print("ERROR: No clean embeddings found!")
    sys.exit(1)

latest_clean = clean_files[-1]
print(f"Loading clean embeddings from: {latest_clean.name}")

clean_embeddings = np.load(latest_clean)
print(f"Shape: {clean_embeddings.shape}")

# The face IDs in order are: 545, 546, 550, 551, 557, 569, 573
face_ids = [545, 546, 550, 551, 557, 569, 573]

# Expected correct embedding for face 545 (from our debug script)
expected_545 = np.array([-0.00793277, 0.01506413, -0.01153619, -0.06624469, 0.04885769])

# Get the actual embedding for face 545 (first in the array)
actual_545 = clean_embeddings[0, :5]

print(f"\nFace 545 comparison:")
print(f"  Expected (correct):  {expected_545}")
print(f"  Actual (from clean): {actual_545}")

diff = np.abs(expected_545 - actual_545)
print(f"  Absolute difference: {diff}")
print(f"  Max difference: {np.max(diff):.6f}")

if np.max(diff) < 1e-6:
    print("\nRESULT: CORRECT - Clean script produced correct embeddings!")
else:
    print("\nRESULT: WRONG - Clean script produced corrupted embeddings!")
    print(f"  Cosine similarity: {np.dot(expected_545, actual_545 / np.linalg.norm(actual_545)):.6f}")

# Also check face 545 vs 546 distance
print(f"\nFace 545 vs 546 distance:")
dist_545_546 = 1.0 - np.dot(clean_embeddings[0], clean_embeddings[1])
print(f"  Distance: {dist_545_546:.4f}")
if dist_545_546 > 0.60:
    print(f"  => CORRECT: Different people (> 0.60)")
elif dist_545_546 < 0.15:
    print(f"  => WRONG: Shows as very similar (< 0.15) - CORRUPTED!")
else:
    print(f"  => BORDERLINE: Between 0.15 and 0.60")
