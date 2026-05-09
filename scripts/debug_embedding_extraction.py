"""
Minimal reproduction script to identify why two methods produce different embeddings.

Tests the same face (545) using both the isolated method (works) and clean extraction method (fails).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2

from face_cluster import InsightFaceEmbedder

def extract_isolated_method(crop_path: Path) -> np.ndarray:
    """
    Extract embedding using the isolated test method (PROVEN TO WORK).
    """
    print("\n" + "="*70)
    print("METHOD 1: ISOLATED TEST (PROVEN WORKING)")
    print("="*70)

    embedder = InsightFaceEmbedder(model_name='buffalo_l')
    print(f"Embedder initialized: {type(embedder)}")

    # Load image
    print(f"Loading: {crop_path}")
    img_pil = Image.open(crop_path)
    print(f"  PIL image: mode={img_pil.mode}, size={img_pil.size}")

    img_np = np.array(img_pil)
    print(f"  Numpy array: shape={img_np.shape}, dtype={img_np.dtype}")
    print(f"  First pixel RGB: {img_np[0, 0]}")

    # Ensure RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        print(f"  Converted grayscale to RGB")
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        print(f"  Converted RGBA to RGB")

    print(f"  Final array: shape={img_np.shape}, dtype={img_np.dtype}")
    print(f"  Final first pixel: {img_np[0, 0]}")

    # Extract embedding
    embedding = embedder.get_embedding(img_np)
    print(f"  Embedding: shape={embedding.shape}, dtype={embedding.dtype}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Norm: {np.linalg.norm(embedding):.6f}")

    return embedding


def extract_clean_method(crop_path: Path) -> np.ndarray:
    """
    Extract embedding using the clean extraction method (PRODUCES CORRUPTED RESULTS).
    """
    print("\n" + "="*70)
    print("METHOD 2: CLEAN EXTRACTION (PRODUCES WRONG RESULTS)")
    print("="*70)

    embedder = InsightFaceEmbedder(model_name='buffalo_l')
    print(f"Embedder initialized: {type(embedder)}")

    # Load image
    print(f"Loading: {crop_path}")
    img_pil = Image.open(crop_path)
    print(f"  PIL image: mode={img_pil.mode}, size={img_pil.size}")

    img_np = np.array(img_pil)
    print(f"  Numpy array: shape={img_np.shape}, dtype={img_np.dtype}")
    print(f"  First pixel RGB: {img_np[0, 0]}")

    # Ensure RGB
    if len(img_np.shape) == 2:
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        print(f"  Converted grayscale to RGB")
    elif img_np.shape[2] == 4:
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        print(f"  Converted RGBA to RGB")

    print(f"  Final array: shape={img_np.shape}, dtype={img_np.dtype}")
    print(f"  Final first pixel: {img_np[0, 0]}")

    # Extract embedding
    embedding = embedder.get_embedding(img_np)
    print(f"  Embedding: shape={embedding.shape}, dtype={embedding.dtype}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Norm: {np.linalg.norm(embedding):.6f}")

    return embedding


def main():
    # Test on face 545 (critical test case)
    face_id = 545

    # Try both source locations
    test_dirs = [
        Path("results/embedding_verification_test"),
        Path("results/Google_Germany/face_crops")
    ]

    for test_dir in test_dirs:
        crop_path = test_dir / f"face_{face_id:04d}_aligned.jpg"

        if not crop_path.exists():
            print(f"SKIP: {crop_path} not found")
            continue

        print("\n" + "="*70)
        print(f"TESTING: {crop_path}")
        print("="*70)

        # Extract using both methods
        embedding1 = extract_isolated_method(crop_path)
        embedding2 = extract_clean_method(crop_path)

        # Compare
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)

        diff = np.abs(embedding1 - embedding2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        if max_diff < 1e-6:
            print("RESULT: IDENTICAL (as expected)")
        else:
            print(f"RESULT: DIFFERENT!")
            print(f"\nMethod 1 first 10 values:\n  {embedding1[:10]}")
            print(f"Method 2 first 10 values:\n  {embedding2[:10]}")

            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
            print(f"\nCosine similarity: {similarity:.6f}")

            if similarity < 0.9:
                print("ERROR: Methods produce COMPLETELY DIFFERENT embeddings!")
            else:
                print("WARNING: Methods produce slightly different embeddings")

        print("\n" + "="*70)
        print()

if __name__ == '__main__':
    main()
