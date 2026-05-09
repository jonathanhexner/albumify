"""
Test if ONNX Runtime is caching embedding computations.

Extract embeddings for the same face multiple times and check if results change.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from face_cluster import InsightFaceEmbedder

def main():
    # Test face 545
    crop_path = Path("results/Google_Germany/face_crops/face_0545_aligned.jpg")

    print("="*70)
    print("ONNX CACHING TEST")
    print("="*70)
    print(f"Testing face: {crop_path}")
    print()

    # Load image once
    img_pil = Image.open(crop_path)
    img_np = np.array(img_pil)

    print(f"Image: shape={img_np.shape}, first pixel={img_np[0, 0]}")
    print()

    # Extract embedding 5 times with SAME embedder instance
    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    embeddings = []
    for i in range(5):
        emb = embedder.get_embedding(img_np)
        embeddings.append(emb)
        print(f"Extraction {i+1}: first 5 values = {emb[:5]}")

    # Check if all identical
    all_identical = True
    for i in range(1, 5):
        if not np.allclose(embeddings[0], embeddings[i]):
            all_identical = False
            print(f"\nExtraction {i+1} differs from extraction 1!")

    if all_identical:
        print("\nAll extractions IDENTICAL (expected for same image)")

    # Now test: what if we process 100 OTHER faces first, then extract face 545 again?
    print("\n" + "="*70)
    print("CACHE POLLUTION TEST")
    print("="*70)
    print("Processing 100 other faces first...")

    crops_dir = Path("results/Google_Germany/face_crops")
    other_crops = sorted(crops_dir.glob("face_*_aligned.jpg"))[:100]

    for crop in other_crops:
        if crop.name == "face_0545_aligned.jpg":
            continue
        img = np.array(Image.open(crop))
        _ = embedder.get_embedding(img)

    print(f"Processed {len(other_crops)} other faces")
    print()

    # Now extract face 545 again
    emb_after_batch = embedder.get_embedding(img_np)
    print(f"Face 545 after batch: first 5 values = {emb_after_batch[:5]}")
    print(f"Original first 5 values:          {embeddings[0][:5]}")

    if np.allclose(embeddings[0], emb_after_batch):
        print("\nRESULT: Embedding UNCHANGED (consistent)")
    else:
        print("\nRESULT: Embedding CHANGED (cache corruption!)")
        similarity = np.dot(embeddings[0], emb_after_batch)
        print(f"Cosine similarity: {similarity:.6f}")

    # Compare to known corrupted value
    corrupted_545 = np.array([-6.0872161e-03, 2.4038328e-02, -6.4672888e-05, -5.7359152e-02, 3.0528182e-02])

    print("\n" + "="*70)
    print("CORRUPTION DETECTION")
    print("="*70)
    print(f"Known corrupted value: {corrupted_545}")

    if np.allclose(emb_after_batch[:5], corrupted_545):
        print("\nERROR: Embedding matches corrupted value!")
        print("ONNX Runtime is returning cached corrupted data!")
    else:
        print("\nOK: Embedding does not match corrupted value")

if __name__ == '__main__':
    main()
