"""Debug embedding mismatch between regenerate script and notebook.

This script:
1. Loads the "FRESH" embeddings from regenerate_embeddings_from_crops.py
2. Loads the same face crop images
3. Extracts embeddings directly using InsightFace
4. Compares to verify they match

Usage:
    python scripts/debug_embedding_mismatch.py \
        --embeddings results/Google_Germany/embeddings_FRESH_*.npy \
        --face-crops results/Google_Germany/face_crops \
        --test-faces 569,573,577,553
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from face_cluster import InsightFaceEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Debug embedding mismatch')
    parser.add_argument('--embeddings', type=Path, required=True, help='FRESH embeddings .npy file')
    parser.add_argument('--face-crops', type=Path, required=True, help='Face crops directory')
    parser.add_argument('--test-faces', type=str, required=True, help='Comma-separated face IDs to test')

    args = parser.parse_args()

    # Parse test faces
    test_face_ids = [int(x.strip()) for x in args.test_faces.split(',')]

    logger.info("="*60)
    logger.info("Embedding Mismatch Debug")
    logger.info("="*60)
    logger.info(f"Embeddings file: {args.embeddings}")
    logger.info(f"Face crops dir: {args.face_crops}")
    logger.info(f"Test faces: {test_face_ids}")

    # Load FRESH embeddings from regenerate script
    logger.info("\n1. Loading FRESH embeddings from regenerate script...")
    fresh_embeddings = np.load(args.embeddings)
    logger.info(f"   Shape: {fresh_embeddings.shape}")

    # Initialize InsightFace
    logger.info("\n2. Initializing InsightFace...")
    embedder = InsightFaceEmbedder(model_name='buffalo_l', ctx_id=-1)

    # Test each face
    logger.info("\n3. Testing each face...")
    for face_id in test_face_ids:
        logger.info(f"\n--- Face {face_id} ---")

        # Find the crop file
        crop_file = args.face_crops / f"face_{face_id:04d}_aligned.jpg"
        if not crop_file.exists():
            logger.error(f"   Crop file not found: {crop_file}")
            continue

        logger.info(f"   Crop file: {crop_file.name}")

        # Load the crop image
        img = cv2.imread(str(crop_file))
        if img is None:
            logger.error(f"   Failed to load image")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.info(f"   Image size: {img_rgb.shape}")

        # Extract embedding FRESH
        logger.info(f"   Extracting fresh embedding...")
        fresh_emb = embedder.get_embedding(img_rgb)
        if fresh_emb is None:
            logger.error(f"   Failed to extract embedding")
            continue

        # Ensure 1D (flatten if needed)
        fresh_emb = np.asarray(fresh_emb).flatten()

        logger.info(f"   Fresh embedding shape: {fresh_emb.shape}")
        logger.info(f"   Fresh embedding norm: {np.linalg.norm(fresh_emb):.6f}")

        # Get embedding from FRESH file
        # QUESTION: How does face_id map to array index?
        if face_id >= len(fresh_embeddings):
            logger.error(f"   Face ID {face_id} out of range (array has {len(fresh_embeddings)} embeddings)")
            continue

        stored_fresh_emb = fresh_embeddings[face_id]
        logger.info(f"   Stored FRESH embedding shape: {stored_fresh_emb.shape}")
        logger.info(f"   Stored FRESH embedding norm: {np.linalg.norm(stored_fresh_emb):.6f}")

        # Compare
        distance = 1 - np.dot(fresh_emb, stored_fresh_emb)
        logger.info(f"   Distance (fresh vs stored FRESH): {distance:.6f}")

        if distance > 0.01:
            logger.error(f"   ❌ MISMATCH! Fresh embedding doesn't match stored FRESH")
            logger.info(f"   First 10 values (fresh):       {fresh_emb[:10]}")
            logger.info(f"   First 10 values (stored FRESH): {stored_fresh_emb[:10]}")
        else:
            logger.info(f"   ✓ Match!")

    # Also check: does the array index match face_id?
    logger.info("\n" + "="*60)
    logger.info("4. Checking face_id to array index mapping...")
    logger.info("="*60)

    # Count how many crops exist for each face_id
    all_crops = sorted(args.face_crops.glob("face_*_aligned.jpg"))
    logger.info(f"Total crops found: {len(all_crops)}")

    if len(all_crops) != len(fresh_embeddings):
        logger.error(f"❌ MISMATCH: {len(all_crops)} crops but {len(fresh_embeddings)} embeddings!")

    # Check if face_ids are sequential
    face_ids_from_files = []
    for crop_file in all_crops[:20]:  # Check first 20
        face_id = int(crop_file.stem.split('_')[1])
        face_ids_from_files.append(face_id)

    logger.info(f"First 20 face IDs from filenames: {face_ids_from_files}")
    logger.info(f"Expected (if sequential): {list(range(20))}")

    if face_ids_from_files != list(range(20)):
        logger.error("❌ Face IDs are NOT sequential! Index-based lookup will fail!")


if __name__ == '__main__':
    main()
