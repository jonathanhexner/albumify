"""
Test face detection and extraction on isolated test set.

Tests ONLY the 15 images that correspond to our ground truth face crops.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from face_cluster import InsightFaceEmbedder

# Register HEIC support
register_heif_opener()


# Mapping: face_id -> (image_name, face_index)
GROUND_TRUTH = {
    545: ("20240819_161357.heic", 1),
    546: ("20240816_142312.jpg", 1),
    550: ("20240816_150903.jpg", 1),
    551: ("20240816_150905.jpg", 2),
    557: ("20240817_110347.jpg", 0),
    558: ("20240817_110348.jpg", 0),
    562: ("20240818_120559.jpg", 0),
    569: ("20240819_131515.heic", 9),
    573: ("20240819_155906.heic", 1),
    580: ("20240819_121804.heic", 0),
    584: ("20240819_125821.jpg", 1),
    587: ("20240819_122719.heic", 0),
    589: ("20240819_142820.heic", 0),
    634: ("20240818_121446.jpg", 2),
    637: ("20240818_121454.jpg", 2),
}


def test_step_by_step():
    """Test pipeline step by step on isolated data."""

    source_dir = Path("test_data/face_clustering/source_images")
    crops_dir = Path("test_data/face_clustering/face_crops")

    print("="*70)
    print("ISOLATED FACE CLUSTERING TEST")
    print("="*70)
    print(f"\nSource images: {source_dir}")
    print(f"Face crops: {crops_dir}")
    print(f"Ground truth: {len(GROUND_TRUTH)} faces")

    # Initialize embedder with detection_order (confidence sorting)
    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

    print("\n" + "-"*70)
    print("STEP 1: Detect faces in each source image")
    print("-"*70)

    detection_results = {}

    for face_id, (image_name, expected_index) in sorted(GROUND_TRUTH.items()):
        image_path = source_dir / image_name

        if not image_path.exists():
            print(f"\nFace {face_id}: ERROR - Image not found: {image_name}")
            continue

        # Load image
        with Image.open(image_path) as pil_img:
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_rgb = np.array(pil_img)

        # Detect faces
        faces = embedder.app.get(img_rgb)

        detection_results[face_id] = {
            'image_name': image_name,
            'image_path': image_path,
            'expected_index': expected_index,
            'num_detected': len(faces),
            'faces': faces,
        }

        status = "OK" if expected_index < len(faces) else "INDEX OUT OF RANGE"
        print(f"  Face {face_id}: {image_name} -> {len(faces)} faces detected, need index {expected_index} [{status}]")

    print("\n" + "-"*70)
    print("STEP 2: Extract embeddings from detected faces")
    print("-"*70)

    pipeline_embeddings = {}

    for face_id, result in sorted(detection_results.items()):
        expected_index = result['expected_index']
        faces = result['faces']

        if expected_index >= len(faces):
            print(f"  Face {face_id}: SKIP - index {expected_index} out of range")
            continue

        # Extract embedding from detected face
        face = faces[expected_index]
        embedding = face.embedding
        embedding_normalized = embedding / np.linalg.norm(embedding)
        pipeline_embeddings[face_id] = embedding_normalized

        print(f"  Face {face_id}: Extracted embedding from index {expected_index}")

    print("\n" + "-"*70)
    print("STEP 3: Load ground truth crop embeddings")
    print("-"*70)

    gt_embeddings = {}

    for face_id in sorted(GROUND_TRUTH.keys()):
        crop_path = crops_dir / f"face_{face_id:04d}_aligned.jpg"

        if not crop_path.exists():
            print(f"  Face {face_id}: ERROR - Crop not found")
            continue

        # Load crop
        crop_img = np.array(Image.open(crop_path))

        # Handle grayscale/RGBA
        if len(crop_img.shape) == 2:
            import cv2
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
        elif crop_img.shape[2] == 4:
            import cv2
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGBA2RGB)

        # Extract embedding
        embedding = embedder.get_embedding(crop_img)
        if embedding is not None:
            gt_embeddings[face_id] = embedding
            print(f"  Face {face_id}: Loaded crop embedding")
        else:
            print(f"  Face {face_id}: ERROR - Failed to extract embedding from crop")

    print("\n" + "-"*70)
    print("STEP 4: Compare pipeline vs ground truth embeddings")
    print("-"*70)

    matches = []
    mismatches = []

    for face_id in sorted(GROUND_TRUTH.keys()):
        if face_id not in pipeline_embeddings:
            print(f"  Face {face_id}: SKIP - not in pipeline embeddings")
            continue

        if face_id not in gt_embeddings:
            print(f"  Face {face_id}: SKIP - not in ground truth embeddings")
            continue

        pipeline_emb = pipeline_embeddings[face_id]
        gt_emb = gt_embeddings[face_id]

        similarity = np.dot(pipeline_emb, gt_emb)

        if similarity > 0.95:
            status = "MATCH"
            matches.append(face_id)
        elif similarity > 0.70:
            status = "BORDERLINE"
            mismatches.append((face_id, similarity))
        else:
            status = "MISMATCH"
            mismatches.append((face_id, similarity))

        print(f"  Face {face_id}: similarity={similarity:.3f} [{status}]")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total faces: {len(GROUND_TRUTH)}")
    print(f"Pipeline extracted: {len(pipeline_embeddings)}")
    print(f"Ground truth loaded: {len(gt_embeddings)}")
    print(f"Matches (>0.95): {len(matches)}")
    print(f"Mismatches (<=0.95): {len(mismatches)}")

    if mismatches:
        print("\nMismatched faces:")
        for face_id, similarity in sorted(mismatches, key=lambda x: x[1]):
            image_name, face_index = GROUND_TRUTH[face_id]
            print(f"  Face {face_id}: {image_name} index {face_index} -> similarity {similarity:.3f}")

    print("\n" + "="*70)

    if len(matches) == len(GROUND_TRUTH):
        print("SUCCESS: All faces match!")
        return True
    else:
        print(f"FAILURE: {len(mismatches)} faces don't match")
        return False


if __name__ == '__main__':
    success = test_step_by_step()
    sys.exit(0 if success else 1)
