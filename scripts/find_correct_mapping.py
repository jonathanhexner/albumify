"""
Find correct mapping between ground truth crops and source images.

1. Run detection on source images from D:\Google_Germany_1
2. Save crops with naming: {image_name}_face_{index}.jpg
3. Compare embeddings with ground truth crops
4. Output correct mapping
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from face_cluster import InsightFaceEmbedder

# Register HEIC support
register_heif_opener()

# Source images based on mapping CSV
SOURCE_IMAGES = [
    "20240819_161357.heic",
    "20240816_142312.jpg",
    "20240816_150903.jpg",
    "20240816_150905.jpg",
    "20240817_110347.jpg",
    "20240817_110348.jpg",
    "20240818_120559.jpg",
    "20240819_131515.heic",
    "20240819_155906.heic",
    "20240819_121804.heic",
    "20240819_125821.jpg",
    "20240819_122719.heic",
    "20240819_142820.heic",
    "20240818_121446.jpg",
    "20240818_121454.jpg",
]

# Ground truth face IDs
GROUND_TRUTH_FACE_IDS = [545, 546, 550, 551, 557, 558, 562, 569, 573, 580, 584, 587, 589, 634, 637]


def main():
    source_dir = Path(r"D:\Google_Germany")
    gt_crops_dir = Path("test_data/face_clustering/face_crops")
    output_crops_dir = Path("test_data/face_clustering/detected_crops")
    output_crops_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("FINDING CORRECT FACE MAPPING")
    print("="*70)
    print(f"Source directory: {source_dir}")
    print(f"Ground truth crops: {gt_crops_dir}")
    print(f"Output crops: {output_crops_dir}")
    print(f"Images to process: {len(SOURCE_IMAGES)}")

    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

    # Step 1: Detect faces and save crops
    print("\n" + "-"*70)
    print("STEP 1: Detect faces in source images and save crops")
    print("-"*70)

    detected_faces = {}  # {image_name: [(face_index, embedding, crop_path), ...]}

    for image_name in SOURCE_IMAGES:
        image_path = source_dir / image_name

        if not image_path.exists():
            print(f"\n{image_name}: NOT FOUND")
            continue

        # Load image
        try:
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_rgb = np.array(pil_img)
        except Exception as e:
            print(f"\n{image_name}: ERROR loading - {e}")
            continue

        # Detect faces
        faces = embedder.app.get(img_rgb)
        print(f"\n{image_name}: {len(faces)} faces detected")

        detected_faces[image_name] = []

        for face_idx, face in enumerate(faces):
            # Extract embedding
            embedding = face.embedding / np.linalg.norm(face.embedding)

            # Get aligned crop
            from insightface.utils import face_align
            aligned_crop = face_align.norm_crop(img_rgb, face.kps)

            # Save crop
            crop_filename = f"{Path(image_name).stem}_face_{face_idx}.jpg"
            crop_path = output_crops_dir / crop_filename
            Image.fromarray(aligned_crop).save(crop_path, 'JPEG', quality=95)

            detected_faces[image_name].append({
                'face_index': face_idx,
                'embedding': embedding,
                'crop_path': crop_path,
                'bbox': face.bbox
            })

            print(f"  Face {face_idx}: saved as {crop_filename}")

    # Step 2: Load ground truth crop embeddings
    print("\n" + "-"*70)
    print("STEP 2: Load ground truth crop embeddings")
    print("-"*70)

    gt_embeddings = {}

    for face_id in GROUND_TRUTH_FACE_IDS:
        crop_path = gt_crops_dir / f"face_{face_id:04d}_aligned.jpg"

        if not crop_path.exists():
            print(f"Face {face_id}: NOT FOUND")
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
            print(f"Face {face_id}: loaded")

    # Step 3: Match ground truth crops to detected faces
    print("\n" + "-"*70)
    print("STEP 3: Match ground truth crops to detected faces")
    print("-"*70)

    matches = {}  # {face_id: (image_name, face_index, similarity)}

    for face_id, gt_emb in sorted(gt_embeddings.items()):
        best_match = None
        best_similarity = -1
        best_image = None
        best_index = None

        # Compare against all detected faces
        for image_name, faces_list in detected_faces.items():
            for face_info in faces_list:
                detected_emb = face_info['embedding']
                similarity = np.dot(gt_emb, detected_emb)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_info
                    best_image = image_name
                    best_index = face_info['face_index']

        if best_similarity > 0.90:
            status = "MATCH" if best_similarity > 0.95 else "WEAK"
            matches[face_id] = (best_image, best_index, best_similarity)
            print(f"Face {face_id}: {best_image} index {best_index} (similarity={best_similarity:.3f}) [{status}]")
        else:
            print(f"Face {face_id}: NO MATCH FOUND (best={best_similarity:.3f})")

    # Step 4: Generate corrected mapping
    print("\n" + "="*70)
    print("CORRECTED MAPPING")
    print("="*70)

    corrected_mapping = []

    for face_id in sorted(GROUND_TRUTH_FACE_IDS):
        if face_id in matches:
            image_name, face_index, similarity = matches[face_id]
            corrected_mapping.append({
                'face_id': face_id,
                'image_name': image_name,
                'face_index': face_index,
                'similarity': similarity
            })
            print(f"{face_id},{image_name},{face_index}  # similarity={similarity:.3f}")
        else:
            print(f"{face_id},UNKNOWN,UNKNOWN  # NO MATCH")

    # Save corrected mapping
    output_csv = Path("test_data/face_clustering/corrected_mapping.csv")
    with open(output_csv, 'w') as f:
        for entry in corrected_mapping:
            f.write(f"{entry['face_id']},{entry['image_name']},{entry['face_index']}\n")

    print(f"\nCorrected mapping saved to: {output_csv}")
    print(f"Detected crops saved to: {output_crops_dir}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Ground truth faces: {len(GROUND_TRUTH_FACE_IDS)}")
    print(f"Matched: {len(matches)}")
    print(f"Unmatched: {len(GROUND_TRUTH_FACE_IDS) - len(matches)}")


if __name__ == '__main__':
    main()
