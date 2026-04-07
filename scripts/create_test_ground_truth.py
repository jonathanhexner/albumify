"""
Create small test ground truth from specific images.

Processes just a few hand-picked images for quick testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from face_cluster import InsightFaceEmbedder

# Register HEIC support
register_heif_opener()

# Hand-picked test images with multiple faces
TEST_IMAGES = [
    "20240816_150903.jpg",  # 5 faces
    "20240816_150905.jpg",  # 4 faces
    "20240818_121446.jpg",  # 4 faces
    "20240817_110347.jpg",  # 3 faces
    "20240817_110348.jpg",  # 3 faces
]


def main():
    source_dir = Path(r"D:\Google_Germany")
    output_dir = Path("test_data/ground_truth_test")
    crops_dir = output_dir / "face_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING TEST GROUND TRUTH")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Test images: {len(TEST_IMAGES)}")

    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

    all_detections = []
    total_faces = 0

    print("\n" + "-"*70)
    print("DETECTING FACES")
    print("-"*70)

    for image_name in TEST_IMAGES:
        image_path = source_dir / image_name

        if not image_path.exists():
            print(f"\n{image_name}: NOT FOUND")
            continue

        print(f"\n{image_name}:")

        # Load image
        try:
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_rgb = np.array(pil_img)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Detect faces
        faces = embedder.app.get(img_rgb)
        print(f"  Detected {len(faces)} faces")

        if len(faces) == 0:
            continue

        # Save crops
        from insightface.utils import face_align

        for face_idx, face in enumerate(faces):
            # Get aligned crop
            aligned_crop = face_align.norm_crop(img_rgb, face.kps)

            # Generate filename
            crop_filename = f"{Path(image_name).stem}_face_{face_idx}.jpg"
            crop_path = crops_dir / crop_filename

            # Save
            Image.fromarray(aligned_crop).save(crop_path, 'JPEG', quality=95)

            # Extract embedding
            embedding = face.embedding / np.linalg.norm(face.embedding)

            # Store detection
            detection = {
                'crop_filename': crop_filename,
                'source_image': image_name,
                'source_path': str(image_path),
                'face_index': face_idx,
                'bbox': face.bbox.tolist(),
                'embedding': embedding.tolist(),
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else None,
            }

            all_detections.append(detection)
            total_faces += 1

            print(f"    [{face_idx}] {crop_filename}")

    # Create mapping CSV
    print("\n" + "-"*70)
    print("CREATING MAPPING CSV")
    print("-"*70)

    mapping_file = output_dir / "mapping.csv"
    with open(mapping_file, 'w') as f:
        f.write("crop_filename,source_image,face_index,source_path\n")
        for det in all_detections:
            f.write(f"{det['crop_filename']},{det['source_image']},{det['face_index']},{det['source_path']}\n")

    print(f"Saved: {mapping_file}")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'n_images': len(TEST_IMAGES),
            'n_faces': len(all_detections),
            'face_ordering': 'detection_order',
            'test_images': TEST_IMAGES,
            'detections': all_detections
        }, f, indent=2)

    print(f"Saved: {metadata_file}")

    # Summary
    print("\n" + "="*70)
    print("SUCCESS")
    print("="*70)
    print(f"Processed: {len(TEST_IMAGES)} images")
    print(f"Detected: {total_faces} faces")
    print(f"Crops: {crops_dir}")
    print(f"Mapping: {mapping_file}")

    # List all crops
    print("\n" + "-"*70)
    print("FACE CROPS CREATED")
    print("-"*70)
    for det in all_detections:
        print(f"  {det['crop_filename']:40s} <- {det['source_image']} face[{det['face_index']}]")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Visually review crops in:", crops_dir)
    print("2. Create labels.json with person identities")
    print("3. Test with test_face_clustering_isolated.py")
    print("4. Verify embeddings match between pipeline and crops")


if __name__ == '__main__':
    main()
