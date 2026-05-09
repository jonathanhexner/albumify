"""
Verify that ground truth face crops match the faces at specified indices in source images.

Creates a visual comparison to debug the face_index mapping.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pillow_heif import register_heif_opener
from face_cluster import InsightFaceEmbedder

# Register HEIC support
register_heif_opener()

# Ground truth mapping (face_id → image_name, face_index)
MAPPING = {
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

def main():
    source_dir = Path("test_data/source_images_ground_truth")
    crops_dir = Path("test_data/face_crops_ground_truth")
    output_dir = Path("test_data/face_index_verification")
    output_dir.mkdir(exist_ok=True)

    embedder = InsightFaceEmbedder(model_name='buffalo_l')

    print("="*70)
    print("VERIFYING FACE INDEX MAPPING")
    print("="*70)

    for face_id, (image_name, expected_index) in sorted(MAPPING.items()):
        print(f"\nFace {face_id}: {image_name} at index {expected_index}")

        # Load source image
        source_path = source_dir / image_name
        if not source_path.exists():
            print(f"  ERROR: Source image not found")
            continue

        with Image.open(source_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_rgb = np.array(pil_img)

        # Detect faces
        faces = embedder.app.get(img_rgb)
        print(f"  Detected {len(faces)} faces")

        if expected_index >= len(faces):
            print(f"  ERROR: Index {expected_index} out of range")
            continue

        # Extract face at expected index
        face = faces[expected_index]
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Crop from source
        detected_crop = img_rgb[y1:y2, x1:x2]
        detected_crop_img = Image.fromarray(detected_crop)

        # Load ground truth crop
        gt_crop_path = crops_dir / f"face_{face_id:04d}_aligned.jpg"
        if not gt_crop_path.exists():
            print(f"  ERROR: Ground truth crop not found")
            continue

        gt_crop_img = Image.open(gt_crop_path)

        # Compute embedding similarity
        emb_detected = embedder.get_embedding(img_rgb)
        if len(faces) > 0:
            emb_detected = faces[expected_index].embedding
            emb_detected = emb_detected / np.linalg.norm(emb_detected)

        gt_crop_array = np.array(gt_crop_img)
        if len(gt_crop_array.shape) == 2:
            import cv2
            gt_crop_array = cv2.cvtColor(gt_crop_array, cv2.COLOR_GRAY2RGB)
        elif gt_crop_array.shape[2] == 4:
            import cv2
            gt_crop_array = cv2.cvtColor(gt_crop_array, cv2.COLOR_RGBA2RGB)

        emb_gt = embedder.get_embedding(gt_crop_array)

        similarity = np.dot(emb_detected, emb_gt) if emb_gt is not None else -1.0
        print(f"  Similarity: {similarity:.3f}")

        # Create comparison image
        # Resize for visualization
        detected_crop_img = detected_crop_img.resize((200, 200), Image.Resampling.LANCZOS)
        gt_crop_img = gt_crop_img.resize((200, 200), Image.Resampling.LANCZOS)

        # Create side-by-side comparison
        comparison = Image.new('RGB', (420, 220), color='white')
        comparison.paste(detected_crop_img, (10, 10))
        comparison.paste(gt_crop_img, (220, 10))

        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((10, 215), f"Detected [idx={expected_index}]", fill='black', font=font)
        draw.text((220, 215), f"Ground Truth", fill='black', font=font)
        draw.text((10, 0), f"Face {face_id} - Sim: {similarity:.3f}", fill='black', font=font)

        # Save comparison
        output_path = output_dir / f"face_{face_id:04d}_comparison.jpg"
        comparison.save(output_path)
        print(f"  Saved: {output_path.name}")

    print(f"\n{'='*70}")
    print(f"Verification images saved to: {output_dir}")
    print("Open these images to visually verify the face_index mapping")


if __name__ == '__main__':
    main()
