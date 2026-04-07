"""
Create ground truth face dataset with proper traceability.

Interactive script to:
1. Select source images
2. Detect and save face crops
3. Create mapping CSV
4. Manual labeling (optional)
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


def list_images(source_dir: Path, limit: int = 50):
    """List available images."""
    image_exts = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    images = sorted([
        f for f in source_dir.iterdir()
        if f.suffix.lower() in image_exts
    ])

    print(f"\nFound {len(images)} images in {source_dir}")
    print("\nShowing first {}: ".format(min(limit, len(images))))

    for i, img_path in enumerate(images[:limit]):
        size_mb = img_path.stat().st_size / (1024 * 1024)
        print(f"  {i+1:3d}. {img_path.name:40s} ({size_mb:.1f} MB)")

    if len(images) > limit:
        print(f"  ... and {len(images) - limit} more")

    return images


def select_images(all_images: list) -> list:
    """Interactively select images to process."""
    print("\n" + "="*70)
    print("SELECT IMAGES TO PROCESS")
    print("="*70)
    print("\nOptions:")
    print("  - Enter image numbers (comma-separated): 1,3,5,7")
    print("  - Enter range: 1-10")
    print("  - Enter 'all' to process all images")
    print("  - Enter 'quit' to exit")

    while True:
        selection = input("\nYour selection: ").strip()

        if selection.lower() == 'quit':
            sys.exit(0)

        if selection.lower() == 'all':
            return all_images

        try:
            selected = []

            # Handle ranges and comma-separated
            for part in selection.split(','):
                part = part.strip()

                if '-' in part:
                    # Range
                    start, end = part.split('-')
                    start_idx = int(start.strip()) - 1
                    end_idx = int(end.strip())
                    selected.extend(all_images[start_idx:end_idx])
                else:
                    # Single number
                    idx = int(part) - 1
                    if 0 <= idx < len(all_images):
                        selected.append(all_images[idx])

            if selected:
                print(f"\nSelected {len(selected)} images:")
                for img in selected:
                    print(f"  - {img.name}")

                confirm = input("\nProceed with these images? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected

        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")
            print("Please try again.")


def detect_and_save_crops(
    images: list,
    output_dir: Path,
    face_ordering: str = 'detection_order'
):
    """Detect faces and save crops with proper naming."""

    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering=face_ordering)

    all_detections = []

    print("\n" + "="*70)
    print("DETECTING FACES")
    print("="*70)
    print(f"Face ordering: {face_ordering}")

    for image_path in images:
        print(f"\n{image_path.name}:")

        # Load image
        try:
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_rgb = np.array(pil_img)
        except Exception as e:
            print(f"  ERROR: Failed to load - {e}")
            continue

        # Detect faces
        faces = embedder.app.get(img_rgb)
        print(f"  Detected {len(faces)} faces")

        if len(faces) == 0:
            print("  Skipping (no faces)")
            continue

        # Save crops
        for face_idx, face in enumerate(faces):
            # Get aligned crop
            from insightface.utils import face_align
            aligned_crop = face_align.norm_crop(img_rgb, face.kps)

            # Generate filename
            image_stem = image_path.stem
            crop_filename = f"{image_stem}_face_{face_idx}.jpg"
            crop_path = output_dir / crop_filename

            # Save crop
            Image.fromarray(aligned_crop).save(crop_path, 'JPEG', quality=95)

            # Extract embedding
            embedding = face.embedding / np.linalg.norm(face.embedding)

            # Store detection info
            detection = {
                'crop_filename': crop_filename,
                'source_image': image_path.name,
                'source_path': str(image_path),
                'face_index': face_idx,
                'bbox': face.bbox.tolist(),
                'embedding': embedding.tolist(),
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else None,
            }

            all_detections.append(detection)

            print(f"    Face {face_idx}: {crop_filename}")

    print(f"\n  Total crops saved: {len(all_detections)}")

    return all_detections


def create_mapping_csv(detections: list, output_file: Path):
    """Create mapping CSV with traceability."""

    print("\n" + "="*70)
    print("CREATING MAPPING CSV")
    print("="*70)

    with open(output_file, 'w') as f:
        # Header
        f.write("crop_filename,source_image,face_index,source_path\n")

        # Write each detection
        for det in detections:
            f.write(f"{det['crop_filename']},{det['source_image']},{det['face_index']},{det['source_path']}\n")

    print(f"Mapping saved to: {output_file}")
    print(f"Total entries: {len(detections)}")


def save_metadata(detections: list, output_file: Path):
    """Save detailed metadata JSON."""

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': str(Path(output_file).stat().st_mtime),
            'n_faces': len(detections),
            'face_ordering': 'detection_order',
            'detections': detections
        }, f, indent=2)

    print(f"Metadata saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create ground truth face dataset')
    parser.add_argument('--source-dir', type=Path, default=Path(r"D:\Google_Germany"),
                       help='Source images directory')
    parser.add_argument('--output-dir', type=Path, default=Path("test_data/ground_truth_fresh"),
                       help='Output directory for crops and metadata')
    parser.add_argument('--face-ordering', default='detection_order',
                       choices=['detection_order', 'reading_order', 'area', 'confidence'],
                       help='Face ordering convention')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-select all images (non-interactive)')

    args = parser.parse_args()

    print("="*70)
    print("GROUND TRUTH CREATION")
    print("="*70)
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Face ordering: {args.face_ordering}")

    # Check source directory exists
    if not args.source_dir.exists():
        print(f"\nERROR: Source directory not found: {args.source_dir}")
        sys.exit(1)

    # List available images
    all_images = list_images(args.source_dir)

    if not all_images:
        print("\nNo images found!")
        sys.exit(1)

    # Select images to process
    if args.auto:
        selected_images = all_images
        print(f"\nAuto-selected all {len(selected_images)} images")
    else:
        selected_images = select_images(all_images)

    if not selected_images:
        print("\nNo images selected!")
        sys.exit(1)

    # Create output directories
    crops_dir = args.output_dir / "face_crops"

    # Detect faces and save crops
    detections = detect_and_save_crops(
        selected_images,
        crops_dir,
        face_ordering=args.face_ordering
    )

    if not detections:
        print("\nNo faces detected!")
        sys.exit(1)

    # Create mapping CSV
    mapping_file = args.output_dir / "mapping.csv"
    create_mapping_csv(detections, mapping_file)

    # Save metadata
    metadata_file = args.output_dir / "metadata.json"
    save_metadata(detections, metadata_file)

    # Summary
    print("\n" + "="*70)
    print("SUCCESS")
    print("="*70)
    print(f"Processed: {len(selected_images)} images")
    print(f"Detected: {len(detections)} faces")
    print(f"Crops saved to: {crops_dir}")
    print(f"Mapping CSV: {mapping_file}")
    print(f"Metadata JSON: {metadata_file}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review the face crops in:", crops_dir)
    print("2. Manually label person identities (create labels.json)")
    print("3. Run test_face_clustering_isolated.py with this data")
    print("4. Once verified, expand to more images")

    print("\nExample labels.json format:")
    print("""{
  "20240816_142312_face_0.jpg": {"person_id": 1, "person_name": "Person A"},
  "20240816_142312_face_1.jpg": {"person_id": 2, "person_name": "Person B"},
  ...
}""")


if __name__ == '__main__':
    main()
