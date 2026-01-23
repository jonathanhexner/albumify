"""
Extract sample face images from CASIA-WebFace for visual inspection.

Saves 5 images each for 4 different identities, with diagnostic info.
"""
import sys
import struct
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.datasets.face_dataset import MXNetRecordDataset


def read_record_with_metadata(rec_file, offset: int) -> dict:
    """
    Read a record and extract all metadata for debugging.

    Returns dict with: label, id1, id2, image_bytes
    """
    rec_file.seek(offset)

    # Read record header
    magic = struct.unpack('I', rec_file.read(4))[0]
    if magic != 0xced7230a:
        raise ValueError(f"Invalid magic at offset {offset}: {hex(magic)}")

    length_flag = struct.unpack('I', rec_file.read(4))[0]
    cflag = (length_flag >> 29) & 7
    length = length_flag & ((1 << 29) - 1)

    # Read data
    if cflag == 0:
        data = rec_file.read(length)
    else:
        data = rec_file.read(length)
        while cflag != 0:
            magic = struct.unpack('I', rec_file.read(4))[0]
            length_flag = struct.unpack('I', rec_file.read(4))[0]
            cflag = (length_flag >> 29) & 7
            length = length_flag & ((1 << 29) - 1)
            data += rec_file.read(length)

    # Parse header: flag(4) + label(4) + id1(8) + id2(8) = 24 bytes
    data_flag = struct.unpack('I', data[0:4])[0]
    label = struct.unpack('f', data[4:8])[0]
    id1 = struct.unpack('Q', data[8:16])[0]  # 8-byte unsigned int
    id2 = struct.unpack('Q', data[16:24])[0]
    image_bytes = data[24:]

    return {
        'data_flag': data_flag,
        'label': int(label),
        'label_raw': label,
        'id1': id1,
        'id2': id2,
        'image_bytes': image_bytes,
        'offset': offset
    }


def extract_samples(rec_path: str, output_dir: str, num_identities: int = 4, images_per_identity: int = 5):
    """
    Extract sample images from the dataset with full metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {rec_path}...")
    dataset = MXNetRecordDataset(rec_path)

    # Group samples by label (identity)
    label_to_indices = defaultdict(list)
    for idx, (offset, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    print(f"Found {len(label_to_indices)} unique identities")

    # Find identities with enough images
    valid_labels = [label for label, indices in label_to_indices.items()
                    if len(indices) >= images_per_identity]
    print(f"Found {len(valid_labels)} identities with >= {images_per_identity} images")

    # Pick spread-out identities (not just first few)
    import random
    random.seed(42)
    selected_labels = random.sample(valid_labels, min(num_identities, len(valid_labels)))

    print(f"\nExtracting images for {len(selected_labels)} identities...")

    # Open file for detailed reading
    with open(rec_path, 'rb') as rec_file:
        for i, label in enumerate(selected_labels):
            person_dir = output_path / f"person_{label}"
            person_dir.mkdir(exist_ok=True)

            indices = label_to_indices[label][:images_per_identity]

            print(f"\n=== Person {label} ({len(label_to_indices[label])} total images) ===")

            for j, idx in enumerate(indices):
                offset, stored_label = dataset.samples[idx]

                # Read with full metadata
                meta = read_record_with_metadata(rec_file, offset)

                # Save image with metadata in filename
                img_path = person_dir / f"img_{j}_id1_{meta['id1']}_id2_{meta['id2']}.jpg"
                with open(img_path, 'wb') as f:
                    f.write(meta['image_bytes'])

                print(f"  [{j}] offset={offset}, label={meta['label']}, id1={meta['id1']}, id2={meta['id2']}")

            # Save metadata file
            meta_path = person_dir / "metadata.txt"
            with open(meta_path, 'w') as f:
                f.write(f"Label: {label}\n")
                f.write(f"Total images with this label: {len(label_to_indices[label])}\n")
                f.write(f"\nSample indices and offsets:\n")
                for j, idx in enumerate(indices):
                    offset, _ = dataset.samples[idx]
                    f.write(f"  img_{j}: idx={idx}, offset={offset}\n")

    print(f"\nDone! Images saved to {output_path}")
    return output_path


def diagnose_label(rec_path: str, label: int, max_images: int = 20):
    """
    Diagnose a specific label - show all metadata for images with that label.
    """
    print(f"\nDiagnosing label {label}...")

    dataset = MXNetRecordDataset(rec_path)

    # Find all samples with this label
    matching = [(idx, offset) for idx, (offset, l) in enumerate(dataset.samples) if l == label]
    print(f"Found {len(matching)} images with label {label}")

    with open(rec_path, 'rb') as rec_file:
        print(f"\nFirst {min(max_images, len(matching))} images:")
        for idx, offset in matching[:max_images]:
            meta = read_record_with_metadata(rec_file, offset)
            print(f"  idx={idx:6d}, offset={offset:10d}, label={meta['label']:5d}, "
                  f"id1={meta['id1']:10d}, id2={meta['id2']:10d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract face samples for inspection")
    parser.add_argument("--rec-path", default="D:/DataSets/casia_Webface/casia-webface/train.rec",
                        help="Path to .rec file")
    parser.add_argument("--output-dir", default="outputs/face/samples",
                        help="Output directory")
    parser.add_argument("--num-identities", type=int, default=10,
                        help="Number of identities to sample")
    parser.add_argument("--images-per-identity", type=int, default=5,
                        help="Images per identity")
    parser.add_argument("--diagnose-label", type=int, default=None,
                        help="Diagnose a specific label (e.g., 1824)")

    args = parser.parse_args()

    if args.diagnose_label is not None:
        diagnose_label(args.rec_path, args.diagnose_label)
    else:
        extract_samples(
            rec_path=args.rec_path,
            output_dir=args.output_dir,
            num_identities=args.num_identities,
            images_per_identity=args.images_per_identity
        )
