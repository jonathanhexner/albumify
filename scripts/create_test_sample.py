"""Copy a deterministic random sample of images from a source album to test_data.

Usage:
    python scripts/create_test_sample.py \
        --source D:/Google_Germany \
        --dest test_data/face_clustering_100 \
        --n 100 \
        --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def main():
    parser = argparse.ArgumentParser(description="Sample images for test data")
    parser.add_argument("--source", required=True, help="Source image directory")
    parser.add_argument("--dest", required=True, help="Destination directory")
    parser.add_argument("--n", type=int, default=100, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    all_images = sorted(
        p for p in source.rglob("*") if p.suffix.lower() in SUPPORTED
    )
    if len(all_images) < args.n:
        raise ValueError(f"Source has only {len(all_images)} images, need {args.n}")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(all_images, args.n))

    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.iterdir())
    if existing:
        print(f"Destination {dest} already has {len(existing)} files — skipping (delete manually to resample)")
        return

    for img in sampled:
        shutil.copy2(img, dest / img.name)
        print(f"  copied {img.name}")

    print(f"\nSampled {len(sampled)} images -> {dest}")


if __name__ == "__main__":
    main()
