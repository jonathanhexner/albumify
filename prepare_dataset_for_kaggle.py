"""
Prepare PhotoTriage dataset for Kaggle upload.

NOTE: The PhotoTriage dataset is already available on Kaggle!
      https://www.kaggle.com/datasets/ericwolter/triage
      
      You likely don't need this script unless you want to create
      your own private copy of the dataset.

This script:
1. Copies images to a clean directory structure
2. Creates a README for the Kaggle dataset
3. Note: CSV is included in sim-bench package, no need to upload

Usage:
    python prepare_dataset_for_kaggle.py --source /path/to/phototriage --output /path/to/kaggle_dataset
"""
import argparse
import shutil
from pathlib import Path


def prepare_dataset(source_dir: Path, output_dir: Path, csv_path: Path = None):
    """Prepare PhotoTriage dataset for Kaggle upload."""
    
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy image directories
    for subdir in ['train_val', 'test']:
        src = source_dir / subdir
        dst = output_dir / subdir
        
        if src.exists():
            print(f"\nCopying {subdir}/...")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            
            # Count images
            img_count = len(list(dst.rglob('*.JPG'))) + len(list(dst.rglob('*.jpg')))
            print(f"  {img_count} images copied")
        else:
            print(f"Warning: {src} not found!")
    
    # Note about CSV file
    print(f"\nNote: CSV file (photo_triage_pairs_embedding_labels.csv) is included in sim-bench package.")
    print(f"      No need to include it in the Kaggle dataset!")
    
    # Create README for Kaggle
    readme_content = """# PhotoTriage Dataset

This dataset contains image pairs from the PhotoTriage paper for training image quality ranking models.

## Contents

- `train_val/train_val_imgs/` - Training and validation images (~12,988 images)
- `test/test_imgs/` - Test images (~2,555 images)
- `photo_triage_pairs_embedding_labels.csv` - Pairwise comparison labels

## CSV Format

The CSV file contains pairwise comparisons with the following columns:
- `series_id` - Photo series identifier
- `compareFile1`, `compareFile2` - Image filenames
- `majority_label` - Ground truth label (which image is better)
- `Agreement` - Annotator agreement score
- Additional quality attribute labels

## Usage

This dataset is designed for training Siamese CNN models for pairwise image quality ranking.

See the sim-bench repository for training code: https://github.com/YOUR_USERNAME/sim-bench

## Citation

If you use this dataset, please cite:

```
@article{zeng2021phototriage,
  title={PhotoTriage: Enhancing Photo Series Through Targeted Prioritization},
  author={Zeng, Hui and Li, Zishuo and Lin, Stephen and Li, Xueting and Zhang, Lei},
  journal={arXiv preprint arXiv:2103.00430},
  year={2021}
}
```

## License

Please refer to the original PhotoTriage paper for license information.
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("\nREADME.md created")
    
    # Verify structure
    print("\n=== Verification ===")
    print(f"Directory structure:")
    for item in sorted(output_dir.rglob('*')):
        if item.is_file() and not item.name.startswith('.'):
            rel_path = item.relative_to(output_dir)
            print(f"  {rel_path}")
            if len(list(output_dir.rglob('*'))) > 100:  # Too many files
                print("  ... (showing first 100)")
                break
    
    # Size estimation
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"\nTotal size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    
    print("\n✓ Dataset prepared for Kaggle upload!")
    print(f"\nNext steps:")
    print(f"1. Go to https://www.kaggle.com/datasets")
    print(f"2. Click 'New Dataset'")
    print(f"3. Upload contents of: {output_dir}")
    print(f"4. Name it 'phototriage-dataset'")
    print(f"5. Add appropriate tags and description")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare PhotoTriage dataset for Kaggle')
    parser.add_argument('--source', required=True, help='Source directory with PhotoTriage data')
    parser.add_argument('--output', required=True, help='Output directory for Kaggle dataset')
    parser.add_argument('--csv-path', default=None, help='Path to photo_triage_pairs_embedding_labels.csv')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    csv_path = Path(args.csv_path) if args.csv_path else None
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    prepare_dataset(source_dir, output_dir, csv_path)


if __name__ == '__main__':
    main()

