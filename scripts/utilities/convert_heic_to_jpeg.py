"""Convert HEIC images to JPEG format for Budapest2025_Google dataset."""
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Path to dataset
dataset_path = Path(r"D:\Budapest2025_Google")

# Find all HEIC files
heic_files = list(dataset_path.glob("*.heic")) + list(dataset_path.glob("*.HEIC"))

print(f"Found {len(heic_files)} HEIC files")
print(f"Converting to JPEG (quality=95)...\n")

converted = 0
skipped = 0
errors = []

for heic_path in tqdm(heic_files, desc="Converting"):
    try:
        jpg_path = heic_path.with_suffix('.jpg')
        
        # Skip if JPEG already exists
        if jpg_path.exists():
            skipped += 1
            continue
        
        # Convert
        img = Image.open(heic_path)
        rgb_img = img.convert('RGB')
        rgb_img.save(jpg_path, 'JPEG', quality=95)
        converted += 1
        
    except Exception as e:
        errors.append((heic_path.name, str(e)))

print(f"\n{'='*60}")
print(f"CONVERSION COMPLETE")
print(f"{'='*60}")
print(f"Converted: {converted}")
print(f"Skipped (already exists): {skipped}")
print(f"Errors: {len(errors)}")

if errors:
    print(f"\nErrors encountered:")
    for filename, error in errors[:10]:  # Show first 10
        print(f"  {filename}: {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")

# Final count
jpg_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.JPG"))
print(f"\nFinal dataset:")
print(f"  Total JPEG files: {len(jpg_files)}")
print(f"  Total HEIC files: {len(heic_files)}")

print(f"\nNote: HEIC files are kept. You can delete them if needed.")





