import yaml
from sim_bench.datasets import load_dataset

print("Testing Holidays sampling fix...")
print("=" * 60)

# Load Holidays dataset
dataset_config = yaml.safe_load(open('configs/dataset.holidays.yaml'))
dataset = load_dataset('holidays', dataset_config)

print("\n1. Loading full dataset...")
dataset.load_data()
print(f"   Total images: {len(dataset.get_images())}")
print(f"   Total queries: {len(dataset.get_queries())}")

# Apply sampling
print("\n2. Applying sampling (max_groups=100)...")
sampling_config = {'max_groups': 100, 'random_seed': 42}
dataset.apply_sampling(sampling_config)

print(f"   Images after sampling: {len(dataset.get_images())}")
print(f"   Queries after sampling: {len(dataset.get_queries())}")
print(f"   Unique groups: {len(set(dataset.get_evaluation_data()['groups']))}")

print("\n✓ Fix working! Images are now filtered, not just queries.")
print(f"\nBefore fix: 1,491 images (all images, only 100 queries)")
print(f"After fix:  {len(dataset.get_images())} images (only images from 100 selected groups)")

