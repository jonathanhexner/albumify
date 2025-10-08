"""Test that base class sampling works for both datasets."""
import yaml
from sim_bench.datasets import load_dataset

print("Testing Base Class Sampling Refactoring")
print("=" * 60)

# Test UKBench
print("\n1. UKBench Dataset:")
print("-" * 60)
ukbench_config = yaml.safe_load(open('configs/dataset.ukbench.yaml'))
ukbench = load_dataset('ukbench', ukbench_config)
ukbench.load_data()

print(f"Before sampling:")
print(f"  Images: {len(ukbench.get_images())}")
print(f"  Queries: {len(ukbench.get_queries())}")

ukbench.apply_sampling({'max_groups': 25, 'random_seed': 42})

print(f"After sampling (max_groups=25):")
print(f"  Images: {len(ukbench.get_images())}")  # Should be 100 (25*4)
print(f"  Queries: {len(ukbench.get_queries())}")
print(f"  Unique groups: {len(set(ukbench.get_evaluation_data()['groups']))}")

assert len(ukbench.get_images()) == 100, f"Expected 100 images, got {len(ukbench.get_images())}"
print("  [PASS] UKBench sampling works correctly!")

# Test Holidays
print("\n2. Holidays Dataset:")
print("-" * 60)
holidays_config = yaml.safe_load(open('configs/dataset.holidays.yaml'))
holidays = load_dataset('holidays', holidays_config)
holidays.load_data()

print(f"Before sampling:")
print(f"  Images: {len(holidays.get_images())}")
print(f"  Queries: {len(holidays.get_queries())}")

holidays.apply_sampling({'max_groups': 100, 'random_seed': 42})

print(f"After sampling (max_groups=100):")
print(f"  Images: {len(holidays.get_images())}")  # Should be ~274 (variable)
print(f"  Queries: {len(holidays.get_queries())}")  # Should be 100
print(f"  Unique groups: {len(set(holidays.get_evaluation_data()['groups']))}")

assert len(holidays.get_queries()) == 100, f"Expected 100 queries, got {len(holidays.get_queries())}"
assert len(holidays.get_images()) < 1491, f"Images should be filtered, got {len(holidays.get_images())}"
print("  [PASS] Holidays sampling works correctly!")

print("\n" + "=" * 60)
print("SUCCESS: Base class sampling works for both datasets!")
print("=" * 60)
print("\nKey benefits of refactoring:")
print("  1. Common logic in base class")
print("  2. Only 2 methods per dataset:")
print("     - _get_group_for_image()")
print("     - _remap_after_sampling()")
print("  3. Consistent behavior across datasets")
print("  4. Easier to add new datasets")

