import yaml

# Check run config
run_config = yaml.safe_load(open('configs/run.yaml'))
print("Run Config:")
print(f"  Dataset specified: {run_config.get('dataset')}")
print(f"  Sampling: {run_config.get('sampling', {})}")
print()

# Calculate expected results
max_groups = run_config.get('sampling', {}).get('max_groups')
dataset_name = run_config.get('dataset')

if dataset_name == 'ukbench':
    if max_groups:
        expected_images = max_groups * 4
        print(f"Expected for UKBench with max_groups={max_groups}:")
        print(f"  Total images: {expected_images} ({max_groups} groups × 4 images)")
    else:
        print("UKBench with NO sampling: 10,200 images (2,550 groups)")
        
elif dataset_name == 'holidays':
    if max_groups:
        print(f"Expected for Holidays with max_groups={max_groups}:")
        print(f"  Total groups (query series): {max_groups}")
        print(f"  Total images: VARIABLE (depends on which groups selected)")
        print(f"  Typical range: {max_groups * 2} to {max_groups * 30} images")
    else:
        print("Holidays with NO sampling: 1,491 images (500 groups)")

print()
print("=" * 60)
print("YOUR OBSERVATION: 1,491 images in 100 groups")
print("=" * 60)
print()

if dataset_name == 'ukbench':
    print("PROBLEM DETECTED!")
    print("  Config says 'ukbench' but 1,491 images is Holidays!")
    print("  UKBench with max_groups=100 should be 400 images.")
    print()
    print("LIKELY CAUSE:")
    print("  - You're actually running Holidays dataset, OR")
    print("  - Dataset path points to Holidays data, OR")
    print("  - You're overriding dataset in CLI")
else:
    print("This makes sense for Holidays!")
    print("  Holidays has 1,491 total images")
    print("  With max_groups=100, you selected 100 query series")
    print("  Each series has varying numbers of images")
    print(f"  Result: All 1,491 images from 100 query series")

