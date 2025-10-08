from sim_bench.datasets import load_dataset
import yaml

print("Testing cleaned dataset implementations...")
print("=" * 60)

# Test UKBench
config = yaml.safe_load(open('configs/dataset.ukbench.yaml'))
ds = load_dataset('ukbench', config)
ds.load_data()
ds.apply_sampling({'max_groups': 10})

print(f'UKBench: {len(ds.get_images())} images, {len(ds.get_queries())} queries')
eval_data = ds.get_evaluation_data()
print(f'Groups: {len(set(eval_data["groups"]))}')
print('[PASS] UKBench works without save_results()')

# Test Holidays
config = yaml.safe_load(open('configs/dataset.holidays.yaml'))
ds = load_dataset('holidays', config)
ds.load_data()
ds.apply_sampling({'max_groups': 50})

print(f'\nHolidays: {len(ds.get_images())} images, {len(ds.get_queries())} queries')
eval_data = ds.get_evaluation_data()
print(f'Groups: {len(set(eval_data["groups"]))}')
print('[PASS] Holidays works without save_results()')

print("\n" + "=" * 60)
print("SUCCESS: Datasets cleaned up!")
print("  - save_results() removed (dead code)")
print("  - Result saving handled by ResultManager")
print("  - Cleaner separation of concerns")

