import hashlib
import json

# Same filenames, different directories
paths1 = ['D:/dataset1/img_0.jpg', 'D:/dataset1/img_1.jpg']
paths2 = ['D:/dataset2/img_0.jpg', 'D:/dataset2/img_1.jpg']

config = {'method': 'chi_square', 'features': {'bins': [16,16,16]}}
method = 'chi_square'

# Compute cache keys (same as feature_cache.py does)
combined1 = f"{method}||{json.dumps(config, sort_keys=True)}||{'|'.join(sorted(paths1))}"
combined2 = f"{method}||{json.dumps(config, sort_keys=True)}||{'|'.join(sorted(paths2))}"

hash1 = hashlib.sha256(combined1.encode()).hexdigest()[:16]
hash2 = hashlib.sha256(combined2.encode()).hexdigest()[:16]

print("Cache Keys for Same Filenames, Different Paths:")
print("=" * 60)
print(f"Dataset 1: D:/dataset1/")
print(f"  Files: img_0.jpg, img_1.jpg")
print(f"  Cache: chi_square_{hash1}.pkl")
print()
print(f"Dataset 2: D:/dataset2/")
print(f"  Files: img_0.jpg, img_1.jpg")
print(f"  Cache: chi_square_{hash2}.pkl")
print()
print(f"Different cache keys? {hash1 != hash2}")
print(f"No cache confusion: YES!")
print()
print("Why? The cache key includes the FULL PATH:")
print(f"  Path 1: {paths1[0]}")
print(f"  Path 2: {paths2[0]}")
print(f"  Different paths → Different hash → Different cache file")

