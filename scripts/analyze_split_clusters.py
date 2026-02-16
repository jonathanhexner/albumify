"""Analyze cases where close faces are in different clusters."""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import pandas as pd

# Load results
results_dir = Path('results/face_clustering_benchmark')
benchmark_files = sorted(results_dir.glob('benchmark_*.json'), reverse=True)
with open(benchmark_files[0], 'r') as f:
    results = json.load(f)

metadata = results['face_metadata']
embeddings_file = results_dir / results['embeddings_file']
embeddings = np.load(embeddings_file)
hdbscan_labels = np.array(results['methods']['hdbscan']['labels'])
hybrid_labels = np.array(results['methods']['hybrid_knn']['labels'])

print("="*70)
print("FINDING OVERSPLIT CASES")
print("="*70)
print(f"Total faces: {len(metadata)}")
print(f"HDBSCAN clusters: {len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)}")
print(f"Noise points: {sum(hdbscan_labels == -1)}")

# Find pairs of faces in different clusters with small distance
print("\n" + "="*70)
print("CLOSE FACES IN DIFFERENT HDBSCAN CLUSTERS")
print("="*70)

splits = []
for i in range(len(embeddings)):
    if hdbscan_labels[i] == -1:  # Skip noise
        continue
    for j in range(i+1, len(embeddings)):
        if hdbscan_labels[j] == -1:  # Skip noise
            continue
        if hdbscan_labels[i] != hdbscan_labels[j]:  # Different clusters
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            cosine_dist = 1 - np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if cosine_dist < 0.35:  # Close enough to be same person
                splits.append({
                    'Face_A': i,
                    'Face_B': j,
                    'Distance': cosine_dist,
                    'Cluster_A': hdbscan_labels[i],
                    'Cluster_B': hdbscan_labels[j],
                    'Image_A': Path(metadata[i]['image_path']).name,
                    'Image_B': Path(metadata[j]['image_path']).name,
                    'Conf_A': metadata[i]['confidence'],
                    'Conf_B': metadata[j]['confidence'],
                })

if splits:
    df = pd.DataFrame(splits).sort_values('Distance')
    print(f"\nFound {len(df)} pairs of close faces in different clusters:")
    print(df.head(20).to_string(index=False))
    
    print("\n" + "="*70)
    print("WORST CASE (closest faces that are split):")
    print("="*70)
    worst = df.iloc[0]
    print(f"Face {int(worst['Face_A'])} (cluster {int(worst['Cluster_A'])}) vs")
    print(f"Face {int(worst['Face_B'])} (cluster {int(worst['Cluster_B'])})")
    print(f"Distance: {worst['Distance']:.4f}")
    print(f"Images: {worst['Image_A']} vs {worst['Image_B']}")
    print(f"\nThese faces should likely be in the SAME cluster!")
    print(f"Use: python scripts/debug_specific_faces.py {int(worst['Face_A'])} {int(worst['Face_B'])}")
else:
    print("\n[OK] No close faces found in different clusters!")
    print("HDBSCAN is doing a good job.")

# Check noise points that are close to clusters
print("\n" + "="*70)
print("NOISE POINTS CLOSE TO CLUSTERS")
print("="*70)

noise_indices = np.where(hdbscan_labels == -1)[0]
non_noise_indices = np.where(hdbscan_labels != -1)[0]

if len(noise_indices) > 0 and len(non_noise_indices) > 0:
    # For each noise point, find closest non-noise point
    noise_embs = embeddings[noise_indices]
    cluster_embs = embeddings[non_noise_indices]
    
    distances = cdist(noise_embs, cluster_embs, metric='cosine')
    min_dists = np.min(distances, axis=1)
    closest_indices = np.argmin(distances, axis=1)
    
    close_noise = []
    for i, (noise_idx, min_dist, closest_idx) in enumerate(
        zip(noise_indices, min_dists, closest_indices)
    ):
        if min_dist < 0.38:  # Within attach threshold
            closest_face = non_noise_indices[closest_idx]
            close_noise.append({
                'Noise_Face': noise_idx,
                'Closest_Face': closest_face,
                'Distance': min_dist,
                'Closest_Cluster': hdbscan_labels[closest_face],
                'Noise_Image': Path(metadata[noise_idx]['image_path']).name,
                'Closest_Image': Path(metadata[closest_face]['image_path']).name,
            })
    
    if close_noise:
        df_noise = pd.DataFrame(close_noise).sort_values('Distance')
        print(f"\nFound {len(df_noise)} noise points close to clusters:")
        print(df_noise.head(15).to_string(index=False))
        print(f"\nThese should have been attached by hybrid algorithm!")
    else:
        print("\n[INFO] No noise points close enough to clusters (all > 0.38)")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("If you see many close faces split:")
print("  1. HDBSCAN is being too aggressive (high min_cluster_size)")
print("  2. Try: min_cluster_size=3, cluster_selection_epsilon=0.35")
print("\nIf noise points should be attached:")
print("  3. Increase singleton_attach_threshold (try 0.42 or 0.45)")
print("  4. Decrease singleton_knn_k (try 2)")
print("\nIf clusters should merge:")
print("  5. Decrease merge_min_links (try 1)")
print("  6. Increase merge_distance_ceiling (try 0.50)")
print("="*70)
