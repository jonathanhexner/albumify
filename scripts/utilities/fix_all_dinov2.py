"""Fix all DINOv2 metrics.csv files."""
import pandas as pd
from pathlib import Path
import csv
from datetime import datetime
import shutil

def fix_metrics_csv(result_dir):
    """Fix a single dinov2 result directory."""
    
    rankings_file = result_dir / "rankings.csv"
    per_query_file = result_dir / "per_query.csv"
    metrics_file = result_dir / "metrics.csv"
    
    if not rankings_file.exists():
        print(f"  [SKIP] No rankings.csv in {result_dir}")
        return False
    
    # Determine dataset type
    dataset_type = 'holidays' if 'holidays' in str(result_dir) else 'ukbench'
    
    print(f"\n{result_dir.relative_to('outputs')}")
    print(f"  Dataset: {dataset_type}")
    
    # Load per_query to get the metrics
    if not per_query_file.exists():
        print(f"  [SKIP] No per_query.csv")
        return False
    
    df = pd.read_csv(per_query_file)
    
    # Calculate metrics
    metrics = {}
    
    if dataset_type == 'ukbench':
        if 'ns_hitcount@4' in df.columns:
            metrics['ns_score'] = df['ns_hitcount@4'].mean()
        if 'ap@10' in df.columns:
            metrics['map@10'] = df['ap@10'].mean()
        
        # Calculate recall from rankings
        rankings_df = pd.read_csv(rankings_file)
        queries = rankings_df['query_filename'].unique()
        
        recall_at_1 = 0
        recall_at_4 = 0
        
        for query_file in queries:
            query_results = rankings_df[rankings_df['query_filename'] == query_file].head(5)
            query_num = int(query_file.replace('ukbench', '').replace('.jpg', ''))
            query_group = query_num // 4
            
            # Recall@1
            rank_1 = query_results[query_results['rank'] == 1]
            if not rank_1.empty:
                result_num = int(rank_1.iloc[0]['result_filename'].replace('ukbench', '').replace('.jpg', ''))
                if result_num // 4 == query_group:
                    recall_at_1 += 1
            
            # Recall@4
            for _, row in query_results.iterrows():
                if row['rank'] == 0:
                    continue
                result_num = int(row['result_filename'].replace('ukbench', '').replace('.jpg', ''))
                if result_num // 4 == query_group:
                    recall_at_4 += 1
                    break
        
        metrics['recall@1'] = recall_at_1 / len(queries)
        metrics['recall@4'] = recall_at_4 / len(queries)
        metrics['num_queries'] = len(queries)
        metrics['num_gallery'] = len(queries)
        
    else:  # holidays
        if 'ap_full' in df.columns:
            metrics['map'] = df['ap_full'].mean()
        if 'ap@10' in df.columns:
            metrics['map@10'] = df['ap@10'].mean()
        if 'ap@50' in df.columns:
            metrics['map@50'] = df['ap@50'].mean()
        if 'recall@10' in df.columns:
            metrics['recall@10'] = df['recall@10'].mean()
        
        metrics['num_queries'] = len(df)
        metrics['num_gallery'] = len(df)
        metrics['recall@1'] = 0.0
        metrics['precision@10'] = 0.0
    
    # Backup original
    if metrics_file.exists():
        backup = metrics_file.parent / "metrics.backup.csv"
        shutil.copy2(metrics_file, backup)
    
    # Write new metrics
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if dataset_type == 'ukbench':
            header = ["method", "ns", "recall@1", "recall@4", "map@10", "num_queries", "num_gallery", "created_at"]
            row = [
                "dinov2",
                f"{metrics.get('ns_score', 0):.6f}",
                f"{metrics.get('recall@1', 0):.6f}",
                f"{metrics.get('recall@4', 0):.6f}",
                f"{metrics.get('map@10', 0):.6f}",
                metrics['num_queries'],
                metrics['num_gallery'],
                datetime.now().isoformat()
            ]
        else:
            header = ["method", "map", "map@10", "map@50", "recall@1", "recall@10", "precision@10", "num_queries", "num_gallery", "created_at"]
            row = [
                "dinov2",
                f"{metrics.get('map', 0):.6f}",
                f"{metrics.get('map@10', 0):.6f}",
                f"{metrics.get('map@50', 0):.6f}",
                f"{metrics.get('recall@1', 0):.6f}",
                f"{metrics.get('recall@10', 0):.6f}",
                f"{metrics.get('precision@10', 0):.6f}",
                metrics['num_queries'],
                metrics['num_gallery'],
                datetime.now().isoformat()
            ]
        
        writer.writerow(header)
        writer.writerow(row)
    
    print(f"  [FIXED] {metrics_file.name}")
    
    # Print key metrics
    if dataset_type == 'ukbench':
        print(f"    N-S Score: {metrics.get('ns_score', 0):.4f}")
        print(f"    Recall@1:  {metrics.get('recall@1', 0):.4f}")
    else:
        print(f"    mAP:       {metrics.get('map', 0):.4f}")
        print(f"    mAP@10:    {metrics.get('map@10', 0):.4f}")
    
    return True

# Find all dinov2 directories
base_dir = Path("outputs/baseline_runs/comprehensive_baseline")
fixed_count = 0

for unified_dir in base_dir.glob("unified_benchmark_*"):
    for dinov2_dir in unified_dir.rglob("dinov2"):
        if dinov2_dir.is_dir():
            if fix_metrics_csv(dinov2_dir):
                fixed_count += 1

print(f"\n{'='*60}")
print(f"[SUCCESS] Fixed {fixed_count} metrics.csv files!")
print('='*60)



