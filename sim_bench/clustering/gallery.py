"""
HTML gallery generator for clustering results.
Creates an interactive web page to visualize clusters.
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader


def generate_cluster_gallery(
    output_dir: Path,
    image_paths: List[str],
    labels: np.ndarray,
    stats: Dict[str, Any],
    experiment_name: str = "Clustering Experiment"
) -> Path:
    """
    Generate an HTML gallery for clustering results.
    
    Args:
        output_dir: Directory containing clusters.csv
        image_paths: List of image paths
        labels: Cluster labels
        stats: Clustering statistics
        experiment_name: Name of the experiment
        
    Returns:
        Path to generated HTML file
    """
    # Group images by cluster
    clusters_dict = defaultdict(list)
    for img_path, label in zip(image_paths, labels):
        clusters_dict[int(label)].append(img_path)
    
    # Prepare clusters for template (sorted by size, noise last)
    clusters = []
    for cluster_id in sorted(clusters_dict.keys()):
        if cluster_id == -1:
            continue  # Add noise at the end
        
        images = []
        for img_path in clusters_dict[cluster_id]:
            images.append({
                'path': img_path,
                'filename': Path(img_path).name
            })
        
        clusters.append({
            'id': cluster_id,
            'images': images
        })
    
    # Sort by cluster size (largest first)
    clusters.sort(key=lambda c: len(c['images']), reverse=True)
    
    # Add noise cluster at the end if it exists
    if -1 in clusters_dict:
        noise_images = []
        for img_path in clusters_dict[-1]:
            noise_images.append({
                'path': img_path,
                'filename': Path(img_path).name
            })
        
        clusters.append({
            'id': -1,
            'images': noise_images
        })
    
    # Prepare template context
    context = {
        'experiment_name': experiment_name,
        'algorithm': stats['algorithm'],
        'params': stats['params'],
        'n_clusters': stats['n_clusters'],
        'total_images': len(image_paths),
        'clusters': clusters
    }
    
    # Add noise stats if available (DBSCAN)
    if 'n_noise' in stats:
        context['n_noise'] = stats['n_noise']
        context['noise_ratio'] = stats['noise_ratio']
    
    # Load Jinja2 template
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template('cluster_gallery.html')
    
    # Render HTML
    html_content = template.render(**context)
    
    # Save to file
    output_path = output_dir / 'clusters.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def open_in_browser(html_path: Path) -> None:
    """Open HTML file in default browser."""
    import webbrowser
    webbrowser.open(f'file:///{html_path}')

