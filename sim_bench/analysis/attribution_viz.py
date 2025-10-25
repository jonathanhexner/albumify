"""
Common visualization functions for feature attribution analysis.

Handles both deep learning (Grad-CAM) and SIFT BoVW (keypoint) visualizations
in a unified interface.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image


def get_top_variance_features(
    diversity_results: Dict[int, Dict[str, Any]], 
    top_k: int = 4
) -> Tuple[List[int], Dict[int, float]]:
    """
    Get top-k features with highest variance across all groups.
    
    Args:
        diversity_results: Results from analyze_within_group_feature_diversity()
        top_k: Number of top features to return
        
    Returns:
        Tuple of (top_feature_ids, all_variances_dict)
    """
    all_variances = {}
    for gid, result in diversity_results.items():
        top_dims = result['top_diverse_dims_by_metric']['variance']
        top_vars = result['top_diverse_values_by_metric']['variance']
        for dim, var in zip(top_dims, top_vars):
            if dim not in all_variances or var > all_variances[dim]:
                all_variances[dim] = var
    
    top_features = sorted(all_variances.keys(), 
                         key=lambda k: all_variances[k], 
                         reverse=True)[:top_k]
    
    return top_features, all_variances


def visualize_deep_attribution(
    extractor,
    query_indices: List[int],
    query_group_ids: List[int],
    image_paths: List[str],
    top_features: List[int],
    all_variances: Dict[int, float],
    plot_attribution_overlay_func
) -> plt.Figure:
    """
    Visualize Grad-CAM attribution for deep learning features.
    
    Args:
        extractor: ResNet50AttributionExtractor instance
        query_indices: Query image indices
        query_group_ids: Group IDs for queries
        image_paths: All image paths
        top_features: Top-k feature dimensions to visualize
        all_variances: Variance values for all features
        plot_attribution_overlay_func: Function to create overlay
        
    Returns:
        Matplotlib figure
    """
    print(f"Visualizing Grad-CAM for top {len(top_features)} high-variance features:")
    for i, dim in enumerate(top_features, 1):
        print(f"  {i}. Feature {dim}: Variance = {all_variances[dim]:.6f}")
    
    n_queries = len(query_indices)
    n_features = len(top_features)
    
    fig, axes = plt.subplots(n_queries, n_features + 1, 
                            figsize=((n_features + 1) * 3, n_queries * 3))
    
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for q_idx, query_idx in enumerate(query_indices):
        query_path = image_paths[query_idx]
        query_gid = query_group_ids[q_idx]
        
        # Show original image
        ax = axes[q_idx, 0]
        img = Image.open(query_path)
        ax.imshow(img)
        ax.set_title(f'Query {query_idx}\nGroup {query_gid}', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Generate Grad-CAM for each high-variance feature
        for f_idx, feature_dim in enumerate(top_features, 1):
            ax = axes[q_idx, f_idx]
            
            try:
                attribution_map, img_array = extractor.compute_attribution(
                    image_path=query_path,
                    feature_indices=[feature_dim]
                )
                
                overlay = plot_attribution_overlay_func(
                    attribution_map=attribution_map,
                    original_image=img_array,
                    alpha=0.5
                )
                
                ax.imshow(overlay)
                ax.set_title(f'Feature {feature_dim}\nVar={all_variances[feature_dim]:.4f}',
                           fontsize=10)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', 
                       ha='center', va='center', fontsize=8)
                ax.axis('off')
    
    plt.suptitle(f'Grad-CAM: High-Variance Features by Query\n' +
                f'(Heatmaps show which pixels activate each feature)',
                fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    return fig


def visualize_sift_attribution(
    extractor,
    query_indices: List[int],
    query_group_ids: List[int],
    image_paths: List[str],
    top_features: List[int],
    all_variances: Dict[int, float],
    draw_keypoints_func
) -> plt.Figure:
    """
    Visualize keypoint locations for SIFT BoVW visual words.
    
    Args:
        extractor: SIFTBoVWAttributionExtractor instance
        query_indices: Query image indices
        query_group_ids: Group IDs for queries
        image_paths: All image paths
        top_features: Top-k visual word IDs to visualize
        all_variances: Variance values for all visual words
        draw_keypoints_func: Function to draw keypoints
        
    Returns:
        Matplotlib figure
    """
    print(f"Visualizing keypoints for top {len(top_features)} high-variance visual words:")
    for i, word_id in enumerate(top_features, 1):
        print(f"  {i}. Visual Word {word_id}: Variance = {all_variances[word_id]:.6f}")
    
    n_queries = len(query_indices)
    n_features = len(top_features)
    
    fig, axes = plt.subplots(n_queries, n_features + 1, 
                            figsize=((n_features + 1) * 3, n_queries * 3))
    
    if n_queries == 1:
        axes = axes.reshape(1, -1)
    
    for q_idx, query_idx in enumerate(query_indices):
        query_path = image_paths[query_idx]
        query_gid = query_group_ids[q_idx]
        
        # Show original image
        ax = axes[q_idx, 0]
        img = Image.open(query_path)
        ax.imshow(img)
        ax.set_title(f'Query {query_idx}\nGroup {query_gid}', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Visualize keypoints for each high-variance visual word
        for f_idx, word_id in enumerate(top_features, 1):
            ax = axes[q_idx, f_idx]
            
            try:
                keypoints_by_word, img_array = extractor.compute_attribution(
                    image_path=query_path,
                    feature_indices=[word_id]
                )
                
                word_keypoints = keypoints_by_word.get(word_id, [])
                
                # Draw keypoints with ENHANCED visibility
                img_with_kp = draw_keypoints_func(
                    image=img_array,
                    keypoints=word_keypoints,
                    color=(255, 0, 0),  # Red
                    draw_rich=True,
                    line_thickness=3,  # Thicker lines
                    circle_radius_multiplier=2.0  # Larger circles
                )
                
                ax.imshow(img_with_kp)
                ax.set_title(f'Word {word_id} ({len(word_keypoints)} kp)\nVar={all_variances[word_id]:.4f}',
                           fontsize=10)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', 
                       ha='center', va='center', fontsize=8)
                ax.axis('off')
    
    plt.suptitle(f'SIFT BoVW: High-Variance Visual Words by Query\n' +
                f'(Red keypoints with orientation arrows show where each word appears)',
                fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    return fig

