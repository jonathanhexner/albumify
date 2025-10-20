"""
Visualization utilities for feature space exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from typing import Optional, Tuple, List, Dict
from pathlib import Path


def plot_feature_distributions(
    features: np.ndarray,
    max_dims: int = 20,
    figsize: Tuple[int, int] = (15, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions of feature dimensions.
    
    Args:
        features: Feature matrix [n_images, feature_dim]
        max_dims: Maximum number of dimensions to plot
        figsize: Figure size
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    n_dims = min(max_dims, features.shape[1])
    
    fig, axes = plt.subplots(4, 5, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        ax.hist(features[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Dim {i}', fontsize=10)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
    
    # Hide unused axes
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.00)
    
    plt.tight_layout()
    return fig


def plot_feature_correlation_heatmap(
    correlation_matrix: np.ndarray,
    max_dims: int = 50,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation heatmap for feature dimensions.
    
    Args:
        correlation_matrix: Correlation matrix [n_dims, n_dims]
        max_dims: Maximum dimensions to display
        figsize: Figure size
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    n_dims = min(max_dims, correlation_matrix.shape[0])
    corr_subset = correlation_matrix[:n_dims, :n_dims]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(title or 'Feature Correlation Matrix', fontsize=14, pad=20)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    
    return fig


def plot_pca_explained_variance(
    explained_variance_ratio: np.ndarray,
    n_components: int = 50,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot PCA explained variance (scree plot and cumulative).
    
    Args:
        explained_variance_ratio: Array of explained variance ratios from PCA
        n_components: Number of components to display
        figsize: Figure size
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    n_show = min(n_components, len(explained_variance_ratio))
    cumsum = np.cumsum(explained_variance_ratio[:n_show])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scree plot
    ax1.bar(range(1, n_show + 1), explained_variance_ratio[:n_show], alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot', fontsize=13)
    ax1.grid(alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, n_show + 1), cumsum, marker='o', linewidth=2)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.00)
    
    plt.tight_layout()
    return fig


def plot_embedding_2d(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = 't-SNE Embedding',
    figsize: Tuple[int, int] = (12, 10),
    alpha: float = 0.6,
    s: int = 20,
    cmap: str = 'tab20'
) -> plt.Figure:
    """
    Plot 2D embedding (t-SNE, UMAP, PCA) with optional color by labels.
    
    Args:
        embedding: 2D embedding coordinates [n_images, 2]
        labels: Optional group labels for coloring
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        cmap: Colormap for labels
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        # Color by labels
        scatter = ax.scatter(
            embedding[:, 0], 
            embedding[:, 1],
            c=labels,
            cmap=cmap,
            alpha=alpha,
            s=s,
            edgecolors='k',
            linewidth=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Group ID')
    else:
        # No labels, uniform color
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=alpha,
            s=s,
            edgecolors='k',
            linewidth=0.5
        )
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_distance_distribution(
    intra_distances: np.ndarray,
    inter_distances: np.ndarray,
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of intra-class vs inter-class distances.
    
    Args:
        intra_distances: Array of intra-class distances
        inter_distances: Array of inter-class distances
        bins: Number of histogram bins
        figsize: Figure size
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(intra_distances, bins=bins, alpha=0.6, label='Intra-class', color='blue', edgecolor='black')
    ax.hist(inter_distances, bins=bins, alpha=0.6, label='Inter-class', color='red', edgecolor='black')
    
    # Add mean lines
    ax.axvline(intra_distances.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Intra mean: {intra_distances.mean():.3f}')
    ax.axvline(inter_distances.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Inter mean: {inter_distances.mean():.3f}')
    
    ax.set_xlabel('Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title or 'Intra-class vs Inter-class Distance Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_statistics_summary(
    stats_df: pd.DataFrame,
    metrics: List[str] = ['mean', 'std', 'min', 'max'],
    figsize: Tuple[int, int] = (15, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot summary statistics for all feature dimensions.
    
    Args:
        stats_df: DataFrame from compute_feature_statistics()
        metrics: List of metric columns to plot
        figsize: Figure size
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        ax.plot(stats_df['dimension'], stats_df[metric], linewidth=1.5)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_title(f'{metric.capitalize()} across dimensions', fontsize=12)
    
    axes[-1].set_xlabel('Feature Dimension', fontsize=12)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout()
    return fig


def plot_nearest_neighbors_grid(
    query_idx: int,
    neighbor_indices: np.ndarray,
    distances: np.ndarray,
    image_paths: List[str],
    labels: Optional[np.ndarray] = None,
    n_show: int = 10,
    figsize_per_image: Tuple[float, float] = (2.5, 2.5),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot query image with its nearest neighbors.
    
    Args:
        query_idx: Index of query image
        neighbor_indices: Indices of nearest neighbors (sorted by distance)
        distances: Distances to neighbors
        image_paths: List of all image paths
        labels: Optional group labels
        n_show: Number of neighbors to show
        figsize_per_image: Size per image
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    from PIL import Image
    
    n_show = min(n_show, len(neighbor_indices) - 1)  # -1 to exclude self
    n_cols = min(5, n_show + 1)
    n_rows = (n_show + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_image[0], n_rows * figsize_per_image[1])
    )
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Plot query
    query_path = image_paths[query_idx]
    img = Image.open(query_path)
    axes[0].imshow(img)
    query_label = labels[query_idx] if labels is not None else '?'
    axes[0].set_title(f'QUERY\nIdx={query_idx}\nGroup={query_label}', 
                      fontsize=10, color='blue', fontweight='bold')
    axes[0].axis('off')
    
    # Plot neighbors (skip first as it's the query itself)
    for i, (nn_idx, dist) in enumerate(zip(neighbor_indices[1:n_show+1], distances[1:n_show+1]), start=1):
        nn_path = image_paths[nn_idx]
        img = Image.open(nn_path)
        axes[i].imshow(img)
        
        nn_label = labels[nn_idx] if labels is not None else '?'
        match = '✓' if labels is not None and nn_label == query_label else '✗'
        color = 'green' if match == '✓' else 'red'
        
        axes[i].set_title(f'#{i} {match}\nIdx={nn_idx}\nGrp={nn_label}\nd={dist:.3f}',
                         fontsize=9, color=color)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_show + 1, len(axes)):
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.00)
    
    plt.tight_layout()
    return fig


def plot_cluster_quality_metrics(
    silhouette_scores: np.ndarray,
    labels: np.ndarray,
    silhouette_avg: float,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot silhouette analysis for clustering quality.
    
    Args:
        silhouette_scores: Per-sample silhouette scores
        labels: Cluster labels
        silhouette_avg: Average silhouette score
        figsize: Figure size
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    y_lower = 10
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        # Get silhouette scores for this cluster
        cluster_scores = silhouette_scores[labels == label]
        cluster_scores.sort()
        
        size_cluster = cluster_scores.shape[0]
        y_upper = y_lower + size_cluster
        
        color = plt.cm.nipy_spectral(float(i) / len(unique_labels))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_scores,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        
        # Label the silhouette plots with cluster numbers
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
        y_lower = y_upper + 10
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(title or f'Silhouette Analysis (avg={silhouette_avg:.3f})', fontsize=14)
    
    # Add vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2)
    
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    
    plt.tight_layout()
    return fig


def plot_queries_by_group(
    query_features: np.ndarray,
    query_indices: List[int],
    query_group_ids: List[int],
    all_features: Optional[np.ndarray] = None,
    method_name: str = "",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize query images in 2D feature space, colored by group.
    
    PCA is fit on the full dataset (if provided) for proper feature space representation.
    
    Args:
        query_features: Query feature matrix [n_queries, feature_dim]
        query_indices: Query indices
        query_group_ids: Group ID for each query
        all_features: Full dataset features for fitting PCA (optional)
        method_name: Method name for title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    
    if len(query_features) < 2:
        raise ValueError("Need at least 2 queries for 2D visualization")
    
    # Fit PCA on full dataset if provided, otherwise on queries only
    pca = PCA(n_components=2)
    if all_features is not None:
        pca.fit(all_features)
        query_2d = pca.transform(query_features)
    else:
        query_2d = pca.fit_transform(query_features)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get unique groups and assign colors
    unique_groups = sorted(set(query_group_ids))
    colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))
    group_to_color = dict(zip(unique_groups, colors))
    
    # Plot each query
    for i, (idx, gid) in enumerate(zip(query_indices, query_group_ids)):
        ax.scatter(query_2d[i, 0], query_2d[i, 1], 
                  c=[group_to_color[gid]], s=200, alpha=0.7, 
                  edgecolors='black', linewidth=2)
        ax.annotate(f'q{idx}\ng{gid}', 
                   (query_2d[i, 0], query_2d[i, 1]),
                   fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_to_color[g], label=f'Group {g}') 
                      for g in unique_groups]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    
    title = f'Query Images in Feature Space (n={len(query_indices)})'
    if method_name:
        title = f'{method_name.upper()}: {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_queries_vs_dataset_by_group(
    all_features: np.ndarray,
    query_features: np.ndarray,
    query_indices: List[int],
    query_group_ids: List[int],
    method_name: str = "",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize queries vs full dataset in 2D, with queries colored by group.
    
    Args:
        all_features: Full dataset features [n_images, feature_dim]
        query_features: Query features [n_queries, feature_dim]
        query_indices: Query indices
        query_group_ids: Group ID for each query
        method_name: Method name for title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_features)
    query_2d = pca.transform(query_features)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot dataset (as "others" in gray)
    ax.scatter(all_2d[:, 0], all_2d[:, 1], 
              c='lightgray', s=10, alpha=0.3, label='Others', rasterized=True)
    
    # Get unique groups and assign colors
    unique_groups = sorted(set(query_group_ids))
    colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))
    group_to_color = dict(zip(unique_groups, colors))
    
    # Plot query images (colored by group)
    for i, (idx, gid) in enumerate(zip(query_indices, query_group_ids)):
        ax.scatter(query_2d[i, 0], query_2d[i, 1], 
                  c=[group_to_color[gid]], s=300, alpha=0.9, 
                  edgecolors='black', linewidth=2.5, 
                  label=f'Group {gid}' if i == query_group_ids.index(gid) else None,
                  zorder=10)
        ax.annotate(f'q{idx}', 
                   (query_2d[i, 0], query_2d[i, 1]),
                   fontsize=9, ha='center', va='center', 
                   fontweight='bold', color='white', zorder=11)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    
    title = f'Queries vs Full Dataset in Feature Space\n({len(query_indices)} queries, {len(all_features)} total images)'
    if method_name:
        title = f'{method_name.upper()}: {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_query_feature_analysis_by_group(
    all_features: np.ndarray,
    query_features: np.ndarray,
    query_indices: List[int],
    query_group_ids: List[int],
    method_name: str = "",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    4-panel plot showing detailed feature analysis for each query, colored by group.
    
    Args:
        all_features: Full dataset features [n_images, feature_dim]
        query_features: Query features [n_queries, feature_dim]
        query_indices: Query indices
        query_group_ids: Group ID for each query
        method_name: Method name for title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    title = 'Feature Analysis: Individual Queries (Colored by Group)'
    if method_name:
        title = f'{method_name.upper()}: {title}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Get colors for groups
    unique_groups = sorted(set(query_group_ids))
    colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))
    group_to_color = dict(zip(unique_groups, colors))
    
    # 1. Feature value distributions by query
    ax = axes[0, 0]
    ax.hist(all_features.flatten(), bins=50, alpha=0.2, label='Full Dataset', 
            density=True, color='gray')
    for i, (qidx, gid) in enumerate(zip(query_indices, query_group_ids)):
        ax.hist(query_features[i], bins=50, alpha=0.4, 
                label=f'q{qidx} (g{gid})', density=True, 
                color=group_to_color[gid])
    ax.set_xlabel('Feature Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Feature Value Distribution by Query', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. L2 norms by query
    ax = axes[0, 1]
    dataset_norms = np.linalg.norm(all_features, axis=1)
    query_norms = np.linalg.norm(query_features, axis=1)
    ax.hist(dataset_norms, bins=50, alpha=0.2, label='Dataset', 
            density=True, color='gray')
    for i, (qidx, gid, qnorm) in enumerate(zip(query_indices, query_group_ids, query_norms)):
        ax.axvline(qnorm, color=group_to_color[gid], linewidth=2.5, 
                   alpha=0.8, label=f'q{qidx} (g{gid})')
    ax.set_xlabel('L2 Norm', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Feature Vector L2 Norms', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Per-dimension activations
    ax = axes[1, 0]
    dims = np.arange(query_features.shape[1])
    dataset_mean = all_features.mean(axis=0)
    ax.plot(dims, dataset_mean, color='lightgray', linewidth=1, 
            alpha=0.5, label='Dataset Mean')
    for i, (qidx, gid) in enumerate(zip(query_indices, query_group_ids)):
        ax.plot(dims, query_features[i], color=group_to_color[gid], 
                linewidth=1.5, alpha=0.7, label=f'q{qidx} (g{gid})')
    ax.set_xlabel('Feature Dimension', fontsize=11)
    ax.set_ylabel('Activation Value', fontsize=11)
    ax.set_title('Per-Dimension Activation Patterns', fontsize=12)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 4. Top-k most active dimensions per query
    ax = axes[1, 1]
    top_k = 20
    width = 0.8 / len(query_indices)
    x_base = np.arange(top_k)
    
    for i, (qidx, gid) in enumerate(zip(query_indices, query_group_ids)):
        top_dims = np.argsort(np.abs(query_features[i]))[-top_k:][::-1]
        top_vals = query_features[i][top_dims]
        x_pos = x_base + i * width
        ax.bar(x_pos, top_vals, width=width, 
               color=group_to_color[gid], alpha=0.7, 
               label=f'q{qidx} (g{gid})')
    
    ax.set_xlabel('Top-k Feature Rank', fontsize=11)
    ax.set_ylabel('Activation Value', fontsize=11)
    ax.set_title(f'Top {top_k} Most Active Features per Query', fontsize=12)
    ax.set_xticks(x_base + width * (len(query_indices) - 1) / 2)
    ax.set_xticklabels([f'{i+1}' for i in range(top_k)])
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_within_group_diversity(
    diversity_results: Dict[int, Dict[str, any]],
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize feature dimensions with highest variance within each group.
    
    Shows which features are inconsistent within groups (potential error sources).
    
    Args:
        diversity_results: Results from analyze_within_group_feature_diversity()
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_groups = len(diversity_results)
    
    if n_groups == 0:
        raise ValueError("No groups with sufficient images for diversity analysis")
    
    # Create subplots (2 columns)
    n_rows = (n_groups + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    fig.suptitle('Within-Group Feature Diversity Analysis\n' + 
                 '(High variance = features inconsistent within group)',
                 fontsize=14, fontweight='bold')
    
    colors = cm.tab10(np.linspace(0, 1, 10))
    
    for idx, (gid, result) in enumerate(sorted(diversity_results.items())):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Plot top diverse feature dimensions
        top_dims = result['top_diverse_dims']
        top_vars = result['top_diverse_variances']
        
        x_pos = np.arange(len(top_dims))
        ax.bar(x_pos, top_vars, color=colors[gid % 10], alpha=0.7)
        
        # Annotate with feature dimension numbers
        for i, (dim, var) in enumerate(zip(top_dims, top_vars)):
            ax.text(i, var, f'{dim}', ha='center', va='bottom', 
                   fontsize=7, rotation=45)
        
        ax.set_xlabel('Rank', fontsize=10)
        ax.set_ylabel('Variance', fontsize=10)
        ax.set_title(f'Group {gid} ({result["n_images"]} images)\n' + 
                    f'Mean var: {result["mean_variance"]:.4f}',
                    fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Show feature dimension IDs on x-axis
        ax.set_xticks(x_pos[::2])  # Show every other to avoid crowding
        ax.set_xticklabels([f'{i+1}' for i in x_pos[::2]], fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(diversity_results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


