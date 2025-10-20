"""
General visualization utilities for feature attribution.

These visualizations work across different attribution methods (ResNet-50, SIFT, etc.)
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

logger = logging.getLogger(__name__)


def plot_attribution_overlay(
    attribution_map: np.ndarray,
    original_image: np.ndarray,
    output_path: Optional[Path] = None,
    alpha: float = 0.5,
    title: Optional[str] = None
) -> np.ndarray:
    """
    Visualize attribution map overlaid on original image.
    
    Works for any attribution method (Grad-CAM, keypoint heatmaps, etc.)
    
    Args:
        attribution_map: Attribution map [H, W] with values in [0, 1]
        original_image: Original image [H, W, 3]
        output_path: Where to save visualization (if None, only returns array)
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        title: Optional title for the plot
        
    Returns:
        Overlaid visualization as numpy array
    """
    if not PLT_AVAILABLE:
        raise ImportError("matplotlib required for visualization")
    
    # Resize attribution map to match image size
    from scipy.ndimage import zoom
    h, w = original_image.shape[:2]
    if attribution_map.shape != (h, w):
        cam_resized = zoom(
            attribution_map,
            (h / attribution_map.shape[0], w / attribution_map.shape[1]),
            order=1
        )
    else:
        cam_resized = attribution_map
    
    # Create heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3] * 255
    heatmap = heatmap.astype(np.uint8)
    
    # Overlay on image
    overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
    
    # Save if path provided
    if output_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Attribution Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return overlay


def plot_attribution_comparison(
    image_paths: List[str],
    attribution_extractor,
    output_path: Path,
    feature_indices: Optional[List[int]] = None,
    title: Optional[str] = None
) -> None:
    """
    Compare attribution maps across multiple images.
    
    Works with any attribution extractor that implements compute_attribution().
    
    Args:
        image_paths: List of image paths to compare
        attribution_extractor: Extractor with compute_attribution() method
        output_path: Where to save comparison
        feature_indices: Which features to visualize (None = auto-detect common)
        title: Optional title
    """
    if not PLT_AVAILABLE:
        raise ImportError("matplotlib required for visualization")
    
    n_images = len(image_paths)
    
    # If no features specified, find common top features
    if feature_indices is None:
        all_top_features = []
        for img_path in image_paths:
            result = attribution_extractor.analyze_feature_importance(img_path, top_k=20)
            all_top_features.extend(result['top_indices'][:5])
        
        # Find most common features
        from collections import Counter
        feature_indices = [f for f, _ in Counter(all_top_features).most_common(3)]
    
    # Create comparison grid
    fig, axes = plt.subplots(
        n_images, len(feature_indices) + 1,
        figsize=(4 * (len(feature_indices) + 1), 4 * n_images)
    )
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_paths):
        # Show original
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        axes[i, 0].imshow(img_array)
        axes[i, 0].set_title(f'{Path(img_path).name}')
        axes[i, 0].axis('off')
        
        # Show attribution for each feature
        for j, feat_idx in enumerate(feature_indices):
            cam, _ = attribution_extractor.compute_attribution(img_path, [feat_idx])
            
            # Resize and overlay
            from scipy.ndimage import zoom
            h, w = img_array.shape[:2]
            cam_resized = zoom(cam, (h / cam.shape[0], w / cam.shape[1]), order=1)
            
            heatmap = cm.jet(cam_resized)[:, :, :3] * 255
            overlay = (0.5 * heatmap + 0.5 * img_array).astype(np.uint8)
            
            axes[i, j + 1].imshow(overlay)
            axes[i, j + 1].set_title(f'Feature {feat_idx}')
            axes[i, j + 1].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16, fontweight='bold')
    else:
        plt.suptitle('Feature Attribution Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    importance_results: dict,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Visualize feature importance analysis.
    
    Works with results from any attribution extractor's analyze_feature_importance().
    
    Args:
        importance_results: Dict with 'top_indices' and 'top_values'
        output_path: Where to save (if None, just returns figure)
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    if not PLT_AVAILABLE:
        raise ImportError("matplotlib required for visualization")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    top_indices = importance_results['top_indices']
    top_values = importance_results['top_values']
    
    x_pos = np.arange(len(top_indices))
    colors = ['green' if v > 0 else 'red' for v in top_values]
    
    ax.bar(x_pos, top_values, color=colors, alpha=0.7)
    ax.set_xlabel('Feature Rank', fontsize=12)
    ax.set_ylabel('Activation Value', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{idx}' for idx in top_indices], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Top {len(top_indices)} Most Active Features', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def visualize_feature_dimensions(
    attribution_extractor,
    image_path: str,
    feature_indices: List[int],
    output_dir: Path
) -> None:
    """
    Create separate attribution visualizations for each feature dimension.
    
    Works with any attribution extractor.
    
    Args:
        attribution_extractor: Extractor with compute_attribution() method
        image_path: Path to image
        feature_indices: List of feature indices to visualize
        output_dir: Directory to save visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_name = Path(image_path).stem
    
    # Load original image once
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Visualize each feature dimension
    for feat_idx in feature_indices:
        cam, _ = attribution_extractor.compute_attribution(image_path, [feat_idx])
        
        output_path = output_dir / f'{img_name}_feature_{feat_idx:04d}.png'
        plot_attribution_overlay(
            cam,
            img_array,
            output_path=output_path,
            alpha=0.5,
            title=f'Feature {feat_idx} Attribution'
        )
    
    print(f"Saved {len(feature_indices)} visualizations to {output_dir}")

