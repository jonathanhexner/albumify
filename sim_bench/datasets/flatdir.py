"""
Flat directory dataset - for clustering without ground truth.
All images in a single directory, each image is its own "group".
"""

import glob
from typing import List, Dict, Any
from pathlib import Path

from sim_bench.datasets.base import BaseDataset


class FlatDirDataset(BaseDataset):
    """
    Flat directory dataset for clustering (no ground truth).
    Each image is treated as its own group (group_id = image_index).
    Useful for exploratory clustering on unlabeled data.
    """
    
    def __init__(self, dataset_config: Dict[str, Any]):
        super().__init__(dataset_config)
        self.groups = None
        
    def load_data(self) -> Dict[str, Any]:
        """Load all images from flat directory."""
        root = Path(self.dataset_config['root'])
        
        # Handle subdirectory if specified
        subdirs = self.dataset_config.get('subdirs', {})
        if subdirs and 'images' in subdirs:
            subdir = subdirs['images']
            if subdir != '.':
                root = root / subdir
        
        # Support multiple patterns if provided
        pattern = self.dataset_config.get('pattern', '*.jpg')
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        
        # Collect all matching files
        files = []
        for pat in patterns:
            files.extend(glob.glob(str(root / pat)))
        
        # Deduplicate (Windows glob is case-insensitive, may match same file multiple times)
        files = sorted(list(set(files)))
        
        # Each image is its own group (no ground truth)
        groups = list(range(len(files)))
        
        self.data = {'images': files, 'groups': groups}
        self.images = files
        self.groups = groups
        self.queries = list(range(len(self.images)))
        
        print(f"Loaded {self.name} dataset:")
        print(f"  Total images: {len(self.images)}")
        print(f"  Note: No ground truth groups (clustering only)")
        
        return self.data
    
    def get_images(self) -> List[str]:
        """Get list of image file paths."""
        if self.images is None:
            self.load_data()
        return self.images
    
    def get_queries(self) -> List[int]:
        """Get list of query image indices."""
        if self.queries is None:
            self.load_data()
        return self.queries
    
    def _get_group_for_image(self, image_index: int) -> int:
        """Get the group ID for an image (each image is its own group)."""
        return self.data['groups'][image_index]
    
    def _remap_after_sampling(self, selected_indices: List[int]) -> None:
        """Update groups list after sampling."""
        # Remap to sequential indices
        self.groups = list(range(len(selected_indices)))
    
    def get_num_relevant(self, query_idx: int) -> int:
        """Get number of relevant images (FlatDir: no ground truth, so 0)."""
        return 0  # No ground truth, each image is its own group
    
    def is_relevant(self, query_idx: int, result_idx: int) -> bool:
        """Check if result is relevant (FlatDir: always False, no ground truth)."""
        return False  # No ground truth
    
    def get_evaluation_data(self) -> Dict[str, Any]:
        """
        Get data needed for metric evaluation.
        Note: Metrics won't be meaningful since there's no ground truth.
        """
        return {
            'groups': self.groups,
            'total_images': len(self.images)
        }

