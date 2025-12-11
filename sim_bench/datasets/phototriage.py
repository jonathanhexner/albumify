"""
Photo Triage dataset implementation.
From paper: "Automatic Triage for a Photo Series" (Kaggle dataset)
"""

import glob
from typing import List, Dict, Any
from pathlib import Path

from sim_bench.datasets.base import BaseDataset


class PhotoTriageDataset(BaseDataset):
    """Photo Triage dataset with photo series (variable-size groups)."""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        super().__init__(dataset_config)
        self.groups = None
        
    def load_data(self) -> Dict[str, Any]:
        """Load Photo Triage dataset."""
        root = Path(self.dataset_config['root']) / self.dataset_config['subdirs']['images']
        files = sorted(glob.glob(str(root / self.dataset_config['pattern'])))
        
        # Parse groups from filenames: 000001-01.JPG -> group 000001
        groups = []
        for file_path in files:
            filename = Path(file_path).stem  # e.g., "000001-01"
            group_id = filename.split('-')[0]  # e.g., "000001"
            groups.append(int(group_id))  # Convert to int for consistency
        
        self.data = {'images': files, 'groups': groups}
        self.images = files
        self.groups = groups
        self.queries = list(range(len(self.images)))
        
        print(f"Loaded Photo Triage dataset:")
        print(f"  Total images: {len(self.images)}")
        print(f"  Query images: {len(self.queries)}")
        print(f"  Unique groups: {len(set(self.groups))}")
        
        # Calculate group sizes
        from collections import Counter
        group_sizes = Counter(self.groups)
        sizes = list(group_sizes.values())
        print(f"  Group sizes: {min(sizes)}-{max(sizes)} (avg: {sum(sizes)/len(sizes):.1f})")
        
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
        """Get the group ID for an image."""
        return self.data['groups'][image_index]
    
    def _remap_after_sampling(self, selected_indices: List[int]) -> None:
        """Update groups list after sampling."""
        self.groups = [self.data['groups'][i] for i in selected_indices]
    
    def apply_sampling(self, sampling_config: Dict[str, Any]) -> None:
        """
        Apply sampling to PhotoTriage dataset.
        
        Supports:
        - num_series: Number of series (groups) to sample (PhotoTriage-specific)
        - max_groups: Number of groups to sample (base class parameter)
        - max_queries: Number of queries to sample (base class parameter)
        
        If num_series is specified, it's converted to max_groups.
        """
        # Convert num_series to max_groups for PhotoTriage
        if 'num_series' in sampling_config and 'max_groups' not in sampling_config:
            sampling_config = sampling_config.copy()
            sampling_config['max_groups'] = sampling_config.pop('num_series')
        
        # Call parent implementation
        super().apply_sampling(sampling_config)
    
    def get_num_relevant(self, query_idx: int) -> int:
        """Get number of relevant images for a query (PhotoTriage: variable group sizes)."""
        query_group = self.groups[query_idx]
        # Count images in same group, excluding query itself
        return sum(1 for g in self.groups if g == query_group) - 1
    
    def is_relevant(self, query_idx: int, result_idx: int) -> bool:
        """Check if result is relevant (same group as query)."""
        return self.groups[query_idx] == self.groups[result_idx]
    
    def get_evaluation_data(self) -> Dict[str, Any]:
        """Get data needed for metric evaluation."""
        return {
            'groups': self.groups,
            'total_images': len(self.images)
        }


