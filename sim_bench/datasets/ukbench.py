"""
UKBench dataset implementation.
"""

import csv
import json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

from sim_bench.datasets.base import BaseDataset
import glob


class UKBenchDataset(BaseDataset):
    """UKBench dataset with groups of 4 similar images."""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        super().__init__(dataset_config)
        self.groups = None
        
    def load_data(self) -> Dict[str, Any]:
        """Load UKBench dataset."""
        root = Path(self.dataset_config['root']) / self.dataset_config['subdirs']['images']
        files = sorted(glob.glob(str(root / self.dataset_config['pattern'])))
        # UKBench groups images by consecutive sets of 4 (indices 0-3, 4-7, etc.)
        groups = [i // 4 for i in range(len(files))]
        
        self.data = {'images': files, 'groups': groups}
        self.images = files
        self.groups = groups
        self.queries = list(range(len(self.images)))
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
        """Get the group ID for an image (UKBench: group = index // 4)."""
        return self.data['groups'][image_index]
    
    def _remap_after_sampling(self, selected_indices: List[int]) -> None:
        """Update groups list after sampling."""
        self.groups = [self.data['groups'][i] for i in selected_indices]
    
    def get_num_relevant(self, query_idx: int) -> int:
        """Get number of relevant images for a query (UKBench: always 3 per group)."""
        return 3  # UKBench has 4 images per group, so 3 are relevant (excluding query)
    
    def is_relevant(self, query_idx: int, result_idx: int) -> bool:
        """Check if result is relevant (same group as query)."""
        return self.groups[query_idx] == self.groups[result_idx]
    
    def get_evaluation_data(self) -> Dict[str, Any]:
        """Get data needed for metric evaluation."""
        return {
            'groups': self.groups,
            'total_images': len(self.images)
        }
