"""
INRIA Holidays dataset implementation.
"""

import csv
import json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

from sim_bench.datasets.base import BaseDataset
import glob
import re


class HolidaysDataset(BaseDataset):
    """INRIA Holidays dataset with variable group sizes."""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        super().__init__(dataset_config)
        self.groups = None  # Will be set after load_data()
        
    def _parse_image_id(self, filename: str) -> int:
        """Extract numeric ID from image filename."""
        stem = Path(filename).stem
        return int(stem)

    def _get_group_id(self, image_id: int) -> int:
        """Get group ID from image ID. Groups are defined by first 4 digits."""
        return image_id // 100

    def _is_query_image(self, image_id: int, all_ids_in_group: List[int]) -> bool:
        """Check if image is a query (first/lowest ID in its group)."""
        return image_id == min(all_ids_in_group)
        
    def load_data(self) -> Dict[str, Any]:
        """Load Holidays dataset."""
        root = Path(self.dataset_config['root'])
        pattern = self.dataset_config.get('pattern', '*.jpg')
        
        # Find all image files
        image_files = sorted(glob.glob(str(root / pattern)))
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {root} with pattern {pattern}")
        
        # First pass: parse all image IDs and group them
        images = []
        groups = []
        id_to_idx = {}
        group_to_ids = {}  # group_id -> list of image_ids
        
        for idx, filepath in enumerate(image_files):
            try:
                image_id = self._parse_image_id(filepath)
                group_id = self._get_group_id(image_id)
                
                images.append(filepath)
                groups.append(group_id)
                id_to_idx[image_id] = idx
                
                # Group images by series
                if group_id not in group_to_ids:
                    group_to_ids[group_id] = []
                group_to_ids[group_id].append(image_id)
                    
            except ValueError:
                # Skip files that don't match expected naming pattern
                continue
        
        # Second pass: make ALL images queries (like UKBench)
        # Each image in a group queries for all OTHER images in the same group
        queries = list(range(len(images)))  # All images are queries
        relevance_map = {}
        
        for group_id, image_ids in group_to_ids.items():
            if len(image_ids) < 2:
                # Skip groups with only 1 image (no relevant images)
                continue
                
            # For each image in the group, other images are relevant
            sorted_ids = sorted(image_ids)
            for query_id in sorted_ids:
                query_idx = id_to_idx[query_id]
                # Relevant = all other images in same group
                relevant_ids = [rid for rid in sorted_ids if rid != query_id]
                relevant_indices = [id_to_idx[rid] for rid in relevant_ids]
                
                relevance_map[query_idx] = relevant_indices
        
        # Validate dataset
        if not queries:
            raise ValueError("No query images found. Check image naming convention.")
        
        print(f"Loaded Holidays dataset:")
        print(f"  Total images: {len(images)}")
        print(f"  Query images: {len(queries)}")
        print(f"  Groups: {len(set(groups))}")
        
        # Print some statistics
        relevance_counts = [len(relevance_map[q]) for q in queries]
        if relevance_counts:
            print(f"  Relevant images per query: {min(relevance_counts)}-{max(relevance_counts)} "
                  f"(avg: {sum(relevance_counts)/len(relevance_counts):.1f})")
        
        self.data = {
            'images': images,
            'groups': groups,
            'queries': queries,
            'relevance_map': relevance_map
        }
        self.images = images
        self.queries = queries
        self.groups = groups  # Expose groups as attribute for consistency
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
        """Get the group ID for an image (Holidays: series ID from filename)."""
        return self.data['groups'][image_index]
    
    def _remap_after_sampling(self, selected_indices: List[int]) -> None:
        """Update Holidays-specific structures (relevance_map and groups) after sampling."""
        # Remap relevance map to new indices
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        
        new_relevance_map = {}
        for old_query_idx, old_relevant_indices in self.data['relevance_map'].items():
            if old_query_idx in old_to_new_idx:
                new_query_idx = old_to_new_idx[old_query_idx]
                new_relevant_indices = [old_to_new_idx[old_idx] for old_idx in old_relevant_indices 
                                       if old_idx in old_to_new_idx]
                new_relevance_map[new_query_idx] = new_relevant_indices
        
        self.data['relevance_map'] = new_relevance_map
        self.data['groups'] = [self.data['groups'][i] for i in selected_indices]
        self.groups = self.data['groups']  # Update groups attribute
    
    def get_num_relevant(self, query_idx: int) -> int:
        """Get number of relevant images for a query (Holidays: variable group sizes)."""
        return len(self.data['relevance_map'].get(query_idx, []))
    
    def is_relevant(self, query_idx: int, result_idx: int) -> bool:
        """Check if result is relevant (in relevance_map for query)."""
        return result_idx in self.data['relevance_map'].get(query_idx, [])
    
    def get_evaluation_data(self) -> Dict[str, Any]:
        """Get data needed for metric evaluation."""
        return {
            'groups': self.data['groups'],
            'queries': self.queries,
            'total_images': len(self.images)
        }
