"""
Abstract base class and factory for datasets in sim-bench.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path


class BaseDataset(ABC):
    """Abstract base class for all datasets."""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        """Initialize dataset with configuration."""
        self.dataset_config = dataset_config
        self.name = dataset_config['name']
        self.data = None  # Store full dataset before sampling
        self.images = None  # Active image list (after sampling)
        self.queries = None  # Active query list (after sampling)
        
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """Load dataset and return data dictionary."""
        pass
    
    @abstractmethod
    def get_images(self) -> List[str]:
        """Get list of image file paths."""
        pass
    
    @abstractmethod
    def get_queries(self) -> List[int]:
        """Get list of query image indices."""
        pass
    
    def get_index_to_path(self) -> Dict[int, str]:
        """
        Return a deterministic mapping from image index to absolute path
        for the active (possibly sampled) dataset.
        """
        images = self.get_images()
        return {i: images[i] for i in range(len(images))}
    
    def resolve_index_to_path(self, index: int) -> str:
        """
        Resolve a single image index to its absolute path for the active dataset.
        """
        return self.get_images()[index]
    
    @abstractmethod
    def _get_group_for_image(self, image_index: int) -> Any:
        """
        Get the group identifier for an image.
        
        Args:
            image_index: Index of the image in the full dataset
            
        Returns:
            Group identifier (can be int, str, or any hashable type)
        """
        pass
    
    @abstractmethod
    def _remap_after_sampling(self, selected_indices: List[int]) -> None:
        """
        Update dataset-specific structures after sampling.
        
        Called after images have been filtered. Subclasses should update
        any internal mappings or data structures to reflect the new indices.
        
        Args:
            selected_indices: List of indices from original dataset that were kept
        """
        pass
    
    def apply_sampling(self, sampling_config: Dict[str, Any]) -> None:
        """
        Apply sampling to limit dataset size (base implementation).
        
        Two supported approaches:
        1. max_groups: Sample by complete groups (recommended for valid metrics)
        2. max_queries: Limit total images/queries (useful for exact count control)
        
        If both are specified, max_groups takes precedence.
        
        Subclasses should implement:
        - _get_group_for_image(): Return group ID for an image
        - _remap_after_sampling(): Update internal structures after filtering
        """
        max_groups = sampling_config.get('max_groups')
        max_queries = sampling_config.get('max_queries')
        
        if not self.images or not self.queries:
            return  # No data loaded yet
        
        if max_groups:
            # Approach 1: Sample by complete groups
            max_groups = int(max_groups)
            
            # Collect unique groups from all images (not just queries)
            # to determine which groups to sample
            all_groups = [self._get_group_for_image(i) for i in range(len(self.images))]
            unique_groups = []
            seen = set()
            for g in all_groups:
                if g not in seen:
                    unique_groups.append(g)
                    seen.add(g)
                    if len(unique_groups) >= max_groups:
                        break
            
            selected_groups = set(unique_groups[:max_groups])
            
            # Find all images belonging to selected groups
            selected_indices = []
            for i in range(len(self.images)):
                if self._get_group_for_image(i) in selected_groups:
                    selected_indices.append(i)
            
            # Filter images
            self.images = [self.images[i] for i in selected_indices]
            
            # Remap query indices to new image list
            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
            self.queries = [old_to_new_idx[q] for q in self.queries if q in old_to_new_idx]
            
            # Let subclass update its specific structures
            self._remap_after_sampling(selected_indices)
            
        elif max_queries:
            # Approach 2: Limit by total images or queries
            # This may create incomplete groups but gives exact count control
            max_queries = int(max_queries)
            
            # For datasets where queries = images (like UKBench), limit images
            # For datasets with separate queries (like Holidays), limit queries first
            if len(self.queries) == len(self.images):
                # All images are queries (UKBench-style)
                selected_indices = list(range(min(max_queries, len(self.images))))
                self.images = [self.images[i] for i in selected_indices]
                self.queries = list(range(len(self.images)))
                self._remap_after_sampling(selected_indices)
            else:
                # Separate queries and gallery (Holidays-style)
                query_indices = self.queries[:max_queries]
                selected_groups = set(self._get_group_for_image(q) for q in query_indices)
                
                selected_indices = []
                for i in range(len(self.images)):
                    if self._get_group_for_image(i) in selected_groups:
                        selected_indices.append(i)
                
                self.images = [self.images[i] for i in selected_indices]
                old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
                self.queries = [old_to_new_idx[q] for q in query_indices if q in old_to_new_idx]
                self._remap_after_sampling(selected_indices)
    
    @abstractmethod
    def get_num_relevant(self, query_idx: int) -> int:
        """
        Get number of relevant images for a query (excluding the query itself).
        
        Args:
            query_idx: Query image index
            
        Returns:
            Number of relevant images for this query
        """
        pass
    
    @abstractmethod
    def is_relevant(self, query_idx: int, result_idx: int) -> bool:
        """
        Check if a result is relevant to a query.
        
        Args:
            query_idx: Query image index
            result_idx: Result image index
            
        Returns:
            True if result is relevant to query, False otherwise
        """
        pass
    
    @abstractmethod
    def get_evaluation_data(self) -> Dict[str, Any]:
        """
        Get data needed for metric evaluation.
        
        Returns:
            Dictionary containing:
            - 'groups': List of group IDs for each image
            - 'total_images': Total number of images
            - Additional dataset-specific fields as needed
        """
        pass


def load_dataset(dataset_name: str, dataset_config: Dict[str, Any]) -> BaseDataset:
    """
    Factory function to load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load ('ukbench', 'holidays')
        dataset_config: Dataset configuration loaded from configs/dataset.{name}.yaml
                       Contains dataset-specific settings like:
                       - root: Path to dataset directory
                       - pattern: File pattern to match (e.g., "*.jpg")
                       - subdirs: Subdirectory structure (if any)
                       - assume_groups_of_four: For UKBench grouping logic
        
    Returns:
        Instantiated dataset object
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name == 'ukbench':
        from sim_bench.datasets.ukbench import UKBenchDataset
        return UKBenchDataset(dataset_config)
    elif dataset_name == 'holidays':
        from sim_bench.datasets.holidays import HolidaysDataset
        return HolidaysDataset(dataset_config)
    elif dataset_name == 'phototriage':
        from sim_bench.datasets.phototriage import PhotoTriageDataset
        return PhotoTriageDataset(dataset_config)
    elif dataset_name in ['flatdir', 'budapest']:  # flatdir-based datasets (no ground truth)
        from sim_bench.datasets.flatdir import FlatDirDataset
        return FlatDirDataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: ukbench, holidays, phototriage, flatdir, budapest")
