"""
Feature caching system to avoid recomputing features.
Caches are stored per dataset and method configuration.
"""

import hashlib
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    """Manages feature caching for methods."""
    
    def __init__(self, cache_root: Path = Path("artifacts/feature_cache")):
        """
        Initialize feature cache.
        
        Args:
            cache_root: Root directory for feature caches
        """
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
    
    def _compute_cache_key(
        self, 
        method_name: str,
        method_config: Dict[str, Any],
        image_paths: List[str]
    ) -> str:
        """
        Compute a unique cache key for this configuration.
        
        Args:
            method_name: Name of the method
            method_config: Method configuration
            image_paths: List of image paths
            
        Returns:
            Cache key string
        """
        # Create a deterministic hash from:
        # 1. Method name
        # 2. Relevant method config (excluding non-feature params)
        # 3. Image paths (sorted for consistency)
        
        # Filter config to only feature-relevant parameters
        relevant_config = {
            'method': method_config.get('method'),
            'features': method_config.get('features'),
            'backbone': method_config.get('backbone'),
            'normalize': method_config.get('normalize'),
            'vocab_size': method_config.get('vocab_size'),
            # Add other feature-affecting params as needed
        }
        
        # Remove None values
        relevant_config = {k: v for k, v in relevant_config.items() if v is not None}
        
        # Create hash
        config_str = json.dumps(relevant_config, sort_keys=True)
        paths_str = '|'.join(sorted(image_paths))
        combined = f"{method_name}||{config_str}||{paths_str}"
        
        cache_key = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"{method_name}_{cache_key}"
    
    def get_cache_path(
        self, 
        method_name: str,
        method_config: Dict[str, Any],
        image_paths: List[str]
    ) -> Path:
        """Get the cache file path for this configuration."""
        cache_key = self._compute_cache_key(method_name, method_config, image_paths)
        return self.cache_root / f"{cache_key}.pkl"
    
    def load(
        self, 
        method_name: str,
        method_config: Dict[str, Any],
        image_paths: List[str]
    ) -> Optional[np.ndarray]:
        """
        Load cached features if available.
        
        Returns:
            Feature matrix if cache exists, None otherwise
        """
        cache_path = self.get_cache_path(method_name, method_config, image_paths)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify the cached data matches
            if cached_data['image_paths'] != image_paths:
                logger.warning(f"Cache mismatch (image paths changed), recomputing features")
                return None
            
            logger.info(f"Loaded cached features from {cache_path.name}")
            print(f"[CACHE] Loaded cached features from {cache_path.name}")  # Keep console feedback
            return cached_data['features']
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, recomputing features")
            return None
    
    def save(
        self, 
        method_name: str,
        method_config: Dict[str, Any],
        image_paths: List[str],
        features: np.ndarray
    ) -> None:
        """
        Save features to cache.
        
        Args:
            method_name: Name of the method
            method_config: Method configuration
            image_paths: List of image paths
            features: Feature matrix to cache
        """
        cache_path = self.get_cache_path(method_name, method_config, image_paths)
        
        try:
            cached_data = {
                'image_paths': image_paths,
                'features': features,
                'method_name': method_name,
                'method_config': method_config,
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved features to {cache_path.name}")
            print(f"[CACHE] Saved features to {cache_path.name}")  # Keep console feedback
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            print(f"[CACHE] Failed to save cache: {e}")
    
    def clear(self, method_name: Optional[str] = None) -> None:
        """
        Clear cached features.
        
        Args:
            method_name: If provided, only clear caches for this method.
                        If None, clear all caches.
        """
        if method_name:
            pattern = f"{method_name}_*.pkl"
            removed = 0
            for cache_file in self.cache_root.glob(pattern):
                cache_file.unlink()
                removed += 1
            logger.info(f"Cleared {removed} cache(s) for method '{method_name}'")
            print(f"Cleared {removed} cache(s) for method '{method_name}'")
        else:
            removed = 0
            for cache_file in self.cache_root.glob("*.pkl"):
                cache_file.unlink()
                removed += 1
            logger.info(f"Cleared all {removed} cache(s)")
            print(f"Cleared all {removed} cache(s)")

