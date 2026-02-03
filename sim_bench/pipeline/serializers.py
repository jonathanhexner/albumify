"""Reusable serialization helpers for pipeline steps.

Steps can use these helpers or implement custom serialization.
"""

import io
import json
import pickle
from typing import Any

import numpy as np


class Serializers:
    """Reusable serialization helpers for common data types."""
    
    @staticmethod
    def json_serialize(value: Any) -> bytes:
        """
        Serialize value to JSON bytes (human-readable, cross-platform).
        
        Good for: scores, metrics, structured data (dicts, lists)
        
        Args:
            value: Any JSON-serializable value
        
        Returns:
            UTF-8 encoded JSON bytes
        """
        return json.dumps(value).encode('utf-8')
    
    @staticmethod
    def json_deserialize(data: bytes) -> Any:
        """
        Deserialize JSON bytes to Python value.
        
        Args:
            data: UTF-8 encoded JSON bytes
        
        Returns:
            Deserialized Python value
        """
        return json.loads(data.decode('utf-8'))
    
    @staticmethod
    def numpy_serialize(array: np.ndarray) -> bytes:
        """
        Serialize numpy array to bytes (compact, fast).
        
        Good for: embeddings, vectors, large arrays
        
        Args:
            array: Numpy array to serialize
        
        Returns:
            Serialized bytes (numpy .npy format)
        """
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()
    
    @staticmethod
    def numpy_deserialize(data: bytes) -> np.ndarray:
        """
        Deserialize bytes to numpy array.
        
        Args:
            data: Serialized numpy bytes
        
        Returns:
            Numpy array
        """
        buffer = io.BytesIO(data)
        return np.load(buffer, allow_pickle=False)
    
    @staticmethod
    def pickle_serialize(obj: Any) -> bytes:
        """
        Serialize Python object with pickle (use sparingly).
        
        Good for: Complex Python objects that aren't JSON-serializable
        
        Warning: Pickle is Python-specific and can be insecure.
        Prefer JSON or numpy when possible.
        
        Args:
            obj: Python object to serialize
        
        Returns:
            Pickled bytes
        """
        return pickle.dumps(obj)
    
    @staticmethod
    def pickle_deserialize(data: bytes) -> Any:
        """
        Deserialize pickled object.
        
        Args:
            data: Pickled bytes
        
        Returns:
            Deserialized Python object
        """
        return pickle.loads(data)
