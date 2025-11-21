"""
Base class for vision-language models in sim-bench.

Provides unified API for models like CLIP, BLIP, LLaVA, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from pathlib import Path


class BaseVisionLanguageModel(ABC):
    """
    Abstract base class for vision-language models.

    Provides unified interface for:
    - Image encoding (image -> embedding)
    - Text encoding (text -> embedding)
    - Image-text similarity
    - Zero-shot classification
    - Semantic retrieval

    Subclasses must implement:
    - encode_images()
    - encode_texts()
    - get_embedding_dim()
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        enable_cache: bool = True
    ):
        """
        Initialize vision-language model.

        Args:
            model_name: Model identifier (e.g., 'ViT-B-32' for CLIP)
            device: Device to run on ('cpu' or 'cuda')
            enable_cache: Whether to cache embeddings
        """
        self.model_name = model_name
        self.device = device
        self.enable_cache = enable_cache

        # Embedding caches
        self._image_cache = {}  # path -> embedding
        self._text_cache = {}   # text -> embedding

    @abstractmethod
    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            Embeddings array of shape [n_images, embedding_dim]
        """
        pass

    @abstractmethod
    def encode_texts(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings array of shape [n_texts, embedding_dim]
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        pass

    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute cosine similarity between images and texts.

        Args:
            image_embeddings: Image embeddings [n_images, dim]
            text_embeddings: Text embeddings [n_texts, dim]
            normalize: Whether to L2-normalize embeddings

        Returns:
            Similarity matrix [n_images, n_texts]
        """
        if normalize:
            # L2 normalize
            image_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
            text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)

            image_embeddings = image_embeddings / (image_norms + 1e-8)
            text_embeddings = text_embeddings / (text_norms + 1e-8)

        # Cosine similarity via dot product
        return image_embeddings @ text_embeddings.T

    def rank_by_text(
        self,
        image_paths: List[str],
        text_query: str,
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[Tuple[int, float]]:
        """
        Rank images by similarity to text query.

        Args:
            image_paths: List of image paths
            text_query: Text query
            top_k: Return top k results (None = all)
            batch_size: Batch size for encoding images

        Returns:
            List of (index, score) tuples sorted by descending similarity
        """
        # Encode images and text
        image_embs = self.encode_images(image_paths, batch_size=batch_size)
        text_emb = self.encode_texts([text_query])

        # Compute similarities
        similarities = self.compute_similarity(image_embs, text_emb)[:, 0]

        # Sort by descending similarity
        sorted_indices = np.argsort(similarities)[::-1]

        if top_k:
            sorted_indices = sorted_indices[:top_k]

        # Return (index, score) tuples
        results = [(int(idx), float(similarities[idx])) for idx in sorted_indices]
        return results

    def zero_shot_classify(
        self,
        image_paths: List[str],
        class_texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Zero-shot classification using text prompts.

        Args:
            image_paths: Images to classify
            class_texts: Text descriptions of classes
            batch_size: Batch size for encoding

        Returns:
            Class indices array [n_images]
        """
        # Encode images and class texts
        image_embs = self.encode_images(image_paths, batch_size=batch_size)
        text_embs = self.encode_texts(class_texts)

        # Compute similarities
        similarities = self.compute_similarity(image_embs, text_embs)

        # Return argmax class for each image
        return np.argmax(similarities, axis=1)

    def zero_shot_classify_probs(
        self,
        image_paths: List[str],
        class_texts: List[str],
        temperature: float = 1.0,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Zero-shot classification with probability scores.

        Args:
            image_paths: Images to classify
            class_texts: Text descriptions of classes
            temperature: Temperature for softmax (default 1.0)
            batch_size: Batch size for encoding

        Returns:
            Probability array [n_images, n_classes]
        """
        # Encode images and class texts
        image_embs = self.encode_images(image_paths, batch_size=batch_size)
        text_embs = self.encode_texts(class_texts)

        # Compute similarities
        similarities = self.compute_similarity(image_embs, text_embs)

        # Apply temperature and softmax
        logits = similarities / temperature
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def batch_image_text_similarity(
        self,
        image_paths: List[str],
        text_prompts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute similarity between multiple images and texts.

        Args:
            image_paths: List of image paths
            text_prompts: List of text prompts
            batch_size: Batch size for encoding

        Returns:
            Similarity matrix [n_images, n_texts]
        """
        image_embs = self.encode_images(image_paths, batch_size=batch_size)
        text_embs = self.encode_texts(text_prompts)

        return self.compute_similarity(image_embs, text_embs)

    def clear_cache(self):
        """Clear embedding caches."""
        self._image_cache.clear()
        self._text_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'image_cache_size': len(self._image_cache),
            'text_cache_size': len(self._text_cache),
            'cache_enabled': self.enable_cache
        }

    def get_config(self) -> Dict[str, any]:
        """Get model configuration."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dim': self.get_embedding_dim(),
            'cache_enabled': self.enable_cache
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device='{self.device}')"
        )
