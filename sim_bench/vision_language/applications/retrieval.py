"""
Semantic image retrieval using vision-language models.

Enables text-based image search: find images matching natural language queries.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from sim_bench.vision_language.base import BaseVisionLanguageModel


class SemanticRetrieval:
    """
    Semantic image retrieval using text queries.

    Supports:
    - Indexing image collections
    - Text-based search
    - Similarity thresholding
    - Multi-query search

    Example:
        >>> from sim_bench.vision_language import CLIPModel
        >>> from sim_bench.vision_language.applications import SemanticRetrieval
        >>>
        >>> clip = CLIPModel("ViT-B-32", device="cuda")
        >>> retrieval = SemanticRetrieval(clip)
        >>>
        >>> # Index images
        >>> retrieval.index_images(all_image_paths)
        >>>
        >>> # Search
        >>> results = retrieval.search("photos taken at sunset", top_k=10)
        >>> for result in results:
        >>>     print(f"{result['path']}: {result['score']:.3f}")
    """

    def __init__(
        self,
        model: BaseVisionLanguageModel,
        enable_cache: bool = True
    ):
        """
        Initialize semantic retrieval.

        Args:
            model: Vision-language model instance
            enable_cache: Whether to cache image embeddings
        """
        self.model = model
        self.enable_cache = enable_cache

        # Image database: path -> embedding
        self.image_database = {}
        self.image_paths_list = []  # Ordered list of paths

    def index_images(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Index images for fast retrieval.

        Args:
            image_paths: List of image paths to index
            batch_size: Batch size for encoding
            verbose: Print progress
        """
        if verbose:
            print(f"Indexing {len(image_paths)} images...")

        # Encode all images
        embeddings = self.model.encode_images(image_paths, batch_size=batch_size)

        # Store in database
        for path, emb in zip(image_paths, embeddings):
            self.image_database[path] = emb
            if path not in self.image_paths_list:
                self.image_paths_list.append(path)

        if verbose:
            print(f"Indexed {len(self.image_database)} images")

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Search indexed images by text query.

        Args:
            query: Text search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold (None = no threshold)

        Returns:
            List of dicts with 'path' and 'score' keys, sorted by score

        Raises:
            ValueError: If no images indexed
        """
        if not self.image_database:
            raise ValueError("No images indexed. Call index_images() first.")

        # Encode query
        query_emb = self.model.encode_texts([query])[0]

        # Compute similarities with all images
        results = []
        for path, img_emb in self.image_database.items():
            similarity = float(np.dot(img_emb, query_emb))

            if threshold is None or similarity >= threshold:
                results.append({
                    'path': path,
                    'score': similarity
                })

        # Sort by descending score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top-k
        return results[:top_k]

    def search_multiple(
        self,
        queries: List[str],
        top_k: int = 10,
        aggregation: str = "max"
    ) -> List[Dict[str, any]]:
        """
        Search with multiple queries, aggregate results.

        Args:
            queries: List of text queries
            top_k: Number of results to return
            aggregation: How to combine scores ('max', 'mean', 'min')

        Returns:
            List of dicts with 'path' and 'score' keys
        """
        if not self.image_database:
            raise ValueError("No images indexed. Call index_images() first.")

        # Encode all queries
        query_embs = self.model.encode_texts(queries)

        # Compute similarities
        aggregated_scores = {}

        for path, img_emb in self.image_database.items():
            similarities = [float(np.dot(img_emb, q_emb)) for q_emb in query_embs]

            if aggregation == "max":
                score = max(similarities)
            elif aggregation == "mean":
                score = np.mean(similarities)
            elif aggregation == "min":
                score = min(similarities)
            else:
                score = max(similarities)

            aggregated_scores[path] = score

        # Sort and format results
        results = [
            {'path': path, 'score': score}
            for path, score in aggregated_scores.items()
        ]
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def get_similar_images(
        self,
        reference_image: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict[str, any]]:
        """
        Find images similar to a reference image.

        Args:
            reference_image: Path to reference image
            top_k: Number of results to return
            exclude_self: Whether to exclude the reference image

        Returns:
            List of dicts with 'path' and 'score' keys
        """
        if reference_image not in self.image_database:
            # Encode reference image
            ref_emb = self.model.encode_images([reference_image])[0]
        else:
            ref_emb = self.image_database[reference_image]

        # Compute similarities
        results = []
        for path, img_emb in self.image_database.items():
            if exclude_self and path == reference_image:
                continue

            similarity = float(np.dot(img_emb, ref_emb))
            results.append({
                'path': path,
                'score': similarity
            })

        # Sort and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def clear_index(self):
        """Clear the image database."""
        self.image_database.clear()
        self.image_paths_list.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get indexing statistics."""
        return {
            'total_indexed': len(self.image_database),
            'embedding_dim': self.model.get_embedding_dim() if self.image_database else 0
        }
