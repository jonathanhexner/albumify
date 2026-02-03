"""Face embedding extraction service using trained ArcFace model.

Provides face embedding extraction for clustering, labeling, and identification.
Integrates with ModelHub for lazy loading.

Example:
    service = FaceEmbeddingService(config)
    embeddings = service.extract_embeddings_batch([Path("face1.jpg"), Path("face2.jpg")])
    similarity = service.compute_similarity(embeddings[0], embeddings[1])
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from sim_bench.models.face_resnet import FaceResNet, create_transform

logger = logging.getLogger(__name__)


class FaceEmbeddingService:
    """
    Face embedding extraction service using trained ArcFace model.

    Lazy loads the model on first use and provides methods for:
    - Single/batch embedding extraction
    - Similarity computation
    - Face clustering
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Configuration dict with 'face' section containing:
                - checkpoint_path: Path to trained ArcFace model
                - embedding_dim: Embedding dimension (default: 512)
                - input_size: Input image size (default: 112)
        """
        self._config = config
        self._face_config = config.get('face', {})
        self._device = config.get('device', 'cpu')

        # Lazy loaded
        self._model = None
        self._transform = None

        logger.info("FaceEmbeddingService initialized")

    def _load_model(self):
        """Lazy load ArcFace model from checkpoint."""
        if self._model is not None:
            return

        checkpoint_path = self._face_config.get('checkpoint_path')
        if not checkpoint_path:
            raise ValueError("No face checkpoint_path specified in config")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Face model checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading face model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        # Get model config from checkpoint
        ckpt_config = checkpoint['config']
        model_config = ckpt_config['model']

        # Create model
        self._model = FaceResNet(model_config)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model = self._model.to(self._device)
        self._model.eval()

        # Create transform
        transform_config = ckpt_config.get('transform', {})
        self._transform = create_transform(transform_config, is_train=False)

        lfw_acc = checkpoint.get('lfw_acc', 'unknown')
        logger.info(f"Loaded face model (LFW acc: {lfw_acc}%)")

    @property
    def model(self):
        """Get loaded model (lazy loads if needed)."""
        self._load_model()
        return self._model

    @property
    def transform(self):
        """Get image transform (lazy loads if needed)."""
        self._load_model()
        return self._transform

    def extract_embedding(self, image: Union[Path, str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract embedding for a single face image.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            L2-normalized embedding array of shape (embedding_dim,)
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        # Apply transform
        tensor = self.transform(image).unsqueeze(0).to(self._device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.model.extract_embedding(tensor)

        return embedding.cpu().numpy().squeeze()

    def extract_embeddings_batch(
        self,
        images: List[Union[Path, str, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings for batch of face images.

        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for inference
            show_progress: Show progress bar

        Returns:
            L2-normalized embeddings array of shape (N, embedding_dim)
        """
        self._load_model()

        all_embeddings = []
        num_batches = (len(images) + batch_size - 1) // batch_size

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting embeddings")

        for i in iterator:
            batch_images = images[i:i + batch_size]

            # Load and transform images
            tensors = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img).convert('RGB')
                tensors.append(self.transform(img))

            batch = torch.stack(tensors).to(self._device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model.extract_embedding(batch)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding (embedding_dim,)
            emb2: Second embedding (embedding_dim,)

        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Embeddings are already L2-normalized
        return float(np.dot(emb1, emb2))

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            embeddings: Array of shape (N, embedding_dim)

        Returns:
            Similarity matrix of shape (N, N)
        """
        # Embeddings are L2-normalized, so similarity = dot product
        return np.dot(embeddings, embeddings.T)

    def is_same_person(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = 0.289
    ) -> bool:
        """
        Determine if two face embeddings are from the same person.

        Args:
            emb1: First embedding
            emb2: Second embedding
            threshold: Similarity threshold (default from LFW evaluation)

        Returns:
            True if same person, False otherwise
        """
        similarity = self.compute_similarity(emb1, emb2)
        return similarity >= threshold

    def cluster_faces(
        self,
        embeddings: np.ndarray,
        method: str = 'agglomerative',
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Cluster face embeddings into identity groups.

        Args:
            embeddings: Array of shape (N, embedding_dim)
            method: Clustering method ('agglomerative')
            n_clusters: Number of clusters (if known)
            distance_threshold: Distance threshold for agglomerative clustering

        Returns:
            Cluster labels array of shape (N,)
        """
        if method == 'agglomerative':
            if n_clusters is not None:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric='cosine',
                    linkage='average'
                )
            labels = clustering.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logger.info(f"Clustered {len(embeddings)} faces into {len(set(labels))} groups")
        return labels

    def find_similar_faces(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar faces in gallery.

        Args:
            query_embedding: Query face embedding
            gallery_embeddings: Gallery embeddings (N, embedding_dim)
            top_k: Number of results to return

        Returns:
            List of dicts with 'index' and 'similarity'
        """
        similarities = np.dot(gallery_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx])
            })

        return results
