"""Protocol definitions for service layer.

These protocols define interfaces that data loaders must implement.
This enables dependency inversion - pages depend on protocols, not implementations.
"""

from typing import Protocol, List, Optional
import numpy as np

from app.face_clustering_debug.models.schemas import FaceInfo, ClusteringResult


class DataLoaderProtocol(Protocol):
    """Interface for loading clustering data from any source.

    Implementations:
    - FileLoader: Loads from benchmark JSON/NPY files
    - DBLoader: Loads from SQLite database
    """

    def load_embeddings(self) -> np.ndarray:
        """Load face embeddings matrix.

        Returns:
            numpy array of shape [N, 512] where N is number of faces
        """
        ...

    def load_faces(self) -> List[FaceInfo]:
        """Load face metadata.

        Returns:
            List of FaceInfo objects, one per face
        """
        ...

    def load_clustering_result(self, method: str) -> Optional[ClusteringResult]:
        """Load pre-computed clustering result for given method.

        Args:
            method: Clustering method name (e.g., 'hdbscan', 'hybrid_hdbscan_knn')

        Returns:
            ClusteringResult if found, None otherwise
        """
        ...

    def get_available_methods(self) -> List[str]:
        """List available clustering methods in this data source.

        Returns:
            List of method names that have results available
        """
        ...

    def get_run_info(self) -> dict:
        """Return metadata about the loaded benchmark run.

        Keys: album_name, album_path, total_faces, timestamp
        """
        ...

    def has_debug_data(self, method: str) -> bool:
        """Return True if this method has detailed debug data (merge/attach decisions).

        Only methods run with collect_debug_data=True will return True.
        """
        ...

    def get_face_crop(self, face_index: int) -> Optional[bytes]:
        """Get aligned face crop image as JPEG bytes (5-point or roll aligned).

        Args:
            face_index: Index of face to retrieve

        Returns:
            JPEG bytes if crop exists, None otherwise
        """
        ...

    def get_face_crop_raw(self, face_index: int) -> Optional[bytes]:
        """Get raw face crop (bbox only, no alignment) as JPEG bytes.

        Args:
            face_index: Index of face to retrieve

        Returns:
            JPEG bytes of unaligned crop, None if unavailable
        """
        ...

    def get_original_image_with_bbox(self, face_index: int, max_size: int = 800) -> Optional[bytes]:
        """Get original image with face bbox drawn, resized for display.

        Args:
            face_index: Index of face
            max_size: Maximum dimension for resized image

        Returns:
            JPEG bytes of original image with bbox overlay, None if unavailable
        """
        ...
