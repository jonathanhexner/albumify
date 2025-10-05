"""
SIFT + Bag-of-Visual-Words method for local feature-based similarity.
Uses configurable distance measures for comparison.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import pickle
from typing import List
from sim_bench.feature_extraction.base import BaseMethod


class SIFTBoVWMethod(BaseMethod):
    """SIFT local features with Bag-of-Visual-Words representation."""
    
    def _sift(self, n_features=800):
        """Create SIFT detector."""
        try:
            return cv2.SIFT_create(nfeatures=n_features)
        except Exception as e:
            raise RuntimeError("SIFT requires opencv-contrib-python.") from e

    def _extract_sift_desc(self, path, n_features=800):
        """Extract SIFT descriptors from a single image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        sift = self._sift(n_features)
        _, des = sift.detectAndCompute(img, None)
        return des  # [N, 128] or None

    def _build_codebook(self, files, cache_dir):
        """Build or load codebook for BoVW."""
        os.makedirs(cache_dir, exist_ok=True)
        codebook_path = os.path.join(cache_dir, 'codebook.pkl')
        if os.path.exists(codebook_path):
            with open(codebook_path, 'rb') as f:
                return pickle.load(f)

        size = int(self.method_config['codebook']['size'])
        max_iter = int(self.method_config['codebook'].get('kmeans_max_iter', 100))
        sample_images = int(self.method_config['codebook'].get('sample_images', 1000))
        desc_per_img = int(self.method_config['codebook'].get('descriptors_per_image', 300))
        n_features = int(self.method_config['local_features']['n_features_per_image'])

        sel = files[:min(sample_images, len(files))]
        all_desc = []
        for f in tqdm(sel, desc="sift_bovw: sampling descriptors"):
            d = self._extract_sift_desc(f, n_features=n_features)
            if d is None or len(d) == 0: 
                continue
            if d.shape[0] > desc_per_img:
                idx = np.random.choice(d.shape[0], desc_per_img, replace=False)
                d = d[idx]
            all_desc.append(d.astype('float32'))
        if not all_desc:
            raise RuntimeError("No SIFT descriptors found to build codebook.")
        D = np.vstack(all_desc)

        km = KMeans(n_clusters=size, max_iter=max_iter, n_init=3, verbose=0, random_state=42)
        km.fit(D)
        codebook = { 'cluster_centers': km.cluster_centers_.astype('float32') }
        with open(codebook_path, 'wb') as f:
            pickle.dump(codebook, f)
        return codebook

    def _assign_hist(self, des, centers):
        """Assign descriptors to histogram bins."""
        if des is None or len(des) == 0:
            return np.zeros((centers.shape[0],), dtype='float32')
        d2 = (np.sum(des**2, axis=1, keepdims=True)
              + np.sum(centers**2, axis=1, keepdims=True).T
              - 2.0 * des.dot(centers.T))
        idx = np.argmin(d2, axis=1)
        K = centers.shape[0]
        hist = np.bincount(idx, minlength=K).astype('float32')
        hist /= (np.linalg.norm(hist) + 1e-12)
        return hist
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract SIFT BoVW features from images."""
        codebook_size = self.method_config['codebook']['size']
        print(f"Extracting SIFT BoVW features (codebook size: {codebook_size})...")
        
        cache_dir = self.method_config.get('cache_dir', 'artifacts/sift_bovw')
        codebook = self._build_codebook(image_paths, cache_dir)
        centers = codebook['cluster_centers']
        n_features = int(self.method_config['local_features']['n_features_per_image'])

        X = []
        for f in tqdm(image_paths, desc="sift_bovw: histograms"):
            des = self._extract_sift_desc(f, n_features=n_features)
            h = self._assign_hist(des, centers)
            X.append(h)
        return np.vstack(X).astype('float32')
