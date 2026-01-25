"""Pair-based verification validation for face recognition.

Implements LFW-style verification protocol:
- Load pre-generated pairs from .bin files
- Extract embeddings for each image
- Compute cosine similarity
- Find optimal threshold and compute accuracy
"""

import io
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Results from pair-based verification."""
    accuracy: float
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    auc_score: float
    fpr: np.ndarray
    tpr: np.ndarray
    # Per-pair predictions for error analysis
    predictions: list  # List of dicts with pair info


def load_bin_file(bin_path: Path) -> tuple:
    """
    Load verification pairs from .bin file.

    Returns:
        images: List of image bytes (2 * num_pairs)
        labels: List of bool labels (num_pairs), True = same person
    """
    with open(bin_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    images = data[0]  # List of JPEG bytes
    labels = data[1]  # List of bool

    return images, labels


def decode_image(img_bytes: bytes, transform=None) -> torch.Tensor:
    """Decode JPEG bytes to tensor."""
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if transform is not None:
        img = transform(img)
    else:
        # Default: convert to tensor and normalize
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

    return img


def extract_embeddings_from_pairs(
    model,
    images: list,
    device: str,
    transform=None,
    batch_size: int = 64
) -> np.ndarray:
    """
    Extract embeddings for all images in pairs.

    Args:
        model: Face recognition model with extract_embedding method
        images: List of image bytes
        device: Device to run on
        transform: Optional image transform
        batch_size: Batch size for inference

    Returns:
        embeddings: (N, embed_dim) array
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]

            # Decode and stack images
            tensors = [decode_image(img, transform) for img in batch_imgs]
            batch = torch.stack(tensors).to(device)

            # Extract embeddings
            embeddings = model.extract_embedding(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_verification_metrics(
    embeddings: np.ndarray,
    labels: list,
    num_pairs: int = None
) -> VerificationResult:
    """
    Compute verification accuracy using cosine similarity.

    Args:
        embeddings: (2*N, embed_dim) embeddings, pairs are consecutive
        labels: N boolean labels (True = same person)
        num_pairs: Number of pairs (defaults to len(labels))

    Returns:
        VerificationResult with accuracy, threshold, confusion matrix, per-pair predictions
    """
    if num_pairs is None:
        num_pairs = len(labels)

    labels = np.array(labels[:num_pairs])

    # Compute cosine similarities for each pair
    similarities = []
    for i in range(num_pairs):
        emb1 = embeddings[2 * i]
        emb2 = embeddings[2 * i + 1]
        sim = np.dot(emb1, emb2)  # Already L2 normalized
        similarities.append(sim)

    similarities = np.array(similarities)

    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)

    # Find threshold that maximizes accuracy (TPR + TNR) / 2
    # Or equivalently minimizes FPR + FNR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Compute predictions at optimal threshold
    predictions_binary = similarities >= optimal_threshold

    # Confusion matrix
    tp = int(np.sum(predictions_binary & labels))
    tn = int(np.sum(~predictions_binary & ~labels))
    fp = int(np.sum(predictions_binary & ~labels))
    fn = int(np.sum(~predictions_binary & labels))

    accuracy = 100.0 * (tp + tn) / num_pairs

    # Per-pair predictions for error analysis
    pair_predictions = []
    for i in range(num_pairs):
        pair_predictions.append({
            'pair_idx': i,
            'similarity': float(similarities[i]),
            'predicted_same': bool(predictions_binary[i]),
            'actual_same': bool(labels[i]),
            'correct': bool(predictions_binary[i] == labels[i])
        })

    return VerificationResult(
        accuracy=accuracy,
        threshold=float(optimal_threshold),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        auc_score=float(auc_score),
        fpr=fpr,
        tpr=tpr,
        predictions=pair_predictions
    )


def validate_lfw(
    model,
    bin_path: Path,
    device: str,
    transform=None,
    batch_size: int = 64
) -> VerificationResult:
    """
    Run LFW-style verification validation.

    Args:
        model: Face recognition model
        bin_path: Path to .bin file with pairs
        device: Device to run on
        transform: Image transform
        batch_size: Batch size for embedding extraction

    Returns:
        VerificationResult
    """
    logger.info(f"Loading pairs from {bin_path}...")
    images, labels = load_bin_file(bin_path)

    num_pairs = len(labels)
    logger.info(f"Loaded {num_pairs} pairs ({sum(labels)} positive, {num_pairs - sum(labels)} negative)")

    logger.info("Extracting embeddings...")
    embeddings = extract_embeddings_from_pairs(
        model, images, device, transform, batch_size
    )

    logger.info("Computing verification metrics...")
    result = compute_verification_metrics(embeddings, labels, num_pairs)

    logger.info(f"Verification accuracy: {result.accuracy:.2f}% @ threshold={result.threshold:.4f}")
    logger.info(f"Confusion: TP={result.tp}, TN={result.tn}, FP={result.fp}, FN={result.fn}")
    logger.info(f"AUC: {result.auc_score:.4f}")

    return result


def save_verification_results(result: VerificationResult, output_path: Path):
    """Save verification results to JSON."""
    import json

    data = {
        'accuracy': result.accuracy,
        'threshold': result.threshold,
        'tp': result.tp,
        'tn': result.tn,
        'fp': result.fp,
        'fn': result.fn,
        'auc_score': result.auc_score,
        'predictions': result.predictions
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved verification results to {output_path}")
