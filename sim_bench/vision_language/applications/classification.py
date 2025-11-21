"""
Zero-shot image classification using vision-language models.

Classify images into categories defined by text prompts, without training.
"""

from typing import List, Dict, Optional
import numpy as np

from sim_bench.vision_language.base import BaseVisionLanguageModel


class ZeroShotClassifier:
    """
    Zero-shot image classification using text prompts.

    Classify images into categories defined by natural language descriptions.
    No training required - works out-of-the-box with any categories.

    Example:
        >>> from sim_bench.vision_language import CLIPModel
        >>> from sim_bench.vision_language.applications import ZeroShotClassifier
        >>>
        >>> clip = CLIPModel("ViT-B-32", device="cuda")
        >>> classifier = ZeroShotClassifier(clip)
        >>>
        >>> # Define classes
        >>> classes = {
        >>>     "indoor": "a photo taken indoors",
        >>>     "outdoor": "a photo taken outdoors",
        >>>     "portrait": "a portrait photograph",
        >>>     "landscape": "a landscape photograph"
        >>> }
        >>>
        >>> # Classify
        >>> result = classifier.classify("photo.jpg", classes)
        >>> print(f"Class: {result['class_name']} ({result['confidence']:.2%})")
    """

    def __init__(
        self,
        model: BaseVisionLanguageModel,
        temperature: float = 1.0
    ):
        """
        Initialize zero-shot classifier.

        Args:
            model: Vision-language model instance
            temperature: Temperature for softmax (default 1.0)
                Lower values make predictions more confident
        """
        self.model = model
        self.temperature = temperature

    def classify(
        self,
        image_path: str,
        classes: Dict[str, str],
        return_probs: bool = False
    ) -> Dict[str, any]:
        """
        Classify single image into one of the given classes.

        Args:
            image_path: Path to image file
            classes: Dict mapping class names to text descriptions
                e.g., {"dog": "a photo of a dog", "cat": "a photo of a cat"}
            return_probs: Whether to return probabilities for all classes

        Returns:
            Dict with:
                - 'class_name': Predicted class name
                - 'class_text': Class text description
                - 'confidence': Confidence score (probability)
                - 'probabilities': All class probabilities (if return_probs=True)
        """
        # Get class names and texts
        class_names = list(classes.keys())
        class_texts = [classes[name] for name in class_names]

        # Encode image and texts
        image_emb = self.model.encode_images([image_path])[0]
        text_embs = self.model.encode_texts(class_texts)

        # Compute similarities
        similarities = self.model.compute_similarity(
            image_emb.reshape(1, -1),
            text_embs
        )[0]

        # Apply temperature and softmax
        logits = similarities / self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Get prediction
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        pred_confidence = float(probs[pred_idx])

        result = {
            'class_name': pred_class,
            'class_text': classes[pred_class],
            'confidence': pred_confidence
        }

        if return_probs:
            result['probabilities'] = {
                class_names[i]: float(probs[i])
                for i in range(len(class_names))
            }

        return result

    def classify_batch(
        self,
        image_paths: List[str],
        classes: Dict[str, str],
        batch_size: int = 32
    ) -> List[Dict[str, any]]:
        """
        Classify multiple images.

        Args:
            image_paths: List of image paths
            classes: Dict mapping class names to text descriptions
            batch_size: Batch size for encoding

        Returns:
            List of classification results (one per image)
        """
        # Get class names and texts
        class_names = list(classes.keys())
        class_texts = [classes[name] for name in class_names]

        # Encode images and texts
        image_embs = self.model.encode_images(image_paths, batch_size=batch_size)
        text_embs = self.model.encode_texts(class_texts)

        # Compute similarities
        similarities = self.model.compute_similarity(image_embs, text_embs)

        # Apply temperature and softmax
        logits = similarities / self.temperature
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get predictions
        pred_indices = np.argmax(probs, axis=1)

        results = []
        for i, pred_idx in enumerate(pred_indices):
            pred_class = class_names[pred_idx]
            results.append({
                'image_path': image_paths[i],
                'class_name': pred_class,
                'class_text': classes[pred_class],
                'confidence': float(probs[i, pred_idx])
            })

        return results

    def get_top_k_classes(
        self,
        image_path: str,
        classes: Dict[str, str],
        k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Get top-k most likely classes for an image.

        Args:
            image_path: Path to image file
            classes: Dict mapping class names to text descriptions
            k: Number of top classes to return

        Returns:
            List of dicts with 'class_name', 'class_text', 'confidence'
        """
        # Get class names and texts
        class_names = list(classes.keys())
        class_texts = [classes[name] for name in class_names]

        # Encode image and texts
        image_emb = self.model.encode_images([image_path])[0]
        text_embs = self.model.encode_texts(class_texts)

        # Compute similarities
        similarities = self.model.compute_similarity(
            image_emb.reshape(1, -1),
            text_embs
        )[0]

        # Apply temperature and softmax
        logits = similarities / self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Get top-k
        top_k_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_k_indices:
            results.append({
                'class_name': class_names[idx],
                'class_text': class_texts[idx],
                'confidence': float(probs[idx])
            })

        return results

    def confusion_matrix(
        self,
        image_paths: List[str],
        true_labels: List[str],
        classes: Dict[str, str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute confusion matrix for labeled images.

        Args:
            image_paths: List of image paths
            true_labels: List of true class names (must be in classes dict)
            classes: Dict mapping class names to text descriptions
            batch_size: Batch size for encoding

        Returns:
            Confusion matrix [n_classes, n_classes]
        """
        # Classify all images
        predictions = self.classify_batch(image_paths, classes, batch_size)

        # Build confusion matrix
        class_names = list(classes.keys())
        n_classes = len(class_names)
        confusion = np.zeros((n_classes, n_classes), dtype=int)

        for true_label, pred in zip(true_labels, predictions):
            true_idx = class_names.index(true_label)
            pred_idx = class_names.index(pred['class_name'])
            confusion[true_idx, pred_idx] += 1

        return confusion
