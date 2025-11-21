"""
CLIP/OpenCLIP implementation for sim-bench vision-language models.

Provides full access to CLIP capabilities:
- Image encoding
- Text encoding
- Image-text similarity
- Zero-shot classification
- Semantic retrieval
"""

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False

from PIL import Image
import numpy as np
from typing import List, Optional
from pathlib import Path

from sim_bench.vision_language.base import BaseVisionLanguageModel


class CLIPModel(BaseVisionLanguageModel):
    """
    OpenCLIP implementation of vision-language model.

    Supports various CLIP architectures:
    - ViT variants: ViT-B-32, ViT-B-16, ViT-L-14, ViT-H-14
    - ConvNext variants: convnext_base, convnext_large
    - RN variants: RN50, RN101

    Pretrained checkpoints:
    - laion2b_s34b_b79k (2B images, best general)
    - laion400m_e32 (400M images, faster)
    - openai (original OpenAI CLIP)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        enable_cache: bool = True
    ):
        """
        Initialize CLIP model.

        Args:
            model_name: Model architecture (e.g., 'ViT-B-32', 'ViT-L-14')
            pretrained: Pretrained checkpoint name
            device: Device to run on ('cpu' or 'cuda')
            enable_cache: Whether to cache embeddings

        Raises:
            ImportError: If torch or open_clip not available
            RuntimeError: If model loading fails
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CLIP. "
                "Install with: pip install torch"
            )

        if not OPENCLIP_AVAILABLE:
            raise ImportError(
                "OpenCLIP is required. "
                "Install with: pip install open-clip-torch"
            )

        super().__init__(model_name, device, enable_cache)

        self.pretrained = pretrained

        # Load CLIP model
        print(f"Loading OpenCLIP: {model_name} ({pretrained})")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
            self.model.eval()
            self.model.to(device)

            # Disable gradients
            for param in self.model.parameters():
                param.requires_grad = False

            self.tokenizer = open_clip.get_tokenizer(model_name)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenCLIP model '{model_name}' "
                f"with pretrained='{pretrained}': {e}"
            )

    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Encode images using CLIP vision encoder.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            Normalized embeddings [n_images, embedding_dim]
        """
        embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_embs = []

            for path in batch_paths:
                # Check cache
                if self.enable_cache and path in self._image_cache:
                    batch_embs.append(self._image_cache[path])
                    continue

                # Load and preprocess image
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        emb = self.model.encode_image(img_tensor)
                        # L2 normalize
                        emb = F.normalize(emb, dim=-1)
                        emb = emb.cpu().numpy()[0]

                    batch_embs.append(emb)

                    # Cache
                    if self.enable_cache:
                        self._image_cache[path] = emb

                except Exception as e:
                    print(f"Warning: Failed to encode {path}: {e}")
                    # Return zero embedding
                    zero_emb = np.zeros(self.get_embedding_dim())
                    batch_embs.append(zero_emb)

            embeddings.extend(batch_embs)

        return np.array(embeddings)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using CLIP text encoder.

        Args:
            texts: List of text strings

        Returns:
            Normalized embeddings [n_texts, embedding_dim]
        """
        embeddings = []

        for text in texts:
            # Check cache
            if self.enable_cache and text in self._text_cache:
                embeddings.append(self._text_cache[text])
                continue

            # Tokenize and encode
            try:
                tokens = self.tokenizer([text]).to(self.device)

                with torch.no_grad():
                    emb = self.model.encode_text(tokens)
                    # L2 normalize
                    emb = F.normalize(emb, dim=-1)
                    emb = emb.cpu().numpy()[0]

                embeddings.append(emb)

                # Cache
                if self.enable_cache:
                    self._text_cache[text] = emb

            except Exception as e:
                print(f"Warning: Failed to encode text '{text}': {e}")
                # Return zero embedding
                zero_emb = np.zeros(self.get_embedding_dim())
                embeddings.append(zero_emb)

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get CLIP embedding dimension."""
        # Access output dimension from visual encoder
        if hasattr(self.model.visual, 'output_dim'):
            return self.model.visual.output_dim
        elif hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        else:
            # Fallback: infer from model name
            if 'ViT-B' in self.model_name:
                return 512
            elif 'ViT-L' in self.model_name:
                return 768
            elif 'ViT-H' in self.model_name:
                return 1024
            else:
                return 512  # Default

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config['pretrained'] = self.pretrained
        return config

    def __repr__(self) -> str:
        return (
            f"CLIPModel("
            f"model_name='{self.model_name}', "
            f"pretrained='{self.pretrained}', "
            f"device='{self.device}')"
        )
