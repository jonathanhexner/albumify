# Vision-Language Model API Proposal

## Problem Statement

Currently, CLIP aesthetic assessment is implemented within `quality_assessment/`, but vision-language models like CLIP have broader applications:

1. **Image-text similarity** (current use: aesthetic assessment via prompts)
2. **Image retrieval** (semantic search: "find photos of sunsets")
3. **Zero-shot classification** (categorize images by text labels)
4. **Image captioning** (generate descriptions)
5. **Cross-modal analysis** (correlate visual and textual features)

**Issue**: The current `QualityAssessor` API is too narrow for general vision-language capabilities.

## Current Architecture Issues

### 1. CLIP in `quality_assessment/`

```python
# Current: CLIP tightly coupled to quality assessment
class CLIPAestheticAssessor(QualityAssessor):
    def assess_image(self, image_path: str) -> float:
        # Returns single quality score
        # Lost: rich semantic capabilities of CLIP
```

**Problems**:
- ❌ Can't access image embeddings directly
- ❌ Can't compute custom text-image similarities
- ❌ Can't use for retrieval or classification
- ❌ Duplicates OpenCLIP loading logic with `feature_extraction/openclip.py`

### 2. OpenCLIP in `feature_extraction/`

```python
# Current: OpenCLIP used for similarity/retrieval
class OpenCLIPMethod(BaseMethod):
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        # Returns image embeddings
        # Lost: text encoding, text-image similarity
```

**Problems**:
- ❌ Only exposes image encoding
- ❌ No text encoding capability
- ❌ No text-image similarity API
- ❌ Separate from quality assessment use case

### 3. Code Duplication

Both implementations load OpenCLIP independently:
- `feature_extraction/openclip.py`: For image similarity
- `quality_assessment/clip_aesthetic.py`: For quality assessment

**Problems**:
- ❌ Duplicated model loading
- ❌ Duplicated preprocessing
- ❌ Inconsistent configurations
- ❌ Wasted memory (two model instances)

## Proposed Solution: New `vision_language` Subpackage

### Directory Structure

```
sim_bench/
├── feature_extraction/      # Traditional image features
│   ├── hsv_histogram.py
│   ├── resnet50.py
│   ├── dinov2.py
│   └── sift_bovw.py
│
├── vision_language/         # NEW: Vision-language models
│   ├── __init__.py
│   ├── base.py              # BaseVisionLanguageModel
│   ├── clip.py              # CLIP/OpenCLIP implementation
│   ├── blip.py              # Future: BLIP
│   ├── llava.py             # Future: LLaVA
│   │
│   └── applications/        # Use-case specific wrappers
│       ├── __init__.py
│       ├── aesthetic.py     # Aesthetic assessment via prompts
│       ├── retrieval.py     # Text-based image retrieval
│       ├── classification.py # Zero-shot classification
│       └── captioning.py    # Image captioning
│
├── quality_assessment/      # Quality-specific methods
│   ├── rule_based.py
│   ├── cnn_methods.py
│   └── clip_aesthetic.py    # MOVED: Thin wrapper over vision_language
│
└── clustering/
    └── ...
```

### Core API Design

#### 1. Base Vision-Language Model

```python
# sim_bench/vision_language/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import numpy as np
import torch

class BaseVisionLanguageModel(ABC):
    """
    Base class for vision-language models (CLIP, BLIP, etc.).

    Provides unified API for:
    - Image encoding
    - Text encoding
    - Image-text similarity
    - Zero-shot capabilities
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        enable_cache: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.enable_cache = enable_cache

        # Caches
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
            Embeddings array [n_images, embedding_dim]
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
            Embeddings array [n_texts, embedding_dim]
        """
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
            image_embeddings: [n_images, dim]
            text_embeddings: [n_texts, dim]
            normalize: Whether to normalize embeddings

        Returns:
            Similarity matrix [n_images, n_texts]
        """
        if normalize:
            image_embeddings = image_embeddings / np.linalg.norm(
                image_embeddings, axis=1, keepdims=True
            )
            text_embeddings = text_embeddings / np.linalg.norm(
                text_embeddings, axis=1, keepdims=True
            )

        return image_embeddings @ text_embeddings.T

    def rank_by_text(
        self,
        image_paths: List[str],
        text_query: str,
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Rank images by similarity to text query.

        Args:
            image_paths: List of image paths
            text_query: Text query
            top_k: Return top k results (None = all)

        Returns:
            Ranked indices (descending similarity)
        """
        image_embs = self.encode_images(image_paths)
        text_emb = self.encode_texts([text_query])

        similarities = self.compute_similarity(image_embs, text_emb)[:, 0]
        ranked_indices = np.argsort(similarities)[::-1]

        if top_k:
            ranked_indices = ranked_indices[:top_k]

        return ranked_indices.tolist()

    def zero_shot_classify(
        self,
        image_paths: List[str],
        class_texts: List[str]
    ) -> np.ndarray:
        """
        Zero-shot classification using text prompts.

        Args:
            image_paths: Images to classify
            class_texts: Text descriptions of classes

        Returns:
            Class indices [n_images]
        """
        image_embs = self.encode_images(image_paths)
        text_embs = self.encode_texts(class_texts)

        similarities = self.compute_similarity(image_embs, text_embs)
        return np.argmax(similarities, axis=1)

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        raise NotImplementedError

    def clear_cache(self):
        """Clear embedding caches."""
        self._image_cache.clear()
        self._text_cache.clear()
```

#### 2. CLIP Implementation

```python
# sim_bench/vision_language/clip.py

import open_clip
import torch
from PIL import Image
from typing import List
import numpy as np

from sim_bench.vision_language.base import BaseVisionLanguageModel


class CLIPModel(BaseVisionLanguageModel):
    """
    OpenCLIP implementation of vision-language model.

    Supports various CLIP variants:
    - ViT-B-32, ViT-B-16
    - ViT-L-14
    - ViT-H-14
    - ConvNext variants
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        enable_cache: bool = True
    ):
        super().__init__(model_name, device, enable_cache)

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.model.to(device)

        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """Encode images using CLIP vision encoder."""
        embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # Check cache
            batch_embs = []
            for path in batch_paths:
                if self.enable_cache and path in self._image_cache:
                    batch_embs.append(self._image_cache[path])
                else:
                    # Load and preprocess
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        emb = self.model.encode_image(img_tensor)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb = emb.cpu().numpy()[0]

                    batch_embs.append(emb)

                    if self.enable_cache:
                        self._image_cache[path] = emb

            embeddings.extend(batch_embs)

        return np.array(embeddings)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using CLIP text encoder."""
        embeddings = []

        for text in texts:
            # Check cache
            if self.enable_cache and text in self._text_cache:
                embeddings.append(self._text_cache[text])
            else:
                tokens = self.tokenizer([text]).to(self.device)

                with torch.no_grad():
                    emb = self.model.encode_text(tokens)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    emb = emb.cpu().numpy()[0]

                embeddings.append(emb)

                if self.enable_cache:
                    self._text_cache[text] = emb

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get CLIP embedding dimension."""
        # ViT-B: 512, ViT-L: 768, ViT-H: 1024
        return self.model.visual.output_dim
```

#### 3. Application-Specific Wrappers

```python
# sim_bench/vision_language/applications/aesthetic.py

from typing import List, Dict
import numpy as np
from sim_bench.vision_language.base import BaseVisionLanguageModel


class AestheticAssessor:
    """
    Assess image aesthetic quality using vision-language models.

    Uses contrastive text prompts to evaluate composition,
    framing, and overall quality.
    """

    CONTRASTIVE_PROMPTS = [
        ("a well-composed photograph", "a poorly-composed photograph"),
        ("a photo with the subject well placed in the frame",
         "a photo with the subject not well placed in the frame"),
        ("a photo that is well cropped", "a photo that is poorly cropped"),
        ("Good Quality photo", "Poor Quality photo"),
    ]

    POSITIVE_PROMPTS = [
        "professional photography",
        "aesthetically pleasing",
    ]

    NEGATIVE_PROMPTS = [
        "amateur snapshot",
        "poor framing",
    ]

    def __init__(
        self,
        model: BaseVisionLanguageModel,
        aggregation: str = "weighted"
    ):
        """
        Initialize aesthetic assessor.

        Args:
            model: Vision-language model instance
            aggregation: How to aggregate scores ('weighted', 'contrastive_only', 'mean')
        """
        self.model = model
        self.aggregation = aggregation

        # Pre-encode all prompts
        self._encode_prompts()

    def _encode_prompts(self):
        """Pre-encode all aesthetic prompts."""
        all_prompts = []

        for pos, neg in self.CONTRASTIVE_PROMPTS:
            all_prompts.extend([pos, neg])
        all_prompts.extend(self.POSITIVE_PROMPTS)
        all_prompts.extend(self.NEGATIVE_PROMPTS)

        self.prompt_embeddings = self.model.encode_texts(all_prompts)

    def assess_image(self, image_path: str) -> float:
        """Assess aesthetic quality of single image."""
        return self.assess_batch([image_path])[0]

    def assess_batch(self, image_paths: List[str]) -> np.ndarray:
        """Assess aesthetic quality of multiple images."""
        # Encode images
        image_embs = self.model.encode_images(image_paths)

        # Compute similarities
        similarities = self.model.compute_similarity(
            image_embs,
            self.prompt_embeddings
        )

        # Aggregate scores
        if self.aggregation == "contrastive_only":
            scores = self._aggregate_contrastive(similarities)
        elif self.aggregation == "weighted":
            scores = self._aggregate_weighted(similarities)
        else:
            scores = np.mean(similarities, axis=1)

        return scores

    def _aggregate_contrastive(self, similarities: np.ndarray) -> np.ndarray:
        """Aggregate using only contrastive pairs."""
        n_contrasts = len(self.CONTRASTIVE_PROMPTS)
        contrastive_scores = []

        for i in range(n_contrasts):
            pos_sim = similarities[:, i*2]
            neg_sim = similarities[:, i*2 + 1]
            contrastive_scores.append(pos_sim - neg_sim)

        return np.mean(contrastive_scores, axis=0)

    def _aggregate_weighted(self, similarities: np.ndarray) -> np.ndarray:
        """Weighted aggregation."""
        contrastive = self._aggregate_contrastive(similarities)

        n_contrasts = len(self.CONTRASTIVE_PROMPTS) * 2
        positive = np.mean(
            similarities[:, n_contrasts:n_contrasts+len(self.POSITIVE_PROMPTS)],
            axis=1
        )
        negative = -np.mean(
            similarities[:, n_contrasts+len(self.POSITIVE_PROMPTS):],
            axis=1
        )

        return 0.5 * contrastive + 0.3 * positive + 0.2 * negative
```

```python
# sim_bench/vision_language/applications/retrieval.py

from typing import List, Dict, Optional
import numpy as np
from sim_bench.vision_language.base import BaseVisionLanguageModel


class SemanticRetrieval:
    """
    Semantic image retrieval using text queries.

    Examples:
        - "photos taken at sunset"
        - "images with dogs"
        - "indoor scenes"
    """

    def __init__(self, model: BaseVisionLanguageModel):
        self.model = model
        self.image_database = {}  # path -> embedding

    def index_images(self, image_paths: List[str], batch_size: int = 32):
        """Index images for fast retrieval."""
        embeddings = self.model.encode_images(image_paths, batch_size)

        for path, emb in zip(image_paths, embeddings):
            self.image_database[path] = emb

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search indexed images by text query.

        Args:
            query: Text search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dicts with 'path' and 'score'
        """
        # Encode query
        query_emb = self.model.encode_texts([query])[0]

        # Compute similarities
        results = []
        for path, img_emb in self.image_database.items():
            similarity = np.dot(img_emb, query_emb)

            if threshold is None or similarity >= threshold:
                results.append({
                    'path': path,
                    'score': float(similarity)
                })

        # Sort and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
```

#### 4. Updated Quality Assessment Wrapper

```python
# sim_bench/quality_assessment/clip_aesthetic.py (UPDATED)

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.vision_language.clip import CLIPModel
from sim_bench.vision_language.applications.aesthetic import AestheticAssessor


class CLIPAestheticAssessor(QualityAssessor):
    """
    Quality assessor using CLIP aesthetic evaluation.

    Thin wrapper over vision_language.applications.aesthetic.AestheticAssessor
    to conform to QualityAssessor API.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        aggregation_method: str = "weighted",
        enable_cache: bool = True
    ):
        super().__init__(device=device, enable_cache=enable_cache)

        # Create CLIP model
        self.clip_model = CLIPModel(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            enable_cache=enable_cache
        )

        # Create aesthetic assessor
        self.aesthetic_assessor = AestheticAssessor(
            model=self.clip_model,
            aggregation=aggregation_method
        )

        self.model_name = model_name
        self.aggregation_method = aggregation_method

    def assess_image(self, image_path: str) -> float:
        """Assess quality of single image."""
        return self.aesthetic_assessor.assess_image(image_path)

    def get_detailed_scores(self, image_path: str) -> Dict[str, float]:
        """Get detailed aesthetic scores (if available)."""
        # Could be implemented in AestheticAssessor
        return {}

    def get_method_name(self) -> str:
        return f"CLIP_Aesthetic_{self.model_name}_{self.aggregation_method}"
```

### Benefits of This Architecture

#### 1. **Separation of Concerns**
- ✅ `vision_language/`: Core VL model capabilities
- ✅ `vision_language/applications/`: Use-case specific logic
- ✅ `quality_assessment/`: Domain-specific wrappers

#### 2. **Reusability**
```python
# Use same CLIP model for multiple tasks
clip = CLIPModel("ViT-B-32", device="cuda")

# Quality assessment
aesthetic = AestheticAssessor(clip)
score = aesthetic.assess_image("photo.jpg")

# Semantic retrieval
retrieval = SemanticRetrieval(clip)
retrieval.index_images(all_images)
results = retrieval.search("sunset photos")

# Zero-shot classification
classes = ["indoor", "outdoor", "portrait", "landscape"]
predictions = clip.zero_shot_classify(images, classes)
```

#### 3. **No Duplication**
- ✅ Single CLIP model instance
- ✅ Shared preprocessing
- ✅ Shared caching
- ✅ Consistent configuration

#### 4. **Extensibility**
```python
# Easy to add new VL models
class BLIPModel(BaseVisionLanguageModel):
    # Implement BLIP-specific logic
    pass

class LLaVAModel(BaseVisionLanguageModel):
    # Implement LLaVA-specific logic
    pass

# Easy to add new applications
class ImageCaptioning:
    def __init__(self, model: BaseVisionLanguageModel):
        ...

    def caption_image(self, image_path: str) -> str:
        ...
```

#### 5. **Backward Compatibility**
```python
# Old API still works!
from sim_bench.quality_assessment import load_quality_method

method = load_quality_method('clip_aesthetic', config)
score = method.assess_image("photo.jpg")
```

### Migration Plan

#### Phase 1: Create New Subpackage (Non-Breaking)
1. Create `sim_bench/vision_language/`
2. Implement base classes and CLIP
3. Implement applications (aesthetic, retrieval)
4. Add tests

#### Phase 2: Deprecate Old Implementation
1. Update `quality_assessment/clip_aesthetic.py` to use new API
2. Add deprecation warnings to old direct usage
3. Update documentation

#### Phase 3: Remove Duplication (Breaking Change)
1. Remove `feature_extraction/openclip.py` vision-language features
2. Keep only feature extraction for similarity tasks
3. Major version bump (2.0.0)

## Recommendation

**Yes, create `sim_bench/vision_language/` subpackage!**

### Immediate Actions

1. **Create new subpackage** with architecture above
2. **Keep current `clip_aesthetic.py` working** (update to use new API internally)
3. **Add new capabilities** (retrieval, classification) as separate modules
4. **Document** the new API with examples

### Long-term Vision

```
sim_bench/
├── vision_language/         # NEW: Vision-language models
│   ├── base.py
│   ├── clip.py
│   ├── blip.py
│   └── applications/
│       ├── aesthetic.py
│       ├── retrieval.py
│       ├── classification.py
│       └── captioning.py
│
├── feature_extraction/      # Pure vision features (no text)
│   ├── dinov2.py
│   ├── resnet50.py
│   └── sift_bovw.py
│
└── quality_assessment/      # Domain wrappers
    ├── clip_aesthetic.py    # Uses vision_language
    ├── nima.py
    └── rule_based.py
```

This architecture prepares sim-bench for:
- ✅ Multi-modal capabilities
- ✅ Semantic search
- ✅ Zero-shot classification
- ✅ Future VL models (BLIP, LLaVA, GPT-4V)
- ✅ Clean, maintainable code

---

**Next Step**: Should I implement this new architecture?
