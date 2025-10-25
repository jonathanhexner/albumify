# AI Burst Image Similarity and Clustering — Literature Review

## 1. Introduction

This literature review summarizes key methods and research directions relevant to building an AI system that can analyze burst photo sequences, identify visually similar images, and ultimately select or generate the best frame. The discussion draws upon techniques used by major technology companies such as **Google**, **Meta (Facebook)**, and **Apple**, as well as modern academic and open-source research.

---

## 2. Background and Motivation

In modern smartphone photography, cameras often capture multiple frames per shutter press (a **burst**) to counteract motion blur, capture fleeting expressions, or improve low-light performance. Systems like **Google HDR+**, **Apple Smart HDR**, and **Samsung Scene Optimizer** use AI pipelines to:
- Detect and cluster near-identical frames,
- Select the sharpest or most aesthetically pleasing image, and
- Merge frames for enhanced quality (HDR or super-resolution).

Building such systems involves three main components:
1. **Feature extraction**: Represent each image as a high-dimensional embedding capturing its visual and semantic properties.
2. **Similarity estimation**: Quantify how close two images are, typically using cosine similarity or Euclidean distance.
3. **Clustering and selection**: Group visually similar images and select the optimal representative per cluster.

---

## 3. Methods Used by Industry Leaders

### 3.1 Meta (Facebook)
Meta employs large-scale **embedding-based clustering** for image search and deduplication. Their system typically uses:
- **Self-supervised vision transformers** such as [DINOv2](https://ai.meta.com/research/publications/dinov2/) or SwAV.
- **FAISS** ([Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)) for high-speed nearest neighbor search and clustering using algorithms like *hierarchical k-means* or *HNSW* (Hierarchical Navigable Small World graphs).

These techniques power applications like visual search, content moderation, and near-duplicate detection.

### 3.2 Google
Google’s image clustering appears in multiple products including **Google Photos**, **Google Lens**, and **Image Search**. Key components include:
- Deep feature extraction using models such as [DELG](https://arxiv.org/abs/2001.05027) (Deep Local and Global Features), [SigLIP](https://arxiv.org/abs/2303.15343), and CLIP-like architectures.
- **ScaNN** ([Scalable Nearest Neighbors](https://github.com/google-research/google-research/tree/master/scann)) for fast similarity retrieval.
- Graph-based clustering such as *HDBSCAN*, *Agglomerative clustering*, or *connected components* to group similar frames.

In products like **Google Photos**, a lightweight on-device model ranks burst frames by sharpness, expression, motion blur, and exposure before deeper clustering or selection occurs.

### 3.3 Apple
Apple’s photo analysis framework performs on-device grouping of images into *moments* and *bursts* using optimized embedding models (likely MobileNet or ViT variants). Clustering relies on cosine similarity between embeddings, applied hierarchically:
- Within bursts (near-duplicate grouping)
- Across time (scene detection)
- Across subjects (face clustering)

Reference: [Apple Vision Framework](https://developer.apple.com/documentation/vision/)

---

## 4. Clustering Algorithms

| Step | Purpose | Common Algorithms / Tools |
|------|----------|---------------------------|
| **1. Feature extraction** | Convert image → vector | ResNet, ViT, DINOv2, CLIP, SigLIP |
| **2. Similarity measure** | Compute pairwise distance | Cosine similarity, Euclidean distance |
| **3. ANN Search** | Efficient retrieval | FAISS, ScaNN, HNSWlib |
| **4. Clustering** | Group by similarity | K-means, HDBSCAN, Agglomerative, Spectral |
| **5. Thresholding** | Filter duplicates | Cosine similarity ≥ 0.9 |

### 4.1 Commonly Used Methods
- **K-Means (Faiss GPU)**: Fast, scalable; requires knowing cluster count.
- **HDBSCAN**: Density-based; automatically infers number of clusters.
- **Agglomerative Clustering**: Hierarchical grouping for bursts or events.
- **Connected Components**: Simple thresholded graph; effective for bursts.

### 4.2 Example Pipeline
```python
# Example: Simple burst clustering using DINOv2 and DBSCAN
from transformers import AutoModel, AutoProcessor
import torch, numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

model = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoProcessor.from_pretrained("facebook/dinov2-base")

def embed_image(img):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model(**inputs).pooler_output
    return emb[0].cpu().numpy()

features = np.stack([embed_image(img) for img in burst_images])
features = normalize(features)
clustering = DBSCAN(eps=0.2, metric="cosine").fit(features)
print(clustering.labels_)
```

---

## 5. Datasets and Benchmarks

| Dataset | Description | Purpose |
|----------|--------------|----------|
| [Google Landmarks v2 (GLDv2)](https://github.com/cvdfoundation/google-landmark) | Millions of landmark images | Tests instance-level grouping |
| [Revisited Oxford/Paris (ROxford/RParis)](https://www.robots.ox.ac.uk/~vgg/data/oxparis/) | Landmark retrieval benchmark | Evaluates embedding quality |
| [HDR+ Burst Dataset](https://arxiv.org/abs/1608.02085) | RAW bursts | Ideal for burst similarity and fusion |
| [INRIA Holidays](https://lear.inrialpes.fr/people/jegou/data.php) | Vacation photos | Classic retrieval benchmark |
| [UKBench](https://www.cs.princeton.edu/~blei/sdm/UKbench.html) | 3-4 similar views per object | Historical clustering baseline |

Metrics: **mAP**, **Recall@K**, **NDCG**, and **pairwise win-rate** (for best photo selection).

---

## 6. Beyond Cosine Similarity — Learning a Similarity Function

### 6.1 Motivation
Instead of relying purely on cosine similarity between pre-trained embeddings, we can *learn* a similarity metric optimized for our domain (e.g., burst images). This is done through **metric learning** using **Siamese** or **Triplet** networks.

### 6.2 Contrastive and Triplet Learning
- **Siamese Network**: Two identical encoders output embeddings for input images; a **contrastive loss** minimizes distance for similar pairs and maximizes for dissimilar pairs.
- **Triplet Network**: Uses an *anchor*, *positive*, and *negative* image; the **triplet loss** enforces that the anchor is closer to the positive than the negative.

📘 Key references:
- [FaceNet (Google)](https://arxiv.org/abs/1503.03832) — Triplet loss for face verification.
- [ArcFace (Deng et al., 2019)](https://arxiv.org/abs/1801.07698) — Margin-based softmax loss for more separable embeddings.
- [SimCLR (Google Research)](https://arxiv.org/abs/2002.05709) — Self-supervised contrastive learning using augmentations.

### 6.3 Application to Burst Clustering
This approach is directly applicable. Using burst sequences:
- **Positive pairs**: frames from the same burst.
- **Negative pairs**: frames from different bursts or scenes.

Training a model this way produces domain-optimized embeddings that better capture subtle variations such as expression, motion, or lighting changes.

### 6.4 When to Use This
| Scenario | Use Learned Similarity? |
|-----------|-------------------------|
| Few, visually distinct bursts | ❌ No — pre-trained embeddings are enough |
| Many subtle variations (expressions, lighting) | ✅ Yes — contrastive learning helps |
| Labeled or structured bursts available | ✅ Yes — can generate positive/negative pairs |

---

## 7. Practical Recommendations

| Stage | Recommended Tools / Methods |
|--------|------------------------------|
| Embeddings | [DINOv2](https://ai.meta.com/research/publications/dinov2/), [SigLIP](https://arxiv.org/abs/2303.15343), [OpenCLIP](https://github.com/mlfoundations/open_clip) |
| ANN Search | [FAISS](https://github.com/facebookresearch/faiss), [ScaNN](https://github.com/google-research/google-research/tree/master/scann) |
| Clustering | HDBSCAN, Agglomerative, Connected Components |
| Quality Scoring | [NIMA](https://arxiv.org/abs/1709.05424), LAION Aesthetic Predictor |
| Datasets | GLDv2, HDR+, ROxford/RParis |

---

## 8. Summary

Modern image similarity and clustering pipelines in industry rely on deep embeddings, efficient approximate search (FAISS/ScaNN), and adaptive clustering. For domain-specific similarity (like burst photography), contrastive or triplet-based learning can further enhance performance. Evaluation typically combines classical retrieval benchmarks (ROxford/RParis) with real burst datasets (HDR+).

In future stages, combining clustering with quality assessment models (NIMA or aesthetics predictors) enables *best-frame selection* or *burst fusion* to produce the optimal output image.

---

*If you want to draft accompanying documentation or write a detailed paper about this work, tools like [Jenni AI](https://jenni.ai/?via=lekys) can assist with writing, citation management, and literature organization for research and technical projects.*

