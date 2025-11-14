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

## 6. Semantic vs Visual Similarity: A Critical Distinction

### 6.1 The Problem with Semantic-Only Approaches

Modern vision transformers like **DINOv2** and **OpenCLIP** excel at capturing **semantic similarity** - understanding that two images contain "cats" or "beaches" even if they look completely different visually. However, this creates challenges for certain applications:

- **Different scenes, same semantics**: Two completely different beach photos might be semantically similar (both contain sand, water, sky) but are visually distinct and should NOT be grouped as duplicates
- **Burst photo detection**: Consecutive frames from the same burst should be visually nearly identical (same scene, lighting, composition) but semantic models may focus on object categories rather than exact visual matches
- **Near-duplicate detection**: Finding edited versions, crops, or re-uploads of the same photo requires visual/geometric matching, not semantic understanding

### 6.2 What Industry Actually Uses

Based on recent information (2024), major companies use **hybrid approaches** that combine both semantic and visual/geometric features:

#### Google's Multi-Level Approach
- **Visual fingerprinting**: Creates unique digital signatures based on colors, shapes, textures, and patterns
- **Semantic embeddings**: Uses models like DELG, SigLIP for high-level understanding
- **ScaNN indexing**: Billion-scale approximate nearest neighbor search in high-dimensional embedding space
- **Google Lens**: Ranks images by similarity and relevance using both visual features and semantic context

Google's system analyzes visual elements first (to find identical or near-identical images) and then uses semantic understanding for broader search queries.

#### Meta's FAISS-Powered System
- **Scale**: Indexes 1.5 trillion 144-dimensional vectors for internal applications
- **Hybrid features**: Combines visual descriptors with semantic embeddings from self-supervised models (DINOv2, SwAV)
- **Applications**: Visual search, content moderation, near-duplicate detection
- **Performance**: 8.5x faster than previous state-of-the-art for billion-scale k-nearest-neighbor search

Meta uses FAISS for both semantic similarity (content recommendations) and visual similarity (duplicate detection) by indexing different types of feature vectors.

### 6.3 The Role of Classical Computer Vision

Traditional techniques like **SIFT**, **ORB**, and **perceptual hashing** remain crucial for specific tasks:

#### When to Use Classical Features:
| Task | Best Approach | Why |
|------|---------------|-----|
| **Exact duplicate detection** | Perceptual hashing (pHash, dHash) | Fast, robust to minor edits |
| **Near-duplicate with transforms** | SIFT/ORB keypoint matching | Handles rotation, scale, perspective |
| **Same-scene burst grouping** | SSIM + deep features | Captures pixel-level similarity |
| **Semantic search** | DINOv2, OpenCLIP | Understands content meaning |
| **Hybrid: duplicate + semantic** | Combine both approaches | Best of both worlds |

#### Classical Geometric Features:
- **SIFT (Scale-Invariant Feature Transform)**: Detects keypoints invariant to scale, rotation, illumination
- **ORB (Oriented FAST and Rotated BRIEF)**: Faster alternative to SIFT, used in real-time applications
- **Perceptual hashing**: Generates compact hash codes for quick similarity checks
- **SSIM (Structural Similarity Index)**: Measures perceived quality difference between images

### 6.4 Visual vs Semantic Similarity Research

Recent research (2024) shows important findings:

- **Visual similarity effect is stronger**: In visual search tasks, visual similarity had a substantially larger effect on distractor fixation than semantic similarity
- **User expectations differ**: Users typically judge similarity based on semantics (what's in the image), while computers naturally measure visual features (how it looks)
- **The semantic gap**: Semantic similarity by human judgment often differs from visual similarity by computer judgment - two images can be:
  - **Semantically similar but visually different**: Two cats that look completely different
  - **Visually similar but semantically different**: A dog and a wolf in similar poses

### 6.5 Recommended Hybrid Pipeline

For robust image similarity systems, industry best practices suggest a **multi-stage pipeline**:

```python
# Stage 1: Fast pre-filtering with perceptual hashing
def quick_duplicate_check(images):
    """Catch exact duplicates early"""
    import imagehash
    hashes = {img: imagehash.phash(img) for img in images}
    # Group images with hash distance < 5
    return group_by_hash_similarity(hashes, threshold=5)

# Stage 2: Geometric verification with keypoint matching
def verify_same_scene(img1, img2):
    """Check if actually the same scene (not just similar content)"""
    import cv2
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # Match keypoints
    matches = match_descriptors(desc1, desc2)
    return len(matches) > 10  # Same scene if many keypoints match

# Stage 3: Deep semantic embeddings for broader grouping
def semantic_clustering(images):
    """Group by high-level content"""
    features = extract_dinov2_features(images)
    return cluster_by_cosine_similarity(features, threshold=0.85)

# Final: Combine all signals
def hybrid_similarity(img1, img2):
    visual_sim = compute_ssim(img1, img2)        # 0-1, pixel-level
    geometric_sim = keypoint_match_ratio(img1, img2)  # 0-1, same scene?
    semantic_sim = cosine_similarity(
        dinov2_embed(img1),
        dinov2_embed(img2)
    )  # 0-1, similar content?

    # Weight based on your use case
    return {
        'visual': visual_sim,
        'geometric': geometric_sim,
        'semantic': semantic_sim,
        'is_duplicate': visual_sim > 0.95 and geometric_sim > 0.8,
        'is_similar_scene': geometric_sim > 0.6,
        'is_similar_content': semantic_sim > 0.85
    }
```

### 6.6 Practical Recommendations by Use Case

| Application | Primary Signal | Secondary Signal | Rationale |
|-------------|---------------|------------------|-----------|
| **Burst photo grouping** | Geometric (SIFT/ORB) | Visual (SSIM) | Must be same scene |
| **Duplicate detection** | Perceptual hash | Visual (SSIM) | Fast + robust to edits |
| **Photo albums/moments** | Semantic (DINOv2) | Temporal clustering | Group by content/events |
| **Image search engine** | Semantic (CLIP/OpenCLIP) | Visual (for refinement) | Users search by concept |
| **Content moderation** | Semantic (DINOv2) | Perceptual hash | Find similar + exact matches |

### 6.7 Key Takeaway

**Don't rely on semantic embeddings alone for visual similarity tasks.** The best production systems (Google, Meta) use **layered approaches**:

1. **Fast pre-filtering**: Perceptual hashing for exact/near-duplicates
2. **Geometric verification**: SIFT/ORB for same-scene detection
3. **Semantic clustering**: Deep embeddings for content-based grouping
4. **Application-specific weighting**: Combine signals based on your use case

This explains why your experiments showed DINOv2 achieving 0.893-0.958 mAP - it's excellent at semantic similarity but may group "visually different but semantically similar" images together. For true visual duplicate detection, you'd want to add geometric or perceptual hash features.

---

## 7. Domain-Specific Networks and Specialized Embeddings

### 7.1 Face Clustering: Same Algorithm, Different Embeddings

**Key insight**: Face clustering uses the **same clustering algorithms** (HDBSCAN, agglomerative clustering, connected components) as general image similarity, but with **face-specific embeddings** optimized for facial features.

#### Face Recognition Networks:
| Network | Embedding Dim | Loss Function | Use Case |
|---------|--------------|---------------|----------|
| **FaceNet** (Google) | 128 | Triplet Loss | Industry standard, robust |
| **ArcFace** | 512 | Additive Angular Margin | State-of-the-art accuracy |
| **CosFace** | 512 | Large Margin Cosine | Similar to ArcFace |
| **SphereFace** | 512 | Angular Softmax | Geometric margin |
| **VGGFace2** | 2048 | Softmax | Pre-trained on 9M images |
| **InsightFace** | 512 | Combined losses | Production-ready, fast |

#### Why Face-Specific Embeddings?
- **Identity-focused**: Trained to distinguish between thousands of individuals
- **Invariant to pose, lighting, expression**: Unlike general vision models
- **Metric learning**: Explicitly trained with triplet/contrastive loss for similarity
- **Compact embeddings**: 128-512 dims vs 768-1536 for DINOv2

**Example Pipeline**:
```python
# Face clustering uses standard algorithms but face-specific embeddings
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

# Extract face embeddings (512-dim, optimized for identity)
app = FaceAnalysis()
app.prepare(ctx_id=0)

face_embeddings = []
for img in images:
    faces = app.get(img)
    if faces:
        face_embeddings.append(faces[0].embedding)  # 512-dim

# Standard clustering on face embeddings
features = np.array(face_embeddings)
clustering = DBSCAN(eps=0.5, metric='cosine').fit(features)
# Each cluster = one person
```

**Key Difference**: The algorithm is identical to image clustering, but face embeddings are trained with:
1. **Triplet loss**: Push same person together, different people apart
2. **Large-margin softmax**: Maximize inter-class separation
3. **Hard negative mining**: Focus on difficult cases

### 7.2 Other Specialized Networks to Consider

Beyond general vision and face recognition, consider these domain-specific models:

#### 1. **Text-Image Models (Multimodal)**
| Model | Provider | Strength | Use Case |
|-------|----------|----------|----------|
| **CLIP** | OpenAI | Text-image alignment | Search by description |
| **OpenCLIP** | Community | Open-source CLIP | Production use |
| **SigLIP** | Google | Improved sigmoid loss | Better scaling |
| **BLIP-2** | Salesforce | Vision-language understanding | Captioning + search |

**When to use**: When you need to search images by text queries ("beach sunset", "red car") or generate descriptions.

#### 2. **Object Detection + Embeddings**
| Model | Purpose | Output |
|-------|---------|--------|
| **YOLO** (v8, v9) | Fast object detection | Bounding boxes + classes |
| **DINO** (Detection) | Transformer-based detection | High-quality boxes |
| **OWL-ViT** | Open-vocabulary detection | Find objects by text |

**When to use**: When you need to find images containing specific objects, then cluster by object type.

**Hybrid approach**:
```python
# Stage 1: Detect objects
detections = yolo_model(image)  # Find "person", "car", etc.

# Stage 2: Extract embeddings per object
for bbox in detections:
    object_img = crop(image, bbox)
    embedding = dinov2(object_img)
    # Cluster similar objects across images
```

#### 3. **Medical/Scientific Imaging**
| Domain | Specialized Models | Why Different? |
|--------|-------------------|----------------|
| **Medical** | MedCLIP, BiomedCLIP | Trained on medical terminology |
| **Satellite** | Prithvi, Satlas | Geospatial understanding |
| **Microscopy** | CellPose, StarDist | Cell/nucleus segmentation |

**Key difference**: These models understand domain-specific concepts (e.g., "cardiomegaly" in X-rays) that general models don't.

#### 4. **Video Understanding**
| Model | Purpose | Strength |
|-------|---------|----------|
| **VideoMAE** | Self-supervised video | Temporal understanding |
| **TimeSformer** | Video classification | Frame-level + temporal |
| **I3D** | Action recognition | Spatio-temporal features |

**When to use**: For video clustering, action recognition, or temporal similarity (not just frame-by-frame).

#### 5. **Fine-Grained Recognition**
| Model | Domain | Use Case |
|-------|--------|----------|
| **iNaturalist models** | Nature (birds, plants, insects) | Species-level classification |
| **Stanford Cars** | Vehicle recognition | Make/model identification |
| **CUB-200** | Bird species | Fine-grained bird classification |

**When to use**: When you need to distinguish between very similar categories (e.g., different bird species, car models).

### 7.3 Choosing the Right Model

Decision tree for selecting specialized models:

```
├─ General images (photos, scenes)?
│  ├─ Need semantic similarity? → DINOv2, OpenCLIP
│  └─ Need visual duplicates? → Perceptual hash + SIFT
│
├─ Faces/people?
│  ├─ Identity clustering? → FaceNet, ArcFace, InsightFace
│  └─ Face detection first? → RetinaFace + FaceNet
│
├─ Text-based search?
│  ├─ Natural language queries? → CLIP, OpenCLIP
│  └─ OCR + similarity? → TrOCR + CLIP
│
├─ Objects/products?
│  ├─ Detect then cluster? → YOLO + DINOv2
│  └─ Fine-grained (cars, animals)? → Domain-specific models
│
├─ Medical/scientific?
│  └─ Use domain-specific models (MedCLIP, BiomedCLIP)
│
└─ Video/temporal?
   └─ VideoMAE, TimeSformer
```

### 7.4 Hybrid Approach: Best of Multiple Models

Production systems often combine multiple specialized models:

```python
# Multi-model pipeline example
def comprehensive_similarity(img1, img2, task='general'):
    results = {}

    if task in ['general', 'all']:
        # Semantic similarity
        results['semantic'] = cosine_sim(
            dinov2_embed(img1),
            dinov2_embed(img2)
        )

        # Visual similarity
        results['visual'] = compute_ssim(img1, img2)

        # Geometric
        results['geometric'] = sift_match_ratio(img1, img2)

    if task in ['faces', 'all']:
        # Face similarity (if faces detected)
        face1 = detect_face(img1)
        face2 = detect_face(img2)
        if face1 and face2:
            results['face_identity'] = cosine_sim(
                arcface_embed(face1),
                arcface_embed(face2)
            )

    if task in ['objects', 'all']:
        # Object-level similarity
        obj1 = detect_objects(img1)
        obj2 = detect_objects(img2)
        results['object_overlap'] = jaccard_similarity(
            obj1['classes'],
            obj2['classes']
        )

    if task in ['text', 'all']:
        # Text-searchable
        results['text_sim'] = cosine_sim(
            clip_embed(img1),
            clip_embed(img2)
        )

    return results
```

### 7.5 Performance Considerations

| Model Type | Embedding Size | Speed | Memory | Best For |
|------------|---------------|-------|--------|----------|
| **Perceptual hash** | 64-256 bits | ⚡️⚡️⚡️ Fastest | 🪶 Minimal | Exact duplicates |
| **SIFT/ORB** | Variable | ⚡️⚡️ Fast | 🪶 Low | Geometric matching |
| **FaceNet/ArcFace** | 128-512 | ⚡️⚡️ Fast | 🪶 Low | Face clustering |
| **ResNet50** | 2048 | ⚡️ Medium | 💾 Medium | General purpose |
| **DINOv2-small** | 384 | ⚡️ Medium | 💾 Medium | Semantic similarity |
| **DINOv2-large** | 1024 | 🐌 Slow | 💾💾 High | Best accuracy |
| **CLIP (ViT-B)** | 512 | ⚡️ Medium | 💾 Medium | Text + image |
| **CLIP (ViT-L)** | 768 | 🐌 Slow | 💾💾 High | Best multimodal |

**Rule of thumb**:
- **Real-time / mobile**: Perceptual hash, MobileNet, small models
- **Batch processing**: DINOv2-large, CLIP-large
- **Hybrid pipelines**: Fast pre-filter (hash) → Medium model (DINOv2-small) → Slow refinement (large model)

---

## 8. Beyond Cosine Similarity — Learning a Similarity Function

### 8.1 Motivation
Instead of relying purely on cosine similarity between pre-trained embeddings, we can *learn* a similarity metric optimized for our domain (e.g., burst images). This is done through **metric learning** using **Siamese** or **Triplet** networks.

### 8.2 Contrastive and Triplet Learning
- **Siamese Network**: Two identical encoders output embeddings for input images; a **contrastive loss** minimizes distance for similar pairs and maximizes for dissimilar pairs.
- **Triplet Network**: Uses an *anchor*, *positive*, and *negative* image; the **triplet loss** enforces that the anchor is closer to the positive than the negative.

📘 Key references:
- [FaceNet (Google)](https://arxiv.org/abs/1503.03832) — Triplet loss for face verification.
- [ArcFace (Deng et al., 2019)](https://arxiv.org/abs/1801.07698) — Margin-based softmax loss for more separable embeddings.
- [SimCLR (Google Research)](https://arxiv.org/abs/2002.05709) — Self-supervised contrastive learning using augmentations.

### 8.3 Application to Burst Clustering
This approach is directly applicable. Using burst sequences:
- **Positive pairs**: frames from the same burst.
- **Negative pairs**: frames from different bursts or scenes.

Training a model this way produces domain-optimized embeddings that better capture subtle variations such as expression, motion, or lighting changes.

### 8.4 When to Use This
| Scenario | Use Learned Similarity? |
|-----------|-------------------------|
| Few, visually distinct bursts | ❌ No — pre-trained embeddings are enough |
| Many subtle variations (expressions, lighting) | ✅ Yes — contrastive learning helps |
| Labeled or structured bursts available | ✅ Yes — can generate positive/negative pairs |

---

## 9. Practical Recommendations

| Stage | Recommended Tools / Methods |
|--------|------------------------------|
| Embeddings | [DINOv2](https://ai.meta.com/research/publications/dinov2/), [SigLIP](https://arxiv.org/abs/2303.15343), [OpenCLIP](https://github.com/mlfoundations/open_clip) |
| ANN Search | [FAISS](https://github.com/facebookresearch/faiss), [ScaNN](https://github.com/google-research/google-research/tree/master/scann) |
| Clustering | HDBSCAN, Agglomerative, Connected Components |
| Quality Scoring | [NIMA](https://arxiv.org/abs/1709.05424), LAION Aesthetic Predictor |
| Datasets | GLDv2, HDR+, ROxford/RParis |

---

## 10. Summary

Modern image similarity and clustering pipelines in industry use **hybrid approaches** that combine multiple signals:

1. **Semantic embeddings** (DINOv2, OpenCLIP, SigLIP) for high-level content understanding
2. **Geometric features** (SIFT, ORB) for same-scene verification and keypoint matching
3. **Perceptual hashing** for fast exact/near-duplicate detection
4. **Visual similarity metrics** (SSIM) for pixel-level comparisons
5. **Domain-specific embeddings** (FaceNet, ArcFace for faces; specialized models for medical, satellite, etc.)

Major companies like Google and Meta **do not rely on semantic embeddings alone**. They use multi-stage pipelines:
- **Google**: Visual fingerprinting + semantic embeddings + ScaNN indexing
- **Meta**: FAISS with 1.5T vectors, combining visual descriptors and semantic features

**Key insights**:

1. **Semantic vs Visual**: DINOv2 and OpenCLIP excel at semantic similarity (understanding "what" is in the image) but may group visually different images with similar content. For true visual similarity tasks like duplicate detection or burst grouping, combine semantic features with geometric or perceptual hash approaches.

2. **Face Clustering**: Uses the **same clustering algorithms** (DBSCAN, HDBSCAN, agglomerative) as general image similarity, but with **face-specific embeddings** (FaceNet, ArcFace, InsightFace). The embeddings are trained with triplet loss to maximize identity separation - this is the key difference, not the clustering algorithm.

3. **Specialized Models**: Different domains require different models - medical imaging needs MedCLIP, satellite imagery needs geospatial models, fine-grained recognition (birds, cars) needs domain-specific classifiers. General-purpose models like DINOv2 won't capture domain-specific nuances.

For domain-specific similarity (like burst photography), contrastive or triplet-based learning can further enhance performance. Evaluation typically combines classical retrieval benchmarks (ROxford/RParis) with real burst datasets (HDR+).

In future stages, combining clustering with quality assessment models (NIMA or aesthetics predictors) enables *best-frame selection* or *burst fusion* to produce the optimal output image.

---

*If you want to draft accompanying documentation or write a detailed paper about this work, tools like [Jenni AI](https://jenni.ai/?via=lekys) can assist with writing, citation management, and literature organization for research and technical projects.*

