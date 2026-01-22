# 📊 sim-bench Retrospective

**Purpose:** Personal reference - what I did, when, and what the data showed

**Repository Started:** October 4, 2025  
**Last Updated:** January 22, 2026

---

## Timeline of What I Actually Did

### Oct 4-8, 2025: Initial Dataset Research & Basic Methods

**Task:** Group similar images together

**Dataset Research:**
- Researched multiple image similarity datasets (see `D:\DataSets\image_similarity_datasets.xlsx`)
- Selected **UKBench** to start (10,200 images, 2,550 objects, 4 images each)
- Why UKBench: Standard benchmark, fixed group size (4 images per object), easy to validate

**Methods Tested:**
1. **EMD (Earth Mover's Distance)** on HSV features
2. **SIFT BoVW** (Bag of Visual Words)
3. **CNN features** (ResNet50)

**Result:** CNN features were best

**Metrics used:** N-S Score, mAP, Recall@k

---

### Oct 20, 2025: EDA & Explainability Analysis

**What I Did:**
- Compared performance across all methods
- Looked at failing cases using analysis notebooks
- Used explainability (Grad-CAM for CNN, keypoint visualization for SIFT) to understand decisions

**Key Findings:**
- **Deep learning significantly superior** to traditional methods
- DL uses semantic features that made more sense for similarity
- **Surprised SIFT worked so poorly** - uses grayscale features
- **Question:** Would SIFT work better on color? Could it complement DL as validation?

**Fisher Criterion Analysis:**
- Identified which features actually discriminate between groups
- Found within-group variance issues in some features

---

### Late Oct 2025: Extended Research - INRIA Holidays, CLIP, DINOv2

**What I Did:**
- Researched what other companies (Google, Meta, Apple) are doing
- Found **INRIA Holidays dataset** (1,491 images, 500 queries - real vacation photos)
- Extended methods to:
  - **OpenCLIP** (vision-language)
  - **DINOv2** (Meta's self-supervised transformer)
- Ran both on UKBench and INRIA Holidays

**Key Findings:**
| Method | UKBench mAP@10 | Holidays mAP@10 | Notes |
|--------|---------------|-----------------|-------|
| **DINOv2** | 0.958 | 0.885 | Best, semantic features |
| **OpenCLIP** | 0.947 | 0.861 | Vision-language |
| **ResNet50** | 0.946 | 0.820 | Still good |
| **SIFT BoVW** | 0.500 | 0.368 | Poor |
| **EMD** | 0.265 | 0.225 | Worse |

**Insights:**
- **UKBench is VERY EASY** - all methods get >0.90 mAP@10
- **DINOv2 slightly better**, uses very semantic features
- **Thought:** Maybe need specialized models? (faces, specific cars, buildings, etc.) for more accurate separation per category

---

### Nov 7, 2025: Clustering Test Application

**What I Did:**
- Built clustering application using DINOv2
- Methods: KMeans, DBSCAN, HDBSCAN
- Tested on Budapest personal photos (310 images)
- HTML gallery visualization

**Result:** Looked good! Ready to move to next step.

**Next goal:** Choose best image per cluster

---

### Nov 14, 2025 - Dec 2025: PhotoTriage Dataset Discovery & Struggles

**Extensive Dataset Research:**
- Found **PhotoTriage dataset** and paper by Huiwen Chang, Fisher Yu, Jue Wang, Douglas Ashley, Adam Finkelstein
- Dataset: 12,988 images in 4,986 series (bursts)
- JSON files explain which image selected and why

**Attempted Categorization:**
- Used ChatGPT to categorize user reasons (poor lighting, composition, cropping, etc.)
- **Problem:** Many reasons difficult to classify into specific category
- Users often gave vague or mixed reasons

**Quality Assessment Attempts:**

1. **IQA (Rule-based):** ~50% accuracy (poor!)

2. **CNN/Transformer variations** based on AI suggestions:
   - Tried many variations
   - Nothing worked
   - Thought I needed Bradley-Terry ranking

3. **Finally read paper myself:**
   - AI was misleading me!
   - Actually need **Siamese U-Net**: Intermediate CNN layers + MLP
   - Implemented this... **didn't work again!**

**Frustration point reached.**

---

### Late Dec 2025: Breakthrough with GitHub Solution

**Found:** [zhenshen-mla/Series-Photo-Selection](https://github.com/zhenshen-mla/Series-Photo-Selection)
- Implementation of ICASSP2020 paper by Jin Huang et al.
- Multi-scale feature aggregation from different network layers
- PAUnit (Parallel Attention Unit) with spatial-channel self-attention
- ResNet backbone (18/50/101) + attention mechanism

**What I Did:**
- Adapted the code in 1-2 days
- **Got 70% accuracy!** (vs my 50%)

**Debugging Process:**
1. Asked AI to find differences between my solution and theirs
2. AI likely **introduced bugs** in the comparison
3. Minimized differences to just the data loader
4. **Critical discovery:** Dataset summary of user decisions DOESN'T correspond to the JSONs they provide!
5. **Dataset itself has a mismatch!**

---

### Jan 5, 2026: Synthetic Degradation Testing

**What I Did:**
- Investigated trained model using synthetic image degradations:
  - Blur
  - JPEG compression
  - Exposure changes
  - Various crops (edge, center, corner, aspect)
- Compared Siamese predictions vs IQA degradation detection

**Result:** Model makes sense! Worked pretty well.

**Degradation Benchmark Results:**
| Degradation Type | Siamese Accuracy |
|-----------------|------------------|
| JPEG | **100%** |
| Crop Edge | **96.5%** |
| Crop Aspect | **94.5%** |
| Crop Center | **94.0%** |
| Crop Corner | **85.5%** |
| Blur | 84.5% |
| Exposure | 74.0% |

---

### Jan 16-17, 2026: AVA Dataset Training

**Found:** AVA dataset - ranking histogram from 1 to 10

**What I Did:**
- Trained ResNet with layer4 connected to MLP predicting histogram
- **Result:** 60% Spearman correlation

**Repeated degradation comparison:**
- Compared AVA predictions vs IQA
- Compared with Siamese

**Conclusion:** **Siamese U-Net works best!**

**Final Degradation Comparison:**
| Model | Overall | Blur | JPEG | Exposure | Crops Avg |
|-------|---------|------|------|----------|-----------|
| **Siamese** | 89.9% | 84.5% | 100% | 74.0% | 92.6% |
| **AVA** | 81.9% | **92.5%** | 95.2% | **91.0%** | 71.0% |
| **IQA** | 68.4% | **100%** | 89.6% | 78.5% | 47.5% |

---

### Jan 19, 2026: Realizations & Unified Framework

**What I Realized:**
- Could have done **self-supervision** on large dataset prior to actual training
- Didn't do **data augmentation** (should have!)
- Initial thoughts of combining more features in MLP:
  - IQA metrics
  - Intermediate CNN features  
  - High-level DINO embeddings
  - Maybe revisit this later?

**Refactored:** Unified image quality models framework

---

### Jan 2026 - Recent: Facial Attributes with MediaPipe

**Goal:** Account for facial attributes in quality score (very important!)
- Is subject smiling?
- Are eyes open?

**Found:** Google's MediaPipe package
- Produces facial landmarks (eye location, lips, etc.)

**Problem:** Didn't find quick dataset for benchmark

**What I Did:**
- Tried on my own photos
- **Result:** Worked reasonably well
- Needed to tune some parameters

---

## 📊 Datasets Compared

| Dataset | Images | Groups | Avg Size | Difficulty | Purpose | When Used |
|---------|--------|--------|----------|------------|---------|-----------|
| **UKBench** | 10,200 | 2,550 | 4 (fixed) | **VERY EASY** | Object retrieval | Oct 2025 (first) |
| **INRIA Holidays** | 1,491 | 500 | 2-20+ | Medium-Hard | Real scenes | Late Oct 2025 |
| **PhotoTriage** | 12,988 | 4,986 | 2.6 | Medium | Bursts + quality | Nov-Dec 2025 |
| **AVA** | Large | N/A | N/A | - | Aesthetic scores | Jan 2026 |
| **Budapest** | 310 | N/A | N/A | - | Personal clustering | Nov 2025 |

---

## 🎯 Key Performance Numbers

### Image Similarity (Oct-Nov 2025)

**UKBench (Easy):**
- DINOv2: **0.958** mAP@10
- OpenCLIP: 0.947
- ResNet50: 0.946
- SIFT: 0.500
- EMD: 0.265

**INRIA Holidays (Harder):**
- DINOv2: **0.885** mAP@10
- OpenCLIP: 0.861
- ResNet50: 0.820
- SIFT: 0.368
- EMD: 0.225

### Quality Assessment (Nov-Dec 2025)

**PhotoTriage - Initial Attempts:**
- IQA (rule-based): **~50%** accuracy
- My Siamese attempts: **~50%** accuracy
- GitHub adapted solution: **~70%** accuracy

**PhotoTriage - After debugging:**
- Rule-based (Sharpness only): **64.95%** Top-1
- Composite methods: **41-42%** (worse than sharpness!)
- NIMA, ViT, CLIP: **41-42%**

### Degradation Benchmark (Jan 2026)

**50 images, 27 degradation variants (1,350 comparisons):**
- Siamese: **89.9%** overall
- AVA: **81.9%** overall
- IQA: **68.4%** overall

### AVA Training (Jan 2026)

- Spearman correlation: **0.637** (60%)

---

## 🔍 What I Learned

### 1. UKBench is Too Easy
- All methods get >0.90 mAP@10
- Not discriminative enough for real-world comparison
- Holidays and PhotoTriage more realistic

### 2. Deep Learning >> Traditional (for Similarity)
- DINOv2 semantic features make sense
- SIFT surprisingly poor (grayscale features?)
- Explainability showed DL focusing on meaningful regions

### 3. Dataset Mismatch in PhotoTriage
- Summary of user decisions ≠ actual JSON data
- Spent time debugging my code when dataset itself had issues
- Critical to validate dataset quality, not just code

### 4. AI Can Mislead
- AI suggested wrong architecture for PhotoTriage paper
- When I read paper myself: actually needed different approach
- AI-generated code comparisons introduced bugs
- **Lesson:** Read papers yourself, validate AI suggestions

### 5. Simple Can Beat Complex (Sometimes)
- Sharpness-only (64.95%) >> Composite methods (41-42%)
- Adding more metrics diluted the signal
- Domain knowledge (sharpness matters most) > model complexity

### 6. Model Specialization Matters
- Siamese (human preferences): Best on composition (92.6% crops)
- AVA (aesthetic scores): Best on technical (91% exposure, 92.5% blur)
- IQA (hand-crafted): Only works on single metrics (100% blur, 47.5% crops)
- **Implication:** Ensemble could be powerful

### 7. Missed Opportunities
- Should have done self-supervision
- Should have done data augmentation
- Could combine IQA + intermediate CNN + DINO features

---

## 🛠️ Tools & Analysis I Built

### Analysis Notebooks (Oct 20, 2025)
1. **methods_comparison.ipynb** - Compare all methods, correlation analysis
2. **method_analysis.ipynb** - Deep-dive single method, visualize failures
3. **feature_exploration.ipynb** - Fisher criterion, feature discriminability, Grad-CAM

### Infrastructure
- **Factory pattern** for all components (methods, datasets, assessors, metrics)
- **Feature caching** (10-300x speedup on reruns)
- **Progress monitoring** with tqdm
- **Quick test mode** (`--quick` flag)
- **Comprehensive logging** (experiment.log + detailed.log)

### Degradation Testing (Jan 5, 2026)
- Synthetic degradation generator (blur, JPEG, exposure, crops)
- Benchmark comparing Siamese vs AVA vs IQA
- Validation that models make sense

---

## ❌ What Didn't Work

1. **IQA on PhotoTriage:** ~50% accuracy (too simple)
2. **Composite quality metrics:** 41-42% vs 64.95% sharpness-only
3. **My Siamese implementations:** ~50% accuracy (bugs, likely from AI suggestions)
4. **SIFT for similarity:** 0.368-0.500 mAP@10 (grayscale features inadequate)
5. **Learned CLIP prompts:** 60% of pairs had no labels, uncertain ground truth
6. **Trusting AI blindly:** Wrong architecture, introduced bugs in code comparison

---

## ✅ What Actually Worked

1. **DINOv2 for semantic similarity:** 0.885-0.958 mAP@10
2. **GitHub Series-Photo-Selection adapted code:** 70% accuracy
   - Multi-scale feature aggregation (different CNN layers)
   - PAUnit attention mechanism (spatial-channel)
   - ResNet backbone architecture
3. **Sharpness baseline:** Simple but effective (64.95%)
4. **Degradation benchmarking:** Revealed model specialization clearly
5. **AVA training:** 60% Spearman correlation
6. **MediaPipe for faces:** Worked reasonably well on my photos
7. **Reading papers myself:** Found what AI missed

---

## 🔮 What I Might Try Next

### Immediate
- Manual analysis of failure cases (100-200 examples)
- Try data augmentation on PhotoTriage
- Try self-supervised pre-training
- Test ensemble (Siamese + AVA)

### Ideas
- Combine IQA + intermediate CNN + DINO features in MLP
- Bradley-Terry ranking for single-pass scoring
- Specialized models per category (faces, cars, buildings)
- Color SIFT as complementary validation to DL
- Benchmark MediaPipe facial attributes if find dataset

---

## 📝 Repository Milestones

| Date | What Happened |
|------|---------------|
| **Oct 4, 2025** | Initial commit - complete framework with factory patterns |
| **Oct 6, 2025** | Major refactoring - clean architecture |
| **Oct 8, 2025** | Performance improvements - caching, logging, analysis |
| **Oct 20, 2025** | Analysis notebooks and feature exploration |
| **Oct 26, 2025** | Documentation consolidation, metrics bug fixes |
| **Nov 7, 2025** | Clustering added (KMeans, DBSCAN, HDBSCAN) |
| **Nov 14, 2025** | Quality assessment module added |
| **Jan 5, 2026** | Synthetic degradation testing, pairwise benchmark, CLIP aesthetic |
| **Jan 16, 2026** | AVA training implementation |
| **Jan 17, 2026** | AVA dataset column mapping fixes |
| **Jan 19, 2026** | Unified image quality models framework |

---

**Current Status:** Working on facial attribute integration with MediaPipe. Have working Siamese model (89.9% on degradations), looking to improve with augmentation/self-supervision.
