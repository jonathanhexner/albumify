# Ground Truth Labeling & AI-Assisted Workflow Proposal

**Date**: 2026-03-29
**Context**: Google_Germany dataset (788 images) - Need to create ground truth for sanity test validation
**Goal**: Label 50 faces from 5 people efficiently and accurately

---

## Questions for Expert Panel

### Question 1: Ground Truth Selection Strategy

**Context**: Need 50 faces from 5 people for validation baseline.

**Options**:

**A. Random Sampling**:
```python
# Randomly select 50 faces from entire dataset
random_faces = random.sample(all_faces, 50)
# Hope they represent 5+ different people
```
❌ **Risk**: Might get 30 faces of one person, 20 of another → poor coverage

**B. Stratified Sampling**:
```python
# 1. Run clustering first (may have errors)
# 2. Select 10 faces from 5 different clusters
# 3. Manually verify each cluster is one person
```
⚠️ **Risk**: If clustering is broken, we're validating against broken clusters

**C. Quality-Stratified Sampling**:
```python
# 1. Select faces from different quality tiers:
#    - 10 frontal, high quality (blur > 150, yaw < 15)
#    - 10 frontal, medium quality (blur > 100, yaw < 30)
#    - 10 profile, high quality (yaw 30-45)
#    - 10 profile, medium quality (yaw > 45)
#    - 10 challenging (blur < 100 or yaw > 60)
# 2. Manually group into people
```
✅ **Pro**: Tests algorithm on diverse quality levels
⚠️ **Risk**: Time-consuming to find specific quality combinations

**D. Exemplar-Based Sampling**:
```python
# 1. Run initial clustering
# 2. Select exemplars from 5 largest clusters
# 3. Manually verify each cluster represents one person
# 4. If cluster has multiple people, split and relabel
```
✅ **Pro**: Fast, uses existing clustering as starting point
✅ **Pro**: Focuses on high-confidence faces (exemplars)

**Expert Question**: Which strategy is most effective for validation baseline?

---

### Question 2: Can AI (Claude) Assist with Labeling?

**Context**: User asked "check with an expert on AI usage on this can can be done best in claude"

**Possible AI-Assisted Workflows**:

**Workflow A: AI Pre-Clustering**:
```python
# 1. Show Claude 50 face crops (in browser via Read tool)
# 2. Claude groups faces: "Faces 1,5,12,23 look like same person"
# 3. Human verifies and corrects
```
✅ **Pro**: Fast initial grouping
⚠️ **Risk**: Claude's face recognition may not be perfect
⚠️ **Risk**: Need to show all 50 faces at once (large context)

**Workflow B: AI-Assisted Pairwise Verification**:
```python
# 1. Human provides initial clustering (via streamlit)
# 2. For uncertain pairs, show Claude: "Are face A and B same person?"
# 3. Claude provides reasoning: "Same person - similar nose, eyes, jawline"
# 4. Human makes final decision
```
✅ **Pro**: Focuses AI on uncertain cases only
✅ **Pro**: Provides reasoning (helps human decision)
⚠️ **Risk**: Still requires human verification

**Workflow C: Human-Only with AI Tool Support**:
```python
# 1. Human labels faces manually via streamlit
# 2. Claude verifies labeling consistency:
#    - "Person A has 3 frontal faces, all blur > 150"
#    - "Person B has 7 faces, mix of frontal and profile"
# 3. Claude flags potential errors:
#    - "Face 23 labeled as Person A, but distance to other Person A faces is 0.82"
```
✅ **Pro**: Human has full control
✅ **Pro**: AI provides quality checks, not labels
✅ **Pro**: Most reliable for ground truth

**Expert Question**:
1. Can Claude reliably perform face recognition on crops?
2. Should we use AI for initial labeling or only for verification?
3. What's the best human-AI collaboration workflow?

---

### Question 3: Labeling Interface Workflow

**Context**: User wants to use streamlit app for labeling.

**Current Streamlit App** (`app/face_clustering_labeling.py`):
- Shows clusters with exemplar faces
- Allows assigning person labels to entire clusters
- Has debug view for within/cross-cluster distances

**Proposed Workflow**:

**Phase 1: Initial Clustering + Export** (10 min)
```bash
# Run clustering on Google_Germany
python scripts/export_clustering_data.py \
    --embeddings results/Google_Germany/embeddings_*.npy \
    --output results/Google_Germany/ground_truth_labeling

# Outputs:
# - faces.csv (all faces with initial cluster assignments)
# - clusters.csv (cluster statistics)
# - face_crops/ (112x112 aligned crops)
```

**Phase 2: Select Ground Truth Subset** (5 min)
```python
# scripts/select_ground_truth_subset.py
# Select 5 clusters with:
# - Size: 8-12 faces each (total ~50 faces)
# - Good quality: median blur > 120
# - Diverse: mix of cluster sizes, diameters
# - Well-separated: min cross-cluster distance > 0.5

# Output: ground_truth_subset.json
# {
#   "cluster_ids": [3, 7, 12, 18, 24],
#   "face_ids": [12, 23, 45, ...],  # 50 total
#   "rationale": "Selected for diversity and quality"
# }
```

**Phase 3: Manual Labeling via Streamlit** (30-45 min)
```bash
# Launch labeling app with ground truth filter
streamlit run app/face_clustering_labeling.py -- \
    --data-dir results/Google_Germany/ground_truth_labeling \
    --ground-truth-filter ground_truth_subset.json

# User workflow:
# 1. App shows only 5 selected clusters
# 2. User reviews each cluster:
#    - View all faces in cluster
#    - Check debug view: within-cluster distances
#    - Assign person label (e.g., "Person_A")
# 3. If cluster has multiple people:
#    - Mark faces that don't belong
#    - Create corrected_identities.json
# 4. Save labels
```

**Phase 4: Validation** (5 min)
```python
# Automated validation checks:
def validate_ground_truth_labels(labels):
    """Ensure ground truth is consistent"""

    # Check 1: All 50 faces labeled
    assert len(labels) == 50

    # Check 2: Exactly 5 unique people
    assert len(set(labels.values())) == 5

    # Check 3: Each person has 8-12 faces
    person_counts = Counter(labels.values())
    for person, count in person_counts.items():
        assert 8 <= count <= 12, f"{person} has {count} faces (expected 8-12)"

    # Check 4: Within-person distances < 0.6
    for person in set(labels.values()):
        faces = [f for f, p in labels.items() if p == person]
        max_dist = compute_max_pairwise_distance(faces, embeddings)
        assert max_dist < 0.6, f"{person} max distance {max_dist:.2f} too high"
```

**Expert Question**: Is this workflow efficient? Any improvements?

---

### Question 4: Quality Checks During Labeling

**Context**: How to ensure ground truth is actually correct?

**Real-Time Checks** (in streamlit app):

1. **Consistency Check**:
   ```python
   # When user assigns label, check distances
   def check_label_consistency(cluster_id, person_label, existing_labels):
       """Warn if inconsistent with existing labels"""

       # Get all faces previously labeled as this person
       existing_faces = get_faces_for_person(person_label)
       current_faces = get_faces_in_cluster(cluster_id)

       # Check cross-distance
       cross_distances = compute_distances(current_faces, existing_faces)
       median_cross = np.median(cross_distances)

       if median_cross > 0.5:
           warn(f"⚠️ Median distance to other {person_label} faces: {median_cross:.2f}")
           warn(f"This cluster may not be the same person")
   ```

2. **Within-Cluster Coherence**:
   ```python
   # Check if cluster is actually one person
   def check_cluster_coherence(cluster_id):
       faces = get_faces_in_cluster(cluster_id)
       distances = compute_pairwise_distances(faces)

       diameter = distances.max()

       if diameter > 0.7:
           warn(f"⚠️ Cluster diameter: {diameter:.2f} (high)")
           warn(f"This cluster may contain multiple people")

           # Show most distant pair
           i, j = np.unravel_index(distances.argmax(), distances.shape)
           show_face_pair(faces[i], faces[j], distance=diameter)
   ```

3. **Coverage Check**:
   ```python
   # Ensure all quality levels represented
   def check_coverage(labeled_faces):
       frontal = sum(1 for f in labeled_faces if f.yaw < 20)
       profile = sum(1 for f in labeled_faces if 30 < f.yaw < 60)
       high_quality = sum(1 for f in labeled_faces if f.blur > 150)
       low_quality = sum(1 for f in labeled_faces if f.blur < 100)

       if frontal < 20:
           warn(f"⚠️ Only {frontal} frontal faces (recommend 20+)")
       if profile < 10:
           warn(f"⚠️ Only {profile} profile faces (recommend 10+)")
   ```

**Expert Question**: What other quality checks should be automated?

---

### Question 5: Tools and Optimizations

**Context**: User asked "If there are any tools I can download to make this more effective please tell me"

**Potential Tools**:

1. **Face Crop Viewer** (for quick scanning):
   ```python
   # Tool: Show grid of 50 faces with quick navigation
   # - Arrow keys to navigate
   # - Number keys (1-5) to assign to person
   # - Drag & drop to group
   # - Undo/redo support
   ```

2. **Embedding Visualizer** (UMAP/t-SNE):
   ```python
   # Tool: 2D scatter plot of 50 faces
   # - Color by person label
   # - Click to see face crop
   # - Drag to select multiple faces
   # - Shows distances on hover
   ```

3. **Batch Similarity Tool**:
   ```python
   # Tool: Given one labeled face, find K most similar unlabeled faces
   # - User labels one face as "Person_A"
   # - Tool suggests next 5 most similar faces
   # - User confirms or skips
   ```

4. **External Tools**:
   - **ImageMagick/montage**: Create contact sheets (all faces in grid)
   - **Gephi**: Visualize kNN graph (see cluster structure before labeling)
   - **Jupyter Lab**: Interactive labeling with inline images

**Expert Question**:
1. Which tool would speed up labeling most?
2. Should we build custom tool or use existing streamlit app?

---

### Question 6: Claude Code Capabilities for Face Recognition

**Context**: Can Claude Code (with image viewing) assist with labeling?

**Claude Code Capabilities**:
- ✅ Can view images via `Read` tool
- ✅ Can analyze visual similarity between faces
- ✅ Can provide reasoning for grouping decisions
- ❓ Unknown: Accuracy on face recognition task

**Proposed Experiment**:
```python
# Test: Can Claude group 10 faces accurately?
# 1. Select 10 faces from 3 people (known ground truth)
# 2. Show Claude all 10 crops
# 3. Ask: "Group these faces by person"
# 4. Compare Claude's grouping vs ground truth
# 5. Measure accuracy
```

**Expert Question**:
1. Should we run this experiment before relying on Claude?
2. What accuracy threshold is acceptable for AI-assisted labeling?
3. Or should human do all labeling with AI only for verification?

---

## Streamlit App Verification Plan

**Context**: User wants to browse different pages, verify no HTTP errors, and validate labeling logic.

### Verification Checklist

**Page 1: Cluster Browser** (Sidebar)
- [ ] App loads without errors
- [ ] Cluster list displays correctly (all clusters shown)
- [ ] Filter works: All / Unlabeled / Labeled
- [ ] Sorting works: By size, diameter, exemplar distance
- [ ] Click cluster navigates to detail view

**Page 2: Cluster Detail - Exemplars Tab**
- [ ] Exemplar faces load and display (no broken images)
- [ ] Image quality is good (no blurry thumbnails)
- [ ] Hover shows face ID and distance info
- [ ] Statistics display correctly (size, diameter, median distance)

**Page 3: Cluster Detail - All Faces Tab**
- [ ] All faces in cluster load (may be many)
- [ ] Pagination works if cluster is large (>20 faces)
- [ ] Can zoom/view individual faces

**Page 4: Cluster Detail - Debug Tab**
- [ ] Within-cluster neighbors display with distances
- [ ] Cross-cluster neighbors display with cluster IDs
- [ ] Exemplar distances to other clusters shown
- [ ] Color coding works: 🟢 green (<0.25), 🟡 yellow (0.25-0.40), 🔴 red (>0.40)

**Page 5: Labeling Interface**
- [ ] Can assign person label to cluster
- [ ] Label persists when navigating away and back
- [ ] "New Person" button creates unique label
- [ ] Save button works (writes corrected_identities.json)
- [ ] Progress bar updates correctly

**Page 6: High-Distance Cluster Investigation** (User's specific test)
- [ ] Identify cluster with highest exemplar distance
- [ ] View all exemplar faces in that cluster
- [ ] Switch to debug tab
- [ ] Check cross-cluster neighbors for each exemplar
- [ ] Compare: Are cross-cluster distances < within-cluster distances?
- [ ] Manually determine: Same person (wide variation) or different people (mis-clustered)?

### HTTP Error Checks
```python
# Automated checks during app usage
def test_app_endpoints():
    """Verify no HTTP errors in streamlit app"""

    # Check 1: Static files load
    - face_crops/*.jpg files accessible
    - CSS/JS loads without 404

    # Check 2: Data files load
    - faces.csv loads without errors
    - clusters.csv loads without errors
    - debug_neighbors.json loads without errors

    # Check 3: No memory errors
    - App doesn't crash after viewing many clusters
    - Image thumbnails don't accumulate in memory
```

**Expert Question**: What else should be verified in streamlit app?

---

## Proposed Timeline

### Phase 1: Expert Review (15 min)
- Review this proposal
- Answer 6 questions above
- Provide recommendations

### Phase 2: Ground Truth Selection (10 min)
- Run initial clustering on Google_Germany
- Select 5 clusters for ground truth (using expert-recommended strategy)
- Export subset for labeling

### Phase 3: Labeling (30-45 min)
- Launch streamlit app
- Label 50 faces into 5 people
- Use real-time consistency checks
- Validate ground truth

### Phase 4: Streamlit Verification (15 min)
- Browse all pages systematically
- Test high-distance cluster investigation
- Document any HTTP errors or bugs

### Phase 5: Sanity Test Execution (30-40 min)
- Run threshold calibration
- Run full pipeline with quality checks
- Analyze high-distance clusters
- Generate report

**Total**: ~2-2.5 hours

---

## Success Metrics

**Ground Truth Labeling Success**:
- ✅ 50 faces labeled into 5 people
- ✅ Each person has 8-12 faces
- ✅ Within-person max distance < 0.6
- ✅ Between-person min distance > 0.5
- ✅ Covers diverse quality levels (frontal, profile, high/low blur)

**Streamlit App Success**:
- ✅ All pages load without HTTP errors
- ✅ Images display correctly (no broken thumbnails)
- ✅ Labeling workflow completes successfully
- ✅ Debug view correctly identifies high-distance clusters
- ✅ Can determine root cause of high distances (validation or limitation)

**Sanity Test Success**:
- ✅ Ground truth subset achieves >80% clustering purity
- ✅ Full pipeline completes without errors
- ✅ Quality metrics within expected ranges
- ✅ High-distance clusters analyzed and diagnosed
- ✅ Clear recommendation: Bug vs expected behavior

---

## Expert Panel: Please Provide Feedback

**For each question (1-6), please provide**:
1. Recommended approach (A, B, C, etc.)
2. Rationale and trade-offs
3. Any modifications to proposal
4. Additional considerations

**Specific requests**:
- **Dr. Sarah Chen**: Ground truth selection strategy (Q1), Claude's face recognition capability (Q6)
- **Dr. Marcus Liu**: Labeling interface workflow optimization (Q3)
- **Jordan Lee**: Quality checks (Q4), tools recommendations (Q5)

**All Experts**: AI-assisted labeling workflow (Q2) - Is this reliable or risky?

---

**Next Steps**: Wait for expert panel feedback, then implement recommended approach.
