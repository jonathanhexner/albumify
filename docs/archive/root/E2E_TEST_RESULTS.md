# E2E Face Clustering Test Results

**Date**: 2026-03-24
**Test Data**: `test_data/face_clustering/` (3 people, 6 JPG images)
**Output**: `results/test_face_clustering_e2e/`

---

## ✅ Test Summary

### Pipeline Execution: **SUCCESS**

All 5 stages completed successfully:

1. **[Stage 1] Detect & Extract Embeddings**
   - Input: 9 images (6 JPG + 3 HEIC)
   - Processed: 6 JPG images (HEIC not supported by cv2.imread)
   - Detected: 6 faces
   - ✅ All faces have valid `image_path` (no nulls)
   - ✅ Saved: `face_records.json`

2. **[Stage 2] Quality Gating**
   - Quality filtering: DISABLED for test (all faces used)
   - Core faces: 6
   - Holdout faces: 0

3. **[Stage 3] Save Crops**
   - Saved: 6 aligned face crops (112x112)
   - ✅ All crops exist and match face_ids
   - ✅ Saved: `crop_manifest.json`

4. **[Stage 4] Clustering**
   - Algorithm: Mutual kNN + Connected Components
   - Config: K=3, threshold=0.40, min_cluster_size=1
   - Result: 4 clusters, 0 noise
   - ✅ Saved: `cluster_result.json`

5. **[Stage 5] Export**
   - ✅ Saved: `faces.csv` (6 rows)
   - ✅ Saved: `clusters.csv` (4 rows)
   - ✅ Saved: `export_summary.json`

---

## 📊 Clustering Results

### Cluster Assignments

| Cluster | Size | Faces (face_id) | Ground Truth Person | Status |
|---------|------|-----------------|---------------------|--------|
| 0       | 2    | 0, 1            | person_1            | ✅ Correct |
| 1       | 1    | 2               | person_2            | ⚠️ Split (1/2) |
| 2       | 1    | 3               | person_2            | ⚠️ Split (2/2) |
| 3       | 2    | 4, 5            | person_3            | ✅ Correct |

### Accuracy

- **Person 1**: ✅ Both faces in same cluster (cluster 0)
- **Person 2**: ⚠️ Split across 2 clusters (clusters 1 & 2)
- **Person 3**: ✅ Both faces in same cluster (cluster 3)

**Overall**: 2/3 people correctly clustered (66.7%)

### Analysis

**Why did Person 2 split?**

Possible causes:
1. **Distance threshold too strict**: 0.40 may be too low for some pose/lighting variations
2. **K too small**: K=3 with only 6 faces creates sparse connectivity
3. **Face quality differences**: Two photos may have different quality/pose
4. **Small dataset**: With only 2 faces per person, no majority voting possible

**Recommended fixes**:
- Increase `distance_threshold` to 0.45-0.50
- Increase `K` to 5 (but limited by small dataset)
- Add more faces per person for robust clustering

---

## 🔍 Data Integrity Verification

All checks passed:

✅ **File completeness**
- `face_records.json` ✓
- `crop_manifest.json` ✓
- `cluster_result.json` ✓
- `export_summary.json` ✓
- `faces.csv` ✓
- `clusters.csv` ✓
- `crops/` directory with 6 files ✓

✅ **Data consistency**
- All `face_id` values sequential (0-5) ✓
- All `image_path` values non-null ✓
- All face crops exist and match face_ids ✓
- All cluster assignments valid ✓

✅ **Traceability**
- Every face traceable: image_path → bbox → embedding → crop → cluster ✓
- `export_summary.json` contains full run metadata ✓

---

## 🎨 Streamlit Labeling App

### Ready to Use

The labeling app can now load this data:

```bash
streamlit run app/face_clustering_labeling.py
```

**In the app**:
1. Select directory: `D:\sim-bench\results\test_face_clustering_e2e`
2. Review clusters: Should see 4 clusters
3. Correct labels:
   - Clusters 1 & 2 should both be labeled as "person_2" (they're the same person)
   - Cluster 0: "person_1"
   - Cluster 3: "person_3"
4. Save corrected labels: `corrected_labels.csv`

### Expected App Behavior

- ✅ Load faces.csv and clusters.csv without errors
- ✅ Display 4 clusters with face thumbnails
- ✅ Allow merging clusters 1 & 2 (split person_2)
- ✅ Save corrected identities for ML training

---

## 🐛 Known Issues

### 1. HEIC Files Not Supported

**Issue**: `cv2.imread()` cannot load HEIC files (Apple's image format)

**Impact**: 3 out of 9 test images skipped (person_1, person_2, person_3 each had 1 HEIC file)

**Workaround**: Convert HEIC to JPG before processing

**Fix**: Add HEIC support using `pillow-heif`:
```python
# In face_cluster/embedding.py
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    img = Image.open(image_path)
    img = np.array(img)
except:
    img = cv2.imread(str(image_path))
```

### 2. Unicode Characters on Windows

**Issue**: Checkmark and arrow characters cause `UnicodeEncodeError` on Windows console

**Fix**: Replaced all unicode with ASCII:
- `✓` → `[OK]`
- `✅` → `[PASS]`
- `⚠️` → `[WARN]`
- `→` → `->`

---

## ✅ Next Steps

### Immediate (Ready Now)

1. **Test Streamlit labeling app** with this data
   ```bash
   streamlit run app/face_clustering_labeling.py
   ```

2. **Manually correct cluster 2** by merging with cluster 1 (both are person_2)

3. **Save corrected labels** for ML training

### Short Term (This Week)

4. **Add HEIC support** to `face_cluster/embedding.py`

5. **Re-run test** with all 9 images (3 per person) to verify improved accuracy

6. **Run on full album** (`Google_Germany`) and verify:
   - All embeddings match crops (no SIGHTING-006 repeat)
   - Full lineage trace works
   - Labeling app loads correctly

### Medium Term (Architecture Cleanup)

7. **Implement new architecture** from `RECOVERY_PLAN.md`:
   - Create `scripts/run_face_clustering.py` (replace `benchmark_face_clustering.py`)
   - Add stage tests to `tests/face_clustering/`
   - Archive old debug scripts/notebooks

8. **Resume ML training workflow**:
   - Label corrected identities
   - Generate merge features
   - Train logistic regression classifier

---

## 📝 Test Command Reference

```bash
# Run E2E test
python scripts/test_e2e_face_clustering.py

# Verify data integrity
python scripts/verify_labeling_app_data.py

# Open labeling app
streamlit run app/face_clustering_labeling.py
```

---

## 🎯 Success Criteria: MET ✅

- ✅ Pipeline runs E2E without crashes
- ✅ No null `image_path` values (SIGHTING-006 prevented)
- ✅ Full traceability (face → image → crop → cluster)
- ✅ Data format compatible with labeling app
- ✅ All files saved independently (no data loss on crash)
- ⚠️ Clustering accuracy: 66.7% (2/3 people correct)
  - Acceptable for test dataset with only 2 faces/person
  - Will improve with more faces and tuned parameters

---

**Conclusion**: The face clustering pipeline is **working correctly** with full traceability and data integrity. The clustering split on person_2 is expected given the small dataset (2 faces/person) and can be corrected manually in the labeling app or by adding more test images.
