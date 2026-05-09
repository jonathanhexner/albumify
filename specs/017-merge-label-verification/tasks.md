# Tasks: Merge Label Verification Tab

**Spec**: `specs/017-merge-label-verification/spec.md`

## Design Notes

### Canonical Runs (one per source dataset)

```python
# Use earliest pre-merge runs — post-merge reclusters have near-zero candidate pairs.
# Germany_10: 748 pairs @0.45. shira_album1/base: 2236 faces, 13 pairs @0.45 / 103 @0.60.
CANONICAL_RUNS = {
    "Google_Germany": {"run": "Germany_12",         "crop_source": None},  # K=50; Germany_10 excluded (K=5)
    "Austria_24":     {"run": "Austria24_1",         "crop_source": None},
    "Shira_Album1":   {"run": "shira_album1/base",   "crop_source": None},
    "Noa_5-7":        {"run": "Noa5-7",              "crop_source": None},
    "Noa_2-5":        {"run": "Noa2-5",              "crop_source": None},
}
```

### Crop Loading
Crops live in `{run_dir}/crops/face_{fid:04d}_aligned.jpg` for all canonical runs.
No cross-run fallback needed (each run has its own crops).

### Tab Placement
Add as a new tab in `app/face_clustering/main.py`, after the ML Training tab.

---

## Task Checklist

### Phase 1: Backend
- [x] Add canonical run config — `configs/label_runs.csv` + `load_canonical_runs()` (CSV-based, editable without restart)
- [x] Add crop fallback helper: `load_crop(face_id, run_dir, fallback_crop_dir)`
- [x] Add `get_labels_for_run(run_id)` + `save_human_label()` to training_db
- [x] Support `label=NULL` / excluded flag in training_db for ignored pairs (source='human', label=NULL)

### Phase 2: Tab UI — Pairs Table
- [ ] Create `app/face_clustering/tabs/label_tab.py`
- [ ] Dataset selector dropdown (one per source dataset)
- [ ] Load run data + compute candidates in background thread
- [ ] Pairs table: all candidates as rows sorted by `min_exemplar_dist` asc
- [ ] Row thumbnails: 5-10 faces per row (exemplars first, then remaining), both clusters separated by divider
- [ ] Row columns: min_exemplar_dist, p10_cross_dist, cluster sizes, heuristic label
- [ ] Row action buttons: Merge / Reject / Ignore (three-way, mutually exclusive)
  - Merge = label=1 (positive training sample)
  - Reject = label=0 (negative training sample)
  - Ignore = excluded from training data entirely
- [ ] Row verified toggle: Verified / Not Verified button
- [ ] Save decision + verified status to training_db on click
- [ ] Progress indicator: "Verified X / Y candidate pairs (Z remaining)"
- [ ] Candidate threshold slider
- [ ] Filter chips: All, Unverified, Heuristic Rejects, Heuristic Merges

### Phase 3: Detail Panel (row click to expand)
- [ ] Click row to expand detail panel below table
- [ ] Show ALL face crops from both clusters (exemplars highlighted with border)
- [ ] Full distance metrics: min_exemplar_dist, p10_cross_dist, p50_cross_dist, support_fraction, post_merge_diameter, diameter_expansion
- [ ] Gate pass/fail badges (from merge_log)
- [ ] Close button / Esc to collapse

### Phase 4: Integration
- [ ] Wire tab into main.py
- [ ] Verify labels are consumed by eda_merge_ml.ipynb via load_training_data()
- [ ] Add label summary table (per-dataset: merge/reject/ignore by source)
- [ ] Export button: CSV with all labeled pairs (ignored pairs excluded)
