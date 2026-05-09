# Spec: Dataset, Album, Run — Terminology & Workflow

**Date**: 2026-05-01
**Status**: Draft
**Problem**: The current app conflates "album" (source photos) with "run" (pipeline execution). Users can't clearly distinguish between input data and processing results. No clear workflow for re-running, comparing runs, or managing datasets.

---

## Proposed Terminology

| Term | Definition | Persistence | Example |
|------|-----------|-------------|---------|
| **Dataset** | A directory of source photos. Immutable input. | Permanent until deleted | `D:\Photos\Austria_2024\` (428 photos) |
| **Run** | A single pipeline execution on a dataset with a specific config. | Permanent until deleted | Run #3 on Austria_2024, K=5, merge=on |
| **Album** | The curated output of a run — selected photos, people, clusters. | Tied to its run | "Austria Best 128" — 128 selected from run #3 |
| **Cache** | Computed features (embeddings, face detections, IQA scores). | Per-dataset, shared across runs | Face embeddings for Austria_2024 |
| **Profile** | A named set of pipeline configuration parameters. | Global, reusable | "strict_merge", "relaxed_quality" |

### Key Distinction

**Dataset** = the photos (input, doesn't change)
**Run** = the processing (config + execution, produces results)
**Album** = the output (selected photos, people assignments)

One dataset can have many runs. Each run produces one album.

---

## Current State vs Target

| Concept | Current | Target |
|---------|---------|--------|
| Source photos | `Album.source_path` | **Dataset** with its own entity |
| Pipeline execution | `PipelineRun` tied to Album | **Run** tied to Dataset |
| Selected output | `PipelineResult.selected_images` | **Album** — first-class output |
| Computed features | `UniversalCache` (global, by image_path) | **Cache** — per-dataset, clearable |
| Config presets | `ConfigProfile` + `ProfileStore` | **Profile** — unchanged |

### Current DB Model
```
Album (source_path, name)
  └─ PipelineRun (status, config, steps)
       └─ PipelineResult (selected_images, metrics, decisions)
       └─ Person (face_instances)
```

### Target DB Model
```
Dataset (source_path, name, created_at, image_count)
  └─ Cache (embeddings, face detections, IQA — shared across runs)
  └─ Run (config, profile, status, created_at)
       └─ Album (name, selected_images, metrics, decisions)
       └─ Person (face_instances)
```

---

## Use Cases

### UC1: First-time analysis of a photo folder

**Actor**: User with a folder of vacation photos

1. User clicks "Add Dataset" → selects folder `D:\Photos\Austria_2024\`
2. System scans folder: 428 images found, shows preview
3. User names it "Austria 2024"
4. User goes to Configure & Run → selects "Austria 2024" dataset
5. User adjusts config (or loads a profile), clicks "Run Pipeline"
6. Pipeline executes → produces Run #1
7. Run #1 produces Album "Austria 2024 — Run 1" with 128 selected photos
8. User views results, explores decisions, browses people

### UC2: Re-run with different config

**Actor**: User unsatisfied with run results (too few selected, wrong people grouping)

1. User goes to Configure & Run
2. Sees Run History for "Austria 2024" — Run #1 (128 selected, K=5, merge=off)
3. Clicks "Load Config" on Run #1 → config populated
4. Adjusts: K=10, merge=on, min_score=0.3
5. Clicks "Run Pipeline" → Run #2 starts
6. **Cache is reused**: embeddings, face detections, IQA scores are NOT recomputed (same dataset)
7. Only clustering, merging, selection steps re-execute
8. Run #2 produces Album "Austria 2024 — Run 2" with 156 selected photos
9. User can compare Run #1 vs Run #2 in run history

### UC3: Compare two runs

**Actor**: User wants to understand impact of config changes

1. User goes to Run History for "Austria 2024"
2. Sees:
   - Run #1: K=5, merge=off → 12 people, 128 selected
   - Run #2: K=10, merge=on → 8 people, 156 selected
3. User can click "View" on either run to see its results
4. (Future) Side-by-side comparison view

### UC4: Edit dataset — remove unwanted images

**Actor**: User realizes some photos shouldn't be included (screenshots, duplicates imported by mistake)

1. User goes to Datasets page
2. Clicks "Edit" on "Austria 2024"
3. Sees thumbnail grid of all 428 images
4. Selects 5 images to exclude (screenshots, blurry test shots)
5. Clicks "Exclude Selected" → dataset now has 423 images
6. Excluded images are marked (not deleted from disk), stored as `Dataset.excluded_images`
7. Next run uses 423 images
8. Old runs (that used 428) are preserved as-is
9. Cache for excluded images can be optionally cleared

### UC5: Add more images to dataset

**Actor**: User finds more photos from the same trip on another device

1. User goes to Datasets → "Austria 2024" → "Add Images"
2. Selects additional folder or files
3. System adds 50 new images → dataset now has 473
4. Cache for new images is empty — next run will compute features for the 50 new ones
5. Old runs unaffected

### UC6: Delete a run

**Actor**: User wants to clean up failed or obsolete runs

1. User goes to Run History
2. Sees Run #1 (completed), Run #2 (failed), Run #3 (completed)
3. Clicks "Delete" on Run #2
4. System deletes: PipelineRun, PipelineResult, Person records for that run
5. Cache is NOT deleted (shared with other runs)
6. Export artifacts on disk are deleted if they exist

### UC7: Delete a dataset

**Actor**: User is done with a photo set

1. User goes to Datasets → "Austria 2024" → "Delete"
2. System warns: "This will delete 3 runs, 2 albums, and all cached features. Source photos are NOT deleted."
3. User confirms
4. System deletes: Dataset, all Runs, all Results, all People, all Cache entries for this dataset
5. Source photos on disk are untouched

### UC8: Clear cache for a dataset

**Actor**: User suspects corrupted embeddings or wants to force recomputation

1. User goes to Datasets → "Austria 2024" → "Cache"
2. Sees cache breakdown:
   - Face embeddings: 428 images (12.4 MB)
   - Face detections: 428 images (3.2 MB)
   - IQA scores: 428 images (0.1 MB)
   - Scene embeddings: 428 images (8.7 MB)
3. User can clear individual cache types or all at once
4. Next run will recompute cleared features

### UC9: Rename an album (run output)

**Actor**: User wants to name a specific run's output

1. User goes to Run History → Run #3
2. Clicks "Rename Album" → enters "Austria Best Selection"
3. Album name updates, visible in Export and Results pages

### UC10: Export from a specific run

**Actor**: User wants to export selected photos from Run #1 (not the latest)

1. User goes to Export page
2. Dataset selector: "Austria 2024"
3. Run selector: dropdown showing all completed runs with date + summary
4. Selects Run #1 (128 selected, 2026-04-30)
5. Configures export options → exports

---

## Cache Architecture

### What's cached (per-dataset, reusable across runs)

| Cache Type | Key | Recompute trigger |
|-----------|-----|-------------------|
| Face detections | `(image_path, model_name)` | Model change, image removed |
| Face embeddings | `(image_path, face_index, model_name)` | Model change, detection rerun |
| IQA scores | `(image_path, model_name)` | Model change |
| AVA scores | `(image_path, model_name)` | Model change |
| Scene embeddings | `(image_path, model_name)` | Model change |
| Person detection | `(image_path, model_name)` | Model change |

### What's NOT cached (per-run, always recomputed)

| Computation | Why not cached |
|------------|----------------|
| Quality filtering | Depends on config thresholds |
| Scene clustering | Depends on algorithm + params |
| Face clustering | Depends on K, distance_threshold, merge params |
| Selection | Depends on max_per_cluster, min_score, siamese |
| People assignment | Depends on clustering output |

### Cache sharing rule

Two runs on the same dataset share cache IF they use the same models. Changing `embedding_backend` from `insightface` to `custom` invalidates face embedding cache.

---

## UI Changes Required

### Datasets Page (replaces current Albums page)

```
Datasets
├── [Add Dataset] button
├── Dataset list:
│   ├── Austria 2024 (428 images, 3 runs, created Apr 30)
│   │   ├── [Edit] [Cache] [Delete]
│   │   └── Runs: #1 (128 sel), #2 (failed), #3 (156 sel)
│   ├── Budapest 2025 (122 images, 1 run, created Apr 29)
│   └── ...
```

### Configure & Run Page changes

```
Configure & Run
├── Dataset selector (was "Album selector")
├── Run History for selected dataset (prominent)
│   ├── Columns: #, Date, Config summary, People, Selected, Status
│   ├── Actions: [View] [Load Config] [Rename] [Delete]
│   └── [Compare Runs] button (future)
├── Config grid (unchanged)
├── [Run Pipeline] → creates new Run
```

### Results Page changes

```
Results
├── Dataset selector
├── Run selector (dropdown: "Run #3 — Apr 30, 156 selected" | "Run #1 — Apr 28, 128 selected")
├── Results for selected run
```

---

## Migration Path

### Phase 1: Rename in UI only (no DB changes)
- Rename "Albums" → "Datasets" in sidebar and pages
- Rename album selector label to "Dataset"
- Add run selector dropdown where needed
- No backend changes

### Phase 2: Separate Dataset from Run output
- Add `Album` table (output naming, independent of run)
- Allow renaming run outputs
- Show run history prominently

### Phase 3: Dataset editing
- Add `excluded_images` field to Dataset
- UI for image inclusion/exclusion
- Add images from additional sources

### Phase 4: Cache management UI
- Cache size display per dataset
- Clear cache per feature type
- Cache sharing indicators

---

## Open Questions

1. **Should "Album" be the output name, or should we use "Selection"?** "Album" is familiar (Google Photos). "Selection" is more precise.
2. **Should deleted runs keep their export artifacts on disk?** Currently exports go to `results/{album}/face_clustering_{ts}/`. Probably should clean up.
3. **Should cache be global or per-dataset?** Currently `UniversalCache` is global (keyed by image_path). Making it per-dataset adds isolation but duplicates if the same image is in multiple datasets.
4. **Should we support dataset versioning?** When images are added/removed, should old "versions" of the dataset be preserved for reproducibility?
