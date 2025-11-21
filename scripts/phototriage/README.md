# PhotoTriage Attribute-Based Contrastive Learning Pipeline

Complete pipeline for extracting quality attributes from PhotoTriage user feedback and preparing training data for contrastive models.

## Overview

This pipeline transforms **34,827 pairwise photo comparisons** with free-form text reasons into a **structured attribute-labeled dataset** suitable for training contrastive models.

**What it does:**
1. Extracts user reasons from 4,986 review JSON files
2. Analyzes reason text patterns
3. Maps reasons to 13 quality attributes using NLU
4. Integrates ground truth rankings
5. Creates train/val/test splits (80/10/10 by series)

**Output:** ~37K attribute-labeled image pairs ready for training

---

## Pipeline Scripts

### 1. Extract Reasons (`01_extract_reasons.py`)

**Purpose**: Parse review JSONs and extract pairwise comparisons

**Input**:
- `D:/Similar Images/.../reviews_trainval/*.json` (4,986 files)

**Output**:
- `data/phototriage/raw_comparisons.jsonl` - All comparisons
- `data/phototriage/raw_comparisons.csv` - Same data as CSV
- `data/phototriage/extraction_stats.json` - Statistics

**Usage**:
```bash
python scripts/phototriage/01_extract_reasons.py
```

**What it extracts**:
```json
{
  "series_id": "000001",
  "compare_id_1": 0,
  "compare_id_2": 1,
  "compare_file_1": "1-1.JPG",
  "compare_file_2": "1-2.JPG",
  "user_choice": "RIGHT",
  "reason_text": "too narrow view; doesn't show enough background",
  "review_file": "000001.json",
  "review_index": 2
}
```

---

### 2. Analyze Reasons (`02_analyze_reasons.py`)

**Purpose**: Analyze reason text patterns to understand user feedback

**Input**:
- `data/phototriage/raw_comparisons.jsonl`

**Output**:
- `data/phototriage/reason_analysis.json` - Comprehensive analysis

**Usage**:
```bash
python scripts/phototriage/02_analyze_reasons.py
```

**What it analyzes**:
- **Keywords**: Most common words (blur, dark, narrow, etc.)
- **Bigrams/Trigrams**: Common phrases ("too dark", "can't see", etc.)
- **Categories**: Groups reasons by theme (focus, composition, exposure, etc.)
- **Negations**: Patterns like "too X", "not X", "can't X"
- **Length statistics**: Character and word count distributions

**Example output**:
```
Top Keywords:
  1. blur              2745
  2. dark              2423
  3. narrow            1234
  ...

Top Bigrams:
  1. too dark          2423
  2. too narrow         899
  3. can't see          873
  ...
```

---

### 3. Map Attributes (`03_map_attributes.py`)

**Purpose**: Map reason texts to structured attribute labels

**Input**:
- `data/phototriage/raw_comparisons.jsonl`

**Output**:
- `data/phototriage/labeled_comparisons.jsonl` - Comparisons with attributes
- `data/phototriage/attribute_stats.json` - Attribute statistics

**Usage**:
```bash
python scripts/phototriage/03_map_attributes.py
```

**What it does**:
- Uses `AttributeMapper` class (NLU-based)
- Maps each reason to 0-N attributes from schema (13 attributes)
- Determines winner for each attribute (A or B)
- Assigns confidence score (0.0-1.0)
- Extracts reason snippet that triggered the attribute

**Example output**:
```json
{
  "series_id": "000001",
  "compare_id_1": 0,
  "compare_id_2": 1,
  "reason_text": "too narrow view; doesn't show enough background",
  "attributes": [
    {
      "name": "field_of_view",
      "winner": "B",
      "confidence": 0.95,
      "reason_snippet": "too narrow view",
      "category": "perspective"
    },
    {
      "name": "background_clutter",
      "winner": "B",
      "confidence": 0.85,
      "reason_snippet": "doesn't show enough background",
      "category": "composition"
    }
  ],
  "num_attributes": 2
}
```

---

### 4. Create Dataset (`04_create_attribute_dataset.py`)

**Purpose**: Create final training dataset with ground truth

**Input**:
- `data/phototriage/labeled_comparisons.jsonl`
- `D:/Similar Images/.../train_pairlist.txt` (ground truth rankings)
- `D:/Similar Images/.../train_val_imgs/` (image directory)

**Output**:
- `data/phototriage/dataset/train_pairs.jsonl` - Training split
- `data/phototriage/dataset/val_pairs.jsonl` - Validation split
- `data/phototriage/dataset/test_pairs.jsonl` - Test split
- `data/phototriage/dataset/dataset_info.json` - Dataset metadata

**Usage**:
```bash
python scripts/phototriage/04_create_attribute_dataset.py
```

**What it does**:
- Loads ground truth rankings (Bradley-Terry preferences)
- Maps comparisons to actual image file paths
- Splits by series_id (80% train, 10% val, 10% test)
- No data leakage (series never split across train/val/test)

**Final output format**:
```json
{
  "pair_id": "000001_0_1",
  "series_id": "000001",
  "image_a_id": "000001-01",
  "image_b_id": "000001-02",
  "image_a_path": "D:/.../000001-01.JPG",
  "image_b_path": "D:/.../000001-02.JPG",
  "chosen_image": "B",
  "reason_raw": "too narrow view",
  "preference_strength": 0.949,
  "rank_a": 1,
  "rank_b": 4,
  "attributes": [
    {
      "name": "field_of_view",
      "winner": "B",
      "confidence": 0.95,
      "reason_snippet": "too narrow view",
      "category": "perspective"
    }
  ],
  "metadata": {
    "review_file": "000001.json",
    "review_index": 2,
    "num_attributes": 1
  }
}
```

---

## Attribute Schema

### 13 Quality Attributes (6 Categories)

#### 1. Focus & Clarity
- `sharpness` - Overall image sharpness and focus quality
- `detail_visibility` - Ability to see important details
- `motion_blur` - Presence/absence of motion blur

#### 2. Composition & Framing
- `framing` - Quality of image framing and boundaries
- `cropping_completeness` - Whether subject is cut off or complete
- `subject_placement` - Positioning of main subject in frame
- `background_clutter` - Amount of distracting background elements

#### 3. Exposure & Lighting
- `exposure_quality` - Overall brightness appropriateness
- `lighting_quality` - Quality and direction of lighting
- `dynamic_range` - Preservation of detail in highlights/shadows

#### 4. Perspective & Field of View
- `field_of_view` - Width of view (narrow vs. wide)
- `distance_appropriateness` - Subject distance suitability

#### 5. Content & Interest
- `subject_interest` - How interesting/engaging the subject is

---

## Full Pipeline Execution

Run all scripts in order:

```bash
# 1. Extract reasons from reviews
python scripts/phototriage/01_extract_reasons.py

# 2. Analyze reason patterns
python scripts/phototriage/02_analyze_reasons.py

# 3. Map to attributes
python scripts/phototriage/03_map_attributes.py

# 4. Create final dataset
python scripts/phototriage/04_create_attribute_dataset.py
```

**Total time**: ~5-10 minutes (depending on system)

---

## Output Directory Structure

```
data/phototriage/
├── raw_comparisons.jsonl          # Step 1 output
├── raw_comparisons.csv
├── extraction_stats.json
├── reason_analysis.json           # Step 2 output
├── labeled_comparisons.jsonl      # Step 3 output
├── attribute_stats.json
└── dataset/                       # Step 4 output
    ├── train_pairs.jsonl
    ├── val_pairs.jsonl
    ├── test_pairs.jsonl
    └── dataset_info.json
```

---

## Expected Results

Based on PhotoTriage dataset statistics:

| Metric | Expected Value |
|--------|---------------|
| Total comparisons | ~37,000 |
| Unique series | 4,986 |
| Comparisons with attributes | ~32,000 (85-90%) |
| Avg attributes/comparison | ~1.5 |
| Total attribute labels | ~48,000-55,000 |
| Train pairs | ~29,600 (80%) |
| Val pairs | ~3,700 (10%) |
| Test pairs | ~3,700 (10%) |

### Attribute Distribution (Estimated)

| Attribute | Expected Count | % of Comparisons |
|-----------|---------------|-----------------|
| sharpness | 8,000-10,000 | 22-25% |
| field_of_view | 6,000-7,000 | 16-18% |
| exposure_quality | 5,000-6,000 | 14-16% |
| cropping_completeness | 4,000-5,000 | 11-13% |
| detail_visibility | 3,500-4,500 | 10-12% |
| background_clutter | 3,000-4,000 | 8-10% |
| (others) | 1,000-3,000 each | 3-8% each |

---

## Troubleshooting

### Error: "Reviews directory not found"
**Solution**: Update the path in `01_extract_reasons.py` line 194:
```python
reviews_dir = Path("YOUR_PATH_HERE/reviews_trainval/reviews_trainval")
```

### Error: "Ground truth file not found"
**Solution**: Update paths in `04_create_attribute_dataset.py` lines 361-362:
```python
image_dir = Path("YOUR_PATH_HERE/train_val_imgs")
ground_truth = Path("YOUR_PATH_HERE/train_pairlist.txt")
```

### Warning: "No attributes extracted"
**Possible causes**:
1. Reason text doesn't match any attribute keywords
2. Reason is too vague ("it's better")
3. Reason mentions specific objects, not quality attributes

**Expected**: ~10-15% of comparisons will have no attributes

### Low attribute counts
**Check**:
- Run `02_analyze_reasons.py` to see keyword distribution
- Verify that common keywords are in `attribute_mapper.py` patterns
- Consider adding domain-specific keywords

---

## Next Steps

After running this pipeline, you can:

1. **Train contrastive model**: Use scripts 05-06 (to be created)
2. **Analyze attribute patterns**: Use the generated statistics files
3. **Refine attribute schema**: Based on analysis, merge/split/adjust attributes
4. **Add more attributes**: Edit `attribute_mapper.py` to add new patterns

---

## Implementation Details

### AttributeMapper Logic

**Keyword Matching**:
- Uses regex with word boundaries (`\bkeyword\b`)
- Case-insensitive matching
- Supports multi-word phrases

**Winner Determination**:
1. If **negation** found ("too dark", "blurry"):
   - Reason describes the **losing** image
   - Winner is the **preferred** image (user_choice)
   - High confidence (0.9)

2. If **positive** mention ("sharp", "clear"):
   - Reason describes the **winning** image
   - Winner is the **preferred** image
   - Medium-high confidence (0.85)

3. If **ambiguous** (just keyword, no negation):
   - Assume describes issue with **losing** image
   - Winner is **preferred** image
   - Lower confidence (0.7)

### Ground Truth Integration

- Loads Bradley-Terry preference ratios (0.0-1.0)
- Loads absolute rankings (1=best in series)
- Handles 1-based to 0-based index conversion
- Provides fallback for missing pairs

### Data Split Strategy

- Splits by `series_id` (not by individual pairs)
- Prevents data leakage (same burst never in train + test)
- Maintains attribute distribution across splits
- Reproducible (fixed random seed)

---

## Design Rationale

### Why 13 attributes?
- Covers all major quality dimensions
- Based on analysis of 34,827 user reasons
- Balances coverage vs. label sparsity
- Each attribute appears in ~5-25% of comparisons

### Why keyword matching (not LLM)?
- Fast and deterministic
- Interpretable (can debug patterns)
- No API costs
- Sufficient for PhotoTriage's structured reasons
- Can upgrade to LLM later if needed

### Why train/val/test split?
- Standard ML practice
- Prevents overfitting
- Enables fair model comparison
- Supports hyperparameter tuning (on val)

---

## References

- **PhotoTriage Dataset**: [Kong et al., 2016](http://vision.cs.utexas.edu/projects/rationales/)
- **Bradley-Terry Model**: Statistical model for pairwise preferences
- **Design Document**: `docs/photo_triage_attribute_contrastive_plan.md`

---

## License

Same as parent project (MIT License)
