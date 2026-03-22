# SIGHTING-006 Debug Tools

Systematic debugging tools for the face crop embedding offset issue.

## Problem

Stored embeddings show a systematic +2 offset: `stored[N]` matches `fresh[N+2]` instead of `fresh[N]`.

## Structure

```
debug_sighting_006/
├── README.md                                 # This file
├── run_debug.py                              # Main entry point (systematic hypothesis testing)
├── debugger.py                               # Main debugger class
├── hypothesis_tests.py                       # All 6 hypothesis test implementations
├── helpers.py                                # Utility functions
├── reporting.py                              # Report generation
├── models.py                                 # Data classes (TestResult, Verdict)
├── trace_crop_save_logic.py                  # Simulates save logic to find skipped faces
├── verify_crop_metadata_consistency.py       # Checks alignment between crops/metadata/embeddings
└── debug_embeddings_comparison.ipynb         # Interactive exploration (original)
```

## Usage

### Automated Testing (Recommended)

Run all hypothesis tests systematically:

```bash
python -m scripts.debug_sighting_006.run_debug \
    --crops results/Google_Germany/face_crops \
    --stored-embeddings "results/Google_Germany/embeddings_2026-*.npy" \
    --fresh-embeddings "results/Google_Germany/embeddings_FRESH_*.npy" \
    --stored-metadata "results/Google_Germany/embeddings_metadata_2026-*.json" \
    --fresh-metadata "results/Google_Germany/embeddings_metadata_FRESH_*.json"
```

### Interactive Exploration

Use the Jupyter notebook for manual investigation:

```bash
jupyter notebook scripts/debug_sighting_006/debug_embeddings_comparison.ipynb
```

## Hypothesis Tests

The automated script tests 6 hypotheses:

1. **H1: Gap in crop files** - Check if face_0000, face_0001 are missing
2. **H2: String vs numeric sorting** - Check if file iteration order is wrong
3. **H3: Metadata index mismatch** - Check if metadata array indices don't match filenames
4. **H4: Extraction order mismatch** - Check if embeddings extracted in wrong order
5. **H5: Staleness check** - Check if stored embeddings are from different crop set
6. **H6: Offset pattern verification** - Verify +2 offset is consistent across samples

Each test returns:
- **Verdict**: PASS / FAIL / SUSPICIOUS / INFO
- **Evidence**: Concrete data supporting the verdict
- **Conclusion**: What the test found
- **Recommendation**: What to do next (if applicable)

## Output Example

```
================================================================================
Test 1: H1: First 2 crop files don't exist (face_0000, face_0001 missing)
================================================================================
Verdict: ❌ FAIL

Conclusion: face_0000 and face_0001 are MISSING. This explains the +2 offset.

Evidence:
  Total crop files: 725
  Actual first face_id: 2
  Missing face_ids: [0, 1]

💡 Recommendation: Check filtering logic in crop generation.

================================================================================
RECOMMENDED NEXT STEPS
================================================================================

ROOT CAUSE IDENTIFIED:
  First 2 faces were filtered out but metadata wasn't adjusted.

IMMEDIATE FIX:
  1. Re-run crop generation with face_id → filename mapping
  2. Use face_id from metadata for filenames, NOT loop counter
```

## Adding New Tests

To add a new hypothesis test:

1. Add method to `HypothesisTests` class in `hypothesis_tests.py`:
   ```python
   def test_h7_your_hypothesis(self) -> TestResult:
       """H7: Brief description of what you're testing."""
       # Your test logic here
       evidence = {...}
       verdict = Verdict.PASS  # or FAIL, SUSPICIOUS, INFO
       conclusion = "What you found"
       recommendation = "What to do about it"

       return TestResult(
           hypothesis="H7: Your hypothesis",
           verdict=verdict,
           evidence=evidence,
           conclusion=conclusion,
           recommendation=recommendation
       )
   ```

2. Add to test sequence in `debugger.py`:
   ```python
   def run_all_tests(self):
       results.append(self.tests.test_h7_your_hypothesis())
   ```

## Additional Debug Tools

### Trace Crop Save Logic

Simulates `benchmark_face_clustering.py::save_face_crops()` to identify which faces were skipped:

```bash
python scripts/debug_sighting_006/trace_crop_save_logic.py \
    --metadata results/Google_Germany/embeddings_metadata_FRESH_*.json \
    --crops results/Google_Germany/face_crops
```

Shows:
- Which metadata entries failed validation (invalid bbox, missing landmarks)
- Mapping: metadata_index → filename_index → expected file
- Whether systematic offset exists

### Verify Consistency

Checks alignment between crops, metadata, and embeddings:

```bash
python scripts/debug_sighting_006/verify_crop_metadata_consistency.py \
    --crops results/Google_Germany/face_crops \
    --metadata results/Google_Germany/embeddings_metadata_FRESH_*.json \
    --embeddings results/Google_Germany/embeddings_FRESH_*.npy
```

Checks:
1. Are crop filenames sequential (0, 1, 2, ...)?
2. Do metadata face_ids match crop filenames?
3. Does embeddings array length match crop count?

## ROOT CAUSE IDENTIFIED

**Location**: `scripts/benchmark_face_clustering.py:340`

**Bug**: Uses `saved_count` (incremental counter) for filenames instead of `face_id` from metadata

**Result**: If first N faces fail validation → all subsequent faces saved with -N offset

**Fix**: Use `face_meta['face_id']` for filenames, not `saved_count`

See `docs/SIGHTINGS.md` SIGHTING-006 for detailed analysis.

## Design Principles

- **One test per hypothesis** - Each method tests a single, specific hypothesis
- **Clear evidence** - Every verdict must be backed by concrete data
- **No debugging needed** - Tests are simple enough to trust their output
- **Actionable results** - Each failure includes a recommendation for next steps
