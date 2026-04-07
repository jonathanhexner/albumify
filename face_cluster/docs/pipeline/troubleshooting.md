# Face Clustering Benchmark - Troubleshooting Guide

## Common Errors and Solutions

### Error: "Failed to load results: Expecting value: line X column Y"

**Symptom**: Streamlit app shows JSON parse error when loading benchmark results

**Cause**: The JSON file is corrupted or incomplete, usually because:
1. Benchmark script crashed while writing results
2. Script was interrupted (Ctrl+C)
3. Data serialization error occurred mid-write

**Solution**:

1. **Check JSON file validity**:
   ```bash
   python scripts/check_json_validity.py results/face_clustering_benchmark/benchmark_*.json
   ```

2. **Delete corrupted file**:
   ```bash
   # Windows PowerShell
   Remove-Item "results\face_clustering_benchmark\benchmark_*.json"
   
   # Or manually delete in File Explorer
   ```

3. **Re-run benchmark**:
   ```bash
   .venv\Scripts\python.exe scripts/benchmark_face_clustering.py --album-path "YOUR_PATH"
   ```

**Prevention**: The benchmark script now uses atomic writes (temp file → rename) and validates JSON before saving.

---

### Error: "Object of type int32 is not JSON serializable"

**Symptom**: Benchmark crashes during "Saving results" with TypeError

**Cause**: Numpy data types (int32, float64, ndarray) cannot be directly serialized to JSON

**Solution**: This is now fixed with automatic type conversion. If you still see this:
1. Make sure you're using the latest benchmark script
2. Check if there are custom data types in face metadata

**Fixed in**: `scripts/benchmark_face_clustering.py` with `convert_numpy_types()` function

---

### Error: "ValueError: Coordinate 'lower' is less than 'upper'"

**Symptom**: Benchmark fails while saving face crops

**Cause**: Invalid bounding box coordinates (negative padding, edge cases)

**Solution**: Now handled gracefully:
- Invalid crops are skipped with debug logging
- Benchmark continues with other faces
- Check logs to see which faces were skipped

**Fixed in**: `save_face_crops()` function with validation

---

### Error: "No benchmark results found"

**Symptom**: Streamlit app sidebar shows "No benchmark results found"

**Cause**: No JSON files in results directory

**Solution**:
1. Run the benchmark first:
   ```bash
   python scripts/benchmark_face_clustering.py --album-path "YOUR_ALBUM"
   ```

2. Check results directory exists:
   ```bash
   dir results\face_clustering_benchmark
   ```

---

### Benchmark Running Very Slowly

**Symptoms**: 
- Takes >10 minutes for small album
- CPU usage is low
- Stuck on embedding extraction

**Solutions**:

1. **Use cached embeddings** (2nd run is much faster):
   - First run: Extracts embeddings (~5-10 min for 250 faces)
   - Subsequent runs: Uses cache (~30 seconds)

2. **Enable GPU** (if available):
   ```yaml
   # In configs/clustering_benchmark.yaml
   pipeline:
     extract_face_embeddings:
       device: cuda  # Change from cpu
   ```

3. **Reduce album size** for testing:
   - Test on subset of photos first
   - Move photos to separate test directory

---

### Benchmark Crashes Mid-Run

**Symptom**: Script stops with no error or incomplete output

**Possible Causes**:
1. Out of memory (too many faces)
2. GPU out of memory (if using CUDA)
3. Python environment issues

**Solutions**:

1. **Check memory usage**:
   - Task Manager (Windows)
   - `htop` (Linux/Mac)

2. **Reduce batch size** (future enhancement)

3. **Use CPU instead of GPU**:
   ```yaml
   device: cpu
   ```

4. **Check logs**:
   ```bash
   # Check terminal output
   type c:\Users\YOUR_USER\.cursor\projects\d-sim-bench\terminals\*.txt
   ```

---

### Hybrid Method Shows No Merges

**Symptom**: Hybrid produces same or more clusters than HDBSCAN

**Cause**: Merge conditions not met (parameters too strict for this dataset)

**Solution**: Adjust parameters in `configs/clustering_benchmark.yaml`:

```yaml
hybrid_knn:
  params:
    # Make merging more permissive
    merge_distance_ceiling: 0.50      # Increase from 0.45
    merge_min_links: 1                # Decrease from 2
    knn_k: 7                          # More neighbors to consider
    
    # Make singleton attachment more permissive
    singleton_attach_threshold: 0.42  # Increase from 0.38
```

**How to verify**: Check merge details in Streamlit app

---

### Streamlit App Not Showing Images

**Symptom**: Cluster view shows placeholders instead of face crops

**Cause**: Face crops directory missing or empty

**Solution**:

1. **Check if crops were saved**:
   ```bash
   dir results\face_clustering_benchmark\face_crops
   ```

2. **Re-run with crop saving enabled**:
   ```yaml
   # In configs/clustering_benchmark.yaml
   output:
     save_face_crops: true
   ```

3. **Check crop errors in logs**:
   - Look for "Invalid bbox" or "Invalid crop coordinates" warnings

---

## Logging and Debugging

### View Detailed Logs

The benchmark script now includes comprehensive logging at every step:

```bash
# Run with full output visible
.venv\Scripts\python.exe scripts/benchmark_face_clustering.py --album-path "YOUR_PATH" 2>&1 | tee benchmark.log
```

Key log messages to look for:
- `✓ Collected N face embeddings` - Pipeline succeeded
- `✓ Saved M/N face crops` - Some crops may have failed
- `✓ JSON validation successful` - Results saved correctly
- `✓ Results saved to: ...` - Benchmark complete

### Check JSON Validity

Use the diagnostic script:

```bash
python scripts/check_json_validity.py results/face_clustering_benchmark/benchmark_*.json
```

Output includes:
- File size
- JSON validity
- Data structure
- Error context (if invalid)
- Truncation detection

### Debug Specific Step

To debug a specific step, run the pipeline manually:

```python
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry
from pathlib import Path
import sim_bench.pipeline.steps.all_steps

# Setup
context = PipelineContext()
context.source_directory = Path("YOUR_ALBUM_PATH")
registry = get_registry()
executor = PipelineExecutor(registry)

# Run specific steps
steps = ['discover_images', 'insightface_detect_faces', 'filter_faces']
result = executor.execute(context, steps)

# Inspect results
print(f"Faces detected: {len(context.insightface_faces)}")
```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check logs first**: Look for ERROR or WARNING messages
2. **Validate JSON**: Use `check_json_validity.py` 
3. **Check file sizes**: Ensure album contains images
4. **Test with small album**: Try 10-20 photos first
5. **Review config**: Ensure paths and parameters are correct

## Prevention Checklist

Before running benchmark:
- [ ] Album path exists and contains images
- [ ] Virtual environment is activated
- [ ] Config file has correct parameters
- [ ] Output directory is writable
- [ ] Previous corrupted files are deleted

After benchmark completes:
- [ ] Check JSON validity with diagnostic script
- [ ] Verify face crops were saved
- [ ] Review log output for warnings
- [ ] Test Streamlit app loads correctly

## Quick Fixes

```bash
# Clean up corrupted results
Remove-Item "results\face_clustering_benchmark\*.json"

# Re-run benchmark (uses cached embeddings if available)
.venv\Scripts\python.exe scripts/benchmark_face_clustering.py --album-path "D:\Budapest2025_Google"

# Check results
python scripts/check_json_validity.py results/face_clustering_benchmark/benchmark_*.json

# View in app
streamlit run app/face_clustering_comparison.py
```
