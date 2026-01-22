# Album Organization MVP - Testing Guide

## Quick Start Testing

### Prerequisites
```bash
# Ensure dependencies installed
pip install streamlit mediapipe pillow pandas numpy opencv-contrib-python

# From project root
cd D:\sim-bench
```

### Run the App
```bash
streamlit run app/album/main.py
```

## Test Scenarios

### Test 1: Basic Workflow (5 minutes)

**Goal**: Verify end-to-end workflow with progress display

1. Open app in browser (http://localhost:8501)
2. Configure settings:
   - Leave defaults (preprocessing enabled)
   - Set min cluster size = 3
3. Enter album info:
   - Source: Path to folder with 20-50 photos
   - Album name: "Test Album"
   - Output: `test_output/`
4. Click "Start Workflow"
5. **Watch for**:
   - Progress bar updates smoothly
   - Status text shows stage names
   - Detail text shows current image name and operation
   - Stats show: Processed count, Rate (img/s), Elapsed time, ETA
   - All 4 metrics update during run

**Expected**: Workflow completes, progress visible throughout

---

### Test 2: Performance & Caching (10 minutes)

**Goal**: Verify thumbnail preprocessing speeds things up

#### First Run (Cold Cache)
1. Delete cache: `rm -rf .cache/album_analysis/`
2. Run workflow with 50 images
3. Note total time (e.g., 120 seconds)
4. **Watch for**:
   - "⚡ Generating thumbnails" stage appears
   - Thumbnails created in `.cache/album_analysis/medium/` and `.cache/album_analysis/large/`

#### Second Run (Warm Cache)
1. Run same workflow again (same source directory)
2. Note total time (should be significantly faster)
3. **Expected**:
   - Thumbnail stage completes almost instantly (cache hit)
   - Overall workflow 50-65% faster than first run

**Verification**:
```bash
# Check cache exists
ls -la .cache/album_analysis/medium/
ls -la .cache/album_analysis/large/

# Should see .jpg files with hash names
```

---

### Test 3: Telemetry & Performance Tab (5 minutes)

**Goal**: Verify timing data captured and displayed

1. Run workflow to completion
2. Check output directory for `telemetry_*.json`
3. Open file - should see:
   ```json
   {
     "run_id": "abc123de",
     "total_duration_sec": 87.5,
     "timings": [
       {
         "name": "discover_images",
         "duration_sec": 0.5,
         "count": 1,
         "avg_per_item": 0.5
       },
       {
         "name": "preprocess_thumbnails",
         "duration_sec": 45.2,
         "count": 50,
         "avg_per_item": 0.904
       },
       ...
     ]
   }
   ```
4. In app, go to "Performance" tab
5. **Expected**:
   - Total time displayed
   - Table shows timing breakdown
   - Bar chart visualizes operation times
   - Slowest operation highlighted

---

### Test 4: Configuration Options (5 minutes)

**Goal**: Verify settings work as expected

#### Test 4A: Disable Preprocessing
1. In config panel, **uncheck** "Enable Thumbnail Preprocessing"
2. Run workflow
3. **Expected**:
   - Workflow still works (falls back to full-res images)
   - Runs slower than with preprocessing
   - No thumbnails generated

#### Test 4B: Adjust Cluster Size
1. Set min cluster size = 1
2. Run workflow
3. **Expected**:
   - More clusters (including singletons)
   - Each single image treated as its own cluster

4. Set min cluster size = 8
5. Run workflow
6. **Expected**:
   - Fewer, larger clusters
   - Smaller groups merged into noise

---

### Test 5: Progress Detail Visibility (3 minutes)

**Goal**: Ensure users understand what's happening

1. Run workflow on 30+ images
2. During "Analyzing quality" stage, **watch detail text**:
   - Should show: `🔧 IQA Quality | 📄 photo_0023.jpg | ⏱️ 1.2s`
   - Then: `🔧 Portrait Detection | 📄 photo_0023.jpg | ⏱️ 0.8s`
   - Updates for each image
3. **Verify**:
   - Can see which specific image being processed
   - Can see which operation (IQA, Portrait, Features)
   - Can see how long each operation takes

---

### Test 6: Error Handling (Optional, 5 minutes)

**Goal**: Verify graceful error handling

1. Enter non-existent directory as source
2. Click "Start Workflow"
3. **Expected**: Clear error message, no crash

4. Run workflow with empty directory
5. **Expected**: Warning about no images found

---

## Performance Benchmarks

### Expected Timings (50 images, ~4MB each)

| Stage | First Run (Cold) | Second Run (Warm) |
|-------|-----------------|-------------------|
| Discover | 0.5s | 0.5s |
| Preprocess | 45s | 2s (cache hit) |
| Analyze | 120s | 65s (thumbnails) |
| Cluster | 8s | 8s |
| Select | 1s | 1s |
| Export | 3s | 3s |
| **Total** | **177.5s** | **79.5s** |
| **Speedup** | - | **55% faster** |

*Actual times vary based on hardware and image sizes*

---

## Verification Checklist

After running all tests, verify:

- [ ] Progress shows current image name during processing
- [ ] Stats display: Processed, Rate, Elapsed, ETA
- [ ] Thumbnails cached in `.cache/album_analysis/`
- [ ] Second run significantly faster (cache hit)
- [ ] Telemetry JSON file created in output directory
- [ ] Performance tab displays timing breakdown
- [ ] Can adjust min_cluster_size in UI
- [ ] Can enable/disable preprocessing
- [ ] Singleton clusters handled correctly (when min_size=1)
- [ ] No crashes or errors during normal operation

---

## Common Issues

### Issue: "No module named 'streamlit'"
**Solution**: `pip install streamlit`

### Issue: "No module named 'mediapipe'"
**Solution**: `pip install mediapipe`

### Issue: Progress doesn't update
**Solution**: Check browser console for errors, refresh page

### Issue: Very slow on first run
**Expected**: First run generates thumbnails, second run much faster

### Issue: Thumbnails not cached
**Check**: 
- `.cache/album_analysis/` directory exists and writable
- Config has `cache_thumbnails: true`

---

## Success Criteria

✅ **MVP is successful if**:
1. Progress clearly shows what's happening
2. Second runs 40%+ faster than first
3. Telemetry data captured and visible
4. Users can adjust settings and see impact
5. No crashes during normal operation

---

## Next Steps After Testing

1. Document any issues found
2. Measure actual performance gains
3. Gather user feedback
4. Consider Phase 2 features:
   - Database layer
   - Interactive editing
   - Selection overrides
