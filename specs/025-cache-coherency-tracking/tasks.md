# spec-025 Tasks

## Part A: Face Distance Calculator
- [x] A1: API endpoint GET /results/{id}/face-embedding
- [x] A2: API endpoint GET /results/{id}/face-distance (cosine distance + verdict)
- [x] A3: "Face Distance" tab in Explore page with image/face selectors
- [x] A4: Color-coded verdict display (green/yellow/red)

## Part B: Face Provenance in Popup
- [x] B1: Show cache key + bbox coords per face in popup Faces tab
- [ ] B2: Show face crop thumbnail per face (from bbox, in popup)
- [ ] B3: Show cluster assignment per face (person name + cluster ID)

## Part C: Embedding Verification (re-extract and compare)
- [ ] C1: "Verify Embeddings" button — samples N faces, for each: open source image → crop face by stored bbox → run InsightFace embedding model → compare fresh embedding to cached embedding (cosine similarity). Requires ~2s model load + ~50ms/face.
- [ ] C2: Display results table: face_key, similarity, verdict (>0.99 = valid, <0.99 = mismatch)
- [ ] C3: Run in background thread (model load is slow, don't block UI)

## Part D: Cluster Sanity Check (nearest neighbor cross-cluster)
- [ ] D1: For each face, find nearest neighbor in embedding space. Flag cases where nearest neighbor is in a DIFFERENT cluster (potential clustering error or embedding mismatch).
- [ ] D2: Display suspicious faces: "Face 42 (Person 3) → nearest: Face 89 (Person 7), distance 0.18"
- [ ] D3: Could run automatically on Face Clustering tab or on-demand

## Verification
- [x] V1: Distance API returns correct values (0.9816 for different people)
- [ ] V2: Test with two faces of same person (distance < 0.3)
- [ ] V3: Nearest neighbors of a correctly clustered face are visually the same person
