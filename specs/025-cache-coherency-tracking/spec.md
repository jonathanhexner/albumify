# spec-025: Face-Embedding Traceability & Verification

**Status**: In Progress
**Date**: 2026-05-03 (revised)

## Problem

There is no way to verify that a cached face embedding actually corresponds to the face it claims to represent. Past bugs (SIGHTING-003, 006) caused embeddings to be stored under wrong face indices. The user needs to:

1. **Trace**: Given any face in the app, see the full chain: source image → face detection bbox → face crop → cached embedding → cluster assignment
2. **Verify**: Pick two faces and confirm their embedding distance makes sense (same person = close, different person = far)
3. **Spot-check**: For any face, visually see the crop that was used to compute the embedding, alongside the embedding's nearest neighbors in the cache

## What This Is NOT

This is not about file modification timestamps. The existing mtime check in `cache_handler.py` handles that. This spec is about verifying the **semantic correctness** of the image → face → embedding → cluster chain.

## User Stories

1. As a user, I pick two faces from any images and see their cosine distance, to verify the clustering decision makes sense.
2. As a user, I click on any face and see: the exact crop used for embedding, the bbox on the source image, the cache key, and which cluster it was assigned to.
3. As a user, I can spot-check a face by seeing its nearest neighbors in embedding space — if the neighbors look like the same person, the embedding is correct.

## Design

### Part A: Face Distance Calculator (DONE)

New "Face Distance" tab in Explore page:
- Two image selectors with face index pickers
- Loads embeddings from UniversalCache
- Computes cosine distance
- Color-coded verdict: green (<0.3 same person), yellow (0.3-0.5 borderline), red (>0.5 different)
- API endpoints: `GET /results/{id}/face-embedding`, `GET /results/{id}/face-distance`

### Part B: Face Provenance in Image Popup

In the tabbed image detail popup, the Faces tab shows per-face:
- Face crop thumbnail (from bbox)
- Bounding box coordinates
- Cache key used for lookup
- Cluster assignment (which person)
- Per-face scores (pose, eyes, expression)

### Part C: Face Nearest Neighbors (NOT YET IMPLEMENTED)

For any selected face, show the top-5 nearest faces in embedding space:
- Load all face embeddings from cache for the album
- Compute distances from selected face to all others
- Show the 5 closest as crops with distance values
- User can visually confirm: do these look like the same person?

This directly catches the corruption scenario: if face_0's embedding is actually face_1's data, the nearest neighbors will look like face_1, not face_0.

## Acceptance Criteria

1. Face Distance tab: select two faces, see distance + verdict
2. Face provenance: click any face in popup, see crop + bbox + cache key + cluster
3. Nearest neighbors: select a face, see 5 closest faces with crops and distances

## Implementation Status

| Task | Status |
|------|--------|
| Face Distance API endpoints | Done |
| Face Distance tab in Explore | Done |
| Face provenance in popup (cache key + bbox) | Done |
| Nearest neighbors view | Not done |
