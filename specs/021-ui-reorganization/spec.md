# spec-021: UI Reorganization — Lifecycle-based Navigation

## Problem
The main album app has structural UI issues that make it unusable for real workflows:
- Pipeline config buried inside Results page (6 tabs crammed together)
- People and Faces are separate pages with unclear distinction
- Nested expanders cause page jumps (SIGHTING-032, 035)
- No per-step observability
- No image detail popup, no bounding boxes
- Face clustering analysis doesn't integrate

## Solution
Reorganize navigation to follow the workflow lifecycle. Replace tab-heavy pages with dedicated pages. Add per-step observability, image detail popup, and bounding box overlays.

## New Navigation
```
Home | Albums | Configure & Run | Results | People & Faces | Explore | Export
```

## Acceptance Criteria
1. All 7 sidebar pages render without error
2. Configure & Run: flat config grid (no expanders), profile load/save, run history table
3. Results: metrics + gallery with filter chips, deep-link to Face Clustering App
4. People & Faces: merged page with Named/Unnamed/Needs Help/Flagged sections
5. Explore: 6 tabs (Image Quality, Person Detection, Face Detection, Scene Clustering, Face Clustering, Selection)
6. Image detail popup: click any image to see all scores, bboxes, selection reason
7. Bounding box overlay on face displays
8. Export: standalone page
9. Playwright E2E tests for each page
10. No nested expanders anywhere in the app

## Edge Cases
- Empty album (no pipeline run): all pages show appropriate empty states
- Failed pipeline run: Results shows error, Explore shows partial data
- Album with no faces: People & Faces shows empty state, Explore face tabs show "no faces detected"
- Multiple runs: Run history shows all, Results shows latest by default

## Mock
See `docs/ui_redesign_mock_v2.html` for interactive mock.

## Sightings Addressed
032, 035, 038, 039, 040, 041, 042, 043, 044, 045, 046
