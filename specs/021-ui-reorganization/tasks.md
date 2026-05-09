# spec-021 Tasks

## Phase A: Navigation skeleton
- [ ] A1: Create stub pages (configure.py, explore.py, people_faces.py, export.py)
- [ ] A2: Update main.py page registry
- [ ] A3: Update sidebar.py navigation buttons
- [ ] A4: Fix cross-page navigation references (home.py, old pages)
- [ ] A5: Playwright test: all 7 sidebar buttons render pages without error

## Phase B: Configure & Run page
- [ ] B1: Move pipeline runner into configure.py
- [ ] B2: Flatten config UI (remove @st.fragment, remove all expanders, 3-column grid)
- [ ] B3: Merge params as full-width section (not expander)
- [ ] B4: Playwright test: config grid renders, profile works, sliders don't jump

## Phase C: Results simplification + Export
- [ ] C1: Strip results.py to viewing only (remove Run Pipeline, Export, Comparisons, Sub-Clusters tabs)
- [ ] C2: Create export.py page (reuse export_panel.py)
- [ ] C3: Playwright test: gallery works, filter chips work, export works

## Phase D: People & Faces merge
- [ ] D1: Create people_faces.py with Named/Unnamed/Needs Help/Flagged sections
- [ ] D2: Deprecate people.py and face_management.py
- [ ] D3: Playwright test: sections render, person detail expands

## Phase E: Image popup + bounding boxes
- [ ] E1: Create bbox_overlay.py utility
- [ ] E2: Create image_popup.py (@st.dialog pattern)
- [ ] E3: Wire popup into gallery.py and main.py
- [ ] E4: Add bboxes to People & Faces person detail
- [ ] E5: Playwright test: click image → popup shows, bboxes visible

## Phase F: Explore page (per-step observability)
- [ ] F1: Image Quality tab
- [ ] F2: Person Detection tab
- [ ] F3: Face Detection & Scoring tab
- [ ] F4: Scene Clustering tab
- [ ] F5: Face Clustering tab (overview + deep-link)
- [ ] F6: Selection tab (reasons + comparisons)
- [ ] F7: Fold debug.py content into explore tabs
- [ ] F8: Playwright test: each tab renders with data
