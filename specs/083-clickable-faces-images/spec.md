# spec-083 — Clickable faces/images (real buttons) + useful Images + pass-filter boxes

**Status:** Implemented
**Date:** 2026-06-06

## Problem (user, verbatim)
> "Images tab is still pretty useless. I can't get any useful information about
> the image. Also I want to be able to analyze images. show all face bounding
> boxes that pass our filtration. Also I still can't click on faces in Face
> Metrics and get face analysis."

### Root cause of "can't click" (reproduced, live browser)
`st.dataframe` row-select is the only drill-in, and **only the ~20px checkbox
column fires it** — clicking the face thumbnail or any data cell does nothing
(glide-data-grid treats it as text-select). Probe result:
- click data cell → stays on Face Metrics (no nav).
- click checkbox col → navigates to Face Analysis.

So the click "works" only on an invisible target. The reliable, intuitive
pattern already in the codebase is `face_grid.py`: a real `st.button("Open")`
under each thumbnail (used by Cluster Analysis, e2e-green).

## Fixes
1. **Face Metrics — clickable.** Add a **Grid** layout (default): each face =
   thumbnail + metric caption (disposition · blur · area%) + **Open** button →
   Face Analysis (real `st.button`, not a canvas checkbox). The sortable
   dataframe stays behind a "Table" toggle for power users.
2. **Images — useful + clickable.** Enrich `ImageRow` with **`n_passed`**
   (faces whose `rejection_reason` is NULL = passed filtration), computed in one
   grouped query. Images **Grid** (default): thumbnail + "N faces · M passed ·
   W×H" + **Open** → Image Analysis. Master-detail *within* the Images tab
   (no Image-Analysis nav page exists): selecting sets `selected_image_path`;
   the tab shows the analysis + a "← Back to images" button.
3. **Image Analysis — boxes that pass filtration.** Default the overlay to draw
   only **passing** faces' boxes (clustered + noise; hides red `filtered`), with
   a checkbox "Show rejected (filtered) faces too" to reveal them. Legend +
   per-image identity list (which clusters appear) added.

## Data
- `sim_bench/run_db/store.py` — `ImageRow.n_passed: int`; `list_images()` left
  outer query counts faces with `rejection_reason IS NULL` per image.
- `face_cluster/views/image_metrics.py` — new `ColumnSpec("n_passed", "Passed")`.

## Out of scope
- Per-image identity column in the *list* (needs the cluster-assignment join;
  shown in the analysis instead).
- Replacing the dataframe everywhere (kept as opt-in Table view).

## Acceptance
- Face Metrics Grid: clicking a face's Open → Face Analysis for that id. ✓ AppTest.
- Images Grid: clicking Open → in-tab Image Analysis for that path. ✓ AppTest.
- `ImageRow.n_passed` correct vs a hand-count on the reference run. ✓ unit.
- Image Analysis hides filtered boxes by default; toggle reveals them. ✓ unit.
- Budapest baseline gate (Gallery / Face Analysis / Merged) green.
- Real-browser screenshots of both grids + an image analysis with boxes.
