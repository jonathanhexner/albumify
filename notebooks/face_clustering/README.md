# Face Clustering — Analysis Notebooks

Notebooks for analysing and improving the face clustering algorithm and merge logic.
Run them from inside this folder (or adjust the `../results/` paths accordingly).

---

## Notebooks

### `eda_merge_explore.ipynb` — Single-run merge sanity check

**Purpose**: Inspect one run in detail before running the ML notebook.

What it does:
- Loads a single result directory (`RUN_DIR`) and its `merge_log.json`
- Displays merged vs rejected pairs as a table
- Computes the feature matrix (`t_local`, `t_global`, `min_exemplar_dist`, …)
- Shows exemplar crop thumbnails side-by-side for each pair
- Plots exemplar distance vs cluster spread, coloured by decision

**When to use**: After a run produces unexpected merge behaviour. Set `RUN_DIR` at the top of cell 2 to the run you want to inspect.

**Guard**: Cell 2 raises `ValueError` immediately if the chosen run has zero merges — pick a run that actually merged something.

---

### `eda_merge_ml.ipynb` — Multi-run ML training & feature importance

**Purpose**: Answer "can a simple model beat the hand-tuned thresholds?"

What it does:
1. Discovers all eligible runs under `../results/` (requires `merge_log.json`, `faces.csv`, `embeddings.npy`)
2. **Skips runs with no merges** — only runs with at least one positive label are loaded
3. Optionally overrides labels with manual corrections from the training DB
4. Balances the dataset, trains LR / DT / RF with 5-fold cross-validation
5. Runs SHAP to identify the top predictive features
6. Compares RF F1 against the best single-threshold heuristic on `min_exemplar_dist`

**When to use**: After collecting several labelled runs, to evaluate whether an ML gate would improve precision/recall over the current rule-based merge logic.

---

## Typical workflow

```
1. Run the pipeline on a new album → produces results/<run>/
2. Open eda_merge_explore.ipynb, set RUN_DIR, run all cells
   → sanity-check the merge decisions visually
3. Once several runs are labelled, open eda_merge_ml.ipynb, run all cells
   → see feature importances and model vs heuristic comparison
```

## Dependencies

Both notebooks use the `face_cluster` package (installed in the project `.venv`).
Launch Jupyter from the project root:

```bash
.venv\Scripts\jupyter notebook notebooks/face_clustering/
```
