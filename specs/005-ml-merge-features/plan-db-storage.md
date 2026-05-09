# Plan: Merge Training Data — SQLite Storage + Dashboard

**Date**: 2026-04-16
**Extends**: spec 005 (ML Merge Features)
**Status**: Draft — pending approval

---

## Problem

Labeled merge decisions + feature vectors are scattered as `merge_features.parquet` inside each run's output directory. To train a classifier you must manually find, load, and concatenate files across runs. There's no visibility into how much training data exists or whether it's sufficient.

## Solution

1. **Central SQLite table** in `~/.sim_bench/sim_bench.db` for all labeled merge training samples
2. **Dashboard tab** in the face clustering app showing training data status

---

## Part 1: Database Table

### Table: `merge_training_data`

```sql
CREATE TABLE merge_training_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,    -- output folder name
    album_path      TEXT,               -- source image directory (traceability)
    cluster_a       INTEGER NOT NULL,
    cluster_b       INTEGER NOT NULL,
    label           INTEGER,            -- 1=approve, 0=reject, NULL=unlabeled
    feature_version INTEGER NOT NULL,   -- FeatureComputer.VERSION
    features_json   TEXT    NOT NULL,   -- full ClusterPairFeatures as JSON
    output_dir      TEXT,               -- full path to run output dir (for crop lookup)
    exemplar_ids    TEXT,               -- JSON: {"a": [12,45,7], "b": [3,88,21]}
    saved_at        TEXT    NOT NULL,   -- ISO-8601
    UNIQUE(run_id, cluster_a, cluster_b)
);
```

**Design choice — JSON blob vs. flat columns:**
- ~40 feature columns that will grow as new feature groups land (kNN graph, context, interactions)
- JSON avoids schema migrations when features are added
- For the dashboard and training, we load into DataFrame anyway: `pd.json_normalize()`
- SQLite `json_extract()` available if we ever need per-feature queries

### New module: `face_cluster/training_db.py`

Thin wrapper around the DB. No ORM — raw SQLite via `sqlite3` stdlib (the face_cluster package is standalone, doesn't depend on `sim_bench.api`).

```python
def get_db_path() -> Path:
    """~/.sim_bench/sim_bench.db"""

def init_training_table(db_path: Path = None) -> None:
    """CREATE TABLE IF NOT EXISTS merge_training_data ..."""

def upsert_training_samples(
    samples: List[Dict],   # each has run_id, cluster_a/b, label, features_json, ...
    db_path: Path = None,
) -> int:
    """INSERT OR REPLACE. Returns count of rows written."""

def load_training_data(db_path: Path = None) -> pd.DataFrame:
    """SELECT * → DataFrame with features_json expanded to columns."""

def training_data_summary(db_path: Path = None) -> Dict:
    """Quick stats: total_rows, n_runs, n_approved, n_rejected, n_unlabeled, feature_versions."""
```

### Integration with existing save flow

**`_save_merge_features_if_available()`** in `app/face_clustering.py`:
- Currently: writes `merge_features.parquet` to run output dir
- Change: also call `upsert_training_samples()` to write to central DB
- Keep parquet write as backup/export (backward compat)

---

## Part 2: Dashboard Tab

Add **"Training Data"** tab (9th tab) to `app/face_clustering.py`.

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Training Data                                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Total Samples]  [Approved]  [Rejected]  [Unlabeled]   │
│       142            87          41           14         │
│                                                         │
│  [Runs Contributing]  [Feature Version]  [Readiness]    │
│         8                  3              ⚠ Need 200+   │
│                                                         │
│  ── Per-Run Breakdown ──────────────────────────────     │
│  | run_id        | album   | approve | reject | total | │
│  | Germany_run_7 | Germany |    12   |   5    |  17   | │
│  | Germany_run_8 | Germany |    15   |   8    |  23   | │
│  | ...           |         |         |        |       | │
│                                                         │
│  ── Label Distribution ─────────────────────────────    │
│  [bar chart: approve vs reject per run]                 │
│                                                         │
│  ── Key Feature Distributions ──────────────────────    │
│  [histogram: min_exemplar_dist colored by label]        │
│  [histogram: post_merge_diameter colored by label]      │
│  [histogram: shared_source_images colored by label]     │
│                                                         │
│  ── Pair Drill-Down ────────────────────────────────    │
│  [Select a row from any table above to inspect]         │
│                                                         │
│  Cluster 5 ←→ Cluster 12   (run: Germany_run_7)        │
│  Label: ✓ Approved   |  Exemplar dist: 0.187           │
│                                                         │
│  Cluster 5 exemplars:      Cluster 12 exemplars:        │
│  [img] [img] [img]         [img] [img] [img]            │
│                                                         │
│  Key features:                                          │
│  | min_exemplar_dist | 0.187 |                          │
│  | shared_source_img |   0   |                          │
│  | post_merge_diam   | 0.42  |                          │
│  | gates passed      |  4/4  |                          │
│                                                         │
│  ── Export ──────────────────────────────────────────    │
│  [Download CSV]  [Download Parquet]                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Readiness indicator

| Condition | Status |
|-----------|--------|
| < 50 labeled samples | "Not enough data" |
| 50–200 samples, class ratio > 3:1 | "Imbalanced — need more of minority class" |
| 200+ samples, class ratio ≤ 3:1 | "Ready to train" |

### Pair drill-down

User selects a row from the per-run table (or clicks a point in a histogram). The drill-down panel shows:

1. **Exemplar thumbnails** for both clusters, loaded from `{output_dir}/crops/` using face IDs from `exemplar_ids` JSON. If the run directory no longer exists, show placeholder captions with face IDs.
2. **Key feature values** — the 5-6 most important features for this pair in a mini-table.
3. **Label** — approve/reject badge.

This lets users audit past decisions visually without switching to the Merge Analysis tab or reloading the original run.

### Key feature histograms

Show 3 features that are most informative for merge/reject separation:
- `min_exemplar_dist` — the primary gate metric
- `shared_source_images` — strong anti-merge signal
- `post_merge_diameter` — quality risk metric

Each histogram: two overlaid distributions (approve=green, reject=red) using plotly.

---

## Files Changed

| File | Change |
|------|--------|
| `face_cluster/training_db.py` | **New** — DB wrapper (init, upsert, load, summary) |
| `app/face_clustering.py` | Add Training Data tab; modify `_save_merge_features_if_available` to also upsert to DB |
| `sim_bench/api/database/models.py` | No change — training_db.py uses raw sqlite3, not ORM |
| `tests/face_clustering/test_training_db.py` | **New** — upsert/load round-trip, summary stats, schema migration |

## Dependencies

- `sqlite3` (stdlib) — no new packages
- Existing: `pandas`, `plotly` (already in project)

## Risks

| Risk | Mitigation |
|------|------------|
| DB schema drift when features change | `feature_version` column; `features_json` avoids column-level migrations |
| Concurrent writes from multiple app sessions | SQLite handles this with WAL mode; upserts are idempotent |
| Large DB size | ~1KB per row × 10K rows = 10MB. Not a concern |

## Not in scope

- Training script / classifier (separate feature)
- kNN graph features in `graph.py` (separate task — needs GraphResult persistence first)
- Automated retraining triggers
