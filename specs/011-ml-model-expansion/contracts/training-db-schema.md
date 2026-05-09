# Contract: Training DB Schema Extension

## Schema migration

```sql
ALTER TABLE merge_training_data ADD COLUMN source TEXT DEFAULT 'human';
```

Applied on first access via `init_training_table()` — same idempotent pattern as table creation. Uses `PRAGMA table_info` to check if column exists before altering.

## Updated _COLS

```python
_COLS = [
    "run_id", "album_path", "cluster_a", "cluster_b", "label",
    "feature_version", "features_json", "output_dir", "exemplar_ids",
    "saved_at", "source",
]
```

## Source values

| Value | Meaning | Written by |
|-------|---------|-----------|
| `"human"` | User explicitly approved/rejected | App merge analysis decisions |
| `"auto_negative"` | Harvested non-candidate pair | Harvest negatives button |

## Backwards compatibility

- Existing rows without `source` column get `DEFAULT 'human'` from the ALTER.
- `load_training_data()` returns `source` column if present; old DBs get `'human'` for all rows.
- `upsert_training_samples()` accepts `source` in the sample dict; defaults to `'human'` if absent.

## training_data_summary() extension

Returns breakdown by source in addition to existing metrics:

```python
{
    "total": 500,
    "positive": 200,
    "negative": 300,
    "by_source": {"human": 250, "auto_negative": 250},
    ...
}
```
