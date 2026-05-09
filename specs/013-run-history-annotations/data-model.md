# Data Model: Run History & Run Annotations (013)

**Date**: 2026-04-23

---

## 1. Database: `action_log` table (extended)

Existing table in `~/.sim_bench/sim_bench.db`. New columns added via `ALTER TABLE … ADD COLUMN` (all NULL-default).

```sql
ALTER TABLE action_log ADD COLUMN source_album   TEXT;     -- original input image dir name
ALTER TABLE action_log ADD COLUMN run_name       TEXT;     -- human-readable label (separate from output dir)
ALTER TABLE action_log ADD COLUMN parent_run_id  TEXT;     -- run_id of the parent action row
ALTER TABLE action_log ADD COLUMN run_kind       TEXT;     -- 'base'|'recluster'|'remerge'|'manual_merge'
ALTER TABLE action_log ADD COLUMN comment        TEXT;     -- user free-text, max 2048 chars
ALTER TABLE action_log ADD COLUMN config_json    TEXT;     -- JSON snapshot of pipeline config at run time
ALTER TABLE action_log ADD COLUMN n_core         INTEGER;  -- faces that passed quality gating

CREATE INDEX IF NOT EXISTS idx_action_log_source_album
    ON action_log(source_album, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_log_comment
    ON action_log(comment);  -- supports LIKE search
```

### Full column inventory (after migration)

| Column | Type | Nullable | Notes |
|---|---|---|---|
| id | INTEGER PK | N | autoincrement |
| action_type | TEXT | N | existing |
| status | TEXT | N | existing: running/complete/failed |
| started_at | TEXT | N | ISO8601, existing |
| ended_at | TEXT | Y | existing |
| duration_s | REAL | Y | existing |
| error | TEXT | Y | existing |
| run_id | TEXT | Y | existing |
| source_dir | TEXT | Y | existing (full path) |
| output_dir | TEXT | Y | existing (full path) |
| album | TEXT | Y | existing — **deprecated**, use source_album |
| n_faces | INTEGER | Y | existing |
| n_clusters | INTEGER | Y | existing |
| n_noise | INTEGER | Y | existing |
| log_file | TEXT | Y | existing |
| payload_json | TEXT | Y | existing |
| **source_album** | TEXT | Y | **NEW** — source input dir name |
| **run_name** | TEXT | Y | **NEW** — human label |
| **parent_run_id** | TEXT | Y | **NEW** — FK to run_id |
| **run_kind** | TEXT | Y | **NEW** — enum |
| **comment** | TEXT | Y | **NEW** — max 2048 chars |
| **config_json** | TEXT | Y | **NEW** — JSON dict |
| **n_core** | INTEGER | Y | **NEW** — post-quality-gate face count |

---

## 2. Python Dataclasses

### `RunRow`
Typed projection of one `action_log` row for the History table and search helper.

```python
@dataclass
class RunRow:
    id: int
    source_album: str | None
    run_name: str | None
    output_dir: str | None
    run_kind: str | None          # 'base'|'recluster'|'remerge'|'manual_merge'
    parent_run_id: str | None
    started_at: str               # ISO8601
    n_faces: int | None
    n_core: int | None
    n_clusters: int | None
    comment: str | None
    config_json: str | None       # raw JSON string
    status: str
```

### `HistoryFilters`
Input container for `run_history.search()`.

```python
@dataclass
class HistoryFilters:
    source_album: str | None = None   # exact match filter
    date_from: str | None = None      # ISO8601 date, inclusive
    date_to: str | None = None        # ISO8601 date, inclusive
    text_query: str | None = None     # LIKE search on source_album, run_name, comment
    limit: int = 200
```

### `ConfigDelta`
One row in a config diff table.

```python
@dataclass
class ConfigDelta:
    field: str
    parent_value: object
    child_value: object
```

### `RunDirSpec`
Input container for `allocate_run_dir`.

```python
@dataclass
class RunDirSpec:
    source_album: str          # used as subdirectory name under results/
    kind: str                  # e.g. 'remerge', 'recluster'
    label: str | None = None   # optional human label stored as run_name
```

---

## 3. `pipeline_run.json` extensions

Additive fields — existing readers ignore unknown keys.

```json
{
  "source_album": "<str>",
  "run_name": "<str | null>",
  "comment": "<str | null>"
}
```

---

## 4. Validation Rules

| Field | Rule |
|---|---|
| `source_album` | Non-empty string if set; inherited from parent or from `session.json.source_album` |
| `run_kind` | Must be one of `{'base', 'recluster', 'remerge', 'manual_merge'}` |
| `comment` | Length ≤ 2 048 characters; enforced in Python write helper |
| `config_json` | Valid JSON dict or NULL; never a non-dict JSON value |
| `parent_run_id` | Must reference an existing `run_id` in `action_log` or be NULL |

---

## 5. Key Relationships

```
action_log row (base run)
  └── source_album = "Noa2_5"
  └── run_kind = "base"
  └── parent_run_id = NULL

action_log row (derived run)
  └── source_album = "Noa2_5"      ← inherited from parent
  └── run_kind = "remerge"
  └── parent_run_id = <base run_id>

action_log row (second derivative)
  └── source_album = "Noa2_5"      ← inherited
  └── run_kind = "remerge"
  └── parent_run_id = <first remerge run_id>
```
