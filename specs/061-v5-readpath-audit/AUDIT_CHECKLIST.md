# V5 Read-Path Audit — Checklist

> Re-runnable script. After any schema change (`SCHEMA_VERSION` bump,
> new producer, layout change), walk these categories and append rows to
> `AUDIT_FINDINGS.md`. Time-box each category to 30 min; "0 hits in 30 min"
> = category complete.

Audit scope: **v2 read paths only.** That means the dependency closure of
`app/face_clustering_v2/` and `face_cluster/views/`, plus
`face_cluster/repositories/`, `sim_bench/db/face_clustering/`,
`sim_bench/run_db/`, `face_cluster/loader.py`. Out of scope:
`app/face_clustering/` (legacy, retiring).

Each category: run the grep, walk the hits, append a row to
`AUDIT_FINDINGS.md` with **v5-compat?** and **Status** (`NO-OP` /
`FIX` / `SIGHTING` / `DEFERRED`).

---

## Category 1 — Hardcoded legacy CSV file references

```
grep -rn "faces\.csv\|clusters\.csv" \
    app/face_clustering_v2/ \
    face_cluster/views/ \
    face_cluster/loader.py \
    face_cluster/repositories/ \
    sim_bench/db/face_clustering/ \
    sim_bench/run_db/
```

Skip rows that describe the legacy layout in an error message (those are
correct after SIGHTING-080). Flag rows that *check for* or *parse* CSVs.

## Category 2 — `RunStore.<method>("final")` calls

```
grep -rn 'RunStore.*"final"\|\.clusters("final"\|\.iteration_count' \
    app/face_clustering_v2/ \
    face_cluster/views/ \
    face_cluster/repositories/ \
    sim_bench/db/face_clustering/ \
    face_cluster/loader.py
```

For each: confirm the caller doesn't depend on RunStore's broken `"final"`
resolver (queries `merge_decisions` instead of `clusters`). If it does,
resolve locally against the `clusters` table (per spec-045 fix).

## Category 3 — Raw `MAX(iteration) FROM merge_decisions`

```
grep -rn "merge_decisions\|MAX(iteration)" face_cluster/ sim_bench/ app/face_clustering_v2/
```

`MAX(iteration) FROM merge_decisions` is the SIGHTING-078 root pattern.
Anywhere it appears is suspect.

## Category 4 — `AsyncHandle` / `threading.Thread` in v2 UI

```
grep -rn "AsyncHandle\|threading\.Thread" app/face_clustering_v2/
```

SIGHTING-079: AsyncHandle in UI = stuck spinner. Streamlit doesn't poll
threads. Sync + `st.spinner` is the right shape for the UI layer.

## Category 5 — `_REQUIRED`, `_ARTIFACTS`, `_COLUMNS`, `_EXPECTED` constants

```
grep -rnE "_(REQUIRED|ARTIFACTS|COLUMNS|EXPECTED)[ :=]" \
    app/face_clustering_v2/ \
    face_cluster/views/ \
    face_cluster/repositories/ \
    sim_bench/db/face_clustering/
```

Skip UI-only constants (display columns, label maps). Flag anything
that claims a *schema* or *file shape*.

## Category 6 — Producer-name string equality checks

```
grep -rnE '== ?"(fc_app|albumify|fc_app_v2)"' \
    app/face_clustering_v2/ \
    face_cluster/views/
```

v2 producers include `fc_app_v2`. Anything checking only the old names
silently excludes v2 runs.

## Category 7 — Direct `sqlite3.connect` on per-run DB without `PRAGMA user_version` check

```
grep -rn "sqlite3\.connect" \
    app/face_clustering_v2/ \
    face_cluster/views/ \
    face_cluster/repositories/ \
    sim_bench/db/face_clustering/
```

Skip connections via `RunStore` (it validates schema_version). Flag
anything opening `face_clustering.db` directly without checking
`PRAGMA user_version == SCHEMA_VERSION`.

## Category 8 — `pipeline_run.json` field reads that assume v4 structure

```
grep -rn "pipeline_run\.json\|pipeline_run\[" \
    face_cluster/views/ \
    app/face_clustering_v2/
```

Confirm each read matches what the v5 `RunExporter._write_pipeline_run_json`
actually writes today (schema_version=5; specific keys).

## Category 9 — Hardcoded run-dir layout assumptions

```
grep -rnE '"_v4"|"crops"' \
    face_cluster/ \
    sim_bench/ \
    app/face_clustering_v2/
```

`crops/` is still v5-correct. `_v4/` is the transitional layout (rare in
practice). Flag any hardcoded assumption about other subdirs that v5 may
not have.

## Category 10 — `os.listdir` / `glob` on run dirs

```
grep -rn "listdir\|\.glob(" \
    face_cluster/views/ \
    app/face_clustering_v2/
```

Look for assumptions about file count or extension presence that the v5
layout breaks. (e.g., "this dir always has N .csv files".)

---

## When to re-run

- After any `SCHEMA_VERSION` bump in `face_cluster/db/schema.py` (or its
  spec-056 successor `sim_bench/run_db/_schema.py`)
- After a new producer tag is added (currently: `fc_app`, `albumify`,
  `fc_app_v2`)
- After any commit touching `RunStore`, `RunExporter`, `load_pipeline_result`,
  or per-run DB ORM models (spec-058)

Each re-run produces a fresh table in `AUDIT_FINDINGS.md` under a new
dated heading.
