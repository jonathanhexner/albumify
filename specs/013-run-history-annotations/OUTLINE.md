# Outline: Run History & Run Comments (spec 013)

**Status**: Draft outline for user review — not yet a full spec-kit spec.
**Created**: 2026-04-22 (revised after user feedback)
**Origin**: Concrete investigation of `results/Noa2_5_1/merge_remerge_1` (see LEARNINGS entries 2026-04-22) showed spec 012 covers *inside-a-run* observability but not *across-run* understanding. User cannot filter by album, cannot leave comments on runs, cannot see what config changed vs the parent run, and Apply + Remerge silently overwrites sibling runs.
**Relates to**: SIGHTING-025 (run overwrite), spec 012 (per-run observability), spec 009 (session pipeline — `session.json` already exists).

---

## Problem Statement

### P0 — Album identity is broken across runs (primary problem)

Every recluster or remerge of the same source album produces a new output directory, and that directory name becomes the de-facto "album" identifier.  As a result:

- Different runs derived from the same source (e.g. `results/Noa2_5_1/merge_snap_1`, `results/Noa2_5_1/merge_remerge_1`) appear as **different albums** in the History tab, even though they are variants of the same album.
- The output folder name leaks into the album column, making the column meaningless for grouping.
- Output folder names collide silently (SIGHTING-025): running Apply + Remerge twice from the same parent overwrites the first result.

**Required fix**: the `album` field stored in `run_history_db` must always reflect the **source input directory** (the original image folder), not the output run folder. All runs derived from the same source are grouped under the same album name in the History table.

### P1 — Additional UX gaps

After running many merge/remerge cycles on an album, the user also cannot:
- Find the right run quickly — the History tab is a flat list without filters.
- Leave a note on a run explaining what it represents ("correct merge: Noa with mom").
- See the parent run of the currently-loaded run, or what config differs between them.

## Goals (scoped down after user feedback)

**G1 (P1)**: Flat searchable History table.
**G2 (P1)**: Per-run free-text comment field.
**G3 (P1)**: Parent pointer + config diff in the loaded-run header.
**G6 (P1)**: Overwrite protection on run directory creation.
**G8 (P1, migrated from spec 012 US4)**: Run Summary dashboard — consumes `pipeline_run.json.quality_summary`, `merge_metadata.json`, and cluster provenance (all persisted by spec 012) to render quality funnel, cluster formation summary, stage timeline, and "0 candidates at threshold X" for empty merge_log runs.

## Explicitly Deferred (not this spec)

- Size-delta warnings on clusters across runs (needs cross-run cluster matching).
- Run-to-run diff view.
- Persistent per-cluster identity labels with propagation.
- Per-cluster annotations / labels.

## Non-Goals

- Changes to the merge algorithm or clustering behavior.
- Any split/undo action.
- Consolidating `session.json` and `run_history_db` — keep both; History reads from the DB.

---

## User Stories

### US1 — Flat searchable History table (G1)

The History tab is a single table — one row per run — with the following columns:

| Column | Source | Notes |
|---|---|---|
| Album | `run_history_db.source_album` | **Source input dir** (e.g. `Noa2_5`) — same for all derived runs |
| Run name | `run_history_db.run_name` | Human-readable label (e.g. `merge_snap_1`, `remerge_tight`) |
| Output folder | relative path of the run's output dir | for disambiguation only |
| Run kind | `base` / `remerge` / `manual_merge` / `recluster` | |
| Parent run | `pipeline_run.json.parent_run_id` (clickable) | |
| Created at | DB timestamp | |
| n_faces / n_core | `pipeline_run.json.summary` | |
| n_clusters | `pipeline_run.json.summary` | |
| Key config | one-line summary: `K=50 d=0.5 merge=on` | |
| Comment | (see US2) | |

`source_album` is written once when a run is first registered. For base runs it equals the input image directory name; for remerge/recluster runs it is inherited from the parent run's `source_album`.

Above the table: **Album filter** (dropdown of distinct `source_album` values), **date range** (from/to), **free-text search** across album + run name + comment. Sort by any column.

**Independent test**: Create 3 base albums × 4 derived runs each; all 4 runs for album A share the same `source_album`; album filter shows 3 distinct albums; selecting one narrows to exactly 4 rows.

### US2 — Per-run comment (G2)

Every run has a single free-text comment field, editable from the History table and from the run's header (when loaded). Stored in `run_history_db` (new `comment` TEXT column) and mirrored into `pipeline_run.json.comment` for portability. Saved inline; no separate Save button needed.

**Independent test**: Add a comment on a run; reload the app; comment still there. Search by comment substring in History finds the run.

### US3 — Parent pointer + config diff in run header (G3)

When any run is loaded into the app, the page header shows:

> **Album**: Noa 2-5  •  **Run**: merge_remerge_1  •  **Parent**: merge_snap_1 (clickable) • **Config Δ**: [see details]

"Config Δ" expands into a two-column table showing only the fields that **differ** from the parent run's config (e.g., `distance_threshold: 0.35 → 0.5`). If parent config is identical, display "No config change vs parent". If no parent, show "Base run".

**Independent test**: Load `Noa2_5_1/merge_remerge_1`; header shows album "Noa 2-5", parent "merge_snap_1" (or whatever its actual parent is), and config diff highlights `distance_threshold`, `K`, `merge_exemplar_threshold` deltas.

### US8 — Run Summary dashboard (G8, migrated from spec 012 US4)

When a run is selected in the History table (or loaded as the active run), a **Run Summary** panel shows:
- **Quality funnel**: total detected -> rejected per gate -> survived (from `pipeline_run.json.quality_summary`).
- **Cluster formation**: `clusters_stage_base.csv` count -> `clusters.csv` count; merges proposed / accepted / rejected (from `merge_log.json`).
- **Stage timeline**: stage names and durations (from `pipeline_run.json.stages`).
- **Empty-merge-log handling**: when `merge_log.json == []`, display "0 candidates at threshold X" using `merge_metadata.json.merge_candidate_threshold` (spec 012 FR-009 guarantees the field is present).

**Dependency**: spec 012 must land first to persist the underlying data.

**Independent test**: Load `results/Noa2_5_1/merge_remerge_1`; Run Summary shows the quality funnel, 0-candidate merge state ("0 candidates at threshold 0.45"), and stage timeline.

---

### US6 — Overwrite protection + unique run naming (G6)

This directly fixes P0's folder collision issue.

A central helper `allocate_run_dir(source_album: str, kind: str, label: str | None) -> Path` guarantees uniqueness:
- Output path = `results/<source_album>/<kind>_<auto_counter>/` (e.g. `results/Noa2_5/remerge_3/`).
- If the computed path already exists, the counter is incremented until a free slot is found.
- An optional human-readable `label` is stored as `run_name` in the DB (separate from the folder name).
- All Apply + Remerge / manual merge / recluster actions route through this helper — none compute their output path locally.
- Before dispatching, the UI shows the final chosen output path so the user can see what will be written.

**Independent test**: Run Apply + Remerge twice from the same parent; the second run lands in a new directory, never overwriting the first. Both rows share the same `source_album` in the DB. Closes SIGHTING-025.

---

## High-Level Design

**Data layer**:
- `run_history_db` gains / renames columns: `source_album` (source input dir, replaces/augments any existing `album` column), `run_name` (human label), `output_dir` (relative path), `parent_run_id`, `run_kind`, `comment`, `config_json`. All indexed for search.
- `pipeline_run.json` gains `source_album`, `run_name`, `comment` (mirrors for portability). Writer updates on every save.
- `session.json` unchanged.

**New helpers** in `face_cluster/`:
- `run_naming.allocate_run_dir(source_album: str, kind: str, label: str | None) -> Path` — unique output-dir + DB registration (G6, P0).
- `config_diff.compute(parent_config: dict, child_config: dict) -> List[ConfigDelta]` — diff config dicts (G3).
- `run_history.search(filters: HistoryFilters) -> List[RunRow]` — queries the DB with album/date/text filters (G1).

**App changes** (`app/face_clustering.py`):
- Rewrite `render_history_tab()` as a filter bar + `st.dataframe` table with click-to-load.
- New run header component rendered at the top of every tab that shows the current run.
- Comment field: inline `st.text_input` bound to DB via a small helper that writes on change.
- Every action-dispatch path calls `allocate_run_dir()` and shows the target path.

## Resolved Decisions

1. **Comment length cap**: hard limit 2 KB.
2. **`parent_run_id` back-fill (migration only)**: existing DB rows predating 013 get `parent_run_id = NULL`. No inference from directory names. All new runs record `parent_run_id` natively.
3. **Config diff expander scope (G3)**: configurable. Default = show all fields that differ from parent; unchanged fields collapsed. Config lives under `configs/pipeline.yaml` in a new `observability` section.
4. **Key-config column format (G1)**: configurable. Default fields = `K`, `distance_threshold`, `merge_exemplar_threshold`, `merge_enabled`. Rendered as a single compact line per row. Config lives in the same `observability` section as Q3.

## Dependencies / Ordering

- **Spec 012 must land first.** 013 consumes: `pipeline_run.json.quality_summary`, `merge_metadata.json` (with `merge_candidate_threshold` and `n_candidates_proposed`), cluster `origin` + `parent_cluster_ids`, `clusters_stage_base.csv`. All four are spec 012 deliverables (FR-002, FR-003, FR-004, FR-009, FR-011).
- Spec 009 infrastructure (`session.json`, `run_history_db`) already provides the base that 013 extends (new columns: `parent_run_id`, `comment`, `config_json`).
- No blocking dependency on spec 010/011.

## Proposed Scope

Scope is already the minimum that solves the user's pain. If we need to shrink further, the cut order is: G3 (config diff) → G1 (table can stay simpler without full search) → G2 (comments) → G6 (overwrite — **must stay**, it's a data-loss bug).

---

## Next Step

On user confirmation of scope + answers to the 4 open questions, proceed to full spec-kit artifacts:
1. `spec.md` (user stories, FRs, success criteria)
2. `plan.md` (phases, design decisions)
3. `research.md` (DB schema decisions, naming helper patterns)
4. `data-model.md` (DB schema, RunRow dataclass, ConfigDelta)
5. `tasks.md` (task list)
6. `checklists/requirements.md`
