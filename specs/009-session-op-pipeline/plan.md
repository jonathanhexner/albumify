# Implementation Plan: Session Operation Pipeline

**Spec**: [spec.md](spec.md)
**Created**: 2026-04-21
**Revised**: 2026-04-21
**Status**: Draft (v3 — labels accumulate, steps materialize)

---

## Current State Analysis

### How folders are created today

| Action | Current folder naming | Location |
|--------|----------------------|----------|
| Run Pipeline | User types any path | Anywhere |
| Recluster | `{source_folder}_recluster_{HH%M%S}` | Sibling of source |
| Manual merge snap | `merge_snap_{rnd}` | Sibling of `result.output_dir` |
| Remerge | `merge_remerge_{rnd}` | Sibling of `result.output_dir` |

### What must NOT regress

- History tab still shows all old runs (pre-session), browsable as before.
- `pipeline_result.output_dir` is still a path to a folder containing `faces.csv`, `clusters.csv`, etc. — all analysis code is unchanged.
- All existing tests pass.
- `FaceClusteringPipeline.run()` and `recluster()` signatures are unchanged.

---

## Design Decisions

### 1. Session folder structure

```
results/my_album/                ← session root (user picks this)
  session.json                   ← source of truth
  base/                          ← base step output (faces, embeddings, quality)
    faces.csv
    embeddings.npy
    quality_scores.csv
    clusters.csv                 ← initial clustering (from pipeline)
  chain_01/                      ← first clustering chain
    step_00_cluster/             ← initial clustering output
      clusters.csv
      cluster_stats.json
    step_01_merge/               ← materialized merge (remerge ran)
      clusters.csv
      merge_log.json
      labels.json                ← accumulated labels that produced this step
    step_02_merge/               ← another materialized merge
      clusters.csv
      merge_log.json
      labels.json
    pending_labels.json          ← labels applied but not yet remerged (accumulates)
  chain_02/                      ← branched from chain_01 step_01
    step_00_cluster/             ← same as chain_01 step_00
    step_01_merge/               ← same as chain_01 step_01
    step_02_recluster/           ← different path: recluster instead of merge
      clusters.csv
```

**Key principle: steps = materialized clustering states, not user clicks.**

A new step folder is only created when the clustering state changes (i.e., after a remerge or recluster). Label-only actions ("Apply only") accumulate in `pending_labels.json` at the chain level. When the user clicks "Apply + Remerge", the pending labels are consumed, a remerge runs, and a new step folder is created with the result. The consumed labels are moved into the step's `labels.json` for reproducibility.

Flow example:
1. User labels 8 pairs → "Apply only" → saved to `chain_01/pending_labels.json` (no new folder)
2. User labels 5 more → "Apply only" → appended to same `pending_labels.json` (13 total)
3. User clicks "Apply + Remerge" → `step_01_merge/` created, `labels.json` contains all 13 pairs, `pending_labels.json` cleared
4. User labels 3 more → "Apply only" → new `pending_labels.json` starts accumulating
5. User clicks "Apply + Remerge" → `step_02_merge/` created with those 3 pairs

### 2. `session.json` schema

```json
{
  "session_id": "my_album_20260421_141500",
  "created_at": "2026-04-21T14:15:00",
  "source_album": "/path/to/album",
  "base": {
    "folder": "base",
    "timestamp": "2026-04-21T14:15:00",
    "params_summary": "120 images, 342 faces, quality_gate=default",
    "n_faces": 342,
    "n_images": 120
  },
  "chains": [
    {
      "chain_id": "chain_01",
      "folder": "chain_01",
      "created_at": "2026-04-21T14:15:30",
      "branched_from": null,
      "has_pending_labels": true,
      "steps": [
        {
          "step_index": 0,
          "folder": "step_00_cluster",
          "type": "cluster",
          "timestamp": "2026-04-21T14:15:30",
          "params": {"K": 5, "distance_threshold": 0.35},
          "result_summary": "28 clusters"
        },
        {
          "step_index": 1,
          "folder": "step_01_merge",
          "type": "merge",
          "timestamp": "2026-04-21T14:31:00",
          "params": {"n_pairs": 13},
          "result_summary": "20 clusters"
        }
      ]
    },
    {
      "chain_id": "chain_02",
      "folder": "chain_02",
      "created_at": "2026-04-21T14:50:00",
      "branched_from": {"chain_id": "chain_01", "step_index": 0},
      "has_pending_labels": false,
      "steps": [
        {
          "step_index": 0,
          "folder": "step_00_cluster",
          "type": "cluster",
          "timestamp": "2026-04-21T14:15:30",
          "params": {"K": 5, "distance_threshold": 0.35},
          "result_summary": "28 clusters"
        },
        {
          "step_index": 1,
          "folder": "step_01_recluster",
          "type": "recluster",
          "timestamp": "2026-04-21T14:50:30",
          "params": {"K": 8, "distance_threshold": 0.32},
          "result_summary": "22 clusters"
        }
      ]
    }
  ],
  "active_chain": "chain_01"
}
```

**Note on merge steps**: The `mode` (apply_remerge vs apply_only) is no longer a step parameter — it determines *whether* a step is created. "Apply only" accumulates labels in `pending_labels.json` (no step). "Apply + Remerge" consumes pending labels and creates a materialized step. The `has_pending_labels` flag on the chain indicates whether uncommitted labels exist.
```

### 3. Chains share the base — no data duplication

All chains read embeddings and face data from `base/`. Each chain step only writes clustering-related outputs (cluster assignments, merge logs, labels). The expensive data (embeddings, crops, quality scores) lives once in `base/`.

### 4. Branching = new chain folder with step definitions copied

When branching from chain_01 step_01:
1. Create `chain_02/` folder.
2. Copy step definitions 0..1 from chain_01 to chain_02's step list in `session.json`.
3. Re-execute steps 0..1 into `chain_02/step_00_cluster/` and `chain_02/step_01_merge/` (cheap — seconds).
4. User then appends new steps to chain_02.

Since clustering is cheap, we re-execute rather than symlink/copy output files. This avoids cross-chain file dependencies and keeps each chain self-contained.

### 5. Merge: labels accumulate, steps materialize

Two user actions, different effects:

| Action | What happens | New step folder? |
|--------|-------------|-----------------|
| **Apply only** | Labels saved to `chain_NN/pending_labels.json` | No |
| **Apply + Remerge** | Pending labels consumed → remerge runs → new `step_NN_merge/` | Yes |

`pending_labels.json` schema:
```json
{
  "labels": [
    {"pair": ["cluster_03", "cluster_07"], "decision": "approve", "timestamp": "..."},
    {"pair": ["cluster_12", "cluster_15"], "decision": "reject", "timestamp": "..."}
  ]
}
```

When "Apply + Remerge" runs:
1. Read `pending_labels.json` (all accumulated labels)
2. Apply approved pairs as merge directives
3. Run remerge on the last materialized step's output
4. Write result to new `step_NN_merge/` folder
5. Copy consumed labels into `step_NN_merge/labels.json` (for reproducibility)
6. Clear `pending_labels.json`

A standalone "Remerge" button (no new labels) is also valid — it creates a step of type `remerge` that re-runs merge detection on the current state without any user labels. This catches cascading effects from earlier merges.

### 6. Backward compatibility

Old runs (no `session.json`) remain fully browsable in the History tab. The session pipeline UI is only shown when a session is active. Loading an old run from History still works as before.

---

## Implementation Tasks

### Task 1 — `SessionManager` data model and disk I/O

**File**: `face_cluster/session_manager.py` (new)

Implement:
- `@dataclass Step`: `step_index`, `folder`, `type` (literal), `timestamp`, `params: dict`, `result_summary: str`
- `@dataclass Chain`: `chain_id`, `folder`, `created_at`, `branched_from: dict | None`, `steps: list[Step]`
- `@dataclass Session`: `session_id`, `created_at`, `source_album`, `session_root: Path`, `base: dict`, `chains: list[Chain]`, `active_chain: str`
- `SessionManager`:
  - `create(session_root: Path, source_album: str, base_summary: dict) -> Session`
  - `load(session_root: Path) -> Session | None`
  - `create_chain(session: Session, branched_from: tuple[str, int] | None = None) -> Chain`
  - `append_step(session: Session, chain_id: str, step_type: str, params: dict) -> tuple[Step, Path]`
  - `update_step_result(session: Session, chain_id: str, step_index: int, result_summary: str)`
  - `get_chain(session: Session, chain_id: str) -> Chain`
  - `get_active_chain(session: Session) -> Chain`
  - `set_active_chain(session: Session, chain_id: str)`
  - `get_step_output_dir(session: Session, chain_id: str, step_index: int) -> Path`
  - `get_base_dir(session: Session) -> Path`
  - `save_pending_labels(session: Session, chain_id: str, labels: list[dict]) -> None` — append labels to `pending_labels.json`
  - `consume_pending_labels(session: Session, chain_id: str) -> list[dict]` — read and clear `pending_labels.json`, return consumed labels
  - `get_pending_labels(session: Session, chain_id: str) -> list[dict]` — read without clearing
  - `has_pending_labels(session: Session, chain_id: str) -> bool`

**Constraints**:
- No UI code. No Streamlit imports.
- Schema defined as module-level constant.
- `_save(session)` atomically writes `session.json` (write to temp, then rename).

**Test file**: `tests/face_clustering/test_session_manager.py`
- `test_create_session` — creates session, verifies `session.json` schema and `base/` folder
- `test_create_chain_and_append_steps` — creates chain, appends cluster + merge steps, verifies folders and json
- `test_branch_chain` — create chain_01 with 3 steps, branch from step_01, verify chain_02 has steps 0..1 copied
- `test_load_session_survives_restart` — write session, re-instantiate, verify all chains and steps present
- `test_multiple_chains_independent` — verify modifying chain_02 doesn't affect chain_01
- `test_pending_labels_accumulate` — save labels twice, verify both batches present; consume, verify cleared
- `test_pending_labels_warn_on_branch` — verify `has_pending_labels` returns True when labels exist

---

### Task 2 — Base step: session creation in Run tab

**File**: `app/face_clustering.py`

**Changes**:
1. Rename "Output directory" → "Session directory".
2. On pipeline success: call `SessionManager.create(output_dir, source_album, base_summary)`.
3. Pipeline writes base output to `session_root/base/`.
4. Create initial chain with step_00 (the initial clustering from the pipeline run).
5. Store session in `st.session_state.active_session`.

**No change** to `FaceClusteringPipeline` signatures.

---

### Task 3 — Chain step execution engine

**File**: `face_cluster/chain_executor.py` (new)

A lightweight executor that takes a chain's step list and base dir, and executes steps from a given index:

```python
def execute_chain_from(session: Session, chain_id: str, from_step: int) -> Path:
    """Execute steps from_step..end, return final output dir."""
```

Each step type maps to a function:
- `cluster` → calls `recluster()` on base embeddings with step params
- `merge` → loads labels from step params, applies them, optionally remerges
- `recluster` → calls `recluster()` on previous step's output
- `split` → placeholder

Input to each step = output of previous step (or `base/` for step 0).

---

### Task 4 — Merge tab: labels accumulate, remerge materializes

**File**: `app/face_clustering.py`

**Changes**:
1. **"Apply only" button** → saves current label decisions to `chain_NN/pending_labels.json` (appends to existing). No new step folder. UI shows "N pending labels" indicator. Analysis tabs remain on the last materialized step.
2. **"Apply + Remerge" button** → consumes all pending labels, runs remerge, creates new `step_NN_merge/` folder. Analysis tabs refresh from the new step's output. Pending labels indicator clears.
3. **"Remerge" button** (no new labels) → creates `step_NN_remerge/` from current state. Useful after accumulating several "apply only" rounds or to catch cascading effects.
4. After any remerge, updated merge candidates are shown (some old unlabeled ones may remain, new ones may appear from centroid shifts).
5. If pending labels exist when user tries to branch or switch chain → warn: "You have N pending labels that haven't been remerged. Discard or remerge first?"

---

### Task 5 — Recluster tab: session-aware

**File**: `app/face_clustering.py`

**Changes**:
1. If session active, hide output dir input. Show "Appending recluster step to chain_XX".
2. On recluster click: `SessionManager.append_step(session, chain_id, "recluster", params)`.
3. Execute the recluster step via `chain_executor`.
4. Update step result summary.

---

### Task 6 — Chain pipeline UI

**File**: `app/face_clustering.py`

Render chain as ordered step list in sidebar or above tabs:

```
Session: my_album
  Base: 120 images, 342 faces

  Chain: chain_01 (active)
    [0] Cluster         K=5, dist=0.35 → 28 clusters
    [1] Merge (8 pairs) apply+remerge → 20 clusters   ◀ viewing
    [2] Merge (5 pairs) apply+remerge → 15 clusters

  [Switch chain ▾]  [Branch from here]
```

- Click a step → load that step's output into analysis tabs
- "Branch from here" → creates new chain from selected step, switches to it
- Chain selector dropdown for switching between chains

---

### Task 7 — Session persistence across restarts

**File**: `app/face_clustering.py`

On app startup:
- If `session_root` is set, `SessionManager.load()` restores the session.
- Active chain and latest step are loaded automatically.
- Old runs without `session.json` use legacy History tab flow.

---

## Integration Points

| Current code | After change |
|-------------|-------------|
| `FaceClusteringPipeline().run(config)` | Unchanged — session wraps pipeline |
| `FaceClusteringPipeline().recluster(...)` | Unchanged — called by chain executor |
| `pipeline_result.output_dir` | Points to active step's output folder |
| History tab | Unchanged — still uses `run_history_db` |
| Analysis tabs | Unchanged — consume `pipeline_result` |
| Merge snapshot / remerge | Called by chain executor inside merge step |

---

## Rollout Order

```
Task 1 (SessionManager) → Task 2 (Base/Run tab) → Task 3 (Chain executor)
  → Task 4 (Merge steps) + Task 5 (Recluster steps)  [parallel]
  → Task 6 (Chain UI) → Task 7 (Persistence)
```

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Old runs break (no session.json) | `SessionManager.load()` returns None; History tab unchanged |
| Chain re-execution produces different results | Steps are deterministic (same params → same output). Log random seeds if any. |
| Many chains → disk bloat | Clustering output is small (~KBs). Base (embeddings) shared. Not a concern. |
| Merge labels reference cluster IDs that shifted | Labels reference the cluster IDs from the *previous* step's output. Chain executor validates. |
| Windows path length limits | Chain/step folder names are short. Session root is user-chosen. |

---

## Files to Create / Modify

| File | Action | Reason |
|------|--------|--------|
| `face_cluster/session_manager.py` | Create | Session/chain/step data model + disk I/O |
| `face_cluster/chain_executor.py` | Create | Step execution engine |
| `face_cluster/__init__.py` | Modify | Export new public API |
| `app/face_clustering.py` | Modify | Session creation, chain UI, step appending |
| `tests/face_clustering/test_session_manager.py` | Create | Unit tests |
| `tests/face_clustering/test_chain_executor.py` | Create | Executor tests |

---

## Out of Scope

- Migrating existing runs to session format (History tab handles them as-is)
- Multi-session parallel editing
- Split step implementation (placeholder type only)
- Export of session as a bundle
- Mode 2 merge (remerge after each individual label — deferred, Mode 3 covers most cases)
