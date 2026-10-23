---
title: "spec-113 REVIEW - remove cluster_people + Albumify regression suite"
---

# spec-113 REVIEW — remove `cluster_people` + Albumify regression suite

*2026-10-16. Reviewer: Claude (against `docs/guides/CODE_REVIEW_CHECKLIST.md`). Scope: the uncommitted change set
on `spec/106-albumify-web-showcase` (73 files, +1,597 / −2,879), i.e. spec-113 plus the related SIGHTING-132 fix,
the LocalRunner encoding fix and the photo guard.*

**Verdict: ACCEPT with follow-ups.** 0 `fail`, 7 `pass-with-followup` (all ticketed). One gate still open before
`Implemented`: the V2 Budapest suite (CLAUDE.md, binding, because `app/face_clustering/tabs/run_tab.py` changed).

## Part 1 — How it works

| Area | Module | Change |
|---|---|---|
| Ordering | `sim_bench/pipeline/steps/{identity_refinement,cluster_by_identity,select_best_per_person}.py` | `depends_on` no longer names `cluster_people` (SIGHTING-132) |
| Removal | `steps/cluster_people.py`, `steps/configs/cluster_people.py`, `steps/face_cluster_bridge.py`, 2 yaml, 4 tests | Deleted; unregistered in `all_steps.py`, `configs/__init__.py` |
| Settings | `configs/pipeline.yaml` `people_clustering:`; `PipelineService._broadcast_clustering_config`; `pipeline_runner.py`; `config_diff.py` | Renamed source block; old key read and wins over yaml defaults (O1) |
| Regression (new) | `sim_bench/album/regression.py` (`AlbumSummarizer`, `AlbumComparator`, `Tolerances`); `scripts/albumify_regression.py`; `tests/regression/albumify/` | Real-album baselines + tolerant comparison through `JobService` |
| Serving | `sim_bench/serving/runners.py` `LocalRunner.run` | Child output read as UTF-8 |
| Repo hygiene | `scripts/check_no_photos.py`, `.gitignore`, `.gitattributes` (nbstripout), `.photo-allowlist`, CI step | Photos never in git |

```
UI / profile  --people_clustering-->  PipelineService._broadcast_clustering_config
                                         FCParams(**block) -> to_step_configs() -> 8 chain steps
default_pipeline: ... quality_gate .. assign_people_clusters -> identity_refinement -> cluster_by_identity -> select_best
regression:  JobService(local) -> album.json -> AlbumSummarizer -> AlbumComparator(baseline) -> PASS/FAIL
```

Dependency direction: `scripts/` -> `sim_bench.album.regression` -> stdlib only; `sim_bench.album.regression` imports
nothing from serving/pipeline. No reverse imports (grep: `regression.py` imports `dataclasses`, `typing`).

## Part 2 — Findings

### §1 Structure — pass-with-followup
- New modules are small with one responsibility: `regression.py` 115 LOC, `albumify_regression.py` 130, `check_no_photos.py` 116.
- Large files touched are pre-existing (`pipeline_service.py` 602, `explore.py` 563, `pipeline_runner.py` 561 — the
  latter shrank by 63 lines). Not introduced here.
- **F1 (low)** dead parameter: `app/streamlit/pages/explore.py:448` `_render_face_clustering_tab(all_decisions, …)`
  no longer uses `all_decisions` after the dead face-decision block was removed. -> TODO.
- Max 4 params everywhere new. No orphan files (removed a failed-clone folder `sim-bench-backup-2026-10-16.git/` from the repo root).

### §2 Code quality — pass-with-followup
- `regression.py`: no try/except; comparisons explicit (`_close` with relative band + absolute floor).
- `albumify_regression.py:63` `except Exception` in `_git_sha` -> "unknown": metadata only, acceptable.
- **F2 (medium)** `scripts/check_no_photos.py:85` skips a text/notebook file when `git show :path` fails
  (`except (OSError, CalledProcessError): continue`). A guard should not pass silently on an unreadable file; image
  files are unaffected (caught by extension first). -> TODO: report it as a problem instead.
- **F3 (low)** `albumify_regression.py` prints results via buffered stdout: with output redirected, PASS/FAIL lines
  appear only at the end of a multi-hour run (seen today). -> TODO: `flush=True` / log through `logging`.

### §3 Naming and package structure — pass
- `people_clustering` names what the block is (a settings source, not a step). `regression.py` declares `__all__`.
- Scripts follow the existing `scripts/` convention (no `__all__`).

### §4 Layering and coupling — pass-with-followup
- Single writer: `context.people_clusters` now has one producer in the default run (`assign_people_clusters`) — the
  core of SIGHTING-132. `TestRealRegistry::test_every_dependency_names_a_registered_step` guards stale deps.
- Old-key fallback is implemented twice with the same rule (`pipeline_service.py`, `pipeline_runner.py`, plus
  `config_diff.py`). Small, documented; acceptable.
- **F4 (medium, existing SIGHTING-133)** two config interpreters still exist: the job path
  (`serving/pipeline_runner.SpecBuilder`) does not apply `people_clustering` at all. Ticketed; owner decision.

### §5 Testability — pass-with-followup
Test inventory (new or changed by this change set):

| Kind | Tests | Count |
|---|---|---|
| Unit | `tests/album/test_regression.py` | 7 |
| Unit | `tests/pipeline/test_fc_analysis_export.py` (+2), `tests/architecture/test_config_parity.py` (+1), `tests/control/test_job_service.py` (+1, real child process) | 4 |
| Registry / build | `tests/pipeline/test_builder.py::TestRealRegistry` (real registry + real yaml) | 2 |
| E2E real data | `tests/regression/albumify/test_albumify_regression.py` -> `albumify_regression.py check` on 4 real albums (48 / 122 / 474 / 715 photos) | 4 datasets, all PASS |
| Removed | 4 old-vs-new / bridge tests; 1 bridge source-inspection test | −5 |

- Real E2E: yes — 4 albums through the product path, baselines on pre-removal code, checks on post-removal code.
- Full suite: **1,584 passed, 2 skipped, 0 failed** (11:47).
- Failure-mode walk-through:
  | Bug class | Test that fails if it recurs |
  |---|---|
  | Deprecated step pulled back in via a stale `depends_on` (SIGHTING-132) | `test_every_dependency_names_a_registered_step` |
  | People steps scheduled after `select_best` (E3) | `test_people_steps_run_after_clustering_and_before_selection` |
  | People settings not reaching the chain after the rename | `test_settings_reach_the_unified_steps_and_source_key_is_dropped` |
  | Old saved profiles silently ignored | `test_pre_spec113_key_overrides_yaml_defaults`, `test_pre_spec113_key_is_still_read` |
  | Album drift on real data | `albumify_regression.py check all` |
  | Child error lost on non-UTF-8 locale | `test_local_runner_keeps_the_error_tail_when_output_is_not_in_the_locale_code_page` (fails without fix) |
  | Photo committed / notebook outputs / inlined image | manual probe 2026-10-16 (guard + hook blocked); **no automated test** -> F5 |
- **F5 (medium)** `check_no_photos.py` has no unit test; verified only by a manual probe. -> TODO.
- **F6 (low)** `test_people_steps_run_after_clustering_and_before_selection` reads `configs/pipeline.yaml` relative
  to the working directory; it breaks if pytest runs from elsewhere. -> TODO (resolve from the repo root).
- Mocks: none in the new tests except the existing `monkeypatch` of `LocalRunner.command` (needed to run a
  controlled child process).
- **F7 (low, existing SIGHTING-134)** `tests/pipeline/test_face_pipeline_e2e.py` (opt-in) was ported but cannot
  run (pre-existing fixture faults). Ticketed.

### §6 Boundary contracts — pass
- `people_clustering` -> `FCParams` (Pydantic): unknown keys are dropped with an INFO log (pre-existing behaviour,
  unchanged); every key the UI writes is an `FCParams` field.
- Config knob -> producer: `export_for_analysis` is routed to `face_cluster_analysis_export` (tested);
  every `people_clustering` field is consumed by a chain step via `to_step_configs()`.
- `Tolerances` is a frozen dataclass; baseline JSON carries `summary_version`.

### §7 Documentation deliverables — pass
- `spec.md`, `tasks.md`, `updates.md` (with verification table), this `REVIEW.md`.
- Architecture HTMLs updated: `classes.html` (removed `ClusterPeopleConfig`, `face_cluster_bridge`; added
  `AlbumSummarizer`/`AlbumComparator`), `data_flow.html` (stage D rewritten), `db_schemas.html`, `db_global.html`,
  `system_onboarding.html`. No new architecture doc -> `index.html` unchanged.
- CHANGES_LOG: entries for SIGHTING-132 fix, spec-113, LocalRunner fix, photo purge, CI pin. LEARNINGS: SIGHTING-132
  and photo-purge entries. MEMORY: `feedback_no_photos_in_git`, `reference_github_repos`, model-cache note.

### §8 Risk register — pass-with-followup
- **Backwards compatibility:**
  - Saved profiles under `cluster_people`: still read (tested). Legacy-method keys (`method`, `min_samples`,
    `pca_components`, …) are ignored (they had no effect since spec-079).
  - A pipeline list or API request naming `cluster_people` as a **step** now fails validation ("Unknown step").
    No shipped config does; documented here.
  - Old run dirs with a `cluster_people` block: `config_diff` still reads them.
- **Behaviour change from the SIGHTING-132 fix (not from the removal):** Slovakia 118 -> 79 people,
  147 -> 247 picks (per-person sub-groups now reach `select_best`). **F8 (owner decision)**: is the larger album
  wanted? -> TODO (owner).
- Hot path: no new per-image work (removal only); regression suite is opt-in (`albumify_regression` marker).
- Workarounds: the old-key fallback should be removable later -> covered by the two O1 tests; remove them together.

## Part 3 — Verdict per area and follow-ups

| Area | Verdict |
|---|---|
| §1 Structure | pass-with-followup (F1) |
| §2 Code quality | pass-with-followup (F2, F3) |
| §3 Naming | pass |
| §4 Layering | pass-with-followup (F4 = SIGHTING-133) |
| §5 Testability | pass-with-followup (F5, F6, F7 = SIGHTING-134) |
| §6 Boundary contracts | pass |
| §7 Documentation | pass |
| §8 Risk register | pass-with-followup (F8) |

Tickets:

| # | Ticket | Where |
|---|---|---|
| F1 | Drop unused `all_decisions` param of `_render_face_clustering_tab` | TODO.md |
| F2 | `check_no_photos.py`: report unreadable files instead of skipping | TODO.md |
| F3 | `albumify_regression.py`: unbuffered progress output | TODO.md |
| F4 | One config interpreter for API and job paths | SIGHTING-133 (existing) |
| F5 | Unit tests for `check_no_photos.py` (image, notebook, inlined image, allowlist) | TODO.md |
| F6 | `test_builder.py` reads the yaml from the repo root | TODO.md |
| F7 | Fix or delete `test_face_pipeline_e2e.py` | SIGHTING-134 (existing) |
| F8 | Owner: accept larger albums from per-person selection, or tune `max_images_per_cluster` | TODO.md (owner) |

**Remaining gate before `Implemented`:** run `pytest -m budapest tests/face_clustering/e2e_budapest/` (V2 baseline
gate; `run_tab.py` help text changed).
