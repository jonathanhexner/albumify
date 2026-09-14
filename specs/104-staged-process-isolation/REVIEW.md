# spec-104 v1 — Code Review (per-step process isolation)

**Branch**: `spec/104-staged-process-isolation` · **Base**: `main` · **Reviewed**: 2026-09-15
**Scope**: executor-only per-step process isolation behind `PipelineConfig.isolate_steps` (default OFF).
**Checklist**: `docs/guides/CODE_REVIEW_CHECKLIST.md`.

---

## Part 1 — How it works

**Module inventory (diff vs main):**
| File | Change |
|---|---|
| `sim_bench/pipeline/config.py` | +`isolate_steps: bool = False` |
| `sim_bench/pipeline/executor.py` | +`_isolated_step_worker` (module-level spawn target), +`_execute_step_isolated`, +1 branch in `_execute_step` |
| `tests/pipeline/test_step_isolation.py` | 6 tests (new) |
| `tests/pipeline/_isolation_fixtures.py` | importable step fixtures (new) |
| `scripts/spec104_isolation_probe.py` | RSS probe (new) |
| `specs/104-.../{spec.md,engineering_design.html}` | v1 section + eng doc |

**Data flow (flag ON):** parent `_execute_step_isolated` → validate in parent → marshal picklable context (minus `on_progress`) → `spawn` child → child re-imports step class (module+qualname), rebuilds context, runs `step.process()` (mutates in place), ships mutated state back over a queue → child **exits** (OS reclaims model RSS) → parent applies state, fires relayed progress, returns `StepResult`. Flag OFF → unchanged in-process path.

**Tests run:** `tests/pipeline/test_step_isolation.py` + `test_executor_step_io_logging.py` + `tests/architecture` = **142 passed, 0 failed**. Probe: parent RSS **+480 MB in-process vs +1 MB isolated** (~100% reclaimed).

---

## Part 2 — Findings by section

**§1 Structure** — `pass-with-followup`
- ⚠ `executor.py` is now ~388 LOC (>300). Single responsibility (the executor) but no size-justifying header. → follow-up: extract isolation machinery into `sim_bench/pipeline/isolation.py`.
- ⚠ `_isolated_step_worker` has 5 params (>4). Justified — it's a `multiprocessing` target and the args ARE the marshaling payload; could become a small namedtuple. Minor.
- No dead code; no reverse imports; `multiprocessing` is stdlib.

**§2 Code quality** — `pass`
- `except Exception` in the worker is the isolation error-capture (ships the failure back, does not swallow). Progress-relay + `release()` swallow-to-warning are on cosmetic/best-effort paths, not load-bearing. No precedence traps. Comments answer "why".

**§3 Naming** — `pass`
- `isolate_steps`, `_execute_step_isolated`, `_isolated_step_worker` are accurate. No `__all__` — consistent with the rest of `sim_bench/pipeline`.

**§4 Layering / coupling** — `pass-with-followup`
- ⚠ `_execute_step_isolated` duplicates the validation-error block and the in/out logging of `_execute_step`. → follow-up: extract shared `_validation_failure_result()` + `_log_io()` helpers.
- Single-writer respected: the two execution paths are mutually exclusive (flag branch); the isolated path replaces context state wholesale from the child's copy.

**§5 Testability** — `pass-with-followup` **← the one item for your decision**
- Test inventory: **6 integration tests** (real subprocesses, not mocks) + **1 probe**. No mocks used.
- ✅ Failure-mode walk-through: the OOM-kill class (the exact SIGHTING-117 scenario) is covered by `test_isolated_crash_parent_survives`; the memory-retention class by the probe; behavior parity by `test_default_off_is_identical_to_in_process`.
- ⚠ **No E2E on the *real* album pipeline with `isolate_steps=True`** (heavy models + data). The mechanism is exercised end-to-end on synthetic steps, but not with the real model-loading steps. Mitigations: flag is **default-OFF** (production path byte-identical); a real-pipeline E2E needs models + `needs_data`. → follow-up ticket: a `slow`/`needs_data` E2E running a small real pipeline isolated, asserting output equivalence + a bounded-RSS assertion.

**§6 Boundary contracts** — `pass`
- No new Pydantic/Pandera models, no DB schema change. The serialization boundary is self-enforcing (a non-picklable field fails loudly); the only non-picklable member (`on_progress`) is excluded + relayed. Config knob → producer: `isolate_steps` → the executor branch exists.

**§7 Documentation** — `pass-with-followup`
- ✅ `spec.md` (v1 section), `engineering_design.html` (the head-of-eng deliverable), `CHANGES_LOG.md`.
- ⚠ `tasks.md` not updated to mark v1 done (still the original stage-phase plan). → follow-up.
- ⚠ Verify `docs/architecture/classes.html` — `PipelineConfig` gained a field + `PipelineExecutor` gained a path; update if those docs cover the pipeline executor (they may not — currently face_cluster/DB-focused). → follow-up.
- ⚠ `LEARNINGS.md` — the durable lesson ("torch CPU RSS isn't returned in-process on Windows; process death is the only reliable free()") warrants an entry. → follow-up.

**§8 Risk register** — `pass`
- Deferred v2 stage-grouping documented (spec + doc) → follow-up ticket.
- No pinned workarounds. Backwards-compat: default-OFF → nothing breaks on legacy runs.
- ⚠ Hot-path perf: per-step spawn + full-context marshal overhead is characterized qualitatively (risk map) but not timed on a real pipeline → covered by the same real-pipeline E2E follow-up.

---

## Part 3 — Verdict & follow-ups

**No high-severity blockers.** No failing tests, no safety mechanism removed, no claimed-but-unenforced contract, no DB/dependency risk. The default-OFF flag means the production path is byte-identical.

**One finding for your explicit call (§5):** the isolated path has **no E2E on the real model-loading pipeline**. Per the checklist's strict reading ("new API without real-data E2E"), this *could* be treated as a blocker; given default-OFF + a fully-exercised mechanism (real subprocesses) + a proven memory probe, the reviewer recommends **Accept-with-follow-up** (ticket the real-pipeline isolated E2E as `slow`/`needs_data`).

**Verdict: ACCEPT (v1, flag-gated) with follow-ups** — pending your decision on §5.

**Follow-up tickets (→ TODO.md):**
1. `slow`/`needs_data` E2E: run a small real pipeline with `isolate_steps=True`; assert output equivalence vs in-process + bounded peak RSS. *(the §5 item)*
2. Extract isolation machinery into `sim_bench/pipeline/isolation.py`; share validation/logging helpers with `_execute_step`.
3. Update `tasks.md` (v1 done); add `LEARNINGS.md` entry (Windows torch RSS); verify `classes.html`.
4. v2: stage-grouping (contiguous steps in one child; disjoint model set per stage).
