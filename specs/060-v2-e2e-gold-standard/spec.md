# spec-060 — v2 end-to-end gold-standard gate

**Created**: 2026-05-29
**Status**: Draft
**Predecessors**: spec-040 (FCAppRunner), spec-042 (v2 tab parity umbrella), spec-045 (Cluster Analysis tab — first big tab where the gate would have caught a regression), spec-050 (per-run UUID dirs), commit `686b635` (spec-implementer subagent — the natural mechanism for invoking the gate automatically).
**Successors**: spec-061 (quality regression band — Phase 3 of this spec, deferred until Phase 1+2 stabilize).
**Trigger**: 2026-05-29 a real bug landed in spec-045 (Cluster Analysis tab crashed with `ValueError: face_clustering.db not found` when spec-050's empty UUID dir was handed to the Repository) — caught only when the user ran the app manually. Every spec-040/042/045/050 commit was unit-test green; the user did the E2E coverage by hand. That's not sustainable as the refactor pace picks up.

---

## Problem

The v2 face-clustering app (`app/face_clustering_v2/`) is mid-refactor: spec-042 lists 8 tabs to rebuild, only 3 are shipped (Run, History, Cluster Analysis), 5 remain (Recluster, Face Analysis, Merged Clusters, Quality, Gallery, Overview). Each new tab + each polish pass on an existing tab is a "considerable change" that can break the happy path in ways unit tests can't see:

- spec-045's resolver bug (Cluster Analysis loaded an allocated-but-empty UUID dir from spec-050) had **0 failing tests** but a guaranteed user-facing crash.
- spec-049 catalogued 13 red tests on `unification/spec-040` that nobody noticed for an unknown number of days because no one was running the full suite.
- spec-040's burn-in gate ("2-week burn-in on at least one real album") has no automated verification — it's "the user runs the app and tells us if it broke."

**There is no single command that answers "does v2 still work end-to-end against a real album?"** That gap is what this spec closes.

## What we build

One pytest test file, opt-in via `@pytest.mark.slow`, that runs the full v2 happy path against a real fixture and asserts on the shape of the output. Three phases of asserting depth:

```
tests/face_clustering/test_v2_e2e_smoke.py    (opt-in: pytest -m slow)

  Phase 1 — pipeline smoke (this spec)
    - Resolves the fixture album dir from env var or conftest fixture.
    - Runs FCAppRunner against a 50-photo subset with a pinned FCParams profile.
    - Asserts: result.success == True; run_dir/face_clustering.db exists;
               result.n_clusters > 0; result.n_faces > 0; no Streamlit imports
               leaked into the pipeline.

  Phase 2 — UI smoke via Streamlit AppTest (this spec)
    - Loads the Phase 1 run_dir into st.session_state.
    - Programmatically renders each v2 tab in turn (Run, History,
      Cluster Analysis — plus any tabs spec-042 has shipped by then).
    - Asserts: no st.exception node on any tab; key widgets populate
      (cluster picker has >=1 option, history table has >=1 row).

  Phase 3 — quality regression band (separate spec-061)
    - Asserts: result.n_clusters in [stable_low, stable_high]
      (initial band wide, e.g. 10-100; tightens after burn-in).
    - Future: pairwise-agreement vs a golden baseline >= 0.95.
```

The test is the **single source of truth.** Pre-push hook, GitHub Action, and the spec-implementer agent all just invoke it.

## What we don't build

- **No new agent.** The spec-implementer subagent (commit `686b635`) already exists; we extend its Implementation gate prompt to invoke this test, no new agent needed. The pytest test does all the work; the agent just runs it.
- **No quality assertion in Phase 1.** "n_clusters > 0" only. The number bands land in spec-061 once we've seen 3-4 runs to characterize variance.
- **No CI matrix.** One platform (Windows, dev machine) for now. Cross-platform CI is a follow-up if the project ever needs it.
- **No fixture bundling.** The Budapest album stays on the user's machine; the test skips cleanly when the env var isn't set, same pattern as spec-045's `v2_budapest_run_dir` fixture.
- **No replacement of the existing `tests/face_clustering/test_legacy_vs_v2_equivalence.py`.** That tests algorithmic equivalence on a 9-face fixture — orthogonal. This one tests the full happy path on a realistic album.

## Locked decisions

1. **Pytest is the gate; the agent invokes it, doesn't replicate it.** A failing test is the unambiguous signal; an agent's "I think it might be broken" judgment isn't.
2. **`@pytest.mark.slow` for opt-in.** Default `pytest` runs stay fast; the gate runs explicitly via `pytest -m slow`. Three triggers: spec-implementer agent's Implementation gate, optional pre-push hook, optional GitHub Action.
3. **Fixture via env var.** `SIM_BENCH_E2E_ALBUM_DIR` (path) + `SIM_BENCH_E2E_PROFILE` (default `profile_4.json`). Test skips with a clear message when unset. No fixture data in the repo.
4. **50-photo subset.** Full Budapest is 5-8 min producer time; the test should run in ≤ 2 min. A subset fixture (first 50 jpgs sorted by name) gives the same code coverage.
5. **Output to `tmp_path`.** Never pollute `~/.sim_bench/runs/`. The session-scoped `isolate_action_log_db` autouse fixture from spec-051 already covers the action_log side.
6. **AppTest is best-effort.** Streamlit's AppTest API has known limitations (some widgets don't render fully). Phase 2 starts as "no exception on tab load"; richer assertions come as the API stabilizes.
7. **CLAUDE.md update is part of this spec.** Add an "E2E gate" subsection under §"Delivery Quality" that names this test as the gate for "considerable changes" (defined: anything touching `face_cluster/`, `sim_bench/pipeline/`, `app/face_clustering_v2/`, or the per-run DB schema).

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `tests/face_clustering/test_v2_e2e_smoke.py` exists with Phase 1 + Phase 2 cases | grep |
| AC2 | `pytest -m slow tests/face_clustering/test_v2_e2e_smoke.py -v` exits 0 when env vars set; skips cleanly when unset | run twice |
| AC3 | Test completes in ≤ 2 min on the user's dev machine for a 50-photo subset | wall-clock during AC2 |
| AC4 | spec-implementer subagent prompt updated to invoke this test as part of the Implementation gate for "considerable changes" | diff `.claude/agents/spec-implementer.md` (or wherever the prompt lives) |
| AC5 | CLAUDE.md §"Delivery Quality" gains an "E2E gate" subsection naming this test + definition of "considerable change" | diff CLAUDE.md |
| AC6 | spec-049 (test-suite recovery) updated to note the new opt-in test exists and how to run it | diff spec-049/spec.md |
| AC7 | One regression demo: stash a known broken state (e.g., revert the spec-045 resolver fix), confirm the test fails with a clear message; restore state | manual |
| AC8 | New session-scoped fixture `v2_e2e_album_dir` in `tests/conftest.py` (env var → Path, skip on missing) | grep |

## Risks

- **AppTest fragility.** Streamlit's testing API has gaps; some widgets / components may not render fully in a headless test runner. Phase 2 starts with the weakest assertion ("no exception") and tightens iteratively.
- **Test runtime creep.** 50 photos → ~2 min producer + clustering. If it grows past 5 min, drop to 25 photos or skip the producer chain (use a pre-built run dir as the Phase 2 input).
- **Profile drift.** If `profile_4.json`'s fields change (FCParams refactor), the test needs an update. Pin the profile name in the env var; document the contract.
- **False sense of security.** "E2E passed" doesn't mean "every tab works for every workflow." It means "the happy path on one album with one profile works." That's the floor, not the ceiling — manual exploratory testing still matters for new features.

## Effort estimate

**~5-7 hours** for the first two phases:
- Phase 1 (pipeline smoke): ~3h. Most of the time is fixture wiring + skip semantics + getting the producer chain to run under `tmp_path`.
- Phase 2 (UI AppTest): ~2-3h. Streamlit AppTest learning curve dominates.
- CLAUDE.md + spec-implementer prompt update + spec-049 cross-reference: ~1h.

Phase 3 (quality band) is spec-061; estimated ~3h after the first 3-4 runs land.

## Open questions

1. **Where does the spec-implementer agent prompt live?** Need to find `.claude/agents/spec-implementer.md` or equivalent. If it's not editable from this repo (lives in user's global Claude config), the agent-invocation piece becomes "document the gate in CLAUDE.md, user wires the agent themselves."
2. **Subset selection: first 50 by name, random seed 42, or labeled subset?** Recommendation: first 50 by name — deterministic, no fixture-management code needed.
3. **Quality band thresholds for Phase 3 / spec-061.** Need 3-4 baseline runs across different profiles to set defensible numbers. Deferred.
