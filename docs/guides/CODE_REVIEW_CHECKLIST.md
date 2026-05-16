# Code Review Checklist

Generic, reusable checklist applied to every spec before its status flips from `In Progress` to `Implemented`. Lives here so the criteria are spec-agnostic; per-spec findings go in the spec's own `REVIEW.md`.

**Invoke via**: `/code-review` slash command (see `.claude/commands/code-review.md`).
**Enforcement gate**: WORKFLOW.md lifecycle — no spec moves to `Implemented` without a green-or-justified review against this list.

---

## How to use this document

For each criterion below, the reviewing agent (or human) records one of:

- `pass` — criterion met
- `pass-with-followup` — minor finding; ticket filed; merge OK
- `fail` — high-severity finding; **blocks handoff** until resolved or explicitly waived

Output lives in `specs/<NNN>-<name>/REVIEW.md` with verdict per area + finding list.

**High-severity findings (any of the following) BLOCK handoff:**

- Critical bug introduced (any failing test that previously passed)
- Removal of a safety mechanism without a replacement (e.g., force-disable removed before the producer it implies exists — see SIGHTING-061)
- A contract claimed but not enforced at runtime (i.e., the test only inspects source, not behavior)
- A new public API with no E2E test exercising it on real data
- DB schema change without a migration story for existing runs
- Dependency added without a Windows + `.venv` installation check

---

## 1 · Structure

- [ ] **No module >300 LOC** unless it has a single, named responsibility and a comment at the top justifying the size.
- [ ] **No module mixes responsibilities** — schema definition + write path + validation in one file is a smell.
- [ ] **No file holds multiple unrelated classes** unless they form a closed type family (e.g., dataclass + its companion enum).
- [ ] **Scattered code with no unified purpose** — if a module's docstring can't summarize it in one sentence, split it.
- [ ] **No function with >4 parameters** without a justification. If unavoidable, group into a typed request object.
- [ ] **No "dead code"**: unused imports, unused private functions, branches whose conditions can't be true, orphan files in the repo root.
- [ ] **Dependency direction respected** — record the intended layering of the change; verify with grep that no new module crosses it.

## 2 · Code quality

- [ ] **Try/except** only around operations that can legitimately fail (I/O, parsing, external APIs). Bare `except:` is a fail. Swallowing exceptions to a warning on a load-bearing path is a fail.
- [ ] **If/else nesting** ≤ 3 levels. Deeper means the function should be split.
- [ ] **No silent defaults** at boundaries — `config.get(key, default)` is OK for genuinely optional values, not for fields the rest of the code depends on.
- [ ] **No operator-precedence-dependent expressions** without parentheses (e.g., `a or b if cond else c` — the ternary doesn't bind the way most readers expect).
- [ ] **Comments answer "why", not "what"**. Per CLAUDE.md.

## 3 · Naming and package structure

- [ ] **Module names match contents** — `config_diff.py` shouldn't grow non-diff helpers; rename or split.
- [ ] **Naming consistency** within a sibling group (e.g., `schemas.py` next to `models.py` is OK; `db_schemas.py` next to `image_detail.py` is inconsistent).
- [ ] **Subpackage when ≥3 files share a concern** — flat files at the package root are fine until they cluster, then promote.
- [ ] **`__all__` declared** for any new public-surface module. Or document the project convention if `__all__` isn't used elsewhere.

## 4 · Layering and coupling

- [ ] **No reverse imports** (low-level package importing high-level).
- [ ] **No duplicated logic** — search for existing helpers before writing a new one (e.g., a "diff two dicts" helper already exists; don't reinvent).
- [ ] **Single writer per piece of state** — if two code paths write to the same DB column / context field, one is wrong.
- [ ] **Two-way writers documented** — if two paths *must* write to the same place, name the contract that ensures they don't conflict.

## 5 · Testability — the load-bearing section

- [ ] **Test inventory recorded** in REVIEW.md: count tests of each kind (static / unit / synthetic / E2E).
- [ ] **At least one real E2E test** exercises the feature on realistic data. "Static + unit only" is a fail for any feature that touches production data flow.
- [ ] **Each test has one responsibility** — name the assertion clearly; reject tests that mix three unrelated checks.
- [ ] **Mock usage justified per test** — mocks are fine but each should answer: "why can't this be exercised against real code?"
- [ ] **Failure mode walk-through**: for each known bug class the feature was supposed to prevent, name the specific test that would fail if the bug recurred. If no such test exists, file a ticket.
- [ ] **Test placement**: unit-shape tests live in `tests/<package>/`; architecture-shape (static) tests live in `tests/architecture/`. Don't hide unit tests in `tests/architecture/` because the name sounds important.

## 6 · Boundary contracts (Pydantic / Pandera era)

- [ ] **Each new boundary** (object construction, DataFrame I/O, API input) has an enforcement mechanism, not just documentation.
- [ ] **`extra="forbid"`** on every new Pydantic BaseModel that represents an external contract.
- [ ] **Pandera `nullable=False`** on every column the schema requires; **`Int64Dtype()`** for nullable int columns; **range / set checks** where the domain is enumerable.
- [ ] **Contract claimed but not invoked** = fail. A schema that's defined but never `.validate()`-called is dead code.
- [ ] **Config knob → producer check**: for every config field that gates downstream behavior, the upstream step that produces the gated value exists. (SIGHTING-061 was this exact failure.)

## 7 · Documentation deliverables

A feature isn't reviewable without these. Reviewer checks each is present in the spec dir:

- [ ] **`spec.md`** — the PRD: problem, user stories with acceptance criteria, edge cases.
- [ ] **`tasks.md`** — design notes + phased task list.
- [ ] **`REVIEW.md`** — this checklist's output, per spec.
- [ ] **`EXECUTIVE_SUMMARY.html`** *(optional but recommended for cross-team-visible features)* — head-of-engineering gloss.
- [ ] **`docs/architecture/db_schemas.html`** updated if a DB column was added / removed / typed differently.
- [ ] **`docs/architecture/classes.html`** updated if a class / Pydantic model / Pandera schema / dataclass field changed.
- [ ] **`docs/architecture/data_flow.html`** updated if a pipeline step, the bridge, the exporter, or the read side changed.
- [ ] **`docs/architecture/index.html`** updated if a new architecture doc was added.
- [ ] **CHANGES_LOG.md** updated with one entry per phase/PR.
- [ ] **LEARNINGS.md** updated when the work surfaced a new failure class.
- [ ] **MEMORY** updated when a piece of guidance applies to all future sessions.

**Documentation update mandate** (failing this section blocks handoff): drift between code and the architecture HTMLs is itself a finding. If the docs are wrong, fix them — don't silently leave them stale.

## 8 · Risk register

- [ ] **Known deferred work**: each "Accept with follow-up" finding has a filed ticket (PRD, sighting, or TODO).
- [ ] **Workarounds named and tested**: any temporary pin / hack is covered by an architecture test that fails if someone "fixes" it without first lifting the constraint that motivated the workaround.
- [ ] **Backwards-compat surface**: list what breaks on legacy data; either fix or document.
- [ ] **Hot-path performance**: if a hot path now does work it didn't before (validation, schema checks), measure overhead on a representative input.

---

## Failure mode lessons baked into this list

Each entry below points at a real incident that produced one of the criteria above. Updates here are expected when a new class of failure shows up.

| Incident | Criterion that would have prevented it |
|---|---|
| **SIGHTING-058** — two writers (JSON + SQLite) for merge_log drifted | §4 "single writer per piece of state" |
| **SIGHTING-059** — bridge silently dropped 5 fields | §2 "no silent defaults at boundaries"; §6 "Pandera nullable=False on required columns" |
| **SIGHTING-060** — `face.area` unit drift across 3 producers | §6 "config knob → producer check" (generalizes to "field unit → producer check") |
| **SIGHTING-061** — bridge removed force-disable without verifying upstream producer | §5 "failure mode walk-through"; §6 "config knob → producer check" |
