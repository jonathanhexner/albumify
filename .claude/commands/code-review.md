---
description: Run the code-review checklist against the active spec; produce REVIEW.md and block handoff on any High-severity finding.
---

Apply `docs/guides/CODE_REVIEW_CHECKLIST.md` to the active spec.

## Procedure

1. **Identify the active spec**:
   - If the user named one, use that.
   - Otherwise: read `.specify/feature.json` if present; else infer from recent git log + currently in-progress entries in `TODO.md`.
   - Confirm the spec dir to the user before starting if there's any ambiguity.

2. **Gather inputs**:
   - `git diff main...HEAD` (or the user-named base) — every file the spec touched.
   - `git log main..HEAD` — every commit on this branch.
   - The spec's `spec.md` and `tasks.md`.
   - Test suite output: `.venv/Scripts/python -m pytest tests/architecture tests/<area>` for the affected area.

3. **Walk every section** of `docs/guides/CODE_REVIEW_CHECKLIST.md`:
   - For each criterion, mark `pass` / `pass-with-followup` / `fail`.
   - File-and-line references for any finding worth a ticket.
   - **Do not skip §5 (Testability)** or §6 (Boundary contracts). These are the most-often-skipped sections and the ones that have prevented past incidents (see the failure-mode table at the bottom of the checklist).

4. **Write `specs/<NNN>-<name>/REVIEW.md`** structured as:
   - Part 1: How it works (module inventory + dependency map + data flow)
   - Part 2: Findings (one subsection per checklist section, with severity tags)
   - Part 3: Accept / reject verdict per area + recommended follow-up tickets

5. **File follow-up tickets** for every `pass-with-followup` and `fail`:
   - Non-trivial features → new spec dir under `specs/NNN-<name>/` (full PRD via WORKFLOW.md)
   - Bug-shaped → entry in `docs/project/SIGHTINGS.md`
   - Trivial housekeeping → line in `TODO.md`
   - **Never** dump into `docs/project/FEATURE_REQUESTS.md` — that file is a deprecated bin.

6. **Block handoff** if any §1-§7 criterion is marked `fail`:
   - Report the failing criteria to the user.
   - Refuse to flip the spec status to `Implemented` in `MASTER_PLAN.md` / `TODO.md` / `CHANGES_LOG.md`.
   - Offer to either (a) fix the failures now, (b) waive with explicit user approval (recorded in REVIEW.md verdict section), or (c) keep the spec at `In Progress` pending follow-up.

## What this command does NOT do

- It does not auto-fix findings. The reviewer flags; the next session (or this one, on user request) fixes.
- It does not skip sections to be polite. Every section gets a verdict.
- It does not write into `FEATURE_REQUESTS.md`. That file is being phased out.

## Output expectations

Concise. Verdict per area + finding list + ticket list. The full REVIEW.md is the artifact; the chat reply should be a 1-paragraph summary plus "REVIEW.md written; N findings; M blockers; awaiting decision on each blocker."
