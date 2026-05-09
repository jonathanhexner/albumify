# Feature Development Workflow

Structured workflow that turns a feature request into a spec, tasks, and implementation. All artifacts live under `specs/<NNN>-<feature-name>/`.

## Quick Reference

```
Feature idea
  |
  v
spec.md    -->  WHAT to build (problem, user stories, acceptance criteria)
  |
  v
tasks.md   -->  HOW to build it (design notes at top, ordered task checklist)
  |
  v
implement  -->  execute tasks phase-by-phase
```

## Required Artifacts (2 files)

### 1. `spec.md` — Define WHAT to build

**Focus**: Problem, user value, acceptance criteria. No implementation details.

**Structure**:
```markdown
# Feature Specification: <Name>
**Created**: YYYY-MM-DD
**Status**: Draft
**Resolves**: SIGHTING-NNN (if applicable)

## Problem Statement
Why this feature exists. What pain point it addresses.

## User Stories
### US1 — <Title> (Priority: P1/P2/P3)
Description of what the user wants and why.
**Acceptance Criteria**:
- Given X, When Y, Then Z
- ...

## Edge Cases
- Bullet list of boundary conditions

## Requirements (brief)
- FR-001: One-line requirement
- ...
```

**Rules**:
- User stories prioritized P1/P2/P3, each independently testable
- Acceptance criteria are concrete and verifiable
- No technology choices, no file paths, no function signatures
- Mark unresolved questions as `[NEEDS CLARIFICATION]` — resolve before tasks

### 2. `tasks.md` — Define HOW to build it

**Focus**: Design decisions, then ordered implementation tasks.

**Structure**:
```markdown
# Tasks: <Feature Name> (NNN)

## Design Notes
Brief technical decisions that inform the tasks below.
- **D1**: <decision and rationale>
- **D2**: <decision and rationale>

## Phase 1: <Name>
- [ ] T001 Description with file path `face_cluster/foo.py`
- [ ] T002 [P] Description (parallelizable)
- [ ] T003 Write tests for T001-T002 in `tests/face_clustering/test_foo.py`

**Checkpoint**: How to verify this phase works.

## Phase 2: <Name>
...
```

**Rules**:
- Design notes replace the old `plan.md` / `research.md` / `data-model.md` / `contracts/` — put it all here
- Tasks include exact file paths
- `[P]` marks tasks that can run in parallel
- Each phase ends with a checkpoint (usually a test command)
- Mark tasks done as `[x]` during implementation

## Optional Artifacts

Create these only when they earn their keep:

| File | When to create |
|------|---------------|
| `research.md` | Genuinely unresolved technical questions requiring investigation before you can write tasks |
| `data-model.md` | Complex schema changes touching 3+ modules (new DB tables, new CSV columns across writer/reader pairs) |
| `contracts/<name>.md` | Public API surface shared across packages (rare in this repo) |
| `checklists/<domain>.md` | Regulated domains, security-sensitive features |

**If you're unsure whether to create an optional file**: don't. Put the content in the Design Notes section of `tasks.md` instead.

## Exemptions (no spec required)

| Change type | What to do instead |
|-------------|-------------------|
| Bug fix | File sighting in `docs/SIGHTINGS.md`, fix, log learning |
| Isolated refactor (no behavior change) | Implement directly, update `CHANGES_LOG.md` |
| Doc-only change | Edit docs directly |

## Integration with Project Trackers

| Event | Tracker | Action |
|---|---|---|
| `spec.md` created | `docs/FEATURE_REQUESTS.md` | Add feature entry |
| `tasks.md` generated | `TODO.md` | Copy key tasks |
| Implementation complete | `CHANGES_LOG.md` | Append entry |
| Bug found during impl | `docs/SIGHTINGS.md` | File sighting |
| Lesson learned | `docs/LEARNINGS.md` | Append (<=5 lines) |
| Architecture changed | `docs/architecture.md` | Update |

## Directory Layout

```
specs/
  NNN-feature-name/
    spec.md          # Required: what to build
    tasks.md         # Required: how to build it
    research.md      # Optional: unresolved questions
    data-model.md    # Optional: complex schema changes
    contracts/       # Optional: shared API surfaces
    checklists/      # Optional: domain-specific quality gates

.specify/
  feature.json       # Active feature pointer (auto-managed)
```

## Legacy Specs

Specs 001–012 were created under the previous 7-file workflow. They remain valid — the extra files (`plan.md`, `research.md`, `data-model.md`, `contracts/`, `checklists/`) just aren't required for new specs going forward.
