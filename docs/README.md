# Documentation Structure

## Where to put things

```
docs/
  project/          Living project tracking — updated continuously
    SIGHTINGS.md      Bug reports and investigations (SIGHTING-NNN)
    LEARNINGS.md      Lessons learned from bugs and incidents
    FEATURE_REQUESTS.md  User feature requests with status

  architecture/     System architecture — update when structure changes
    overview.md        Top-level architecture overview
    PIPELINE_*.md      Pipeline engine design docs
    SELECT_BEST_*.md   Selection algorithm docs
    FACE_*.md          Face detection/clustering architecture

  guides/           How-to guides — for humans getting started
    GETTING_STARTED.md   First-time setup
    APPS.md              All Streamlit apps and how to start them
    TROUBLESHOOTING.md   Common problems and solutions
    ML_DEVELOPER_SKILLS.md  ML development best practices

  design/           Specs, mocks, proposals, code reviews — scoped by app
    app/                      Main album app
    face_clustering/          Face clustering app
    PROPOSAL_*.md             Cross-cutting proposals

  reference/        Algorithms, APIs, config — lookup material
    face_clustering.md          Face clustering algorithm docs
    merge_criteria_reference.md Merge gate documentation
    requirements.md             Feature requirements log

  image_similarity/  Image similarity benchmarking docs (standalone topic)

  archive/          Stale docs — kept for history, not maintained
```

## File Naming Rules

### Design docs (`docs/design/<app>/`)
Always include date and scope:
```
YYYY-MM-DD_<topic>.md       — analysis, review, proposal
YYYY-MM-DD_<topic>.html     — interactive mock
```
Examples:
- `docs/design/app/2026-04-30_explore_page_code_review.md`
- `docs/design/app/2026-05-01_selection_observability_proposal.md`
- `docs/design/face_clustering/2026-03-15_merge_ui_mock.html`

### Architecture docs (`docs/architecture/`)
No date prefix — these describe the *current* state, not a point in time:
```
<TOPIC>.md                  — e.g. PIPELINE_ARCHITECTURE.md
```

### Project tracking (`docs/project/`)
Single living files — append new entries at the top, newest first.

### Guides (`docs/guides/`)
No date prefix — guides should stay current:
```
<TOPIC>.md                  — e.g. GETTING_STARTED.md
```

### Reference (`docs/reference/`)
No date prefix — reference material for the current system:
```
<topic>.md                  — e.g. merge_criteria_reference.md
```

## Placement Rules

1. **New sighting** → append to `docs/project/SIGHTINGS.md`
2. **New learning** → append to `docs/project/LEARNINGS.md`
3. **New feature request** → append to `docs/project/FEATURE_REQUESTS.md`
4. **Architecture changed** → update relevant file in `docs/architecture/`
5. **New spec/mock/proposal/review** → `docs/design/<app>/YYYY-MM-DD_<topic>.md`
6. **How-to guide** → `docs/guides/<TOPIC>.md`
7. **Algorithm/API reference** → `docs/reference/<topic>.md`
8. **Stale doc** → move to `docs/archive/`
9. **Never** create docs at `docs/` root level (except this README)
10. **Never** create new `.md` files at the repo root — use the appropriate `docs/` subfolder

## Root-level files (NOT in docs/)

These stay at the repo root because tools and workflows reference them directly:

| File | Purpose | Updated by |
|------|---------|-----------|
| `README.md` | Project overview, getting started | On major changes |
| `CLAUDE.md` | Claude Code instructions | On workflow changes |
| `TODO.md` | Task tracking | On every task start/complete |
| `CHANGES_LOG.md` | Change history | After every code change |
| `WORKFLOW.md` | Feature development workflow | On process changes |
| `RECOVERY_PLAN.md` | Face clustering module responsibilities | On architecture changes |
| `GEMINI.md` | Gemini integration notes | As needed |
| `MILESTONES.md` | Project milestones | On milestone completion |
