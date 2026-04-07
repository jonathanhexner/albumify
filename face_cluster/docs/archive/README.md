# Archived Face Clustering Documentation

**Purpose**: Historical documentation that has been superseded or is no longer relevant.

---

## Why Documents Are Archived

Documents are moved here when they are:
1. **Superseded** - Replaced by newer, better documentation
2. **Outdated** - Describes old implementations that no longer exist
3. **Completed** - Planning documents where the plan has been executed
4. **Historical** - Useful for understanding past decisions but not current state

---

## Archive Contents

### Superseded Documentation

**IMPLEMENTATION_SUMMARY.md**
- Superseded by: `design/PHASE_1A_SUMMARY.md`
- Reason: Newer summary is more complete and accurate

**CURRENT_IMPLEMENTATION_STATUS.md**
- Superseded by: `ARCHITECTURE.md` + `design/PHASE_1A_SUMMARY.md`
- Reason: Status is now documented in architecture doc

**ARCHITECTURE_DOCUMENTATION_COMPLETE.md**
- Superseded by: `ARCHITECTURE.md`
- Reason: Final architecture doc replaces this completion marker

### Completed Planning Documents

**SPRINT_PLANS_CLUSTERING_DEBUG.md**
- Status: Completed
- Date: 2026-03-27
- Reason: Sprint plans were executed, now documented in Phase 1A summary

**PLAN_ML_CLUSTER_MERGING.md**
- Status: Completed
- Date: 2026-03-27
- Reason: ML merging plan executed, now documented in workflows/

**DOCUMENTATION_ORGANIZATION_PROPOSAL.md**
- Status: Executed
- Date: 2026-03-28
- Reason: Proposal was accepted and implemented (you're reading the result!)

### Outdated Implementation Docs

**FACIAL_CLUSTERING_DEBUG.md**
- Reason: Describes old debug approach, superseded by ui/debug_view.md

**FACE_FILTERING_PLAN.md**
- Reason: Old filtering approach, superseded by algorithms/quality_gating.md (TODO)

**FACE_MANAGEMENT_MODULES.md**
- Reason: Old module structure, superseded by ARCHITECTURE.md

**FACE_MANAGEMENT_UI_PLAN.md**
- Reason: Old UI plan, superseded by ui/ documentation

**FACE_RECOGNITION_FIX_PLAN.md**
- Reason: Bug fix completed, no longer relevant

**PROPOSAL_FACE_ALIGNMENT_REFACTOR.md**
- Reason: Refactor completed, now documented in ARCHITECTURE.md

---

## How to Use Archive

**When to read archived docs**:
- Understanding historical context for current design decisions
- Investigating why something was implemented a certain way
- Researching past bugs and how they were fixed

**When NOT to use archived docs**:
- Building new features (use current docs instead)
- Onboarding new team members (start with main README)
- Debugging current issues (use pipeline/troubleshooting.md)

---

## Restoring Archived Documents

If an archived document becomes relevant again:
1. Review if it's still accurate
2. Update to match current state
3. Move back to appropriate directory
4. Update this README to reflect the change

---

**Last Updated**: 2026-03-28
