---
name: prd
description: Use when the user asks to write, draft, re-plan, split out or check a spec / PRD in this repo (e.g. "write spec-116 for X", "/prd", "re-plan spec-112 in the new format", "check spec-115"). Produces specs/NNN-<name>/{spec,tasks,updates}.md in the mandated diagram-first format, grounds every class/file claim in the real code, renders the HTML, and runs scripts/prd_lint.py before showing it to the owner. Does not implement anything.
version: 1.0.0
user-invocable: true
argument-hint: "[what to spec]  |  check <spec dir or spec.md>"
---

# Write (or check) a PRD

The owner reads PRDs as HTML and wants them **readable at a glance: diagrams and tables, not prose**. The format
is fixed in `WORKFLOW.md` → "### 1. `spec.md`" (template, colour rules, limits). Read that section first, every
time; this skill is the procedure, WORKFLOW.md is the contract. `specs/115-album-workbench/spec.md` is the
reference example.

Mode: `check <spec>` → only steps 6-7 on an existing spec, then report. Otherwise steps 1-8.

## 1. Ground it before writing a word
- Read `ROADMAP.md` (stage, neighbouring specs), the specs this one touches, and **the code it will change**.
  Use an Explore agent when that spans more than a few files.
- List, from the code, every class and file the work touches, with what exists **today**. The class diagram and
  the file tables are claims about the code; the lint checks them.

## 2. Number and place it
- Next free number: `ls specs`. Directory `specs/NNN-<short-name>/`. A spec split out of another keeps a link
  both ways (old spec: one-line pointer instead of the moved content).

## 3. Write `spec.md` in the six sections, in order
1. **Objective**: at most 3 lines, what changes for the owner and why now.
2. **Requirements**: at most 15 rows `# | Status | Requirement | How we know it works`. Each check names a test
   file/test (`tests/x.py::test_y`) or a real-data result ("real ..."). Add one **Definition of done** line under
   the table. Cover, or state why not needed: errors/failure end states, security (who can read what), cost/limits,
   migration of existing data.
3. **Plan**: one mermaid flowchart; each step lists the modules it touches and ends on a check node naming the
   requirement ids it closes. Every requirement id must appear in some check node.
4. **Class diagram**: new / changed classes plus the existing ones they talk to. Colour each class with a `style`
   line (green `fill:#E3F3E5,stroke:#3E8A4B`, amber `fill:#FFF1D6,stroke:#C98A1B`, grey for context). Members:
   `✚` new, `✎` changed, `✖` removed, unmarked = unchanged; marker **before** the member. `✎`/`✖`/unmarked members
   must exist in the code today. Pages and routers get `<<page>>` / `<<API>>`. Use `direction TB`.
5. **New files**: `File | Class | Responsibility | Methods | Attributes | Status`. No test files here.
6. **Modified files**: `File | Class / function | Today | After this spec | Req | Status`.
Then short bullets: decisions, risks, non-goals. Status everywhere: ● done · ◐ partly · ○ planned (never colour).
Copy the legend line from the template.

## 4. `tasks.md` and `updates.md`
- tasks.md orders the work (W1, W2 …), each task names its files and test and ends with its requirement ids.
- updates.md: a dated "drafted" entry; later entries carry a Claim → Evidence table.
- Add a row to `ROADMAP.md` (Specs table) and, if it is next, to "Current position".

## 5. Render
`.venv/Scripts/python scripts/md2html.py specs/NNN-x/spec.md specs/NNN-x/tasks.md specs/NNN-x/updates.md ROADMAP.md`

## 6. Lint until clean
`.venv/Scripts/python scripts/prd_lint.py specs/NNN-x/spec.md --render`
Fix every ERROR (format limits, unnamed checks, requirements missing from the plan, file tables vs disk,
diagram members that do not exist, diagrams that did not draw). Warnings: fix, or say in one line why not.

## 7. Look at it
Open the HTML in Playwright at 1400 px (screenshots stay on disk, outside git). Check the diagrams are legible,
nothing wraps badly, and the first screen answers "what, why, how do we know". Aim for under ~6,000 px.

## 8. Hand over: do not implement
Reply with the link, the objective, the requirement count by status, open decisions as questions, and the lint
result. Code starts only after the owner approves (CLAUDE.md "Implementation gate"). Log a CHANGES_LOG [DOCS] entry.
