# PRD format review — spec-115, 2026-10-23

Two independent reviews of `specs/115-album-workbench/spec.html` (neither saw the other), plus the impeccable
detector:
- **A. Design critique (impeccable method):** Nielsen's 10 heuristics adapted to a reading document, cognitive-load checklist.
- **B. Senior engineer + UI review:** clarity, size, focus, readability, accuracy against the code, recommendations for the PRD rule.
- **Detector** (`npx impeccable --json`): 33 findings before fixes (27 dark-mode contrast, 6 cramped highlights), **0 after**.

## Scores (before fixes)

| Review | Result |
|---|---|
| Design health (Nielsen, /40) | **21/40, "acceptable"**: navigation 2, consistency 1, phone 1 were the lows |
| Cognitive load (8 checks) | **4 failures, "critical"**: no chunking of 15 + 13 rows, banner and TOC above the title, R/W codes to remember, no progressive disclosure |
| Engineer: clarity | a new engineer could mostly start W2; would still ask about crop orientation pass criteria, box coordinate frame, real flag names |
| Engineer: accuracy | 2 methods marked "changed" did not exist (`spawn`, `download_results`); a wrong signature; members missing |

Both reviews praised: the **Today → After** columns, the **plan flowchart** (each step closes on its check), and
the concrete **"How we know it works"** column.

## Fixed on 2026-10-23 (in this round)

| Issue | Fix |
|---|---|
| Green meant both NEW and DONE | Hue = type of change only (green NEW, amber CHANGED, grey EXISTING, red REMOVED); progress = ● done · ◐ partly · ○ planned; plan steps: thick border done, dashed planned |
| Dark mode: badges 2.4-3:1 contrast, white diagram slabs | Badge colours are theme tokens with dark variants; diagrams sit on a paper card |
| Whole cells highlighted (no signal) | `<mark>` removed from tables; rule: only for a single changed word in prose |
| "R / 1" wraps, paths break mid-word | ID column nowrap; tables and diagrams widen to 1,240 px; status cells are a glyph |
| Wrong diagram markers | `spawn` / `download_results` now ✚, `stage` / `curate` / `download` ✖, real `_stage` signature, `path` added; no ✓ after members (mermaid shows it as a return type) |
| Redundant Type column, test rows in §5 | Removed; tests live in tasks.md per task. Phone page 18,000 → 10,400 px |
| Wrong flag name in tasks.md | `write_displays` + `_write_bundle` crops |

## Open: spec-115 content (engineer review, needs owner decision)

| Sev | Finding | Proposed fix |
|---|---|---|
| high | Cancel is underspecified: no `cancelled` status; partial files, cache commit, L4 queue slot | R4 states the end state of each |
| high | No requirement for file-serving security (shared `_inputs/` served through a job's route) | new requirement: route serves only files listed in that job's manifest/results; traversal 404 (tests exist) |
| med | No spend cap | R2 names the default `max_photos_l4` and a per-run ceiling |
| med | Old Modal jobs on `albumify-jobs` (kept forever) | migrate, purge, or a stated non-goal |
| med | No single definition of done | one line under §2 |
| med | Crop orientation has no pass criterion | T2.1 pass = PC crop overlaps worker crop on ≥ 19/20 faces, else keep worker crops |
| low | Vague checks (R3, R5, R11, R12 say "Playwright"/"tests") | name the test file and assertion |
| low | R6 / R15 are engineering guarantees, not outcomes | group under "Guarantees" |

Adding rows would exceed 15; merging R6+R7 and R14+R15 keeps it at 15.

## Recommendations for the PRD rule (WORKFLOW.md) and a skill

**Done 2026-10-23 (owner: apply content fixes, build /prd):** content fixes applied to spec-115 (14 requirements);
1 = `/prd` skill (`.claude/skills/prd/SKILL.md`); 2 = `scripts/prd_lint.py` (all checks above except "NEW files do
not exist" is refined to "unless marked ●"; desktop height is a warning); 4 = definition of done + the
errors/security/cost/migration coverage rule are in the skill. **Still open:** 3 (generate colours and status),
5 (progress strip, TOC position), 6 (plan as a table), 7 (phone layout).

1. **Make it a `/prd` skill** (a packaged instruction set): draft from the template, then a **code-grounding pass**
   that checks every ✎ / ✖ member exists in the named class (Python AST) and every ✚ does not, before asking for
   approval. This would have caught the `spawn` / `download_results` error. Plus `/prd-check` for an existing spec.
2. **Lint in `md2html.py --check`** (fail the render): objective ≤ 3 lines; ≤ 15 requirements; every requirement has a
   check that names a file, test, or real-data result (bare "tests" fails); every R appears in a plan check node and
   in a §6 Req cell; CHANGED files exist, NEW files do not yet; warn when the desktop page is over 6,000 px.
3. **Generate, don't hand-write:** class colours from `<<new>>` / `<<changed>>` markers (today: 13 hand-written
   `style` lines that can drift); status glyphs from tasks.md checkboxes (one source of truth); a
   requirement → task → evidence matrix from tasks.md "(Rn)" and updates.md tables.
4. **Template additions:** a one-line **Definition of done**; a fixed **Errors · Security · Cost · Migration** row
   set in §2 (each answered or "N/A because…"); REMOVED rows in §6.
5. **First screen:** a progress strip under the title (W1-W5 ●/○, "4/15 requirements proven"); TOC after the title
   (or sticky); short H1 ("spec-115 · Album workbench") with the subtitle as a lede.
6. **Size:** plan as a 5-row table (step · modules · check · status) or `flowchart LR` saves ~1,000 px; drop
   context-only boxes from the class diagram.
7. **Phone:** decide whether it matters for a single-owner doc. If yes, tables become stacked cards under 700 px;
   if no, say so and scroll tables sideways.
