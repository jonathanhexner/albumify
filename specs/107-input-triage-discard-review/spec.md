# spec-107 — Input Triage & Discard Review

**Status:** Implemented (P0–P2) 2026-10-01 — REVIEW.md ACCEPT, architecture+triage suites green (149).
P0 metadata tier + P1 content tier + P2 pipeline steps & `set_aside` contract shipped; end-to-end
verified **0 junk in the curated album** (vs 6/20 before) on the real Slovakia roll. **P3 (behavioral
singletons/intent) + P4 (review-UX contract) remain** — P3 depends on spec-108; both tracked below.
**Owner:** Jonathan / Claude
**Priority:** P1 (proven necessary; blocks honest curation on real uploads)
**Relation:** feeds / is fed by [[spec-108]] (theme-relevance signal); surfaced in [[spec-106]] UI.

## 1. Problem (measured, not assumed)

Real camera rolls contain photos that are **not album moments**: auto-saved screenshots,
photographed documents/posters/receipts/maps, and utilitarian "notes-to-self" (the rental car, a
store product, a parking spot). Measured on a real trip (`D:\Slovakia26`, 2026-10-01 report):

- **6 of 20** curated album picks were junk (5 screenshots + 1 photographed poster).
- A **Gmail screenshot scored 0.703 — the highest in the whole album.**
- **Junk out-scored real photos** (mean composite 0.54 vs 0.41).

Root cause: screenshots are flat, sharp, well-exposed, face-free → the quality scorers (IQA/AVA)
**reward** them. You cannot filter junk by scoring it lower; it must be removed **before** curation.

## 2. Goal / non-goals

**Goal:** a pre-curation triage stage that identifies irrelevant photos, **sets them aside** (never
silent-delete), surfaces a per-item reason, and lets the user review/restore — so the album is
curated only from real moments.

**Non-goals:** theme/interestingness judgment (that signal comes from spec-108); the review-UI build
itself (spec-106 owns the surface); hard deletion of any user photo.

## 3. Scope — three tiers, cheapest first

| Tier | Catches | Signal | Validated on Slovakia26 |
|---|---|---|---|
| 1 · Metadata | screenshots, downloads | no camera EXIF (Make/Model), filename `Screenshot_*`, PNG, screen-res dims | caught all 8 screenshots, 0 false positives |
| 2 · Content | photographed docs/posters/receipts/maps | RAM++ junk tags (`screenshot, document, poster, receipt, map, menu…`) | caught 2 real-photo posters metadata missed |
| 3 · Behavioral | utilitarian notes (rental car, product) | scene-cluster singleton + no-person (+ theme-relevance from spec-108) | inconclusive on a sampled subset → test on full roll |

## 4. Approach

A triage step (or small step group) that runs **before** the scoring/curation steps, reads the
discovered image set, computes the three signals, and writes a `set_aside` list with
`{image, reason, confidence}` into the pipeline context. Conservative thresholds (only high-confidence
junk). The curation pipeline then runs only on the survivors. Nothing is deleted; `set_aside` is a
first-class, restorable output.

## 5. Acceptance criteria

- **AC1** On a real roll, ≥95% of screenshots/documents are set aside; **~0 real-photo false
  positives** at default thresholds.
- **AC2** No junk reaches the curated album in the Slovakia26 re-run (0/K, vs 6/20 today).
- **AC3** Nothing is ever silently deleted; every set-aside item carries a human-readable reason and
  is restorable.
- **AC4** `set_aside` (with reasons + counts) is exposed in the result contract for the UI.
- **AC5** Metadata tier adds negligible runtime; content tier (RAM++) cost is measured and bounded.

## 6. Dependencies & risks

- Depends on RAM++ content tagging (validated 2026-10-01) and spec-108 for the behavioral/theme
  tier's interestingness signal (optional at first).
- Risk: a real keeper wrongly set aside → mitigated by conservative thresholds + mandatory review UI.
- Risk: RAM++ adds a model to the memory floor → runs as its own isolated step (spec-104).

## 7. Definition of done
Triage step implemented + tests; Slovakia26 re-run shows 0 junk in album; `set_aside` contract +
UI hook; REVIEW.md; CHANGES_LOG. See `tasks.md`.
