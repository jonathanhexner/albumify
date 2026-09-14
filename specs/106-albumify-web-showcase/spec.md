# spec-106 — Albumify Web Showcase (the launchable front door)

**Status:** Draft
**Owner:** Jonathan / Claude
**Branch (when started):** `spec/106-albumify-web-showcase`

> **Two decisions assumed here — override before we start if you disagree:**
> 1. **Frontend = Next.js/React on Vercel** (product-grade polish + full-stack signal). Alternative: polished Streamlit (faster, weaker signal).
> 2. **Phase 1 = explore precomputed real trips** (bulletproof, no live models). Live upload is Phase 2, optional, and never a dependency of Phase 1.

---

## 1. Problem

The technical work behind Albumify is strong (real CV pipeline, benchmarked against a frontier VLM, 16 completed experiments) but **invisible and unshippable as a portfolio piece**: three separate Streamlit apps, a 640-line internal README, a two-terminal launch, and a pipeline that OOMs above ~120 images. A hiring manager sees a research monorepo, not a product.

We need **one running, publicly deployed app** that presents Albumify as a real product — and that **cannot look half-baked**: every state designed, every interaction working, nothing that errors or OOMs in front of a stranger.

## 2. Goal / non-goals

**Goal:** A deployed, product-grade web app at a public URL that lets anyone explore Albumify curating three real trips — the culled album, the people it found, and *why* each photo was chosen — with a clear "how it works / the engineering behind it" narrative.

**Non-goals (Phase 1):**
- No live model inference, no photo upload, no accounts, no database.
- No dependency on the FastAPI backend or the heavy pipeline at runtime.
- Not a rewrite of the existing apps — they stay as internal tools.

## 3. Users & the single job

| User | Their job on the page | Success = |
|---|---|---|
| Recruiter / hiring manager (2–5 min) | "Is this person's work real and finished?" | Leaves convinced it's a shipped product with real depth behind it |
| Engineer / interviewer (deeper) | "How does it actually work, and are the claims sound?" | Can drill into the pipeline decisions + honest benchmark framing |

## 4. What ships (Phase 1)

A Next.js app, deployed to Vercel, driven entirely by **precomputed JSON + 768px images** already on disk (`D:\albumify_vs_vlm\{budapest,austria,germany}\`):

- `albumify_picks.json` — the curated K=20 with composite scores + per-photo reason
- `scene_clusters.json` — moment groupings (dedup evidence)
- `vlm_picks.json` + `vlm_annotation.json` — the VLM arm: narrative roles, captions, album_type
- `manifest.json` — per-image metadata
- `imgs768/` — web-ready images

**Screens:**
1. **Landing / hero** — the drafted contact-sheet culling animation (`design_drafts/albumify_landing.html`), ported to React. States the thesis + routes to the trips.
2. **Trip album view** — the curated album as a designed gallery: ordered K=20, each photo openable to show *why it was picked* (composite score breakdown: sharpness / aesthetic / occlusion / tilt / face coverage) and *what it beat* (the near-duplicates from its scene cluster that got dropped).
3. **People view** — the face clusters for the trip (e.g. Budapest 340 faces → 15 people) as a browsable set.
4. **"How it works"** — the 4-pass pipeline explainer (already drafted) + a link out to the benchmark/case-study posts.
5. **Honest "vs a frontier VLM" panel** — the head-to-head, framed explicitly as a *preliminary pilot* (verdict pending, N=1) — never as a settled win.

## 5. Acceptance criteria (the anti-half-baked gate)

Phase 1 is **not done** until every one of these holds on the deployed URL, on a machine that is not Jonathan's:

- **AC1 — It loads and works first try.** Public URL, cold, no console errors, all three trips render.
- **AC2 — No dead ends.** Every link/button does what it says; no route 404s; no "TODO" / placeholder text visible.
- **AC3 — Every state is designed.** Loading, image-not-found fallback, and the "no faces / small cluster" cases all have intentional UI — never a blank or a stack trace.
- **AC4 — Data honesty.** Numbers shown match the source JSON (scores, pick counts, cluster sizes); the VLM comparison carries its "preliminary pilot, N=1" caveat inline.
- **AC5 — Responsive + themed.** Works on a phone-width viewport and in both light/dark; body never scrolls sideways.
- **AC6 — Fast.** Album view interactive < 2s on a mid-tier connection (images sized/lazy-loaded).
- **AC7 — Reproducible build.** `npm run build` clean; deploy is one command / one Vercel push; a short README says how.
- **AC8 — Nothing live to break.** No runtime model call, no unbounded input path anywhere in Phase 1.

## 6. Phase 2 (optional, later, gated separately)

A bounded "try your own photos" path: small N (≤~40 imgs), the light pipeline variant (no OOM), hosted FastAPI on a model-capable host (Modal / Render / Fly / HF Spaces). **Only merged once it clears the same AC bar on its own.** Its own spec. Phase 1 must stand complete without it.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Precomputed JSON is missing a field the UI wants (e.g. per-photo reason text) | Task 0 audits the 3 JSON sets against the UI's data needs; any gap is filled by a one-off offline script, never at runtime |
| Images total ~13 MB+/trip → slow | Generate web thumbnails (e.g. 400px) at build time; lazy-load full 768px on open |
| VLM comparison over-claims | AC4 pins the "preliminary pilot" caveat as a hard requirement |
| Scope creep back into "live app now" | Phase 1/2 split is binding; live path cannot block the deploy |

## 8. Definition of "shipped"

Deployed public URL meeting AC1–AC8, `REVIEW.md` produced via `/code-review` with no unresolved High findings, CHANGES_LOG entry, and the URL linkable from a resume.
