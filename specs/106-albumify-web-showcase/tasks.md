# spec-106 — Tasks

**Status:** In Progress (planning) — **no implementation until spec.md is approved.**

## Design notes

- **Stack:** Next.js (App Router) + TypeScript, static-export friendly, deployed on Vercel. No server needed for Phase 1 — data is JSON + images shipped as static assets.
- **Data flow (build-time, not runtime):** the 3 trips' JSON + images are *ingested once* by an offline script into `public/trips/<trip>/` (thumbnails + a single normalized `trip.json` the UI consumes). The heavy pipeline is never invoked by the app. This is what makes it un-OOM-able.
- **Normalized contract:** define one `Trip` / `Photo` / `Person` TypeScript type the UI reads; the ingest script is the only place that knows the raw `albumify_picks.json` / `scene_clusters.json` shapes. UI stays decoupled from pipeline internals.
- **Design source of truth:** `design_drafts/albumify_landing.html` (palette, type, contact-sheet motif) — port its tokens into the app's CSS/Tailwind theme so landing + app read as one system.
- **Honesty guardrail:** the VLM panel's "preliminary pilot / N=1" caveat is a component prop that cannot render empty (typed required).

## Phase 0 — Data audit & ingest (de-risk before any UI)
- [ ] T0.1 Audit `albumify_picks.json`, `scene_clusters.json`, `vlm_*.json`, `manifest.json` for all 3 trips; map each UI field (pick order, composite-score breakdown, per-photo reason, scene-cluster members, face clusters, VLM roles/captions) to a source field. **Record gaps.**
- [ ] T0.2 For any gap (e.g. no per-photo "reason" string, or face-cluster data lives elsewhere), write a one-off offline script to derive/emit it. Face clusters: locate the run artifacts (per memory, `face_clustering.db` / export dirs) and confirm 340→15 for Budapest is reproducible into JSON.
- [ ] T0.3 Ingest script: normalize the 3 trips → `public/trips/<trip>/trip.json` + generate 400px thumbnails alongside the 768px originals. Deterministic, re-runnable.
- [ ] T0.4 Define `Trip`/`Photo`/`Person`/`SceneCluster`/`VlmView` TS types; validate the emitted `trip.json` against them (zod or a tiny checker).

## Phase 1 — App scaffold + landing
- [ ] T1.1 `create-next-app` (TS, App Router); port design tokens from `design_drafts/albumify_landing.html` into the theme (both light/dark).
- [ ] T1.2 Landing route `/` — port the hero + contact-sheet culling animation to React; wire "explore trips" nav. Respect `prefers-reduced-motion`.
- [ ] T1.3 "How it works" section (4-pass pipeline explainer) + outbound link slot for case-study posts.

## Phase 2 — Trip album view (the core)
- [ ] T2.1 `/trip/[slug]` route; load `trip.json`; render the ordered K=20 album as a designed gallery (lazy-loaded thumbs).
- [ ] T2.2 Photo detail: composite-score breakdown (sharpness/aesthetic/occlusion/tilt/face) + "what it beat" (dropped near-dupes from its scene cluster).
- [ ] T2.3 Empty/fallback states: image-not-found placeholder, missing-score graceful render (AC3).

## Phase 3 — People view + VLM panel
- [ ] T3.1 `/trip/[slug]/people` — face clusters as browsable groups (cluster size, sample faces). Small-cluster state designed (AC3).
- [ ] T3.2 "vs a frontier VLM" panel — head-to-head with the **required** "preliminary pilot, N=1" caveat (AC4).

## Phase 4 — Harden + ship
- [ ] T4.1 Responsive pass (phone-width) + both themes; verify no horizontal body scroll (AC5).
- [ ] T4.2 Perf pass: image sizing, lazy-load, album interactive < 2s (AC6).
- [ ] T4.3 Deploy to Vercel; verify AC1–AC8 on the public URL from a second device/browser.
- [ ] T4.4 App README (run + deploy in one command each); CHANGES_LOG entry.
- [ ] T4.5 `/code-review` → `REVIEW.md`; resolve High findings; flip spec to Implemented.

## Explicitly out of scope for this spec
- Live upload / model inference (→ separate Phase 2 spec).
- Any change to the existing Streamlit apps or the pipeline.
