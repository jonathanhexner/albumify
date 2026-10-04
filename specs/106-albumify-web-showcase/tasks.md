# spec-106 — Tasks (Direction B — Live Curator)

**Status:** In Progress — build started 2026-10-04. **Design of record: [`architecture.html`](architecture.html)**
(four planes, warm-up∥upload lifecycle, resolution tiers, show-everything, secrets, job state machine,
observability/durability). Phase detail: [`plan.html`](plan.html). Host decided: **Modal** (GPU worker
scaffolded in spec-109).

## Design notes (decisions, from architecture.html)
- **Two shapes, one reuses the engine.** The **image pipeline** (38 steps in `curate`) reuses
  `PipelineExecutor` as-is on Modal (device=cuda). The **job orchestration** is a NEW small
  **event-driven state machine** (control plane) — async/cross-service/long-lived; the existing
  synchronous executor does not fit it.
- **GPU is mandatory, not optional.** The full-pipeline CPU baseline is **9 s/img → non-interactive**
  (spec-109 T0.1). The live product runs on the Modal L4 worker. (This corrects the earlier
  "CPU ships v1, GPU is a later optimization" note — that was wrong.)
- **Resolution tiers:** browser downscales to **768px** for ALL photos (analysis + review); ~1600px
  display for keepers; original download-only. Don't upload full-res for analysis.
- **Warm-up ∥ upload:** fire the GPU warm-up when upload STARTS so the cold start hides under the upload.
- **Storage-reference, not inline bytes:** browser uploads 768px directly to storage via a short-lived
  presigned URL; the API passes `job_id`; `curate` pulls from storage (idempotent → free resume).
- **Show everything:** curation is a proposal, not a verdict. `album.json` carries EVERY photo + its drop
  reason; all are browsable + one-click promotable. Extends spec-107 `set_aside`.
- **Secrets:** Modal token + object-store creds on trusted hosts only (Vercel env / Modal Secret); NEVER
  in git / image / browser. Browser gets only scoped, expiring presigned URLs.
- **Observability/durability:** every stage + step writes to the DB (`jobs`, `job_events`,
  `pipeline_steps`); the job is config too (`job_pipeline.yaml`-style); resume = idempotent re-run from
  the persisted stage. No heavyweight workflow engine.
- **Design tokens:** `design_drafts/albumify_landing.html` → the app theme (both light/dark).

## Build order (prove each seam before stacking the next — architecture.html §08)

### P1 — Prove the GPU seam  (depends on user Modal auth)
- [ ] T1.1 `modal setup` → `gpu_smoke` (CUDA proof). | user
- [ ] T1.2 **storage-reference `curate(job_id)`** — pull 768px images from storage, run pipeline (device=cuda),
      write `album.json` + displays back. Replaces the inline-bytes scaffold. Upload models to the Volume.
- [ ] T1.3 First real `curate` on a stored trip → per-image **GPU latency** + $/album (proves ≤3–5 min).

### P2 — Storage + presigned upload
- [ ] T2.1 Object store (S3/R2) + per-job prefixes + TTL.
- [ ] T2.2 Presign endpoint (scoped, expiring write URL); browser PUTs 768px directly.
- [ ] T2.3 Prove: browser → storage (presigned) → `curate` pulls. No keys reach the browser.

### P3 — Control-plane API + job state machine
- [ ] T3.1 Job state machine (config-driven stages + `needs`); `jobs`/`job_events`/`pipeline_steps` DB;
      **write-to-DB on every transition** (status, duration, error) — the telemetry "what happened".
- [ ] T3.2 `POST /jobs` (bounds guard → presign → **fire warm-up** → enqueue), `GET /jobs/{id}` (status from
      DB), `GET /jobs/{id}/result`. Stateless orchestrator advanced by events.
- [ ] T3.3 Bounds guard: count/size/MIME + decode probe, rate limit, TTL; typed rejections (413/415/429).
- [ ] T3.4 Secrets wiring: Vercel env (API) + `modal.Secret` (worker); nothing in git/image/browser.
- [ ] T3.5 Resume: on restart, read job row → run from first incomplete stage; `curate` idempotent.

### P4 — Next.js front end
- [ ] T4.1 `create-next-app` (TS, App Router); port `albumify_landing.html` tokens; both themes; reduced-motion.
- [ ] T4.2 Dropzone + **in-browser downscale to 768px** + client-side bounds; direct-to-storage upload.
- [ ] T4.3 Live progress from `job_events`; **album + show-everything viewer** (picks + dropped-with-reasons + promote).
- [ ] T4.4 A designed screen for every job state: queued / rejected / timeout / failed / expired.

### P5 — Showcase seed
- [ ] T5.1 Ingest the 3 precomputed trips into the `album.json` shape → instant first paint.
- [ ] T5.2 "How it works" explainer + honest vs-VLM panel ("preliminary pilot, N=1" caveat).

### P6 — Harden &amp; ship
- [ ] T6.1 Adversarial uploads (HEIC/oversized/zero-faces); cold-start budget; concurrency UX.
- [ ] T6.2 Cost/budget alarm; privacy pass (TTL purge verified); responsive + perf.
- [ ] T6.3 AC1–AC10 (plan.html) on the public URL from a second device.
- [ ] T6.4 `/code-review` → `REVIEW.md`; CHANGES_LOG + README; flip to Implemented.

## Open decisions (deciding factors)
- **Content-safety stance** (public anon upload) — ToS + TTL minimum, or add a safety pre-check? Blocks public launch.
- **Shareable/persistent albums in v1?** If no → originals stay on device, zero full-res upload (simplest).
  If yes → upload full-res of keepers lazily after selection. Decides whether keeper-upload exists.

## Future hooks (noted, deferred)
- AI "configure" stage in the job state machine (album-type tuner, spec-108) — parametrizes the run,
  never sequences the pipeline. Escalation, not orchestration.

## Explicitly out of scope (v1)
Accounts / login / persistence beyond a TTL'd anon session · billing · heavyweight workflow engine ·
any change to the image pipeline beyond device=cuda + the global top-K cap.
