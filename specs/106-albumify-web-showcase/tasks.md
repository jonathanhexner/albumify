# spec-106 — Tasks (Direction B — Live Curator)

**Status:** Draft — **no implementation until spec.md is approved and the §8 open decisions are made
(host + content-safety stance are the two blockers).**
Full rationale + diagrams: [`plan.html`](plan.html). Superseded the Direction-A task list.

## Design notes
- **Reuse the engine, don't touch it.** Workers call `execute_spec(spec, ctx, isolate_steps=True)`
  (`sim_bench/pipeline/run.py`) on the `default_pipeline` from `configs/pipeline.yaml`. No step changes
  except the new global top-K cap (T0.1).
- **Normalized `album.json`** is the single contract the UI reads — identical for live jobs and the
  precomputed showcase seed, so both render through the same components.
- **The in-process `_jobs` dict + `BackgroundTasks` (`sim_bench/api/pipeline_service.py`) is retired**,
  replaced by a durable queue + job store.
- **Design source of truth:** `design_drafts/albumify_landing.html` tokens (amber accent, warm dark
  ground, serif/sans/mono) → the app theme, both light/dark.

## P0 — Contract & harness (de-risk before infra)
- [ ] T0.1 Add a global top-K album cap over `select_best`'s per-cluster picks (no global K today) + test.
- [ ] T0.2 Define `Album`/`Pick`/`Person` TS types (see `plan.html` §7); write the normalizer from
      `context.selected_images`/`people_clusters`/`scene_clusters` + the `RunExporter` v5 dir → `album.json`.
- [ ] T0.3 Headless script: folder → `execute_spec(isolate_steps=True)` → `album.json`, run on all 3 real trips.
- **Gate:** one command turns a folder into a valid `album.json` matching the showcase shape.

## P1 — Containerized worker + durable queue
- [ ] T1.1 Package pipeline + models into a worker image on the chosen host (Modal recommended).
- [ ] T1.2 Durable queue + job store; worker pulls jobs, calls `execute_spec`, emits step-complete events, owns persistence.
- [ ] T1.3 Adversarial memory test: max-N (150) upload under `isolate_steps` → bounded peak RSS, no OOM; a killed step → clean job failure.
- **Gate:** hand-enqueued 150-img job → album artifacts in storage; RSS bounded; step-kill fails cleanly.

## P2 — Control-plane API
- [ ] T2.1 `POST /jobs` (bounds guard → presign upload → enqueue), `GET /jobs/{id}` (status+progress), `GET /jobs/{id}/result`.
- [ ] T2.2 Progress relay (SSE or poll) sourced from worker step events.
- [ ] T2.3 Bounds guard: count/size/MIME + decode probe, rate limit, TTL/expiry; typed rejections (413/415/429) not 500s.
- **Gate:** curl an upload end-to-end; every bound returns a clean typed rejection.

## P3 — Next.js front end
- [ ] T3.1 `create-next-app` (TS, App Router); port `albumify_landing.html` tokens; both themes; `prefers-reduced-motion`.
- [ ] T3.2 Dropzone with client-side bounds; live progress (four passes); album + people + why-picked viewers.
- [ ] T3.3 A designed screen for every job state: queued / rejected / timeout / failed / expired (AC3).
- **Gate:** a real browser upload renders an album; all states designed.

## P4 — Showcase seed
- [ ] T4.1 Ingest the 3 precomputed trips (`D:\albumify_vs_vlm\{budapest,austria,germany}\`) into `album.json` shape as static seed.
- [ ] T4.2 "How it works" (four-pass explainer) + honest vs-VLM panel with the **required** "preliminary pilot, N=1" caveat.
- **Gate:** first paint shows curated trips instantly; live + showcase share one component set.

## P5 — Harden & load-test
- [ ] T5.1 Cold-start budget; concurrency test (queue-backlog UX); adversarial uploads (HEIC, oversized, zero-faces).
- [ ] T5.2 Cost/budget alarm; privacy pass (verify TTL purge of uploads + results).
- [ ] T5.3 Responsive + perf pass; no horizontal body scroll; album interactive fast.
- **Gate:** AC1–AC10 hold on the public URL from a device that isn't Jonathan's.

## P6 — Review & ship
- [ ] T6.1 `/code-review` → `REVIEW.md`; resolve High findings.
- [ ] T6.2 CHANGES_LOG + README (run + deploy each one command); flip spec to Implemented.
- **Gate:** public URL linkable from a resume; REVIEW.md clean.

## Explicitly out of scope (v1)
Accounts / login / persistence beyond a TTL'd anon session · billing · GPU (CPU ships v1; GPU is a P5
optimization) · any change to the Streamlit apps or the engine beyond T0.1.
