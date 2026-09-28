# spec-106 — Albumify Live Curator (the launchable product)

**Status:** Draft — awaiting approval
**Owner:** Jonathan / Claude
**Branch:** `spec/106-albumify-web-showcase`
**Direction:** **B — Live Curator** (settled; see `direction_decision.html`). Supersedes the earlier
Direction-A "precomputed showcase" draft of this spec.
**Full plan (rich diagrams):** [`plan.html`](plan.html) — architecture, job lifecycle, memory model,
hosting, cost, phased roadmap, risks. This spec.md is the summary; `plan.html` is the source of truth.

---

## 1. Problem

Albumify's engine is real, benchmarked, and green in CI, but it ships as a research monorepo, not a
product. The landing page already promises the live product — *"Bring a trip. Leave with an album."*
We need to actually deliver it: a stranger uploads their own trip and gets a curated, face-tagged,
story-ordered album back, from the real 28-step pipeline running live — **without ever OOM-ing,
hanging, or showing a stack trace.**

The old draft dodged this by shipping *precomputed* trips (Direction A). That's a case study, not the
"0→100, launched a service" signal the portfolio needs. This rewrite commits to the live service.

## 2. Goal / non-goals

**Goal:** A deployed, product-grade web app at a public URL where anyone can upload a bounded trip and
receive a live-curated album (picks + why + people), plus the three precomputed trips as instant
showcase seed and an honest "how it works / vs a frontier VLM" narrative.

**Non-goals (v1):** no accounts / login / persistence beyond a TTL'd anonymous session; no billing; no
GPU requirement (CPU ships v1); no change to the existing Streamlit apps; not a rewrite of the engine.

## 3. What makes this buildable now

**spec-104 process isolation (merged to main).** Each pipeline step runs in a spawned child that exits
and returns 100% of its model memory to the OS, with a per-step wall-clock timeout. Measured capacity
(spec-104 report): peak RSS is model-floor bound (~2.5 GB + ~0.67 MB/img), runtime-bound at ~1.6 s/img
CPU, ~4 GB/worker. That turns "a live pipeline is reckless" into "a live pipeline is bounded."

## 4. Architecture (summary — see `plan.html` §3)

```
Next.js (Vercel, static/edge)  →  FastAPI control plane (bounds guard, /jobs)
   →  durable queue + job store  →  scale-to-zero worker pool
        [execute_spec(spec, ctx, isolate_steps=True)]  →  object storage  →  viewer
```

Reused: the 28-step engine, executor + isolation, `execute_spec`/`run_pipeline`, `RunExporter`, result
schemas. New: durable queue (replaces the in-process `_jobs` dict + `BackgroundTasks`), bounds guard,
worker packaging, storage, and the Next.js front end.

## 5. Acceptance criteria (AC1–AC10)

Full text in `plan.html` §11. In brief: loads first try (AC1); a real ≤150-photo upload returns a live
album (AC2); every job state designed — queued/rejected/timeout/failed/expired (AC3); bounds enforced
server-side with clean rejections (AC4); never OOMs under adversarial N (AC5); truthful progress (AC6);
data honesty incl. N=1 VLM caveat + no determinism over-claim (AC7); TTL privacy (AC8); responsive +
themed + fast (AC9); idle-cheap + spend-capped (AC10). All verified on the public URL from a device
that is not Jonathan's.

## 6. Bounded-input contract (proposed defaults — see `plan.html` §6)

Max 150 photos / 25 MB per file / 1.5 GB total; JPEG·HEIC·PNG (decode-probed); per-step timeout 600 s;
rate limit per IP/hour; upload TTL 1 h, result TTL 24 h. Server-enforced before any compute.

## 7. Phased roadmap (see `plan.html` §10)

P0 contract & harness (incl. new global top-K selection cap) · P1 containerized worker + durable queue ·
P2 control-plane API · P3 Next.js front end · P4 showcase seed · P5 harden & load-test · P6 review & ship.
Each phase ends green and standalone.

## 8. Open decisions (blocking P0 — see `plan.html` §13)

Host (Modal recommended) · input cap · content-safety stance · GPU now/later · accounts (assume none) ·
album size K. Proposed defaults let me proceed on everything except host + safety stance.

## 9. Definition of "shipped"

Deployed public URL meeting AC1–AC10, `REVIEW.md` via `/code-review` with no unresolved High findings,
CHANGES_LOG + README (run + deploy in one command each), URL linkable from a resume.
