# Albumify — Roadmap

**This is the single roadmap file. Read it first every session; every spec below links back here.**
Last revised: 2026-10-09, reconciled to the saved CPU/GPU runs and current worker code. Links point to rendered HTML. **Open scope decisions still await the owner's revision.**

Readable view: `ROADMAP.html` (generated — run `.venv/Scripts/python scripts/md2html.py ROADMAP.md` after editing this file).

---

## Goal

Ship Albumify as a finished portfolio piece: a public URL where a stranger uploads a bounded set of photos and gets back a good, honestly-framed curated album, live, from the real pipeline on a scale-to-zero GPU worker.

Two kinds of "done": **it runs reliably** (engineering, known path) and **the picks are good** (the bet, must be measured).

## Stages and gates

```
STAGE A  prove the picks ──► GATE 1 (go/no-go: quality)
STAGE B  finalize engine ──► gate: folder -> album.json, one command
STAGE C  build service   ──► blocked until host + content-safety decided
STAGE D  harden + ship   ──► SHIP GATE: cold public URL, stranger upload, AC1-AC10
         narrative (parallel, low effort)
```

| Stage | What | Gate | Status (2026-10-09) |
|---|---|---|---|
| A — Prove the picks | spec-102 real trips (Slovakia26, Budapest, Austria, Germany), judged against the owner's keeper list | GATE 1: "would I show these picks to a recruiter" | **No human verdict.** Harness and trip reports exist; PhotoTriage/CUFED are not this gate (owner decision 2026-10-09). |
| B — Finalize the engine | global top-K cap; fix only what A flags; lock production profile; album.json contract | one command: folder → valid album.json, predictable size | Partly done ahead of A: album.json normalizer (106 P0), triage (107), semantic tags + theme (108 P0/P1). Top-K cap not built. |
| C — Build the service | containerized worker (110) → GPU (109) → control-plane API + job state machine → Next.js front end → showcase seed | host + content-safety decided | Modal auth and L4 smoke passed; full CPU/GPU pipeline ran on 12 real photos with matching picks/theme. **2026-10-09:** storage-reference `curate_job(job_id)` + persisted normalized `album.json` built; **AC3 met on Modal** (48-photo GPU run: 13 picks, 25 dropped-with-reason, 7 scenes, full theme). Idempotence (AC4) failed on that run only on pick order — cause fixed (SIGHTING-126), one re-run closes it. Open: AC5 model-cache reuse, persistent deployment, and SIGHTING-128 (album omits photos dropped before scoring). API and front end not started. |
| D — Harden, polish, ship | adversarial uploads, cold-start budget, cost alarm, TTL privacy, both themes, AC1–AC10 on public URL, REVIEW.md | SHIP GATE | Not started. |
| Narrative (parallel) | "how it works" explainer, honest vs-VLM panel (N=1 caveat), short technical write-up | — | Not started. |

## Current position

- Last commit 2026-10-09: roadmap and rendered documentation. Latest worker fixes and run artifacts are still uncommitted.
- The last week worked Stages B and C (and the v1.5 agentic layer) **ahead of Stage A**. Gate 1 is still open.
- **Modal account blocker cleared:** auth, billing, image imports, and L4 smoke passed. Owner chose the shared GPU image for spec-110 acceptance, with CPU proof retained.
- Saved GPU retry: **142.186 s pipeline / 313.8 s driver wall**, four picks matching CPU; face detection + embeddings consume **101.850 s (72%)**. This 12-photo result does not establish the 100/300-photo latency gate.
- Next worker work (owner action): re-run `scripts/modal_curate_driver.py --src D:/Slovakia26 --start 300 --n 48 --job-id slovakia48 --twice` to close spec-110 AC4 (the ordering fix landed after the 2026-10-09 run), then once more to close AC5 (models must not re-download). Then profile face-model startup versus inference and benchmark 100/300 photos and cost. Built since: storage-reference `curate_job(job_id)` → normalized persisted `album.json`, Volume-backed model caches, and the local no-spend proof path (`--local-root`). See [109 updates](specs/109-parallel-gpu-execution/updates.html) and [110 tasks](specs/110-containerized-worker-modal-cpu/tasks.html).
- Other unblocked engineering: spec-106 P3 (control-plane API + job state machine). Spec-111 T1.4 remains a v1.5 task pending the scope decision.

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Quality bar for Gate 1 | Open — agree the bar before running Stage A |
| 2 | Worker host | **Decided 2026-10-04: Modal** (CPU first via spec-110, GPU via spec-109) |
| 3 | Content-safety stance for anonymous uploads | Open — rec: minimal ToS + short TTL for v1 |
| 4 | Agentic layer in v1 or v1.5 | Drifted: spec-111 P0/P1 built 2026-10-05. Needs an explicit call. |
| 5 | MVP scope: anonymous, bounded, TTL'd, no accounts | Open — rec: yes |
| 6 | GPU in v1 | **Reversed 2026-10-04:** CPU baseline 9.0 s/img makes GPU mandatory (was "WON'T v1") |
| 7 | Live DB + storage | **Decided 2026-10-10: Neon Postgres (free) + Cloudflare R2**; where the pipeline runs (local / Modal GPU / Modal CPU) is a config setting (spec-112) |

## Scope tiers

- **MUST (v1):** reliable live curation on bounded upload · quality proven · deployed URL with every job state designed · honest framing.
- **SHOULD (v1):** people view · why-picked score breakdown · showcase seed trips · "how it works" explainer.
- **COULD (v1.5):** AI verification layer (111) · album-type tuner · captions (108 P2) · thematic grouping.
- **WON'T (v1):** accounts · billing · live model training · multi-user persistence.

## Specs (PRDs)

| Spec | Title | Stage | Status | Progress & evidence |
|---|---|---|---|---|
| [102](specs/102-albumify-vs-vlm/spec.html) | Albumify vs VLM blind A/B | A | In Progress — harness done, no verdict | [updates](specs/102-albumify-vs-vlm/updates.html) · reports: [Budapest](reports/2026-07-17_albumify_vs_vlm_budapest/report.html) · [Austria](reports/2026-07-18_albumify_vs_vlm_austria/report.html) · [Germany](reports/2026-07-18_albumify_vs_vlm_germany/report.html) · [Slovakia26](reports/2026-10-01_slovakia_analysis/report.html) |
| [104](specs/104-staged-process-isolation/spec.html) | Staged process isolation | B (enabler) | Implemented v1, merged to main 2026-09-28 | [updates](specs/104-staged-process-isolation/updates.html) · [REVIEW](specs/104-staged-process-isolation/REVIEW.html) · [capacity report](reports/2026-09-21_spec104_capacity/report.html) |
| [107](specs/107-input-triage-discard-review/spec.html) | Input triage + discard review | B | Implemented P0–P2 (2026-10-01); P3/P4 open | [updates](specs/107-input-triage-discard-review/updates.html) · [REVIEW](specs/107-input-triage-discard-review/REVIEW.html) · [Slovakia26 report](reports/2026-10-01_slovakia_analysis/report.html) |
| [108](specs/108-album-semantic-layer/spec.html) | Album semantic layer | B / v1.5 | P0–P1 done; P2 code-complete, live VLM awaits GPU | [updates](specs/108-album-semantic-layer/updates.html) · [REVIEW](specs/108-album-semantic-layer/REVIEW.html) · [RAM++ vs CLIP](reports/2026-10-01_ram_vs_clip_tagging/report.html) |
| [106](specs/106-albumify-web-showcase/spec.html) | Albumify Live Curator (the product) | C–D | In Progress — P0 done; [architecture](specs/106-albumify-web-showcase/architecture.html) · [plan](specs/106-albumify-web-showcase/plan.html) · [tasks](specs/106-albumify-web-showcase/tasks.html) | [updates](specs/106-albumify-web-showcase/updates.html) |
| [110](specs/110-containerized-worker-modal-cpu/spec.html) | Containerized worker on Modal (CPU) | C | In Progress — AC1-AC3 met (AC3 on a 48-photo Modal GPU job run); AC4 needs one re-run after the SIGHTING-126 fix; AC5 unproven; SIGHTING-128 open | [updates](specs/110-containerized-worker-modal-cpu/updates.html) |
| [112](specs/112-live-db-storage-runner/spec.html) | Live DB + storage, runner as config (Neon + R2) | C | In Progress — Phase A A1–A5 built 2026-10-10, real-data proof pending; Phase A local (SQLite + folder), Phase B Neon + R2 by config; photos/faces/people/scenes as DB rows; takes spec-106 T2.1 + DB half of T3.1 | [updates](specs/112-live-db-storage-runner/updates.html) · [tasks](specs/112-live-db-storage-runner/tasks.html) |
| [113](specs/113-remove-cluster-people/spec.html) | Remove the deprecated `cluster_people` step | B (clean-up) | Implemented 2026-10-18 — removal + regression suite (4 albums PASS), review ACCEPT, Budapest gate 11/11 | [updates](specs/113-remove-cluster-people/updates.html) · [tasks](specs/113-remove-cluster-people/tasks.html) |
| [114](specs/114-compare-jobs-view/spec.html) | Compare two jobs in the app (regression results, CPU vs L4, before/after) | C | Implemented 2026-10-17 — Compare runs page; regression runs in the app's job list; review ACCEPT (F1–F6 TODO) | [updates](specs/114-compare-jobs-view/updates.html) · [tasks](specs/114-compare-jobs-view/tasks.html) |
| [109](specs/109-parallel-gpu-execution/spec.html) | GPU enablement + parallelism | C | In Progress — 12-photo GPU run matches CPU picks/theme; face stack dominates; full-size gate pending | [updates](specs/109-parallel-gpu-execution/updates.html) · [CPU baseline](reports/2026-10-04_spec109_baseline/summary.html) · [Modal explainer](specs/109-parallel-gpu-execution/modal_explainer.html) |
| [111](specs/111-ai-verification-layer/spec.html) | AI verification layer | v1.5 | In Progress — P0 + P1 core built 2026-10-05 (owner call pending on v1 vs v1.5) | [updates](specs/111-ai-verification-layer/updates.html) |
| [105](specs/105-vlm-merge-arbiter/spec.html) | VLM merge arbiter | — | Superseded by 111 checkpoint #1 | [updates](specs/105-vlm-merge-arbiter/updates.html) |
| [103](specs/103-scene-clustering-geo-time-fusion/spec.html) | Scene clustering geo/time fusion | B (optional) | In Progress — gated off by default | [updates](specs/103-scene-clustering-geo-time-fusion/updates.html) · [fusion report](reports/2026-07-21_scene_clustering_fusion/report.html) |

## Definition of done

- Public URL loads cold and works first try on a device that isn't the owner's.
- A stranger's bounded upload returns a good curated album, live.
- Curation quality validated with a number and honest framing.
- Every job state designed: queued / rejected / timeout / failed / expired.
- Never OOMs under adversarial input; spend is capped.
- AC1–AC10 (spec-106 plan.html §11) hold on the deployed URL.
- "How it works" + honest vs-VLM narrative live.
- REVIEW.md clean · CHANGES_LOG · README with one-command run and deploy.
- v1.5 ideas parked here, not half-built in v1.

## Risks

| Risk | Level | Mitigation |
|---|---|---|
| Scope creep (every subsystem has a rabbit hole) | highest | Freeze v1 scope; every task must answer "does this serve shipping?" |
| Quality shortfall on fresh photos | high | Stage A first; targeted fix to the weak pass, not a rebuild |
| First-run trust (slow/failed first upload) | med | Showcase seed + honest progress + cold-start budget |
| Cost blowup on a public endpoint | med | Scale-to-zero + caps + TTL + budget alarm |
| Solo bandwidth | med | Strict critical-path order; defer v1.5 |

## Related trackers

- `TODO.md` — task-level items and review follow-ups.
- `docs/project/SIGHTINGS.md` — open bugs.
- `CHANGES_LOG.md` — every change, newest first.
