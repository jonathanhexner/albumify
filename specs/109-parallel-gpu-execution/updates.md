# spec-109 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [modal_explainer.html](modal_explainer.html) · [modal_q_and_a.html](modal_q_and_a.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — re-scoped 2026-10-05 to GPU enablement + parallelism only (spec-110 proves the worker first).

## How it was verified so far
| Item | Evidence |
|---|---|
| T0.1 CPU baseline of the REAL default pipeline (38 steps) | `reports/2026-10-04_spec109_baseline/`: **9.0 s/img** on 57 Slovakia images; RAM++ triage 3.0 s + GeoCalib tilt 2.6 s = 62%, face stack 2.0 s = 23% |
| T0.2 "defer face embeddings" idea | Rejected with evidence: `select_best` defaults to face sub-clusters, which need embeddings on all faces |
| P2 Modal worker scaffold (`sim_bench/serving/modal_app.py`) | Constructs against modal 1.6.1 (import verified). **Not deployed, not run** — needs owner's Modal account |

## Decisions
- Host = Modal (owner approved 2026-10-04). L4 GPU, ~1–2 cents per album, $0 idle.
- Whole pipeline runs on one GPU box; CPU-only steps are ~2% of time, not worth splitting.
- Storage-reference input and the warm-up trigger moved to spec-110 / spec-106.

## Open
- Everything after the scaffold is blocked on `modal setup` by the owner, then spec-110's CPU proof.
- Per-step device=cuda verification (MediaPipe may be CPU-only).

## Timeline
- 2026-10-05 — re-scoped to GPU-only; CPU proof split into spec-110.
- 2026-10-04 — baseline report; Modal approved; worker scaffolded; two explainer HTMLs.
- 2026-10-01 — PRD created from the Slovakia26 findings.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [CPU baseline summary](../../reports/2026-10-04_spec109_baseline/summary.html) | full 38-step pipeline, 57 Slovakia imgs: 9.0 s/img; RAM++ 3.0 s + GeoCalib 2.6 s = 62%, face stack 2.0 s = 23%. Raw per-step numbers in `timing.json` (timing only, no image report) |
| [Slovakia26 analysis](../../reports/2026-10-01_slovakia_analysis/report.html) | the earlier lean `faces` number (2.2 s/img, face stack 75%) this baseline corrected |
| [Modal explainer](modal_explainer.html) | what Modal is, our architecture and implementation, deploy lifecycle, cost table |
| [Modal Q&A](modal_q_and_a.html) | whole pipeline on one GPU box; storage-reference input; reactive warm-up |
| Worker scaffold | `sim_bench/serving/modal_app.py` + `README.md` (commands: setup → smoke → volume put → curate → deploy) |
