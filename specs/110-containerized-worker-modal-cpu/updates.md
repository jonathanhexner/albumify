# spec-110 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Draft — nothing built or verified. Blocked on the owner running `modal setup` (T0.1).

## Purpose
Prove the serving data path on CPU before touching GPU: image build with the full dependency set,
code mount, storage-reference `curate(job_id)`, valid `album.json` out, idempotent re-run.

## Verification plan (acceptance criteria)
- AC1 `gpu_smoke` returns from a Modal run.
- AC2 pipeline imports cleanly inside the container.
- AC3 a small real trip (≤40 images staged in storage) produces a valid `album.json`.
- AC4 same `job_id` twice → identical result.
- AC5 model weights reused from the Volume across runs.

## Timeline
- 2026-10-05 — spec created (split out of spec-109).

## Reports & artifacts
Nothing run yet. Background reading: [Modal explainer](../109-parallel-gpu-execution/modal_explainer.html) ·
[Modal Q&A](../109-parallel-gpu-execution/modal_q_and_a.html) · worker scaffold `sim_bench/serving/modal_app.py` ·
dependency pitfalls recorded in LEARNINGS (2026-07-13 dry-run gate, 2026-10-02 VLM env isolation).
