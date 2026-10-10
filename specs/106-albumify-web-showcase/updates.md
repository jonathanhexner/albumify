# spec-106 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [architecture.html](architecture.html) · [plan.html](plan.html) · roadmap stage: **C–D (the product)** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — P0 done; P1 L4 smoke and inline-byte pipeline proof passed. Storage-reference result, full-size latency and cost remain open.

## 2026-10-09 — The album contract gained an ordering guarantee (from spec-110)

`normalize()` now owns ordering, because the worker's two-run proof showed album.json was not
reproducible (SIGHTING-126): scenes are renumbered `0..n-1` by their earliest photo, picks are
ordered by EXIF capture time (falling back to the id, so `Pick.order` really is the album's reading
order), and `beat` / `beaten_by` are sorted. The pipeline's own `scene_cluster` ids and
`selected_images` order follow an embedding-dict order that varies between runs, so the front end
must not treat them as identifiers — the album's numbering is the stable one. Scene 0 is the trip's
first scene.

| How it was verified | Evidence |
|---|---|
| Order-insensitivity of the contract | `tests/album/test_normalizer.py::ut_AlbumDeterminism` — 6 tests (permuted cluster ids + reversed selected order → identical album; EXIF-time ordering; noise stays -1; duplicate in `selected_images` picked once) |
| Real-data album through the worker path | [local job report](../../reports/_local_job_localproof12.json) — 4 picks with score breakdowns + captions, 8 dropped with reasons, displays written |


## 2026-10-09 — Follow-up: Modal blocker cleared, production seam still incomplete
- Modal auth/billing, image imports, L4 smoke, and CPU/GPU 12-photo full-pipeline runs passed.
- Saved CPU/GPU outputs match on input filenames, pick filename sets, theme and set_aside.
  GPU: 142.186 s pipeline / 313.8 s driver wall. This does not establish the 100-photo latency gate.
- `_curate` returns an inline-byte pipeline summary; the normalizer is still not invoked by the worker.
  Storage-reference input, normalized persisted album.json/displays, idempotence and complete cache reuse
  remain spec-110 work before the control plane consumes the worker.
- Verified: saved [CPU result](../../reports/_modal_curate_last.json) and
  [GPU result](../../reports/_modal_curate_gpu12_retry2.json), worker source inspection, and 37 local
  focused regression tests (including the six normalizer tests); one slow test deselected.

## Progress by phase
| Phase | State | Verified by |
|---|---|---|
| P0 album.json normalizer | **Done 2026-10-05** | `tests/album/test_normalizer.py` (6 tests): pure classify, synthetic-context normalize, straighten-derivative mapping, JSON round-trip. **Real-data validation pending** the worker run |
| P1 prove the GPU seam | Smoke + inline-byte real-data proof passed; storage result, latency/cost gate pending | [109 updates](../109-parallel-gpu-execution/updates.html), [110 updates](../110-containerized-worker-modal-cpu/updates.html); GPU 12-photo result matches CPU picks/theme |
| P2 storage + presigned upload | Not started | — |
| P3 control-plane API + job state machine | Not started (unblocked, no Modal needed) | — |
| P4 Next.js front end | Not started | — |
| P5 showcase seed | Not started | — |
| P6 harden & ship | Not started | — |

## Design decisions on record (architecture.html)
- Four planes with a trust boundary: untrusted browser → FastAPI control plane → Modal worker → storage.
- Warm-up fires when upload starts so the cold start hides under upload time.
- Resolution tiers: 768px for analysis and review, ~1600 for keeper display, originals download-only.
- Show-everything: every photo visible with a drop reason and promotable; nothing silently deleted.
- Secrets only on trusted hosts (Vercel env, Modal Secret); browser gets short-lived presigned URLs.
- The job is a small event-driven state machine written to DB on every stage and step transition;
  the pipeline executor is reused as-is.
- **GPU is mandatory for v1** (reversed 2026-10-04 after the 9.0 s/img CPU baseline).

## Timeline
- 2026-10-05 — P0 normalizer built + 6 tests.
- 2026-10-04 — architecture.html (design of record) incl. §09 Observability & Durability; tasks.md rewritten to the build order.
- 2026-10-01 — finalization_plan.html (now superseded as roadmap-of-record by `ROADMAP.md`).
- 2026-09-2x — rewritten to Direction B (Live Curator) + plan.html.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [architecture.html](architecture.html) | design of record: four planes, job lifecycle, warm-up parallel to upload, resolution tiers, show-everything, secrets, telemetry, resume |
| [plan.html](plan.html) | detailed build plan, bounded-input contract, cost, AC1–AC10 (§11) |
| [direction_decision.html](direction_decision.html) | why Direction B (Live Curator) over the precomputed showcase |
| [agentic_layer_design.html](agentic_layer_design.html) | pre-spec design note for the AI layer (became spec-111) and candidate eval datasets |
| [finalization_plan.html](finalization_plan.html) | 2026-10-01 critical path; superseded as roadmap-of-record by [ROADMAP](../../ROADMAP.html) |
| Showcase seed inputs | the spec-102 trip reports: [Budapest](../../reports/2026-07-17_albumify_vs_vlm_budapest/report.html) · [Austria](../../reports/2026-07-18_albumify_vs_vlm_austria/report.html) · [Germany](../../reports/2026-07-18_albumify_vs_vlm_germany/report.html) |
| Runtime evidence behind "GPU mandatory" | [spec-109 CPU baseline](../../reports/2026-10-04_spec109_baseline/summary.html): 9.0 s/img |
