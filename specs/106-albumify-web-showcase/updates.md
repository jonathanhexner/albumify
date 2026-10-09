# spec-106 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [architecture.html](architecture.html) · [plan.html](plan.html) · roadmap stage: **C–D (the product)** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — build started 2026-10-04. P0 done; P1 blocked on Modal auth.

## Progress by phase
| Phase | State | Verified by |
|---|---|---|
| P0 album.json normalizer | **Done 2026-10-05** | `tests/album/test_normalizer.py` (6 tests): pure classify, synthetic-context normalize, straighten-derivative mapping, JSON round-trip. **Real-data validation pending** the worker run |
| P1 prove the GPU seam | Blocked — needs owner's `modal setup` | — |
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
