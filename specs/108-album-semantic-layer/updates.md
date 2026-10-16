# spec-108 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [REVIEW.md](REVIEW.html) · roadmap stage: **B / v1.5** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — P0 + P1 Implemented; P2 code-complete, awaiting live VLM on the GPU worker.

## How it was verified
| Increment | Evidence |
|---|---|
| P0 semantic tags (reuses cached RAM++ tags, zero added model cost) | 10 tests; real Slovakia subset gives a sensible family-trip facet mix |
| P1 tag-based album theme (no model) | 7 tests; real Slovakia → "family trip — towns & nature" (people 34 / nature 20 / urban 11 / food 1) |
| P2 per-keeper + album captions via the VLM seam | 12 tests with a fake provider (26 semantic tests total); seam proven with `--stub`. **No live caption yet** |
| MiniCPM-V 4.5 provisioning | Loads in isolated `.venv-vlm` (`ready: true`), but a single CPU caption never completed. LEARNINGS 2026-10-02 |
| Code review | REVIEW.md: P0 + P1 ACCEPT; P2 ACCEPT as code-complete, not Implemented |

## Decisions
- VLM = MiniCPM-V 4.5, local (no per-token cost, photos never leave the box). Runs on the spec-109/110
  Modal worker, not the dev box.
- Caption steps are opt-in (`enabled: false`) so model-less environments are unaffected.

## Open
- T3 (TODO.md): live caption E2E on the GPU worker gates P2 → Implemented.
- `album_theme` two-writer contract (tags vs VLM) to be documented (T1); architecture HTML drift (T2).
- P3 theme-relevance and P4 not started.

## Timeline
- 2026-10-02 — REVIEW.md; VLM cache moved off C:; CPU inference abandoned for the GPU worker.
- 2026-10-01 — P0, P1, P2 built.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [RAM++ vs CLIP tagging](../../reports/2026-10-01_ram_vs_clip_tagging/report.html) | why the semantic tags are RAM++-based: names the real subject (waffle/syrup, mallard duck) where CLIP forces wrong buckets |
| [CLIP tagging method](../../reports/2026-10-01_clip_tagging_method/report.html) | per-category softmax fixes CLIP scene reads (landmark 99%, group 85%); kept as the near-free fallback |
| [RAM++ for defect detection](../../reports/2026-10-01_ram_defect_detection/report.html) | RAM++ tags carry no quality signal → semantic layer stays separate from IQA/AVA scoring |
| [Slovakia26 analysis](../../reports/2026-10-01_slovakia_analysis/report.html) | the real-tag subset the P0 facets and P1 theme ("family trip — towns & nature") were validated on |
| [REVIEW](REVIEW.html) | P0/P1 ACCEPT; P2 code-complete |
