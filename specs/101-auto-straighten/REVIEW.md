# REVIEW — spec-101 (auto-straighten, early roll correction + inscribed crop)

**Reviewed**: 2026-07-17 · **Scope**: the straighten feature only (T1–T3). T4 (report) not yet built.
**Verdict**: **components ACCEPT; §5 has one FAIL that blocks `Implemented`** — the integrated
pipeline (reorder + image-path rebinding + downstream scoring on straightened images) has not been
run end-to-end on real data, and the step is **on by default**.

---

## Part 1 — How it works

```
score_tilt → detect_persons → straighten_images(EARLY) → score_iqa → … → select_best
                                     │  per image:
                                     │  decide(roll,conf,w,h,person_bbox,cfg)  [straighten_gate.py — pure]
                                     ├─ straighten → straighten(rgb,roll) [straighten.py] → derived JPEG
                                     │               → rebind context.image_paths / active_images
                                     │               → straightened_from[derived]=orig
                                     └─ decline/not-tilted → keep original path (tilt_penalty demotes it)
```

**New:** `straighten.py` (geometry, 122−? — 3 pure fns), `straighten_gate.py` (65 LOC, `decide`),
`straighten_images.py` (122 LOC, the step). **Edited:** `context.py` (+`straightened_from`),
`all_steps.py`, `configs/pipeline.yaml` (reorder + config), spec-034 contract, classes/data_flow HTML.
**Dependency direction:** pipeline → quality_assessment (no reverse imports).

## Part 2 — Findings

**§1 Structure** — `pass-with-followup` (low). (a) `decide()` takes 6 positional params (roll, conf,
w, h, person_bbox, cfg) — over the >4 guideline; group `(w,h)` + `(roll,conf)` or pass a small
request object. (b) `straighten_images.py` is 122 LOC vs the ≤80 thin-step convention — justified
(it's a transform, not a scorer; gate+rotate logic lives in helpers) but note it.

**§2 Code quality** — `pass`. `except Exception` sites wrap real I/O (`Image.open`) and degrade to
passthrough (fail-safe). No silent boundary defaults on load-bearing values.

**§3 Naming** — `pass`. `straighten` / `straighten_gate` / `straighten_images` consistent; `__all__`
omitted per project convention (registration via `@register_step` import).

**§4 Layering & coupling** — `pass-with-followup` (**medium, important**). `context.image_paths` was
effectively **write-once** (`discover_images`); `straighten_images` now **rebinds** it mid-pipeline.
Two consequences to document/handle: (1) name the two-writer contract (discover produces;
straighten_images may rewrite, once, early). (2) **Any consumer that persists or displays image paths
downstream must consult `straightened_from`** — otherwise it will treat a `~/.sim_bench/image_cache`
file as the user's original (wrong provenance in DB/export/UI). That consumer wiring is not in this
spec; flagged as a required follow-up before the feature is user-visible.

### UPDATE 2026-07-17 — E2E run (blocker resolution attempt (a)) FOUND A REAL BUG

Ran the reordered pipeline on 2 real images (`discover → score_tilt → detect_persons →
straighten_images → score_iqa`). Result: `success=True` but **score_iqa scored the ORIGINALS, not
the straightened deliverable**. Root cause: `PipelineExecutor` orders steps by **`depends_on` edges
only** (builder `_topological_sort`; `requires`/`produces` do NOT create ordering edges; independent
steps tie-break alphabetically). `straighten_images` has nothing depending on it → it is scheduled
**last** (after every scorer + face detection). The yaml `default_pipeline` order is cosmetic.

**Consequence:** the "early rebind" design cannot work without the downstream image-consumers
(`score_iqa/ava/occlusion`, `insightface_detect_faces`, …) declaring `depends_on:
[straighten_images]`. But auto-resolve then **injects `straighten_images` into every pipeline
containing those steps** — a cross-app side effect. There is no produces/requires-only fix.

**Resolution (user chose option A, 2026-07-17): REDESIGNED to terminal + fixability penalty.**
`tilt_penalty` now scales by fixability (small FOV cost if cleanly straightenable, full angle penalty
if a prominent person would be clipped — via `straighten_gate.decide`), so *selection* accounts for
the crop. `straighten_images` runs **terminal** (`depends_on select_best`) on the winners only —
verified: it schedules last, and `score_iqa` alone resolves to `[discover_images, score_iqa]` (no
drag). **Committed real-data E2E** (`test_straighten_terminal_e2e.py`): landscape winner straightened
& GeoCalib-confirmed level (|roll|<1.5), portrait declines, penalty splits fixable/unfixable. 44
component tests green. **§5 blocker CLEARED.** Re-enabled `enabled: true`.

**§5 Testability** — ~~`fail`~~ → **`pass` (blocker cleared 2026-07-17)** — real-data E2E committed.
Original finding retained above for history. Unit/component coverage is strong: 16 (`straighten`, incl.
brute-force max-area + sentinel-border + real-GeoCalib round-trip), 8 (gate), 4 (step, synthetic).
**But there is no E2E test exercising the *integrated* feature on real data** — the reordered
pipeline running `score_tilt → straighten_images → score_iqa → … → select_best` on real images has
never been executed; verification stopped at component tests + a dependency-graph check + a
scratchpad decline trace. The checklist lists "a new feature with no E2E test on real data" as
high-severity. Also missing as committed tests: the straighten↔penalty interaction (verified inline
only) and a real-data step trace (scratchpad only).

**§6 Boundary contracts** — `pass`. `straightened_from` registered in spec-034 §3 → drift test green.
Config→producer holds (`tilt_angles`←score_tilt, `persons`←detect_persons, both upstream). Frozen
dataclasses (no new Pydantic external boundary).

**§7 Documentation** — `pass-with-followup` (low). spec/tasks/classes/data_flow/contract/CHANGES_LOG
updated. Owed: the spec's own **A7 before/after report** (retained-area stats on real tilts) — T4,
not built.

**§8 Risk register** — `pass-with-followup` (**medium**). The step is **`enabled: true` by default**,
so it reorders the albumify pipeline and runs face detection on straightened images for *every* run —
with no baseline verification that clustering/selection is unchanged. `e2e_budapest` (face_cluster
app) is untouched by this diff (separate pipeline) so that gate is unaffected, but Albumify's own
behavior is not validated. Backwards-compat: first run after enabling re-scores on new derived paths
(cache keys change) — expected, worth noting.

## Part 3 — Verdict & tickets

| Area | Verdict |
|---|---|
| §2, §3, §6 | ACCEPT |
| §1, §4, §7, §8 | ACCEPT with follow-up |
| **§5 Testability** | **FAIL — blocks Implemented** |

**To clear the blocker (pick one):**
- **(a)** Run the albumify pipeline end-to-end on a real folder with `straighten_images` on; add it as
  a committed slow test; confirm downstream steps consume straightened paths and selection is sane.
  Plus a committed straighten↔penalty test. *(recommended)*
- **(b)** Default `enabled: false` until (a) is done — ships the code dark, no behavior change, clears
  the "on-by-default unvalidated" risk.
- **(c)** Hold at `In Progress`.

**Follow-up tickets (to file):**
- TODO — `decide()` param grouping (§1).
- TODO — provenance wiring: DB/export/UI must map derived→original via `straightened_from` (§4) —
  likely its own small spec when the consumer surface is built.
- T4 — before/after report + retained-area stats (§7 / spec A7).
- TODO — architecture/E2E test of the reordered pipeline on real data (§5).
