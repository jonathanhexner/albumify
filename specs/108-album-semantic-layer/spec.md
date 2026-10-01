# spec-108 — Album Semantic Layer (theme · captions · semantic scene clustering)

**Status:** Draft — awaiting approval
**Owner:** Jonathan / Claude
**Priority:** P2 (quality depth + the agentic-AI capability; v1.5 of the product)
**Relation:** produces the interestingness signal consumed by [[spec-107]]; shares VLM infra with
[[spec-105]] (VLM merge arbiter); builds on the model-to-task findings in
`specs/106-albumify-web-showcase/agentic_layer_design.html`.

## 1. Problem

Curation today judges **individual photos technically** (sharpness, aesthetics) but has no grasp of:
1. **The album's theme** — is this a family trip, a hike, a city break? (album-level, not per-photo.)
2. **A photo's interestingness / relevance to that theme** — a sharp rental-car or store-product shot
   scores *high* on quality yet isn't a *moment*. Measured leaks on Slovakia26: a car+license-plate
   shot, a rack of CDs, a construction worker on a ladder all reached the top-14.
3. **Human-readable captions** — no album caption, no per-photo "what this is / why picked" text.

Separately, scene clustering is **purely visual** (DINOv2); semantic tags could optionally improve it
(thematic grouping across moments).

These are one coherent gap: **semantic, album-level understanding** — the content axis that RAM++
(validated 2026-10-01) and a VLM are the right tools for.

## 2. Goal / non-goals

**Goal:** an album-level semantic layer that (a) infers the trip theme, (b) captions the album and the
keeper photos, (c) scores each photo's theme-relevance/interestingness (feeding spec-107's set-aside),
and (d) optionally augments scene clustering with semantic signal.

**Non-goals:** replacing visual dedup (DINOv2 stays the backbone — captions *over-merge* dedup, proven
in the design note); running a VLM on every image (cost) — VLM only on a sample / survivors / ambiguous
cases (the escalation pattern); auto-deleting "uninteresting" photos (always set-aside-for-review).

## 3. Scope

- **Theme inference** — a VLM reads a representative sample of the roll → structured theme
  (`{type: family-trip, place: Slovakia, motifs: [kids, landscapes, towns]}`).
- **Captioning (the captioning request)** — album-level caption + per-keeper captions (Tier-2, ~20
  images, cheap); powers "why picked", scene labels ("Day 3 — Spiš Castle"), and search text.
- **Theme-relevance / interestingness score** — per photo, relevance to the inferred theme → low-score
  utilitarian shots flow to spec-107's set-aside (review, not delete).
- **Optional semantic scene-clustering boost** — cheap RAM++/CLIP tags (content axis) as an auxiliary
  signal to the visual clusterer; opt-in, measured against pure-visual dedup (must not regress).

## 4. Approach (escalation, not orchestration)

Cheap signals first: RAM++ content tags on all images (near the resident-model cost). A VLM (local open
— Qwen3-VL / MiniCPM-V, or an API) runs only on: a sample (theme), the ~20 keepers (captions), and the
ambiguous "real-but-maybe-utilitarian" subset (interestingness). Never per-image on the full roll.
Human-in-the-loop for the subjective calls.

## 5. Acceptance criteria

- **AC1** Theme inferred correctly on the 3 real trips (+ Slovakia26), human-judged.
- **AC2** Keeper captions are accurate and useful (spot-checked); album caption reads true.
- **AC3** Theme-relevance flags ≥ the known utilitarian leaks (car/CDs/ladder) with low false-positive
  rate; output feeds spec-107 set-aside.
- **AC4** Semantic scene-clustering boost is opt-in and **does not regress** duplicate-survival vs
  visual-only (measured).
- **AC5** Token/compute cost is bounded (sample + survivors + ambiguous only); measured $/album.

## 6. Dependencies & risks

- RAM++ (content tags, validated); a VLM (local open preferred for cost/privacy — see agentic design);
  spec-107 consumes the relevance signal; spec-105 shares VLM infra.
- Risk: interestingness is **subjective** → set-aside-for-review, conservative, never auto-delete.
- Risk: VLM cost/latency → scoped to sample/survivors/ambiguous; local model to kill per-token cost.
- Risk: distillation trap (over-tuning to a VLM) → VLM is a reference/judge, not a training target
  (per agentic design §AI-as-reference).

## 7. Definition of done
Theme + captions + relevance implemented behind flags; AC1–AC5 met on real trips; opt-in clustering
boost measured; REVIEW.md; CHANGES_LOG. See `tasks.md`.
