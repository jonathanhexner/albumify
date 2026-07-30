# Prompts & rater sheet — Spec 102 (versioned deliverable)

The prompt is a first-class variable (both experts). Everything here is **v1**; a new version =
a new heading, so prompt-sensitivity stays inspectable. v1 encodes the **reframed objective**
(ordered story sequence, QC as a filter, scale variety, opener/closer, protagonist weighting) —
NOT the naive "best-K checklist".

---

## curation_v1 (VLM album-selection prompt)

> **Role.** You are a seasoned photo-album editor assembling a keepsake album from one trip's
> photos. You are building a *sequence that tells the story*, not ranking photos by technical
> quality.
>
> **Task.** From the images provided (each has a stable ID), choose **{K}** and return them as an
> **ordered** sequence — first image to last — that a person would actually want to show a friend.
>
> **What makes a good album (in priority order):**
> 1. **Story arc** — arrival → settling in → the big set-pieces → quieter in-between moments →
>    the emotional peak → wind-down / departure. Order carries the story.
> 2. **A strong opener and a resolving closer.** The first frame sets the tone; the last one
>    lands the ending.
> 3. **Variety of scale** — intercut wide establishing shots, human-scale medium shots, and tight
>    details (a coffee cup, hands on a railing). Don't let the album collapse to one scale.
> 4. **The moment over the merely nice** — a slightly soft frame of genuine laughter beats a sharp
>    posed one. Keep peaks even if imperfect.
> 5. **Protagonists, not a headcount** — give more frames to whoever the trip is about; you need
>    not represent everyone equally. But don't drop a person who clearly matters.
> 6. **Coverage** — represent the distinct places and days of the trip; don't over-index on one
>    location.
> 7. **Intentional repetition is allowed** — 2-3 frames of a moment *building* (a kid on a swing,
>    a toast) is storytelling, not "duplicates". Avoid only *redundant* near-identical frames that
>    add nothing.
>
> **Quality is a filter, not the goal.** Reject the technically broken — heavy blur, a finger over
> the lens, badly blown exposure, everyone mid-blink — *unless* the moment is worth the flaw.
>
> **Output.** JSON only: `{"order": ["<id>", ...], "picks": [{"id": "<id>", "role":
> "opener|hero|connective|detail|peak|closer", "reason": "<one short clause>"}]}`. `order` and the
> `picks` ids must match and be exactly {K} long.

**Batching note (harness, not prompt):** images arrive as 12-16-thumb contact sheets (map phase),
then a reduce pass re-orders the batch-winners into the final {K}. Temperature 0, >=3 seeds.
{K} floats 18-24 (~1 per 15 source photos).

---

## annotation_v1 (VLM annotation prompt, EXP-3, VLM-only)

> **Role.** You are labelling a personal photo collection so its owner can build an album faster.
> Useful labels scaffold an album; literal descriptions ("a bridge over a river") are noise —
> don't produce them.
>
> **Task.** Group the images into meaningful moments and annotate. Return **JSON only**:
>
> ```json
> {
>   "album_type": "trip | kids-growing | family | wedding | milestone-celebration | everyday |
>                  pet | project | memorial | event | mixed",
>   "trip_subtype": "city | road | beach | hike | null",
>   "narrative": "<one sentence: the arc of this collection>",
>   "groups": [
>     {
>       "label": "Day 3 - Buda Castle & the funicular",
>       "day": "YYYY-MM-DD",
>       "scene_type": "<place/activity, human-readable>",
>       "moment_type": "meal | transit | landmark | candid | goofing | golden-hour |
>                       posed-group | detail | the-one-where-X",
>       "people": ["<who, if identifiable/nameable by role>"],
>       "image_ids": ["<id>", ...],
>       "best_id": "<the single best frame in this group>",
>       "reason": "<why that frame is the group's best MOMENT, in the owner's terms>"
>     }
>   ]
> }
> ```
>
> Prefer human-readable day/place labels and moment tags over object lists. `best_id.reason`
> should be about the moment ("everyone's actually looking and laughing"), not the pixels.

---

## Rater instruction sheet (EXP-1 judging) — v1, solo N=1 pilot

> You will see two albums, **A** and **B** (which system made which is hidden; left/right is
> randomized). Each is shown **in its intended order**. Spend ~10 seconds forming a gut reaction
> before analysing.
>
> 1. **Overall:** Which album would you actually want to **show someone** — A or B? (forced)
> 2. **The one moment:** What single image or moment made you pick it?
> 3. **Story vs slideshow:** Does the winner feel like a *story* or a *pile of nice photos*?
>    Rate each album on that axis (1 = pile … 5 = story).
> 4. **Sub-ratings (1-5 each, per album):** coverage (people & places) · photo quality ·
>    redundancy (5 = no wasted near-dups) · narrative (flow/pacing/opener-closer).
> 5. **Human rescue:** Name up to **5 photos you'd add back** that BOTH albums missed (and which
>    trip source folder they're in). This captures the "draft quality" framing — the frames the
>    metrics wanted to cut.
>
> Record answers in the judging sheet the viewer writes; do not look up which system is which
> until all trips are judged.

**Honesty note (in the report):** v1 is **N=1, the pipeline's owner** — an illustrative anecdote,
not a measured win-rate. Self-recognition leakage is possible and is disclosed. No inter-rater
agreement until the multi-rater upgrade.

---

## EXP-2 within-cluster pick — prompt fragment

> From this set of near-identical frames (same moment), pick the **single best** to keep in an
> album, and give a one-clause reason. Prefer eyes-open, in-focus, best expression, cleanest
> framing — but the *best moment* wins ties. Output: `{"best_id": "<id>", "reason": "<clause>"}`.

Reference truth = 3 blind human best-picks per cluster (majority); report top-1 accuracy + Kendall
tau. (In the N=1 pilot, the owner is the single rater; upgrade to 3 raters later.)
