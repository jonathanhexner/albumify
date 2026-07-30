# Expert reviews — Spec 102 (Albumify vs VLM)

Two independent critiques of the draft experiment design (2026-07-17), preserved verbatim, each
followed by how the point was addressed in `spec.md`.

---

## Reviewer A — Senior CV Researcher (measurement validity)

### Must-Fix

**1. N=1 blind A/B is not blind and has near-zero power.** The project owner both built Albumify and
judges the albums. Even with hidden labels and L/R randomization, the owner recognizes their own
pipeline's stylistic fingerprint (crop tendencies, exposure profile, which faces it favors) — this is
*self-recognition leakage*, not blindness. And a single rater gives you no way to separate "the album
is better" from "this rater prefers this style." Fix: recruit >=5 raters who have never seen either
system's output; ideally include >=1 rater per trip who was *on the trip* (they can judge
coverage/person-completeness, which strangers cannot). Pre-register the decision rule. Report per-rater
and inter-rater agreement (Cohen's kappa / Krippendorff's alpha); if alpha < ~0.4 the "which won"
verdict is uninterpretable regardless of N.

**2. Batching is confounded with the model.** The VLM never sees the full album — it sees 12-16-thumb
contact sheets then a reduce pass, at 768px. Albumify scores globally at (presumably) higher res. So
EXP-1 does not compare "pipeline vs VLM"; it compares "global high-res scorer vs windowed-low-res-
thumbnail scorer." A loss could be entirely the harness. Fix: (a) run at least one ablation where the
VLM sees all photos in one context (accept the cost on one trip) to bound the batching penalty; (b) feed
Albumify the *same* 768px inputs so resolution is held constant; (c) log and report batch composition,
since near-duplicates split across sheets can't be deduped in the map phase — that's a systematic
handicap you must measure, not hide.

**3. Same-K is necessary but not sufficient for fairness.** Both pick K=20, good. But "best album"
conflates two axes the brief itself lists: *selection quality* (are these good photos?) and *coverage*
(every person/scene/day?). A single side-by-side forced choice lets one dominate unpredictably across
raters. Fix: collect structured sub-ratings (5-pt: coverage, photo quality, redundancy, narrative)
*plus* the overall pick. This also gives you something analyzable with N small.

### Should-Fix — Reproducibility gaps
- VLM nondeterminism: temperature, seed, model snapshot date, and the fact that grid tiling order
  changes picks — none specified. Fix temperature=0, log model version, run >=3 seeds/trip and report
  pick stability.
- "Best K default 20" but albums are 200-400 photos of differing content density — K should scale or be
  justified per trip.
- No stated randomization protocol, no rater instructions text, no tie-handling. Ship the exact prompt,
  the exact rater sheet, and the seed list.

### EXP-2: not meaningful as designed
Comparing two systems' within-cluster picks with no ground truth just tells you *whether they agree*,
not who is *right* — agreement can be high and both wrong. Minimal rigorous version: sample ~50 clusters,
have 3 blind raters rank the shots in each cluster (or pick the best). That human "best" is your
reference; report each system's top-1 accuracy and rank correlation (Kendall tau) against it.

### Statistical power — honest claim
Three trips = n=3 experimental units (trips, not photos — photos within a trip are not independent). With
1 judge you can honestly claim *nothing inferential*; it is an anecdote/case study. Smallest change that
materially helps: 5 blind raters x 3 trips lets you at least report a per-trip win-rate with a CI and
rater agreement. To claim generality you need ~10-15 trips; short of that, frame the whole thing as a
*pilot* and say so in the title.

### Metrics to add
1. Coverage precision/recall against a hand-labeled roster of persons and distinct scenes/days per trip.
2. Duplicate rate: fraction of near-dup pairs both surviving in the chosen K (from Albumify's clusters).
3. Defect rate: blur/occlusion/eyes-closed count in each chosen set, scored by existing detectors.

**How addressed**: >=5 blind raters + on-trip rater (D6/EXP-1); Krippendorff alpha (A8, T4.3); same-768px
inputs (D5/A1) + full-context ablation + logged batch/dup-split (A5/T3.1,T3.3); structured sub-ratings
(EXP-1/T4.1); temp=0/seeds/version (A4/T3.2); K floats (spec objective); EXP-2 reframed to 3-rater truth
+ top-1 acc + Kendall tau (D3/EXP-2/A7); pilot framing (D6/A8); three objective metrics (A6/T4.2).

---

## Reviewer B — Senior Album Editor (craft validity)

### 1. What a real editor optimizes that the criteria list misses
The list is a *quality-control checklist* masquerading as an *editorial brief*. QC is table stakes — how
you *reject* frames, not how you *build* an album. Top 5 under-weighted:
1. **Narrative arc / story beats (must-have).** An album is a sequence, not a set. Arrival -> settling ->
   set-pieces -> a quiet in-between -> emotional peak -> wind-down. "Best 20" with no ordering is a
   real-estate-open-house slideshow.
2. **Variety of scale — wide / medium / detail (must-have).** Editors intercut establishing wide,
   human-scale medium, tight detail. "Best" collapses toward one scale (pretty medium-wide scenics).
3. **Emotional peak over the "nice" (must-have).** A slightly soft genuine-laughter frame beats a razor-
   sharp posed one. The criteria actively *punish* the peak (motion blur, closed eyes mid-laugh, flare).
   The single most dangerous bias in the list.
4. **Opening and closing shots (high-leverage).** Positional roles no per-image score assigns.
5. **Connective tissue vs hero shots.** ~15% heroes, ~85% "boring but necessary" glue. All-heroes is
   exhausting and flattens the peaks.

### 2. Is "best K, cover everyone, no duplicates" the right objective?
No — technically-correct, soulless. "No near-duplicates" throws away intentional repetition (the kid on
the swing across 4 frames *is* the story). "Cover everyone equally" is wrong — albums have protagonists.
"K=20 flat" is arbitrary. Reframe to: **"assemble an ordered N-image sequence that tells the trip's
story, with QC as a filter not an objective"** — target scale-distribution, required opener/closer,
permission to keep intentional repetition, protagonist-weighting, K floats.

### 3. Judging — what decides it in the first 10 seconds
Not per-image re-inspection — **flow and variety.** Story or a pile? Rhythm (wide/tight/wide) or monotony?
Inviting opener? One or two frames that make me *feel* something? Sharpness/exposure only noticed when
*bad*. Change the "why" prompt to force the real reason: *"Which album would you actually want to show
someone, and what's the one image or moment that made you pick it?"* + a *story vs slideshow* axis. And
capture ordering — a great set in bad order loses to a good set in good order; unordered grids don't test
the thing that matters most.

### 4. Annotation — signal vs noise
Helpful (must-have): day/place segmentation with a human label ("Day 3 — Buda Castle & the funicular");
moment-type tags (meal / transit / landmark / candid-goofing / golden-hour / the-one-where-X-happened);
best-moment-per-group with a one-line *reason*. Noise (skip): literal captions ("a bridge over a river"),
object-detection dumps, sentiment adjectives. Taxonomy missing the big ones: milestone/celebration,
everyday/slice-of-life, pet, project/build, memorial, and **mixed** (most real folders); plus trip
sub-types (city break / road trip / beach / hike) that imply different pacing.

### 5. The one thing no metric will ever capture
**Meaning that lives outside the frame.** The blurry breakfast photo that's the last morning before
Grandpa got sick. Albums are memory prosthetics; value is personal and contextual, not visual. Be honest:
every pipeline optimizes photographic quality + coverage as a *proxy for personal meaning*, loosest
exactly at the emotional peaks. Measure tools on **"how good a first-pass draft"** a human then reorders
and rescues 3-5 frames into — bake a human-rescue step into the eval.

**How addressed**: objective reframed to story-sequence + QC-as-filter (spec Objective); ordered output
required both arms (A2) + ordered A/B viewer (T4.1); "why" prompt + story-vs-slideshow + human-rescue on
the rater sheet (T4.1); annotation schema = day/place labels + moment_type tags + best-moment reason,
noise graded down (EXP-3/T6); album_type taxonomy expanded incl. mixed + trip_subtype (EXP-3); proxy-gap
+ "grade the draft" stated in Threats to Validity; scale-variety / opener-closer / protagonist /
intentional-repetition encoded in curation_v1 prompt (T1.3).
