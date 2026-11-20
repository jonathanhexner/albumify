# Slovakia26 picks review, run 52776cc5 (2026-11-13)

**Goal:** Gate 1 evidence: are the app's picks good, does it skip good photos, are the people right?
**Method:** owner reviewed the K=51 run `52776cc5` (715 photos, 302 picks) in `review.html`: Keep / Drop + reason per
pick, "Should pick" per skipped photo, a verdict per person. 101 of 106 pages (all 96 scenes, all 6 "in no scene"
pages, 1 of 5 "removed before scenes" pages). Answers: `picks_review.json`. Built by `build_review.py`.

**Result**

| Measure | Value |
|---|---|
| Picks judged | 299 of 302 |
| Kept | 219 (73%) |
| Dropped | 80 (27%): 57 of 217 in scenes, 23 of 82 "in no scene" |
| Top drop reason | near-duplicate of another pick: 47 of 80 (41 in scenes, 7 alone) |
| Other drop reasons | other / none 18, dull 6, eyes closed / not looking 5, framing 4, stranger 4, blurry 2 |
| Should have picked | 29: **20 of 20** on the one "removed before scenes" page reviewed, 9 inside scenes (7 outranked, 2 below 0.4) |
| People | 4 main people right; person 4 is a split of person 2; persons 5-7 are strangers |

**Findings**
1. Near-duplicate picks are the biggest problem. Removing them alone would take kept picks from 73% to about 88%.
2. The sharpness gate (`filter_quality` `min_sharpness: 0.2`) removed 20 of 20 reviewed photos the owner wants
   (sharpness 0.02-0.20, all with faces). This matches the 2026-11-06 label study (59 of 60 removals wrong).
3. Removed photos showed "Person detected" as their reason, not the sharpness rejection: SIGHTING-149.
4. Owner notes: a finger in frame (scene 82; was it detected?), a dull dryer photo (needs a "not interesting"
   filter), a kid pulling a face spoils a picture (scene 39).
5. Scenes are visual only (DINOv2 + HDBSCAN, no time): scene 5 mixes Aug 18 and Aug 29.

Images: `img/` and `crops/` (downscaled, untracked); originals in `D:\Slovakia26`.
