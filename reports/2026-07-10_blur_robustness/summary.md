# Blur robustness: own albums + RealBlur (2026-07-10)

**Question**: does real (non-synthetic) blur fool the occlusion detector, and does
training on RealBlur negatives fix it? (spec-097 research thread; both detection P
and explainability measured.)

| Part | Data | Result |
|---|---|---|
| A. natural blur, own albums | 40 blurriest of 776 negatives | **0/40 over gate** (max 0.21); pop. scan re-found the hidden Germany positive at P=0.91 |
| B. RealBlur-J real shake | 702 imgs / 234 scenes | **v1 FAILS: 52 imgs (38 scenes) over gate**, median P 0.54 |
| C. retrain +140 RB negatives (1/scene) | 282 holdout imgs | **31 -> 0 over gate**; PR-AUC 0.777->0.774; recall 56/56; synthetic 0/60x3; explain 13/26->10/26 |

**Verdict**: synthetic blur was too easy; real handshake blur is a genuine v1 failure mode
(7.4% over gate) and 140 well-chosen negatives eliminate it on held-out scenes at zero
benchmark/recall cost. Candidate artifact: `D:\occlusion_dataset\realblur\clip_b32_gmax_v2rb.npz`
(NOT promoted to production — user decision pending).

Details + galleries: [report.html](report.html)
