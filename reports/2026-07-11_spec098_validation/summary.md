# spec-098 validation: noise-aware quality scoring (2026-07-11/12)

**Goal**: verify the spec-098 fix against pre-registered gates by re-running the 2026-07-10
defect benchmarks unchanged (SIDD noise, RealBlur blur) + Budapest face-gate check.

**Shipped formula** (first attempt — median blur only — FAILED gates; fixed via 3 sweeps):
`sharpness = max(LaplacianVar(median3(img)) - 5*sigma^2, 0)`, noise component `1/(1+sigma/2)`,
weights `.30/.25/.05/.10/.30` (colorfulness cut — chroma noise inflates it).

| Gate | Before | After | Target | Verdict |
|---|---|---|---|---|
| A1 SIDD overall pair acc | 0.0% | **99.4%** (hi-ISO 98.3%) | >=95% | PASS |
| A2 sharpness noise-inflation | 39x | **0x** | <2x | PASS |
| A3 RealBlur blur pair acc | 99.43% | **99.29%** | >=99% | PASS |
| A4 Budapest face gate | — | blur_min 150->73.3, within tolerance bands | recalibrate+justify | PASS (user-approved) |
| A5 unit tests | — | 13/13 | pass | PASS |
| A6 sigma perf @16MP | — | 33 ms | <150 ms | PASS |

**A4 detail**: corrected face-crop scores are ~0.45x old scale -> `blur_min` 150->73.3 in
profile_4/profile_5 (19/340 gate flips, minimal). Headless comparison: baseline@150 = 7 clusters
[26,20,12,7,3,2,2]; spec-098@73.3 = 8 clusters [29,19,13,7,3,2,2,2] — inside `_budapest_baseline`
tolerance bands. **Pre-existing**: the 15-cluster UI reference is unreproducible headlessly even on
baseline code (SIGHTING-113 item 3 evidence); 3 e2e scenarios fail on baseline too (UI-level).

**Migration**: scale any custom nonzero `blur_min` by ~0.49; `score_iqa` cache bumped to
`rule_based_v2` (unit-test-pinned) so stale v1 scores are never served.
