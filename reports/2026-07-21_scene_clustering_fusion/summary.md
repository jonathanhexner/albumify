# Scene fusion (spec-103) — before/after summary

Visual-only vs time+geo fused scene clustering. Noise = unplaceable orphans (HDBSCAN -1).

| Trip | Photos | GPS | Visual noise | +Time | +Time+Geo | Rescued |
|---|---|---|---|---|---|---|
| budapest | 122 | 36% | 27 | 18 | 15 | 12 |
| austria | 474 | 73% | 104 | 77 | 69 | 52 |
| germany | 797 | 35% | 191 | 102 | 94 | 129 |

Time is the backbone (does most of the noise reduction); geo refines. Germany (lowest
GPS) benefits most because time carries it — graceful degradation shown empirically.
No reference labels yet: this shows change direction/magnitude, not correctness.