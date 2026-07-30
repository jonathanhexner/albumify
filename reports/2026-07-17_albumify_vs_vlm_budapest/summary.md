# Albumify vs VLM — budapest pilot (spec-102)

- Input: 122 photos -> 768px (hash 878d0c14331d19a3), both arms, K=20.
- Albumify arm: `default` pipeline (full product path; occlusion + tilt penalties active, SIGHTING-117 fixed).
- VLM arm: claude-opus-4-8, map(shortlist)->reduce(order); 93784/3291 tokens.
- **Pick overlap: 6/20 identical** (systems disagree on 14/20).
- Duplicate-survival (Albumify scene clusters as truth): Albumify 0.25, VLM 0.2 (VLM slightly more diverse, and it never saw the clusters).
- VLM picked 3 shot(s) Albumify's quality gate rejected: 20250822_122632, 20250822_125147, 20250822_194259.
- VLM produced narrative roles (opener/hero/peak/closer) + editorial captions.
- **Human blind A/B verdict: PENDING** (viewer built; solo N=1 pilot, no inferential claim).
