# Albumify vs VLM — austria pilot (spec-102)

- Input: 122 photos -> 768px (hash 42756fcc088ed9fc), both arms, K=20.
- Albumify arm: `minimal` pipeline (full 33-step OOMs this box at occlusion CLIP load; occlusion/tilt penalties skipped).
- VLM arm: claude-opus-4-8, map(shortlist)->reduce(order); 356859/9244 tokens.
- **Pick overlap: 3/20 identical** (systems disagree on 17/20).
- Duplicate-survival (Albumify scene clusters as truth): Albumify 0.0, VLM 0.0 (VLM slightly more diverse, and it never saw the clusters).
- VLM picked 5 shot(s) Albumify's quality gate rejected: 20230817_180846, 20230817_212315, 20230821_130239, 20230824_144214, 20230826_125819.
- VLM produced narrative roles (opener/hero/peak/closer) + editorial captions.
- **Human blind A/B verdict: PENDING** (viewer built; solo N=1 pilot, no inferential claim).
