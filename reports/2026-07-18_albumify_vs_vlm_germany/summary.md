# Albumify vs VLM — germany pilot (spec-102)

- Input: 122 photos -> 768px (hash 19b384f1fda78994), both arms, K=20.
- Albumify arm: `minimal` pipeline (full 33-step OOMs this box at occlusion CLIP load; occlusion/tilt penalties skipped).
- VLM arm: claude-opus-4-8, map(shortlist)->reduce(order); 190712/2786 tokens.
- **Pick overlap: 0/20 identical** (systems disagree on 20/20).
- Duplicate-survival (Albumify scene clusters as truth): Albumify 0.0, VLM 0.05 (VLM slightly more diverse, and it never saw the clusters).
- VLM picked 6 shot(s) Albumify's quality gate rejected: 20240816_152405, 20240820_133757, 20240820_164646, IMG_20240824_110517, IMG_20240824_180959, IMG_20240826_170113.
- VLM produced narrative roles (opener/hero/peak/closer) + editorial captions.
- **Human blind A/B verdict: PENDING** (viewer built; solo N=1 pilot, no inferential claim).
