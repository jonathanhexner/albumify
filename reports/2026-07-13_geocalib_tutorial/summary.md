# GeoCalib tilt — step-by-step tutorial (spec-100)

A CV tutorial walking GeoCalib's single-image roll estimation stage by stage, worked on two real
Budapest photos side by side: HIGH tilt (20250822_123528, roll +16.5 deg) vs LOW tilt
(20250822_112359, roll -0.01 deg). Every intermediate image is the model's real output.

Steps illustrated (each shown on both photos + calculations):
1. Dense up-field û(x,y) — the learned perspective field (semantic "which way is up").
2. Latitude field φ + horizon (φ=0 contour); slope = roll, height = pitch.
3. Per-pixel confidence — concentrates on man-made structure; collapses on textureless regions.
4. LM optimization: fit θ=(focal, roll, pitch) to the confidence-weighted fields. Fitted params
   HIGH (r+16.5, p+6.6, vfov 79.3) vs LOW (r-0.01, p+12.4, vfov 35.8). Sidebar: why solve, not
   average (naive mean up = 4.6 deg, true roll 16.5).
5. Read roll = angle(fitted up, screen up).
6. Uncertainty sigma_roll = sqrt(Cov[roll,roll]); both ~0.9 deg (strong structure); kaleidoscope
   ~10-30 deg (abstains).
7. conf = exp(-sigma/2); penalty = 0 unless conf>=0.5 AND |roll|>3, else -min(0.02(|roll|-3), 0.15).
   HIGH -> -0.15 (capped); LOW -> 0.
8. Straighten (rotate by -roll) as visual proof.

View: report.html. Source: scripts/geocalib_tutorial.py.
