"""Human-readable reference for each analysis method (spec-094).

Drives the in-app "What do these columns mean?" legend. IMPORTANT: the studio
table shows every quality score DIRECTION-NORMALIZED so **higher = better** for
all of them (BRISQUE / NIQE are distortion metrics — lower raw is better — so the
app negates them; a shown -19.7 beats -30.2). ``raw_range`` / ``raw_dir`` describe
the underlying metric; ``in_table`` describes what you read in this app.
"""

# key -> (what it measures, typical raw range, raw direction, what "better" means in the table)
METHOD_INFO = {
    # --- image quality (all shown higher = better) -------------------------
    "iqa": ("Overall technical quality — blend of sharpness, exposure, contrast, colorfulness",
            "0 – 1", "higher = better", "higher ↑"),
    "sharpness": ("Edge / detail sharpness (focus, motion blur)",
                  "0 – 1", "higher = better", "higher ↑"),
    "ava": ("Aesthetic appeal (composition / beauty), trained on the AVA photo-rating dataset",
            "1 – 10", "higher = better", "higher ↑"),
    "maniqa": ("Perceptual no-reference quality (transformer, trained on KonIQ-10k). Most accurate of the group",
               "~0 – 1", "higher = better", "higher ↑"),
    "musiq": ("Multi-scale perceptual quality — robust across image resolutions / content",
              "~0 – 100", "higher = better", "higher ↑"),
    "hyperiqa": ("Content-adaptive perceptual quality (lightweight CNN, fast)",
                 "~0 – 1", "higher = better", "higher ↑"),
    "brisque": ("Naturalness / distortion — blur, noise, JPEG artefacts (classical, no deep net)",
                "0 – 100 raw", "LOWER raw = better", "higher ↑ (app negates: −19.7 beats −30.2)"),
    "niqe": ("Deviation from natural-image statistics (classical, no training)",
             "~0 – 25 raw (natural ≈ 2–8)", "LOWER raw = better", "higher ↑ (app negates)"),
    "clipiqa": ("CLIP-based 'is this a good photo?' likelihood (prompt-driven, no training)",
                "0 – 1", "higher = better", "higher ↑"),
    "clip_occlusion": ("EXPERIMENTAL — CLIP 'clear vs finger-over-lens' prompt score. "
                       "Tested: does NOT reliably flag corner occlusion (whole-image CLIP is "
                       "dominated by the main subject). Configurable prompts",
                       "0 – 1 P(clear)", "higher = clearer", "higher ↑ (weak signal)"),
    # --- geo & caption -----------------------------------------------------
    "exif": ("GPS coordinates + capture time read from the file's EXIF (ground truth)",
             "lat/lon + timestamp", "n/a — this is truth", "—"),
    "streetclip": ("Predicted CITY from image content (StreetCLIP)",
                   "confidence 0 – 1", "higher = surer (NOT more accurate)", "confidence, not correctness"),
    "geoclip": ("Predicted latitude/longitude from image content (GeoCLIP)",
                "prob 0 – 1; distance km vs EXIF", "higher prob = surer; closer km = better", "closer to EXIF = better"),
    "blip": ("Scene caption describing the photo (BLIP)",
             "free text", "n/a", "—"),
}
