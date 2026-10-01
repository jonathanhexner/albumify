"""spec-107 Tier-1 metadata triage — unit tests (pure, no files, no model)."""
from __future__ import annotations

from sim_bench.triage.metadata import TriageInputs, calc, classify_file


class ut_MetadataTriage:
    """calc() decisions from metadata alone."""

    def test_screenshot_filename_is_set_aside(self):
        r = calc(TriageInputs("Screenshot_20260823_104703_Gmail.jpg", has_camera_exif=False))
        assert r.decision == "set_aside"
        assert "screenshot" in r.reason.lower()
        assert r.confidence >= 0.9

    def test_screenshot_name_wins_even_with_exif(self):
        # A file named Screenshot_* is junk regardless of (spoofable) EXIF.
        r = calc(TriageInputs("screenshot_foo.jpg", has_camera_exif=True))
        assert r.decision == "set_aside"

    def test_camera_photo_is_kept(self):
        r = calc(TriageInputs("20260822_162306.jpg", has_camera_exif=True, fmt="JPEG",
                              width=4032, height=3024))
        assert r.decision == "keep"
        assert r.confidence == 0.0

    def test_png_at_screen_res_no_exif_is_set_aside(self):
        r = calc(TriageInputs("image.png", has_camera_exif=False, fmt="PNG", width=1170, height=2532))
        assert r.decision == "set_aside"
        assert "screen" in r.reason.lower()

    def test_png_no_exif_non_screen_is_review(self):
        r = calc(TriageInputs("graphic.png", has_camera_exif=False, fmt="PNG", width=512, height=512))
        assert r.decision == "review"

    def test_jpg_no_exif_non_screen_is_gentle_review(self):
        # Could be an edited/exported real photo -> review, never silently dropped.
        r = calc(TriageInputs("edited.jpg", has_camera_exif=False, fmt="JPEG", width=4000, height=3000))
        assert r.decision == "review"
        assert r.confidence < 0.6

    def test_download_like_name_no_exif_is_review(self):
        r = calc(TriageInputs("WhatsApp Image 2026.jpg", has_camera_exif=False, fmt="JPEG",
                              width=1600, height=1200))
        assert r.decision == "review"
        assert "no-camera-exif" in r.signals

    def test_screen_dimension_matches_either_orientation(self):
        landscape = calc(TriageInputs("x.png", has_camera_exif=False, fmt="PNG", width=2532, height=1170))
        assert landscape.decision == "set_aside"

    def test_classify_file_missing_path_is_review_not_crash(self):
        r = classify_file("D:/does/not/exist/nope.jpg")
        assert r.decision == "review"
        assert "unreadable" in r.signals
