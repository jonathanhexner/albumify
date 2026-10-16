"""spec-107 Tier-2 content triage — unit tests (pure calc on tags, no model)."""
from __future__ import annotations

from sim_bench.triage.content import calc


class ut_ContentTriage:
    """calc() decisions from RAM++ content tags."""

    def test_screenshot_content_is_set_aside(self):
        r = calc(["screenshot", "text", "text message", "smartphone"])
        assert r.decision == "set_aside"
        assert r.confidence >= 0.8

    def test_document_is_set_aside(self):
        r = calc(["document", "paper", "checklist", "note"])
        assert r.decision == "set_aside"

    def test_pure_poster_is_set_aside(self):
        # printed material, no moment context -> excluded (recoverable)
        r = calc(["poster", "label", "magazine", "package"])
        assert r.decision == "set_aside"
        assert "poster" in r.reason

    def test_sign_within_a_scene_is_review_not_set_aside(self):
        # a road sign in a landscape is a real moment -> keep it, just flag
        r = calc(["road sign", "mountain", "tree", "landscape"])
        assert r.decision == "review"

    def test_real_photo_is_kept(self):
        r = calc(["boat", "rowboat", "lake", "mountain", "calm"])
        assert r.decision == "keep"
        assert r.confidence == 0.0

    def test_people_photo_is_kept(self):
        r = calc(["person", "child", "family", "creek", "goggles"])
        assert r.decision == "keep"

    def test_empty_tags_is_kept(self):
        assert calc([]).decision == "keep"

    def test_case_insensitive(self):
        assert calc(["SCREENSHOT", "Text"]).decision == "set_aside"
