"""spec-108 P1 — tag-based album theme inference (pure, no model)."""
from __future__ import annotations

from sim_bench.semantic.theme import infer_theme


def _tags(primary, tags, facets):
    return {"primary_facet": primary, "tags": tags, "facets": facets}


class ut_InferTheme:
    def test_family_trip_towns_and_nature(self):
        st = {
            "a": _tags("people", ["family", "child"], {"people": ["family", "child"]}),
            "b": _tags("nature", ["lake", "mountain"], {"nature": ["lake", "mountain"]}),
            "c": _tags("urban", ["castle", "plaza"], {"urban": ["castle", "plaza"]}),
        }
        theme = infer_theme(st)
        assert theme.album_type == "family trip — towns & nature"
        assert theme.facet_distribution == {"people": 1, "nature": 1, "urban": 1}
        assert theme.source == "tags"

    def test_nature_only(self):
        st = {k: _tags("nature", ["mountain"], {"nature": ["mountain"]}) for k in "abc"}
        assert infer_theme(st).album_type == "landscape / nature trip"

    def test_motifs_are_faceted_tags_only(self):
        st = {
            "a": _tags("nature", ["lake", "lake", "sky"], {"nature": ["lake"]}),
            "b": _tags("nature", ["lake", "mountain"], {"nature": ["lake", "mountain"]}),
        }
        theme = infer_theme(st)
        assert "lake" in theme.motifs                     # faceted + frequent
        # 'sky' not in any facet hits here -> not a motif
        assert "sky" not in theme.motifs

    def test_empty_is_mixed(self):
        theme = infer_theme({})
        assert theme.album_type == "mixed / general"
        assert theme.facet_distribution == {}

    def test_other_facet_ignored_in_distribution(self):
        st = {"a": _tags("other", ["artifact"], {}), "b": _tags("people", ["person"], {"people": ["person"]})}
        theme = infer_theme(st)
        assert "other" not in theme.facet_distribution
        assert theme.facet_distribution == {"people": 1}
