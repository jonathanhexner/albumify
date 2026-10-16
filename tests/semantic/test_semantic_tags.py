"""spec-108 P0 — semantic tag cleaning + faceting (pure, no model)."""
from __future__ import annotations

from sim_bench.semantic.tags import clean, categorize


class ut_CleanTags:
    def test_drops_verb_fragment_noise(self):
        raw = ["waffle", "syrup", "fork", "eat", "bite", "table", "plate"]
        out = clean(raw)
        assert "waffle" in out and "fork" in out
        assert "eat" not in out and "bite" not in out          # verbs dropped

    def test_drops_vague_and_color_tags(self):
        out = clean(["mountain", "image", "photo", "pink", "red", "lake"])
        assert out == ["mountain", "lake"]

    def test_dedup_preserves_order(self):
        assert clean(["tree", "Tree", "tree", "flower"]) == ["tree", "flower"]

    def test_empty(self):
        assert clean([]) == []


class ut_Categorize:
    def test_facets_assigned(self):
        st = categorize(["mallard duck", "lake", "mountain", "water", "swim"])
        assert "nature" in st.facets
        assert "animal" in st.facets and "mallard duck" in st.facets["animal"]
        assert "swim" not in st.tags                            # verb dropped before faceting

    def test_primary_facet_is_majority(self):
        st = categorize(["mountain", "lake", "tree", "forest", "person"])
        assert st.primary_facet == "nature"                    # 4 nature vs 1 people

    def test_people_facet(self):
        st = categorize(["person", "child", "family", "creek"])
        assert "people" in st.facets and "nature" in st.facets

    def test_no_known_facet_is_other(self):
        st = categorize(["artifact", "showcase", "vase"])
        assert st.primary_facet == "other"
        assert st.tags == ["artifact", "showcase", "vase"]     # kept, just unfaceted
