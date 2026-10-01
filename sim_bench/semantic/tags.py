"""Semantic content tags (spec-108 P0): clean RAM++ open-vocab tags into usable labels.

RAM++ (run + cached by the spec-107 ``triage_content`` step) emits ~12 tags/image, but mixes genuine
content nouns ("waffle", "mallard duck", "pedestrian bridge") with low-value verbs/fragments
("out", "break", "stand", "walk", "shine") observed on the 2026-10-01 Slovakia bake-off. This module
drops that noise and groups the survivors into coarse facets for thematic grouping + album-type.

Pure ``clean(raw_tags) -> list[str]`` and ``categorize(tags) -> SemanticTags`` (spec-053: domain
module, framework-agnostic). The step is the translator that reads ``context.ram_tags``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["clean", "categorize", "SemanticTags"]

# Non-content tags RAM++ emits that don't describe *what's in* the photo: verbs/actions, postures,
# and bare fragments. Curated from the observed RAM++ output; conservative (only clear non-nouns).
_STOPLIST = {
    # actions / verbs
    "out", "break", "bite", "push", "shine", "run", "beam", "hang", "walk", "play", "draw", "eat",
    "float", "swim", "bend", "squat", "ride", "pose", "make", "stand", "sit", "flow", "spray",
    "stretch", "climb", "laugh", "smile", "wear", "bend", "mark",
    # vague / meta
    "image", "photo", "picture", "collage", "display", "view", "scene", "background", "foreground",
    "color", "red", "pink", "green", "gold", "blue", "white", "black", "yellow", "orange",
}

# Coarse facets for thematic grouping / album-type. A tag may match more than one; first match wins
# the primary facet, but it is recorded under every matching facet.
_FACETS: Dict[str, set] = {
    "people": {"person", "people", "man", "woman", "child", "boy", "girl", "baby", "family",
               "crowd", "group", "selfie", "portrait", "couple", "daughter", "son"},
    "nature": {"mountain", "lake", "river", "sea", "ocean", "beach", "forest", "tree", "flower",
               "garden", "creek", "stream", "valley", "hill", "waterfall", "sky", "sunset", "cloud",
               "park", "pond", "water", "rock", "stone", "cliff", "meadow", "field"},
    "urban": {"building", "city", "street", "plaza", "road", "bridge", "tower", "castle", "church",
              "cathedral", "house", "architecture", "monument", "square", "wall", "skyline"},
    "food": {"food", "waffle", "cake", "coffee", "restaurant", "meal", "drink", "plate", "dish",
             "breakfast", "lunch", "dinner", "dessert", "fruit", "bread"},
    "animal": {"animal", "dog", "cat", "bird", "duck", "pet", "horse", "cow", "fish", "butterfly",
               "chicken", "hen"},
    "activity": {"boat", "bicycle", "helmet", "goggles", "playground", "amusement park", "fair",
                 "museum", "swimming pool", "pool"},
}


def clean(raw_tags: List[str]) -> List[str]:
    """Drop non-content noise; dedup; preserve order. Pure."""
    out: List[str] = []
    seen = set()
    for t in raw_tags or []:
        low = t.strip().lower()
        if not low or low in _STOPLIST or low in seen:
            continue
        seen.add(low)
        out.append(low)
    return out


@dataclass
class SemanticTags:
    """Cleaned content tags for one image, plus coarse facet membership."""
    tags: List[str]
    facets: Dict[str, List[str]] = field(default_factory=dict)
    primary_facet: str = "other"


def categorize(raw_tags: List[str]) -> SemanticTags:
    """Clean tags and bucket them into coarse facets (people/nature/urban/food/animal/activity)."""
    tags = clean(raw_tags)
    facets: Dict[str, List[str]] = {}
    for facet, vocab in _FACETS.items():
        # A tag matches if any facet word appears as a whole token in it, so compound RAM++ tags
        # ("mallard duck" -> animal, "pedestrian bridge" -> urban) are caught, not just exact tags.
        hits = [t for t in tags if vocab & set(t.split())]
        if hits:
            facets[facet] = hits
    # primary = facet with the most hits (stable order by _FACETS definition on ties)
    primary = max(facets, key=lambda f: len(facets[f])) if facets else "other"
    return SemanticTags(tags=tags, facets=facets, primary_facet=primary)
