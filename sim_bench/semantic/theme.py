"""Album theme inference (spec-108 P1).

Two tiers, matching the cost discipline of the whole semantic layer:

* ``infer_theme(semantic_tags)`` — a cheap, pure, no-model baseline computed from the P0 facet
  distribution + top tags already on the context. Fully deterministic and testable.
* a VLM refinement (``sim_bench.vlm``) that reads a representative image sample for a richer,
  free-text theme — layered on top when the worker env is available (see spec.md status).

``AlbumTheme`` is the structured result both tiers produce, so downstream (captions, relevance,
UI copy) is decoupled from which tier produced it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["AlbumTheme", "infer_theme"]

# Facet-combination → human album-type label. Checked most-specific first.
_TYPE_RULES = [
    (lambda f: f.get("people", 0) and f.get("nature", 0) and f.get("urban", 0), "family trip — towns & nature"),
    (lambda f: f.get("people", 0) and f.get("nature", 0), "outdoors with family/friends"),
    (lambda f: f.get("people", 0) and f.get("urban", 0), "city break with people"),
    (lambda f: f.get("nature", 0) and not f.get("people", 0), "landscape / nature trip"),
    (lambda f: f.get("urban", 0) and not f.get("people", 0), "architecture / city sightseeing"),
    (lambda f: f.get("people", 0), "people & gatherings"),
    (lambda f: f.get("food", 0), "food & dining"),
]


@dataclass
class AlbumTheme:
    """Structured album theme. `source` records which tier produced it."""
    album_type: str
    facet_distribution: Dict[str, int]
    top_tags: List[str]
    motifs: List[str] = field(default_factory=list)
    source: str = "tags"             # "tags" (baseline) or "vlm" (refined)
    caption: str = ""                # free-text, VLM-only


def infer_theme(semantic_tags: Dict[str, dict], top_n_tags: int = 12, top_n_motifs: int = 5) -> AlbumTheme:
    """Infer an album theme from the P0 per-image semantic tags. Pure, no model.

    Aggregates each image's ``primary_facet`` into a distribution, picks the album-type label from
    the dominant facet combination, and surfaces the most common content tags as motifs.
    """
    facet_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    for rec in (semantic_tags or {}).values():
        pf = rec.get("primary_facet")
        if pf and pf != "other":
            facet_counts[pf] += 1
        for t in rec.get("tags", []):
            tag_counts[t] += 1

    facets = dict(facet_counts)
    album_type = next((label for cond, label in _TYPE_RULES if cond(facets)), "mixed / general")
    top_tags = [t for t, _ in tag_counts.most_common(top_n_tags)]
    # motifs = most common tags that belong to a recognized facet (content, not generic)
    faceted = {t for rec in (semantic_tags or {}).values() for hits in rec.get("facets", {}).values() for t in hits}
    motifs = [t for t, _ in tag_counts.most_common() if t in faceted][:top_n_motifs]

    return AlbumTheme(album_type=album_type, facet_distribution=facets,
                      top_tags=top_tags, motifs=motifs, source="tags")
