"""Spec-102 T6 (EXP-3) — VLM annotation value-add: structured, product-shaped labels.

This is the scouting arm, not a competition: the VLM groups a trip's photos into MOMENTS and emits
structured JSON (per-group caption / scene_type / moment_type / people / best_id + reason; per-album
album_type / trip_subtype / narrative). Useful labels scaffold an album; literal captions are graded
as noise (PROMPTS.md annotation_v1). Grouping runs per-day (day segmentation is free from filenames),
then one cheap text-only call classifies the whole album. Reuses the VLM-arm client plumbing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sim_bench.albumify_vs_vlm.vlm_arm import (
    VLMArmConfig,
    _call,
    _content_for,
    _load_key,
)

logger = logging.getLogger(__name__)

_DAY_PROMPT = (
    "You are labelling ONE day of a personal trip's photos so the owner can build an album fast. "
    "Above are {n} photos, each preceded by its ID. Group them into meaningful MOMENTS (a meal, a "
    "landmark visit, a walk, a goofing-around beat). Useful labels scaffold an album; literal "
    'descriptions like "a bridge over a river" are noise — do not produce them.\n'
    'Return JSON ONLY: {{"groups": [{{"label": "<human-readable, e.g. \'Buda Castle & the '
    "funicular'>\", \"scene_type\": \"<place/activity>\", \"moment_type\": \"meal|transit|landmark|"
    'candid|goofing|golden-hour|posed-group|detail|the-one-where-X", "people": ["<role/name if '
    'clear>"], "image_ids": ["<id>", ...], "best_id": "<the single best frame>", "reason": '
    '"<why that frame is the group\'s best MOMENT, in the owner\'s terms>"}}]}}.'
)

_ALBUM_PROMPT = (
    "Here are the moment-groups discovered in one personal photo collection (label — moment_type):\n"
    "{groups}\n\n"
    "Classify the WHOLE collection. Return JSON ONLY: "
    '{{"album_type": "trip|kids-growing|family|wedding|milestone-celebration|everyday|pet|project|'
    'memorial|event|mixed", "trip_subtype": "city|road|beach|hike|null", '
    '"narrative": "<one sentence: the arc of this collection>"}}.'
)


def _days_from_roster(roster: dict) -> list[tuple[str, list[str]]]:
    out = []
    for d in roster.get("days", []):
        imgs = d.get("images", [])
        if imgs:
            out.append((d.get("date", "?"), imgs))
    return out


def run_annotation(imgs_dir: Path, roster: dict, trip: str,
                   input_set_hash: str, config: VLMArmConfig | None = None) -> dict:
    """Per-day moment grouping + album-level classification. Returns the structured annotation."""
    import anthropic

    cfg = config or VLMArmConfig(max_tokens=8000)
    client = anthropic.Anthropic(api_key=_load_key())
    in_tok = out_tok = 0
    groups: list[dict] = []

    for date, stems in _days_from_roster(roster):
        prompt = _DAY_PROMPT.format(n=len(stems))
        content = _content_for(stems, imgs_dir) + [{"type": "text", "text": prompt}]
        parsed, (i_t, o_t) = _call(client, cfg, content)
        in_tok += i_t
        out_tok += o_t
        for g in parsed.get("groups", []):
            g["day"] = date
            groups.append(g)
        logger.info("annotate day %s: %d groups", date, len(parsed.get("groups", [])))

    # Album-level classification from the group labels (cheap text-only call).
    group_lines = "\n".join(f"- {g.get('label','?')} ({g.get('moment_type','?')})" for g in groups)
    parsed, (i_t, o_t) = _call(
        client, cfg,
        [{"type": "text", "text": _ALBUM_PROMPT.format(groups=group_lines)}],
    )
    in_tok += i_t
    out_tok += o_t

    annotation = {
        "trip": trip,
        "input_set_hash": input_set_hash,
        "model": cfg.model,
        "album_type": parsed.get("album_type"),
        "trip_subtype": parsed.get("trip_subtype"),
        "narrative": parsed.get("narrative"),
        "n_groups": len(groups),
        "groups": groups,
        "meta": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    logger.info("annotation: type=%s subtype=%s, %d groups, tokens in=%d out=%d",
                annotation["album_type"], annotation["trip_subtype"], len(groups), in_tok, out_tok)
    return annotation


def save_annotation(annotation: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("wrote annotation -> %s", path)
