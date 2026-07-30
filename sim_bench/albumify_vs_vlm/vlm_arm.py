"""Spec-102 T3 — the VLM arm: Claude Opus 4.8 curates an ordered K-sequence.

Raw-images-only (D1): the model sees the same 768px working set as Albumify, nothing else.
Two phases (batching held fixed so it's a logged variable, not a hidden confound — CV review):

  MAP     each batch of ~15 images -> shortlist the best few (cheap, parallel-in-spirit)
  REDUCE  all shortlisted images -> select AND order exactly K, with role + reason per pick

Returned IDs are validated against the known stem set, so a hallucinated filename can't enter the
album; a short result is padded from the shortlist in map-order. Client plumbing mirrors
`sim_bench/occlusion_bench/vlm_label.py` (encode / request / retry).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from sim_bench.albumify_vs_vlm.schema import AlbumResult, Pick

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-opus-4-8"

_MAP_PROMPT = (
    "You are a seasoned photo-album editor. Above are {n} photos from ONE trip, each preceded by "
    "its ID. Shortlist up to {m} that most deserve a place in a keepsake album: strong moments, "
    "good composition, eyes open, sharp, and VARIETY of subject/scale (wide scenes, people, "
    "details). Keep genuine moments even if slightly imperfect; drop redundant near-duplicates and "
    "technically broken frames (heavy blur, finger over lens, blown exposure). "
    'Return JSON ONLY: {{"shortlist": [{{"id": "<id>", "reason": "<short clause>"}}]}}.'
)

# The reduce prompt encodes the reframed objective (PROMPTS.md curation_v1).
_REDUCE_PROMPT = (
    "You are a seasoned photo-album editor assembling a keepsake album from ONE trip. Above are "
    "{n} candidate photos, each preceded by its ID. Choose exactly {k} and return them as an "
    "ORDERED sequence (first to last) that tells the trip's story a person would want to show a "
    "friend.\n"
    "Priorities: (1) a story arc arrival->peak->wind-down; (2) a strong opener and resolving "
    "closer; (3) variety of scale (wide / medium / detail); (4) the real moment over the merely "
    "sharp; (5) give more frames to whoever the trip is about, but don't drop someone who matters; "
    "(6) cover the distinct places/days; (7) intentional repetition of a moment building is fine, "
    "redundant near-identical frames are not. Quality is a FILTER, not the goal.\n"
    'Return JSON ONLY: {{"order": ["<id>", ...], "picks": [{{"id": "<id>", '
    '"role": "opener|hero|connective|detail|peak|closer", "reason": "<short clause>"}}]}}. '
    "`order` and `picks` ids must match and be exactly {k} long."
)


@dataclass
class VLMArmConfig:
    target_k: int = 20
    model: str = MODEL
    batch_size: int = 15
    shortlist_per_batch: int = 6
    # Reduce can't send the whole shortlist at once (API ~32MB request cap -> 413 on big trips), so
    # the shortlist is narrowed in chunks of this many images before the final ordering call.
    max_reduce_images: int = 45
    max_retries: int = 3
    max_tokens: int = 4000


def _load_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (checked .env and environment)")
    return key


def _img_block(path: Path) -> dict:
    b64 = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def _content_for(stems: list[str], imgs_dir: Path) -> list[dict]:
    """Interleave `ID: <stem>` text then the image block, so the model ties picks to IDs."""
    content: list[dict] = []
    for stem in stems:
        content.append({"type": "text", "text": f"ID: {stem}"})
        content.append(_img_block(imgs_dir / f"{stem}.jpg"))
    return content


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:120]}")
    return json.loads(text[start : end + 1])


def _call(client, cfg: VLMArmConfig, content: list[dict]) -> tuple[dict, tuple[int, int]]:
    last_exc: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            # Opus 4.8 deprecated the `temperature` param (400 if sent); it decodes near-greedily
            # by default, which is what we want for reproducibility.
            msg = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            parsed = _parse_json(msg.content[0].text)
            return parsed, (msg.usage.input_tokens, msg.usage.output_tokens)
        except Exception as exc:  # transient API / parse errors -> backoff + retry
            last_exc = exc
            logger.warning("VLM call attempt %d failed: %s", attempt + 1, str(exc)[:160])
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"VLM call failed after {cfg.max_retries} attempts: {last_exc}")


def _batches(stems: list[str], size: int) -> list[list[str]]:
    return [stems[i : i + size] for i in range(0, len(stems), size)]


def _map_phase(client, cfg: VLMArmConfig, imgs_dir: Path, stems: list[str],
               known: set) -> tuple[list[str], int, int]:
    shortlist: list[str] = []
    in_tok = out_tok = 0
    batches = _batches(stems, cfg.batch_size)
    for bi, batch in enumerate(batches):
        prompt = _MAP_PROMPT.format(n=len(batch), m=cfg.shortlist_per_batch)
        content = _content_for(batch, imgs_dir) + [{"type": "text", "text": prompt}]
        parsed, (i_t, o_t) = _call(client, cfg, content)
        in_tok += i_t
        out_tok += o_t
        picked = [p.get("id") for p in parsed.get("shortlist", []) if p.get("id") in known]
        shortlist.extend(dict.fromkeys(picked))
        logger.info("map batch %d/%d: %d shortlisted", bi + 1, len(batches), len(picked))
    return list(dict.fromkeys(shortlist)), in_tok, out_tok


def _reduce_phase(client, cfg: VLMArmConfig, imgs_dir: Path, shortlist: list[str],
                  known: set) -> tuple[dict, int, int]:
    """Narrow the shortlist in chunks (API request-size cap) then order the survivors to exactly K."""
    in_tok = out_tok = 0
    pool = shortlist
    # Hierarchical narrowing: while the pool won't fit one request, semifinal-reduce each chunk.
    while len(pool) > cfg.max_reduce_images:
        chunks = _batches(pool, cfg.max_reduce_images)
        # keep ~cap/n_chunks per chunk so the next pool lands near the cap (>=K for real choice)
        # and shrinks every round -> converges to a single final call.
        keep_per = max(1, cfg.max_reduce_images // len(chunks))
        nxt: list[str] = []
        for ci, chunk in enumerate(chunks):
            prompt = _MAP_PROMPT.format(n=len(chunk), m=keep_per)
            content = _content_for(chunk, imgs_dir) + [{"type": "text", "text": prompt}]
            parsed, (i_t, o_t) = _call(client, cfg, content)
            in_tok += i_t
            out_tok += o_t
            nxt.extend(p.get("id") for p in parsed.get("shortlist", []) if p.get("id") in known)
        new_pool = list(dict.fromkeys(nxt))
        logger.info("reduce narrowing: %d -> %d (chunks=%d keep=%d)",
                    len(pool), len(new_pool), len(chunks), keep_per)
        if len(new_pool) >= len(pool):  # safety: no progress -> hard-truncate
            new_pool = new_pool[: cfg.max_reduce_images]
        pool = new_pool

    prompt = _REDUCE_PROMPT.format(n=len(pool), k=cfg.target_k)
    content = _content_for(pool, imgs_dir) + [{"type": "text", "text": prompt}]
    parsed, (i_t, o_t) = _call(client, cfg, content)
    return parsed, in_tok + i_t, out_tok + o_t


def run_vlm_arm(
    imgs_dir: Path,
    stems: list[str],
    trip: str,
    input_set_hash: str,
    config: VLMArmConfig | None = None,
    shortlist_cache: Path | None = None,
) -> AlbumResult:
    """Run the two-phase VLM curation and return the ordered K-sequence.

    If `shortlist_cache` is given, the map-phase shortlist is saved there and reused on a re-run
    (so a reduce-phase failure never wastes the expensive map phase again).
    """
    import anthropic

    cfg = config or VLMArmConfig()
    client = anthropic.Anthropic(api_key=_load_key())
    known = set(stems)
    in_tok = out_tok = 0

    # --- MAP (cached) ------------------------------------------------------------------------
    if shortlist_cache and shortlist_cache.exists():
        cached = json.loads(shortlist_cache.read_text(encoding="utf-8"))
        shortlist = [s for s in cached.get("shortlist", []) if s in known]
        logger.info("map phase: reused %d shortlisted from cache", len(shortlist))
    else:
        shortlist, in_tok, out_tok = _map_phase(client, cfg, imgs_dir, stems, known)
        if shortlist_cache:
            shortlist_cache.write_text(json.dumps({"shortlist": shortlist}, indent=2),
                                       encoding="utf-8")
    if not shortlist:
        raise RuntimeError("VLM map phase shortlisted nothing")

    # --- REDUCE (hierarchical, order exactly K) ----------------------------------------------
    parsed, r_in, r_out = _reduce_phase(client, cfg, imgs_dir, shortlist, known)
    in_tok += r_in
    out_tok += r_out

    reason_by_id = {p.get("id"): p.get("reason", "") for p in parsed.get("picks", [])}
    role_by_id = {p.get("id"): p.get("role", "") for p in parsed.get("picks", [])}
    order = [s for s in parsed.get("order", []) if s in known]
    order = list(dict.fromkeys(order))[: cfg.target_k]
    # Pad from the shortlist (map order) if the reduce returned fewer than K valid ids.
    if len(order) < cfg.target_k:
        for s in shortlist:
            if s not in order:
                order.append(s)
            if len(order) == cfg.target_k:
                break

    picks = [
        Pick(id=s, score=0.0, scene_cluster=-1,
             reason=reason_by_id.get(s, ""), role=role_by_id.get(s, ""))
        for s in order
    ]
    logger.info("VLM arm: %d picks (K=%d) from %d shortlisted; tokens in=%d out=%d",
                len(order), cfg.target_k, len(shortlist), in_tok, out_tok)
    return AlbumResult(
        trip=trip, arm="vlm", input_set_hash=input_set_hash, k=cfg.target_k,
        pipeline=cfg.model, order=order, picks=picks,
        meta={"input_tokens": in_tok, "output_tokens": out_tok,
              "n_shortlisted": len(shortlist),
              "n_batches": (len(stems) + cfg.batch_size - 1) // cfg.batch_size,
              "batch_size": cfg.batch_size},
    )
