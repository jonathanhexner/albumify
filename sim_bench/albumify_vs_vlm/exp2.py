"""Spec-102 EXP-2 — within-cluster best-frame pick.

The clean test of *selection judgment*, isolated from coverage and ordering (the confounds in
EXP-1): given the SAME cluster of same-moment frames, each system picks the single best one, and a
blind human pick is the reference truth. We report **top-1 accuracy** (system pick == human) per
system + system-vs-system agreement. Kendall tau is deferred (needs a full human ranking; v1 asks
only for the single best, per D6 solo-N=1).

The "cluster" is Albumify's own scene cluster — the exact unit `select_best` chooses from — so
Albumify's pick is just `argmax(composite_score)` within it (no re-run of the selector needed once
the per-image composite scores are dumped). The VLM pick is one call per cluster over the raw frames.

Framework-agnostic (spec-053): config in `__init__` / dataclasses, `calc`-style entry points, no
PipelineContext. The CLI harness (`scripts/experiment_albumify_vs_vlm.py`) is the only translator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Scene clustering leaves one oversized catch-all bucket (Budapest 15, Austria 85, Germany 168) that
# is NOT a set of same-moment frames — it is "everything that didn't group." Picking a single best
# from it is meaningless, so clusters larger than this are excluded from EXP-2.
MAX_CLUSTER_SIZE = 30
MIN_CLUSTER_SIZE = 2  # a singleton has a trivial pick (itself) — no judgment to test


@dataclass
class ClusterCase:
    cluster_id: int
    stems: list[str]                 # every frame in the cluster (presentation handled by the viewer)
    albumify_best: str               # argmax composite_score within the cluster
    vlm_best: str = ""               # VLM's single-best pick
    vlm_reason: str = ""
    human_best: str = ""             # filled in after blind judging (reference truth)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Exp2Result:
    trip: str
    input_set_hash: str
    cases: list[ClusterCase] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # vlm tokens, sampling params

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def select_cluster_cases(
    scene_clusters: dict[int, list[str]],
    composite_scores: dict[str, float],
    max_size: int = MAX_CLUSTER_SIZE,
    min_size: int = MIN_CLUSTER_SIZE,
) -> list[ClusterCase]:
    """Multi-frame scene clusters (excluding the oversized catch-all), each with Albumify's best.

    Albumify's best-of-cluster == the frame `select_best` would keep first == highest composite
    score. Frames with no composite score (dropped before scoring) are ignored for the argmax but
    kept in `stems` only if they were scored — an unscored frame can't be a fair pick target.
    """
    cases: list[ClusterCase] = []
    for cid, stems in sorted(scene_clusters.items()):
        if int(cid) < 0:
            # Scene clustering labels unclustered/noise frames -1: a leftover bucket of
            # UNRELATED images, not a same-moment group. "Best frame" among them is meaningless.
            continue
        scored = [s for s in stems if s in composite_scores]
        if not (min_size <= len(scored) <= max_size):
            continue
        best = max(scored, key=lambda s: composite_scores[s])
        cases.append(ClusterCase(cluster_id=int(cid), stems=scored, albumify_best=best))
    logger.info("EXP-2: %d cluster cases (from %d scene clusters)", len(cases), len(scene_clusters))
    return cases


_BEST_PROMPT = (
    "Above are {n} photos of the SAME scene/moment from one trip, each preceded by its ID. Pick the "
    "SINGLE best frame to represent this moment in a keepsake album. Prefer: in focus, well composed, "
    "flattering expressions with eyes open, no finger/strap over the lens, no motion blur or blown "
    "exposure. Between technically-similar frames, prefer the more alive/genuine moment. "
    'Return JSON ONLY: {{"best_id": "<id>", "reason": "<short clause>"}}.'
)


def vlm_best_per_cluster(imgs_dir: Path, cases: list[ClusterCase], model: str | None = None):
    """Fill each case's `vlm_best` via one VLM call per cluster over the raw frames.

    Reuses the vlm_arm client plumbing (key load, image blocks, retrying JSON call). Mutates the
    cases in place and returns (input_tokens, output_tokens).
    """
    import anthropic

    from sim_bench.albumify_vs_vlm.vlm_arm import (
        VLMArmConfig,
        _call,
        _content_for,
        _load_key,
    )

    cfg = VLMArmConfig(model=model or VLMArmConfig.model, max_tokens=400)
    client = anthropic.Anthropic(api_key=_load_key())
    in_tok = out_tok = 0
    for i, case in enumerate(cases):
        known = set(case.stems)
        content = _content_for(case.stems, imgs_dir) + [
            {"type": "text", "text": _BEST_PROMPT.format(n=len(case.stems))}
        ]
        parsed, (i_t, o_t) = _call(client, cfg, content)
        in_tok += i_t
        out_tok += o_t
        pick = parsed.get("best_id")
        if pick not in known:  # hallucinated id -> fall back to the first frame, logged
            logger.warning("EXP-2 cluster %d: VLM returned unknown id %r; using first frame",
                           case.cluster_id, pick)
            pick = case.stems[0]
        case.vlm_best = pick
        case.vlm_reason = str(parsed.get("reason", ""))[:120]
        logger.info("EXP-2 cluster %d/%d (%d frames): VLM best=%s",
                    i + 1, len(cases), len(case.stems), pick)
    return in_tok, out_tok


def top1_accuracy(cases: list[ClusterCase]) -> dict:
    """Top-1 accuracy vs the human reference, per system, over judged cases only.

    Returns Nones (not a crash) when no case is judged yet, so the harness can build the report
    skeleton before the human step is done.
    """
    judged = [c for c in cases if c.human_best]
    n = len(judged)
    if n == 0:
        return {"n_judged": 0, "n_total": len(cases), "albumify_top1": None,
                "vlm_top1": None, "system_agreement": _agreement(cases)}
    alb = sum(1 for c in judged if c.albumify_best == c.human_best)
    vlm = sum(1 for c in judged if c.vlm_best == c.human_best)
    return {
        "n_judged": n,
        "n_total": len(cases),
        "albumify_top1": round(alb / n, 3),
        "vlm_top1": round(vlm / n, 3),
        "albumify_hits": alb,
        "vlm_hits": vlm,
        "system_agreement": _agreement(cases),
    }


def _agreement(cases: list[ClusterCase]) -> float | None:
    """Fraction of clusters where Albumify and the VLM chose the same best frame (needs no human)."""
    both = [c for c in cases if c.vlm_best]
    if not both:
        return None
    return round(sum(1 for c in both if c.albumify_best == c.vlm_best) / len(both), 3)


def save_exp2(res: Exp2Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.to_json(), indent=2), encoding="utf-8")
    logger.info("wrote EXP-2 -> %s", path)


def load_exp2(path: Path) -> Exp2Result:
    d = json.loads(path.read_text(encoding="utf-8"))
    cases = [ClusterCase(**c) for c in d.get("cases", [])]
    return Exp2Result(trip=d["trip"], input_set_hash=d.get("input_set_hash", ""),
                      cases=cases, meta=d.get("meta", {}))
