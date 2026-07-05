"""Track A — Haiku VLM occlusion labeler (spec-096).

Labels every dataset image with Claude Haiku: occluded? severity 0-3 + one-line
reason. Dual role: benchmark candidate (its score) AND second validator (its
disagreements with the user's labels get human review — surfacing occlusions
hiding in the album negatives).

- Results append to ``haiku_labels.csv`` incrementally -> fully resumable.
- The API key comes from the environment / Windows User registry; it is never
  logged, printed, or written to any file.
- Images are downscaled to <=1024px JPEG before upload (cost + speed).
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_SIDE = 1024

PROMPT = (
    "You are auditing personal photos for CAMERA LENS OBSTRUCTIONS - typically a "
    "finger, hand or strap partially covering the lens. Such a defect appears as a "
    "blurry, out-of-focus blob (often warm/skin-toned or dark) usually at a corner "
    "or edge of the frame.\n"
    "Look carefully at all corners and edges. Answer ONLY with JSON:\n"
    '{"occluded": true|false, "level": 0|1|2|3, "reason": "<one short sentence>"}\n'
    "level: 0=no obstruction, 1=slight (small corner blob, photo still fine), "
    "2=moderate (noticeable, part of the scene blocked), 3=severe (photo ruined).\n"
    "Do NOT flag: ordinary bokeh/depth-of-field, motion blur of subjects, walls or "
    "sky that are merely plain, or objects far from the lens."
)


def _api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # Windows: read the User env var from the registry (no shell restart needed)
        r = subprocess.run(
            ["powershell", "-Command",
             "[Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')"],
            capture_output=True, text=True, timeout=20)
        key = r.stdout.strip()
        return key or None
    except Exception:
        return None


def _encode(path: str) -> Optional[str]:
    try:
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning("encode failed for %s: %s", path, e)
        return None


def label_image(client, path: str) -> Optional[dict]:
    b64 = _encode(path)
    if b64 is None:
        return None
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=150,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": PROMPT},
                ]}])
            text = msg.content[0].text.strip()
            start, end = text.find("{"), text.rfind("}")
            out = json.loads(text[start:end + 1])
            return {"occluded": bool(out.get("occluded")),
                    "level": int(out.get("level", 0)),
                    "reason": str(out.get("reason", ""))[:200],
                    "in_tokens": msg.usage.input_tokens,
                    "out_tokens": msg.usage.output_tokens}
        except Exception as e:
            wait = 5 * (attempt + 1)
            logger.warning("label_image retry %d for %s: %s", attempt + 1,
                           os.path.basename(path), e)
            time.sleep(wait)
    return None


def run(dataset_root: str, limit: Optional[int] = None) -> dict:
    """Label all manifest rows not yet in haiku_labels.csv. Resumable."""
    import anthropic
    from sim_bench.occlusion_bench.dataset import load_manifest

    key = _api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not available (env or User registry)")
    client = anthropic.Anthropic(api_key=key)

    out_csv = os.path.join(dataset_root, "haiku_labels.csv")
    done = set()
    if os.path.isfile(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}

    rows = [r for r in load_manifest(dataset_root) if r["id"] not in done]
    if limit:
        rows = rows[:limit]
    logger.info("labeling %d images (%d already done)", len(rows), len(done))

    new_file = not os.path.isfile(out_csv)
    stats = {"n": 0, "occluded": 0, "failed": 0, "in_tokens": 0, "out_tokens": 0}
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id", "user_label", "haiku_occluded", "haiku_level", "haiku_reason"])
        for i, r in enumerate(rows):
            sub = "positives" if r["label"] == "1" else "negatives"
            res = label_image(client, os.path.join(dataset_root, sub, r["id"]))
            if res is None:
                stats["failed"] += 1
                continue
            w.writerow([r["id"], r["label"], int(res["occluded"]), res["level"], res["reason"]])
            f.flush()
            stats["n"] += 1
            stats["occluded"] += int(res["occluded"])
            stats["in_tokens"] += res["in_tokens"]
            stats["out_tokens"] += res["out_tokens"]
            if (i + 1) % 25 == 0:
                logger.info("progress %d/%d (occluded so far: %d)",
                            i + 1, len(rows), stats["occluded"])
    return stats
