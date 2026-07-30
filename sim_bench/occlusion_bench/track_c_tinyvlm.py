"""Track C — tiny local VLM (Moondream2) zero-shot occlusion scoring (spec-096).

CPU-viable ~2B VLM. Asks for a 0-3 severity per image; resumable CSV. Uses 4
torch threads so it can run alongside the CNN training without starving it.
Defensive: any per-image failure logs + continues; a model-load failure writes
results_track_c.json with the error so the benchmark reports it honestly.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

PROMPT = ("Does a finger, hand, strap or other object cover part of the camera lens, "
          "appearing as a blurry out-of-focus blob at a corner or edge of this photo? "
          "Answer with just one digit: 0 (no), 1 (slight blob at edge), "
          "2 (moderate, part of scene blocked), 3 (severe).")


def run(dataset_root: str) -> dict:
    import torch
    torch.set_num_threads(4)
    from PIL import Image
    from sim_bench.occlusion_bench.dataset import load_manifest

    out_csv = os.path.join(dataset_root, "tinyvlm_scores.csv")
    res_json = os.path.join(dataset_root, "results_track_c.json")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
        model.eval()
    except Exception as e:
        logger.error("moondream load failed: %s", e)
        with open(res_json, "w") as f:
            json.dump({"error": f"model load failed: {e}"}, f)
        return {"error": str(e)}

    done = set()
    if os.path.isfile(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    rows = [r for r in load_manifest(dataset_root) if r["id"] not in done]
    logger.info("tinyvlm: scoring %d images (%d done)", len(rows), len(done))

    new = not os.path.isfile(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "label", "level", "raw"])
        for i, r in enumerate(rows):
            sub = "positives" if r["label"] == "1" else "negatives"
            try:
                img = Image.open(os.path.join(dataset_root, sub, r["id"])).convert("RGB")
                img.thumbnail((768, 768))
                with __import__("torch").no_grad():
                    enc = model.encode_image(img)
                    ans = model.answer_question(enc, PROMPT, tok)
                m = re.search(r"[0-3]", str(ans))
                lvl = int(m.group(0)) if m else 0
                w.writerow([r["id"], r["label"], lvl, str(ans)[:60]])
                f.flush()
            except Exception as e:
                logger.warning("tinyvlm failed on %s: %s", r["id"], e)
            if (i + 1) % 25 == 0:
                logger.info("tinyvlm %d/%d", i + 1, len(rows))
    with open(res_json, "w") as f:
        json.dump({"status": "scored", "csv": out_csv}, f)
    return {"status": "ok"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset"))
