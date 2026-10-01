"""MiniCPM-V worker — runs INSIDE .venv-vlm (transformers==4.51), isolated from the main env.

A long-lived process: load the model once, then serve one request per stdin line and reply one
line on stdout (JSON-lines protocol), so the 8B model is loaded once, not per image.

Protocol (one JSON object per line):
  request  : {"id": <int>, "image": "<abs path>", "prompt": "<text>"}
  response : {"id": <int>, "ok": true, "text": "<caption>"}   |   {"id": <int>, "ok": false, "error": "..."}
  ready    : on startup the worker prints {"ready": true, "model": "<name>"} once.

Run modes:
  --stub   : no model; echoes a canned caption. Used to verify the IPC/plumbing end-to-end.
  (default): load MiniCPM-V 4.5 via transformers and answer for real.

This file imports transformers/torch ONLY in real mode, so --stub runs even before the model env
is fully provisioned.
"""
from __future__ import annotations

import argparse
import json
import sys

MODEL_ID = "openbmb/MiniCPM-V-4_5"


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class _StubModel:
    name = "stub"

    def caption(self, image: str, prompt: str) -> str:
        from pathlib import Path
        return f"[stub caption of {Path(image).name} for prompt: {prompt[:40]}]"


class _MiniCPMModel:
    name = MODEL_ID

    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self._model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).eval().to(device)
        self._tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._device = device

    def caption(self, image: str, prompt: str) -> str:
        from PIL import Image
        img = Image.open(image).convert("RGB")
        msgs = [{"role": "user", "content": [img, prompt]}]
        with self._torch.no_grad():
            return self._model.chat(image=None, msgs=msgs, tokenizer=self._tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="echo canned captions, no model")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        model = _StubModel() if args.stub else _MiniCPMModel(device=args.device)
    except Exception as e:  # model load failed -> report and exit non-zero
        _emit({"ready": False, "error": f"{type(e).__name__}: {e}"})
        sys.exit(1)

    _emit({"ready": True, "model": model.name})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        if req.get("cmd") == "shutdown":
            break
        try:
            text = model.caption(req["image"], req.get("prompt", "Describe this photo."))
            _emit({"id": rid, "ok": True, "text": text})
        except Exception as e:
            _emit({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
