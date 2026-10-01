"""SubprocessVLMProvider (spec-108) — run MiniCPM-V in the isolated .venv-vlm, from the main env.

Implements the ``VLMProvider`` seam by spawning ``vlm_worker.py`` with the ``.venv-vlm`` interpreter
(transformers==4.51) and talking JSON-lines over stdin/stdout. The model loads once in the child and
stays warm. The main env (transformers 4.57) is never touched — this is how we dodge the version
conflict without a worker host (spec-109 will run the same worker on GPU).

Lazy: the child is spawned on first ``caption`` call. Per-image failures return "" (never raise), so
one bad image can't fail a run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

__all__ = ["SubprocessVLMProvider"]

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "sim_bench" / "vlm" / "vlm_worker.py"
_DEFAULT_VENV_PY = ROOT / ".venv-vlm" / "Scripts" / "python.exe"


class SubprocessVLMProvider:
    """VLMProvider backed by a long-lived MiniCPM-V worker in .venv-vlm."""

    def __init__(self, venv_python: Optional[Path] = None, stub: bool = False,
                 device: str = "cpu", startup_timeout_s: float = 600.0):
        self._py = Path(venv_python) if venv_python else _DEFAULT_VENV_PY
        self._stub = stub
        self._device = device
        self._startup_timeout = startup_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 0

    # ------------------------------------------------------------------ lifecycle
    def _ensure(self) -> None:
        if self._proc is not None:
            return
        if not self._py.exists():
            raise FileNotFoundError(f"VLM venv python not found: {self._py} (create .venv-vlm first)")
        cmd = [str(self._py), str(WORKER), "--device", self._device] + (["--stub"] if self._stub else [])
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, encoding="utf-8",
        )
        ready = self._proc.stdout.readline()  # worker emits a ready line once the model is loaded
        info = json.loads(ready) if ready.strip() else {"ready": False, "error": "no ready line"}
        if not info.get("ready"):
            raise RuntimeError(f"VLM worker failed to start: {info.get('error')}")

    def _rpc(self, image: Optional[str], prompt: str) -> str:
        self._ensure()
        self._next_id += 1
        rid = self._next_id
        req = {"id": rid, "prompt": prompt}
        if image:
            req["image"] = image
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("VLM worker closed unexpectedly")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown VLM error"))
        return resp["text"]

    # ------------------------------------------------------------------ VLMProvider
    def caption(self, image_path: str, prompt: str) -> str:
        try:
            return self._rpc(image_path, prompt)
        except Exception as e:  # degrade, never fail the run on one image
            print(f"SubprocessVLMProvider.caption failed for {image_path}: {e}", file=sys.stderr)
            return ""

    def caption_batch(self, image_paths: List[str], prompt: str) -> List[str]:
        return [self.caption(p, prompt) for p in image_paths]

    def generate(self, prompt: str) -> str:
        try:
            return self._rpc(None, prompt)
        except Exception as e:
            print(f"SubprocessVLMProvider.generate failed: {e}", file=sys.stderr)
            return ""

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
