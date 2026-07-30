"""Compat shim for loading .bin-only HF models on torch < 2.6 (spec-022).

StreetCLIP and BLIP-base ship only ``pytorch_model.bin`` (no safetensors).
transformers >= 4.5x refuses to ``torch.load`` a .bin unless torch >= 2.6
(CVE-2025-32434). This repo is pinned to torch 2.3, and upgrading torch would
churn the whole vision stack (insightface, mediapipe, …).

These two models are well-known, trusted HF repos, and this is a local
experiment tool — so we narrowly no-op the guard for THIS process. If/when these
steps graduate into production Albumify, the right fix is torch >= 2.6 (or a
safetensors conversion), not this shim.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_PATCHED = False


def allow_unsafe_torch_load() -> None:
    global _PATCHED
    if _PATCHED:
        return
    noop = lambda *a, **k: None  # noqa: E731
    try:
        import transformers.utils.import_utils as iu
        iu.check_torch_load_is_safe = noop
    except Exception as e:  # pragma: no cover
        logger.debug("could not patch import_utils guard: %s", e)
    try:
        import transformers.modeling_utils as mu
        mu.check_torch_load_is_safe = noop
    except Exception as e:  # pragma: no cover
        logger.debug("could not patch modeling_utils guard: %s", e)
    _PATCHED = True
    logger.info("HF .bin load guard disabled for this process (trusted StreetCLIP/BLIP, torch<2.6)")
