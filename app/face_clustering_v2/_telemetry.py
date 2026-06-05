"""spec-068 — v2 tab render telemetry.

One ``tab.start`` / ``tab.done`` / ``tab.skipped`` line per tab render, on a
dedicated ``fc_app_v2.tabs`` logger. Driver-agnostic: the app writes these
whether AppTest or a real browser drives it, so a single log line answers
"which tab ran, with what run dir, producing what counts?" — the question
screenshots cannot.

Format is plain ASCII key=value (Windows CLI safe), one line per event.
This module does NO Streamlit I/O — it only logs.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("fc_app_v2.tabs")

# Single kill-switch for frontend observability. Default ON. Disable with
# FC_V2_TAB_TELEMETRY=0 (or raise the "fc_app_v2.tabs" logger level). Read
# once at import — flip the env before launching streamlit to take effect.
_ENABLED = os.getenv("FC_V2_TAB_TELEMETRY", "1") != "0"


def telemetry_enabled() -> bool:
    """True when tab/component telemetry lines are emitted (spec-068/066)."""
    return _ENABLED


def _kv(counts: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in counts.items())


def tab_start(name: str, run_dir: Any) -> None:
    """Log that tab ``name`` began rendering against ``run_dir``."""
    if _ENABLED:
        logger.info("tab.start name=%s run_dir=%s", name, run_dir)


def tab_done(name: str, **counts: Any) -> None:
    """Log that tab ``name`` finished, with its key counts as k=v pairs."""
    if _ENABLED:
        logger.info("tab.done name=%s %s", name, _kv(counts))


def tab_skipped(name: str, reason: str) -> None:
    """Log that tab ``name`` returned early without rendering its body."""
    if _ENABLED:
        logger.info("tab.skipped name=%s reason=%s", name, reason)


def component_render(tab: str, component: str, **counts: Any) -> None:
    """Log that a sub-component of ``tab`` rendered, with its counts.

    Finer-grained than tab.done — answers "did the Gallery strip actually
    paint thumbnails for cluster N?" without a screenshot (spec-066)."""
    if _ENABLED:
        logger.info("component.render tab=%s name=%s %s", tab, component, _kv(counts))
