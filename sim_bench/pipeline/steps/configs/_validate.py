"""spec-033 P-G: shared validation helper for typed step configs.

Steps call ``validate_step_config(name, config)`` at the top of ``process()``.
A typo'd key in the incoming dict (from UI or pipeline.yaml) raises
``pydantic.ValidationError`` with the exact field name — no more silent
defaults. The returned model exposes typed attribute access; for steps
mid-migration, the original dict can keep being read alongside.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

from sim_bench.pipeline.steps.configs import STEP_CONFIG_MODELS


def validate_step_config(step_name: str, config: Dict[str, Any]) -> Optional[BaseModel]:
    """Validate ``config`` against the Pydantic model registered for ``step_name``.

    Returns the typed model instance, or ``None`` if the step has no model yet
    (full pipeline migration is out of P-G's scope per the master plan).
    Raises ``pydantic.ValidationError`` on unknown / out-of-range keys.
    """
    model = STEP_CONFIG_MODELS.get(step_name)
    if model is None:
        return None
    return model.model_validate(config or {})
