"""spec-041 Phase 1 drift guard.

If anyone adds a knob to ``FCConfig`` without a matching field on
``FCParams`` (or vice versa), this test fails at import / collection
time — long before a confused user hits a profile that drops a knob
on the floor.

The runtime-only fields on ``FCConfig`` (``stages``, ``source_dir``,
``output_dir``, ``on_progress``) are intentionally NOT mirrored on
``FCParams`` and are excluded from the comparison.
"""
from __future__ import annotations

import dataclasses

from face_cluster.config import PipelineConfig as FCConfig
from face_cluster.fc_params import RUNTIME_FIELDS, FCParams


def test_fcparams_fields_match_fcconfig_fields():
    fc_fields = {f.name for f in dataclasses.fields(FCConfig)} - RUNTIME_FIELDS
    p_fields = set(FCParams.model_fields)
    missing_in_params = fc_fields - p_fields
    missing_in_config = p_fields - fc_fields
    assert not missing_in_params and not missing_in_config, (
        "FCParams ↔ FCConfig field drift detected.\n"
        f"  in FCConfig but missing from FCParams: {sorted(missing_in_params)}\n"
        f"  in FCParams but missing from FCConfig: {sorted(missing_in_config)}\n"
        "Fix: add the missing field on the lagging side, or add it to "
        "face_cluster.fc_params.RUNTIME_FIELDS if it's intentionally "
        "runtime-only."
    )


def test_runtime_fields_actually_exist_on_fcconfig():
    """If a name appears in RUNTIME_FIELDS, it must really be on FCConfig."""
    fc_fields = {f.name for f in dataclasses.fields(FCConfig)}
    for name in RUNTIME_FIELDS:
        assert name in fc_fields, (
            f"RUNTIME_FIELDS lists {name!r} but it's not a field on FCConfig — "
            "stale entry?"
        )
