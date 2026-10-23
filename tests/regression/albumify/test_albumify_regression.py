"""spec-113 R1 - the default pipeline on real albums stays close to the saved baselines.

Runs ``scripts/albumify_regression.py check <dataset>`` (stage -> default pipeline -> load, local
runner). Opt-in: `pytest -m albumify_regression`. Skips a dataset whose photos or baseline are absent.
Re-baseline deliberately with `scripts/albumify_regression.py capture <dataset>`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
DATASETS = yaml.safe_load((HERE / "datasets.yaml").read_text(encoding="utf-8"))

pytestmark = [pytest.mark.albumify_regression, pytest.mark.needs_data]


@pytest.mark.parametrize("name", list(DATASETS))
def test_album_close_to_baseline(name):
    if not Path(DATASETS[name]["source"]).is_dir():
        pytest.skip(f"{DATASETS[name]['source']} not on this machine")
    if not (HERE / "baselines" / f"{name}.json").exists():
        pytest.skip(f"no baseline for {name}")
    sys.path.insert(0, str(REPO / "scripts"))
    import albumify_regression

    assert albumify_regression.main(["check", name]) == 0, f"{name} drifted from its baseline (see output)"
