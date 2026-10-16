"""Offline regression tests for RAM++ input device routing."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from sim_bench.triage.content import RamTagger


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tag_moves_input_to_model_device(tmp_path, monkeypatch, device):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    inference = Mock(return_value=("person | lake", ""))
    monkeypatch.setitem(sys.modules, "ram", SimpleNamespace(inference_ram=inference))
    tagger = RamTagger.__new__(RamTagger)
    tagger.device = device
    tagger._model = object()
    tagger._tf = Mock()
    batch = tagger._tf.return_value.unsqueeze.return_value

    assert tagger.tag(image_path) == ["person", "lake"]

    batch.to.assert_called_once_with(device)
    inference.assert_called_once_with(batch.to.return_value, tagger._model)
