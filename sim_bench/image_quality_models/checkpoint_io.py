"""Portable checkpoint loading.

Checkpoints trained on Windows may pickle `pathlib.WindowsPath` objects (e.g. a training
`config.output_dir`). Unpickling those on Linux raises
``NotImplementedError: cannot instantiate 'WindowsPath' on your system`` — which is how the
Siamese checkpoint broke the Modal worker (spec-110, 2026-10-09). The weights are fine; only the
pickled path class is foreign. This loader aliases the foreign path class to the native one for the
duration of the load.
"""
from __future__ import annotations

import logging
import os
import pathlib
from contextlib import contextmanager
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)


@contextmanager
def _foreign_paths_as_native() -> Iterator[None]:
    """Make the other OS's concrete Path class unpickle as this OS's class."""
    if os.name == "nt":
        foreign, native = "PosixPath", pathlib.WindowsPath
    else:
        foreign, native = "WindowsPath", pathlib.PosixPath
    original = getattr(pathlib, foreign)
    setattr(pathlib, foreign, native)
    try:
        yield
    finally:
        setattr(pathlib, foreign, original)


def load_checkpoint(path: str | os.PathLike, device: str = "cpu") -> Any:
    """`torch.load` that tolerates path objects pickled on the other OS."""
    with _foreign_paths_as_native():
        return torch.load(path, map_location=device, weights_only=False)
