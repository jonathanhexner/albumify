"""ut_CheckpointIO — portable checkpoint loading (Windows-pickled paths on Linux and back)."""
import pathlib
import pickle

import pytest
import torch

from sim_bench.image_quality_models import checkpoint_io


def test_loads_checkpoint_with_native_path(tmp_path):
    ck = {"model_state_dict": {"w": torch.zeros(2)}, "config": {"output_dir": pathlib.Path("outputs/x")}}
    f = tmp_path / "ck.pt"
    torch.save(ck, f)
    out = checkpoint_io.load_checkpoint(f, "cpu")
    assert torch.equal(out["model_state_dict"]["w"], torch.zeros(2))
    assert isinstance(out["config"]["output_dir"], pathlib.Path)


def test_foreign_path_class_is_aliased_during_load_only():
    foreign = "PosixPath" if checkpoint_io.os.name == "nt" else "WindowsPath"
    before = getattr(pathlib, foreign)
    with checkpoint_io._foreign_paths_as_native():
        assert getattr(pathlib, foreign) is type(pathlib.Path())
    assert getattr(pathlib, foreign) is before


def _foreign_path_pickle() -> bytes:
    """A pickle that instantiates the OTHER OS's concrete Path class, built from opcodes
    (GLOBAL pathlib.<Foreign> ; MARK ; BINUNICODE 'outputs/x' ; TUPLE ; REDUCE ; STOP)."""
    foreign = b"PosixPath" if checkpoint_io.os.name == "nt" else b"WindowsPath"
    arg = b"outputs/x"
    GLOBAL, MARK, BINUNICODE, TUPLE, REDUCE, STOP, NL = b"c", b"(", b"X", b"t", b"R", b".", b"\n"
    return (GLOBAL + b"pathlib" + NL + foreign + NL
            + MARK + BINUNICODE + len(arg).to_bytes(4, "little") + arg + TUPLE
            + REDUCE + STOP)


def test_foreign_path_pickle_fails_without_alias_and_loads_with_it():
    payload = _foreign_path_pickle()
    with pytest.raises(NotImplementedError):
        pickle.loads(payload)  # the exact failure the Modal worker hit
    with checkpoint_io._foreign_paths_as_native():
        out = pickle.loads(payload)
    assert isinstance(out, pathlib.Path) and out.name == "x"
