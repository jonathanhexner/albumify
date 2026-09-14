"""Importable step fixtures for the spec-104 process-isolation tests.

These MUST live in a real, importable module (not as classes defined inside a
test function) because the 'spawn' child re-imports each step by its module +
qualname. Filename is deliberately not ``test_*`` so pytest doesn't collect it.
"""
from sim_bench.pipeline.base import BaseStep, StepMetadata


class WriteValueStep(BaseStep):
    """Writes a known value into a produced context key (proves the child's
    mutations are marshalled back onto the parent context)."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="iso_write", display_name="Write", description="",
            category="test", requires=set(), produces={"iqa_scores"}, depends_on=[],
        )

    def process(self, context, config) -> None:
        context.iqa_scores = {"a.jpg": 0.9, "b.jpg": 0.1}


class ChildPidStep(BaseStep):
    """Records the PID it runs under + allocates a large array, so the test can
    assert (a) it ran in a different process and (b) the parent didn't inherit
    the allocation."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="iso_pid", display_name="Pid", description="",
            category="test", requires=set(), produces={"metrics"}, depends_on=[],
        )

    def process(self, context, config) -> None:
        import os
        import numpy as np
        big = np.ones((60_000_000,), dtype=np.float64)  # ~480 MB, lives only in the child
        context.metrics = {"child_pid": os.getpid(), "checksum": float(big[:3].sum())}
        del big


_LEAKED = []  # module global that survives the step call — stands in for a torch model


class LeakStep(BaseStep):
    """Stashes ~480 MB in a module global that is NOT freed when the step returns
    — the way a torch model's RSS lingers (SIGHTING-117). In-process this bloats
    the parent; isolated, it dies with the child and the parent's RSS is unchanged."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="iso_leak", display_name="Leak", description="",
            category="test", requires=set(), produces=set(), depends_on=[],
        )

    def process(self, context, config) -> None:
        import numpy as np
        _LEAKED.append(np.ones((60_000_000,), dtype=np.float64))  # ~480 MB, retained


class RaiseStep(BaseStep):
    """Raises inside the child — should come back as a clean failed StepResult."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="iso_raise", display_name="Raise", description="",
            category="test", requires=set(), produces=set(), depends_on=[],
        )

    def process(self, context, config) -> None:
        raise ValueError("boom in child")


class HardCrashStep(BaseStep):
    """Simulates an OOM-kill / segfault: the child dies WITHOUT shipping a result.
    The parent must survive and report a failed StepResult (today an OOM kills the
    whole run silently — that's the crash-handling this proves)."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="iso_crash", display_name="Crash", description="",
            category="test", requires=set(), produces=set(), depends_on=[],
        )

    def process(self, context, config) -> None:
        import os
        os._exit(137)  # hard exit, no terminal message on the queue
