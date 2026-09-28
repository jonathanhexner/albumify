"""spec-104 v1 probe: parent-process RSS when a step leaks ~480 MB.

Runs the same leaking step in-process (flag off) vs isolated (flag on) and prints
the parent's RSS delta. This is the SIGHTING-117 headline: a subprocess that exits
returns 100% of the leaked memory to the OS.

Run:  .venv/Scripts/python scripts/spec104_isolation_probe.py
(Must be run as a file, not piped stdin — 'spawn' bootstraps the child from __main__.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for tests.* + spawn children

from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import StepRegistry
from tests.pipeline import _isolation_fixtures as fx


def _rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def _run(isolate: bool):
    registry = StepRegistry()
    registry.register(fx.LeakStep)
    executor = PipelineExecutor(registry)
    ctx = PipelineContext()
    before = _rss_mb()
    result = executor.execute(ctx, ["iso_leak"], config=PipelineConfig(isolate_steps=isolate))
    after = _rss_mb()
    return before, after, after - before, result.success


def main():
    print("=== spec-104: parent RSS after a step that leaks ~480 MB ===")
    b1, a1, d1, ok1 = _run(False)
    print(f"IN-PROCESS (flag off): {b1:6.0f} -> {a1:6.0f} MB   delta +{d1:6.0f} MB   ok={ok1}")
    b2, a2, d2, ok2 = _run(True)
    print(f"ISOLATED   (flag on):  {b2:6.0f} -> {a2:6.0f} MB   delta +{d2:6.0f} MB   ok={ok2}")
    reclaimed = d1 - d2
    print(f"\n==> isolation reclaimed ~{reclaimed:.0f} MB of the {d1:.0f} MB leaked "
          f"({100 * reclaimed / max(d1, 1):.0f}%); isolated step succeeded={ok2}")


if __name__ == "__main__":
    main()
