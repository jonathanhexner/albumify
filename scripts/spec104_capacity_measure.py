"""spec-104 capacity measurement (the §5 real-pipeline probe).

Runs the representative-heavy `faces` pipeline on N real images (Germany 768px) and
records the PEAK process-tree RSS — parent + any isolated step-children — for
in-process vs isolated. Answers "how many images can we run now".

Long run: real models (YOLO / InsightFace / DINOv2 / AVA / pyiqa) load and process
every image, cold cache. Isolated runs first so the capacity numbers land first.

Run:  .venv/Scripts/python scripts/spec104_capacity_measure.py
"""
import sys
import time
import json
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (+ spawn children)

GERMANY = Path(r"D:\albumify_vs_vlm\germany\imgs768")
MATRIX = [(100, True), (300, True), (500, True), (100, False), (300, False), (500, False)]


def _faces_steps_and_configs():
    import yaml
    from sim_bench.albumify_vs_vlm.albumify_arm import _FACES_DROP
    doc = yaml.safe_load(open("configs/pipeline.yaml"))
    steps = [s for s in doc["default_pipeline"] if s not in _FACES_DROP]
    step_configs = {s: (doc.get(s) or {}) for s in steps}
    return steps, step_configs


def _pipeline_worker(N, isolate, result_path):
    try:
        import sim_bench.pipeline.steps.all_steps  # noqa: F401 — registers steps
        from sim_bench.pipeline.context import PipelineContext
        from sim_bench.pipeline.config import PipelineConfig
        from sim_bench.pipeline.executor import PipelineExecutor
        from sim_bench.pipeline.registry import get_registry

        steps, step_configs = _faces_steps_and_configs()
        step_configs.setdefault("discover_images", {})["limit"] = N

        ctx = PipelineContext()
        ctx.source_directory = GERMANY
        cfg = PipelineConfig(step_configs=step_configs, fail_fast=False,
                             isolate_steps=isolate, isolate_step_timeout_s=300.0)
        res = PipelineExecutor(get_registry()).execute(ctx, steps, cfg)
        Path(result_path).write_text(json.dumps({
            "success": res.success,
            "n_images": len(getattr(ctx, "image_paths", []) or []),
            "n_faces": len(getattr(ctx, "face_records", []) or []),
            "failed_step": res.failed_step,
            "error": (res.error_message or "")[:160],
        }))
    except Exception as e:
        import traceback
        Path(result_path).write_text(json.dumps(
            {"success": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-400:]}))


def _tree_rss_mb(ps):
    import psutil
    try:
        total = ps.memory_info().rss
        for c in ps.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total / 1e6
    except psutil.Error:
        return 0.0


def _measure(N, isolate):
    import psutil
    result_path = Path(f"scripts/_cap_{N}_{'iso' if isolate else 'inp'}.json")
    if result_path.exists():
        result_path.unlink()
    proc = mp.get_context("spawn").Process(target=_pipeline_worker, args=(N, isolate, str(result_path)))
    t0 = time.time()
    proc.start()
    ps = psutil.Process(proc.pid)
    peak = 0.0
    while proc.is_alive():
        peak = max(peak, _tree_rss_mb(ps))
        time.sleep(0.15)
    proc.join()
    dur = time.time() - t0
    if result_path.exists():
        res = json.loads(result_path.read_text())
        result_path.unlink()
    else:
        res = {"success": False, "error": f"worker died (exitcode={proc.exitcode}) — likely OOM"}
    return {"N": N, "isolate": isolate, "peak_mb": round(peak), "dur_s": round(dur), **res}


def main():
    rows = []
    for N, isolate in MATRIX:
        print(f"running N={N} {'isolated' if isolate else 'in-process'} ...", flush=True)
        r = _measure(N, isolate)
        rows.append(r)
        print("  ->", json.dumps(r), flush=True)
    print("\n=== spec-104 capacity — peak process-tree RSS (faces pipeline, Germany 768px) ===")
    print(f"{'N':>5} {'mode':>11} {'peak_MB':>8} {'dur_s':>6} {'ok':>6} {'imgs':>5} {'faces':>6}")
    for r in rows:
        print(f"{r['N']:>5} {'isolated' if r['isolate'] else 'in-process':>11} "
              f"{r['peak_mb']:>8} {r['dur_s']:>6} {str(r.get('success')):>6} "
              f"{r.get('n_images','-'):>5} {r.get('n_faces','-'):>6}")


if __name__ == "__main__":
    main()
