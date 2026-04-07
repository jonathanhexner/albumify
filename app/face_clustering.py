"""Face Clustering Streamlit app — 5-tab analysis hierarchy.

Architecture: all heavy computation (pipeline run, UMAP, ClusterView, FaceView)
runs in a daemon background thread. The UI thread never blocks — it polls
session_state and calls st.rerun() to refresh until work is complete.

Tabs:
  1. Run             - Execute a new pipeline run with live log stream
  2. History         - Browse past runs, inspect files, load into analysis
  3. Run Overview    - Full-run health: quality funnel, cluster table, UMAP
  4. Cluster Analysis - Per-cluster: exemplars, outliers, split signal, nearest clusters
  5. Face Analysis   - Per-face: attributes, closest same/other cluster faces

Usage:
    .venv/Scripts/streamlit run app/face_clustering.py
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from face_cluster import FaceClusteringPipeline, PipelineConfig
from face_cluster.analysis_views import ClusterView, ClusterDebugView, FaceView, RunOverview
from face_cluster.loader import load_pipeline_result

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Schema descriptions for every file the pipeline produces.
_RUN_FILE_DEFS = [
    {
        "file":        "pipeline_run.json",
        "format":      "JSON",
        "description": "Run record: config, per-stage status/timing, final summary, errors",
        "schema":      "run_id, source_album, started_at, config{distance_threshold,blur_min,...}, "
                       "stages{name:{status,elapsed_s}}, status, summary{n_faces,n_core,n_clusters,n_noise}",
    },
    {
        "file":        "faces.csv",
        "format":      "CSV",
        "description": "One row per detected face (core + holdout)",
        "schema":      "face_id, image_path, image_id, crop_path, cluster_id, is_core, "
                       "blur_score, area, yaw, pitch, roll",
    },
    {
        "file":        "clusters.csv",
        "format":      "CSV",
        "description": "One row per cluster",
        "schema":      "cluster_id, size, exemplar_face_ids, diameter",
    },
    {
        "file":        "embeddings.npy",
        "format":      "NumPy",
        "description": "L2-normalised ArcFace embeddings, float32, one row per face",
        "schema":      "shape (n_faces x 512) — row order matches faces.csv",
    },
    {
        "file":        "embedding_face_ids.npy",
        "format":      "NumPy",
        "description": "face_id for each row in embeddings.npy",
        "schema":      "shape (n_faces,) int32",
    },
    {
        "file":        "crop_manifest.json",
        "format":      "JSON",
        "description": "Mapping from face_id (str) to relative path of its aligned crop",
        "schema":      '{"0": "crops/face_0000_aligned.jpg", "1": "crops/face_0001_aligned.jpg", ...}',
    },
    {
        "file":        "export_summary.json",
        "format":      "JSON",
        "description": "High-level summary saved by the export stage",
        "schema":      "source_album, run_id, created_at, n_faces, n_core, n_clusters, n_noise, config",
    },
    {
        "file":        "crops/",
        "format":      "JPEG",
        "description": "Aligned face crops, 112x112 px (ArcFace input size), one per face",
        "schema":      "crops/face_XXXX_aligned.jpg  (XXXX = zero-padded face_id)",
    },
    {
        "file":        "logs/",
        "format":      "Text",
        "description": "Per-run pipeline log capturing all stages at DEBUG level",
        "schema":      "logs/run_YYYYMMDD_HHMMSS.log",
    },
]

_LOG_LIVE_LINES = 30    # lines shown in live log during a run
_LOG_MAX_STORED = 500   # cap on accumulated log lines in session_state


# ---------------------------------------------------------------------------
# Logging: QueueHandler routes face_cluster.* log records to the UI
# ---------------------------------------------------------------------------

class _QueueHandler(logging.Handler):
    """Puts formatted log records into a queue for the UI to consume."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(levelname)-8s %(name)s  %(message)s"))

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AsyncWorker: run any callable in a background daemon thread
# ---------------------------------------------------------------------------

class _AsyncState:
    """Shared state between background thread and UI thread.

    The background thread writes only to self.result / self.error / self.status
    and self.log_q.  Session state is never written from the thread — the render
    thread applies results after observing worker.is_done.
    """
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"

    def __init__(self):
        self.status  = self.IDLE
        self.result  = None
        self.error   = None
        self.log_q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self): return self.status == self.RUNNING
    @property
    def is_done(self):    return self.status == self.DONE
    @property
    def has_error(self):  return self.status == self.ERROR

    def start(self, fn, *args, **kwargs):
        if self.is_running:
            return
        self.status = self.RUNNING
        self.result = None
        self.error  = None
        while not self.log_q.empty():
            try: self.log_q.get_nowait()
            except queue.Empty: break

        handler = _QueueHandler(self.log_q)
        fc_logger = logging.getLogger("face_cluster")
        fc_logger.addHandler(handler)
        fc_logger.setLevel(logging.DEBUG)

        def _run():
            try:
                self.result = fn(*args, **kwargs)
                self.status = self.DONE
            except Exception as exc:
                self.error = str(exc)
                self.status = self.ERROR
                import traceback
                self.log_q.put_nowait("ERROR: " + traceback.format_exc())
            finally:
                fc_logger.removeHandler(handler)

        self._thread = threading.Thread(target=_run, name="_run", daemon=True)
        self._thread.start()

    def drain_logs(self) -> list[str]:
        lines = []
        while True:
            try:
                lines.append(self.log_q.get_nowait())
            except queue.Empty:
                break
        return lines


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "pipeline_result":   None,   # PipelineResult
        "pipeline_worker":   None,   # _AsyncState for pipeline run
        "pipeline_log":      [],     # accumulated log lines for current run
        "active_run_dir":    None,   # Path to output_dir of the currently running pipeline
        "overview_worker":   None,   # _AsyncState for RunOverview.compute
        "cluster_worker":    None,   # _AsyncState for ClusterView.compute
        "cluster_debug_worker": None, # _AsyncState for ClusterDebugView.compute
        "face_worker":       None,   # _AsyncState for FaceView.compute
        "selected_cluster":  None,
        "selected_face":     None,
        "manifest_cache":    None,   # (output_dir_str, manifest_dict)
        "faces_df_cache":    None,   # (output_dir_str, faces_df)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(output_dir: Path) -> dict:
    cached = st.session_state.manifest_cache
    if cached and cached[0] == str(output_dir):
        return cached[1]
    path = output_dir / "crop_manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    st.session_state.manifest_cache = (str(output_dir), m)
    return m


def _load_faces_df(output_dir: Path) -> pd.DataFrame:
    """Read faces.csv once per run, cached in session_state."""
    cached = st.session_state.faces_df_cache
    if cached and cached[0] == str(output_dir):
        return cached[1]
    df = pd.read_csv(output_dir / "faces.csv")
    st.session_state.faces_df_cache = (str(output_dir), df)
    return df


def _crop_for_face(face_id: int, output_dir: Path) -> Optional[Image.Image]:
    manifest = _load_manifest(output_dir)
    rel = manifest.get(str(face_id))
    if not rel:
        return None
    p = output_dir / rel
    return Image.open(p) if p.exists() else None


def _invalidate_run_caches():
    """Clear all per-run cached state when a new run is loaded."""
    st.session_state.overview_worker       = None
    st.session_state.cluster_worker        = None
    st.session_state.cluster_debug_worker  = None
    st.session_state.face_worker           = None
    st.session_state.selected_cluster = None
    st.session_state.selected_face    = None
    st.session_state.manifest_cache   = None
    st.session_state.faces_df_cache   = None


def _breadcrumb():
    result = st.session_state.pipeline_result
    parts = []
    if result:
        parts.append(f"Run: **{Path(result.output_dir).name}**")
    if st.session_state.selected_cluster is not None:
        parts.append(f"Cluster **{st.session_state.selected_cluster}**")
    if st.session_state.selected_face is not None:
        parts.append(f"Face **{st.session_state.selected_face}**")
    if parts:
        st.caption(" > ".join(parts))


def _no_result():
    st.info("No run loaded. Use **Run** to start a new pipeline run, or **History** to load a past run.")


def _render_live_log(log_lines: list[str]):
    """Show last N log lines directly (no expander) — always visible during a run."""
    tail = log_lines[-_LOG_LIVE_LINES:]
    st.code("\n".join(tail) if tail else "(no log output yet)", language=None)


def _render_log_expander(log_lines: list[str], label: str = "Log"):
    if log_lines:
        with st.expander(label, expanded=True):
            st.code("\n".join(log_lines[-200:]), language=None)


_STAGE_ORDER = ["discover", "embed", "quality", "crops", "cluster", "exemplars", "export",
                "split", "merge", "attach"]

_STATUS_ICON = {
    "pending": "...",
    "running": ">>",
    "done":    "OK",
    "failed":  "!!",
}

def _render_stage_plan(run_dir: Optional[Path]):
    """Read pipeline_run.json and render a live stage status table."""
    if run_dir is None:
        return
    run_json = run_dir / "pipeline_run.json"
    if not run_json.exists():
        st.caption("Waiting for pipeline to write stage data...")
        return

    try:
        with open(run_json, encoding="utf-8") as f:
            rec = json.load(f)
    except Exception:
        return  # file mid-write — skip this render cycle

    stages = rec.get("stages", {})
    rows = []
    for name in _STAGE_ORDER:
        if name not in stages:
            continue
        s = stages[name]
        status = s.get("status", "pending")
        started = s.get("started_at", "")
        started_fmt = started[11:19] if len(started) >= 19 else ""   # HH:MM:SS only
        elapsed = s.get("elapsed_s")
        elapsed_fmt = f"{elapsed:.1f}s" if elapsed is not None else ("..." if status == "running" else "")
        rows.append({
            "Stage":   name,
            "Status":  _STATUS_ICON.get(status, status),
            "Started": started_fmt,
            "Elapsed": elapsed_fmt,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Run files panel (shared by Run tab and History tab)
# ---------------------------------------------------------------------------

def _render_run_files_panel(run_dir: Path, key_prefix: str = ""):
    """Show all data files produced by a pipeline run: path, size, content, schema.

    key_prefix must be unique per call site so Streamlit can distinguish widgets
    rendered in different tabs during the same render pass.
    """
    st.markdown(f"**Run directory:** `{run_dir.resolve()}`")

    rows = []
    for fd in _RUN_FILE_DEFS:
        p = run_dir / fd["file"]
        exists = p.exists()
        size_str = ""
        items_str = ""

        if exists:
            if p.is_file():
                sz = p.stat().st_size
                size_str = f"{sz / 1024:.1f} KB" if sz < 1_000_000 else f"{sz / 1_000_000:.1f} MB"
                if fd["file"].endswith(".csv"):
                    try:
                        with open(p, encoding="utf-8") as fh:
                            items_str = f"{sum(1 for _ in fh) - 1} rows"
                    except Exception as e:
                        items_str = f"read error: {e}"
                elif fd["file"] == "crop_manifest.json":
                    try:
                        with open(p, encoding="utf-8") as fh:
                            items_str = f"{len(json.load(fh))} entries"
                    except Exception as e:
                        items_str = f"read error: {e}"
                elif fd["file"].endswith(".npy"):
                    try:
                        arr = np.load(p)
                        items_str = f"shape {arr.shape}"
                    except Exception as e:
                        items_str = f"read error: {e}"
            elif p.is_dir():
                files = list(p.rglob("*"))
                n_files = sum(1 for f in files if f.is_file())
                total_sz = sum(f.stat().st_size for f in files if f.is_file())
                items_str = f"{n_files} files"
                size_str = f"{total_sz / 1_000_000:.1f} MB"

        rows.append({
            "File":        fd["file"],
            "Format":      fd["format"],
            "Status":      "OK" if exists else "MISSING",
            "Size":        size_str,
            "Items":       items_str,
            "Description": fd["description"],
            "Schema":      fd["schema"],
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True)

    col1, col2 = st.columns(2)
    run_json = run_dir / "pipeline_run.json"
    if run_json.exists():
        with col1:
            st.download_button(
                "Download pipeline_run.json",
                run_json.read_bytes(),
                file_name=f"{run_dir.name}_pipeline_run.json",
                mime="application/json",
                key=f"{key_prefix}dl_run_json",
            )

    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("run_*.log"), reverse=True)
        if log_files:
            with col2:
                st.download_button(
                    f"Download {log_files[0].name}",
                    log_files[0].read_bytes(),
                    file_name=f"{run_dir.name}_{log_files[0].name}",
                    mime="text/plain",
                    key=f"{key_prefix}dl_log",
                )


# ---------------------------------------------------------------------------
# Tab 1: Run
# ---------------------------------------------------------------------------

def render_run_tab():
    st.header("Run Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        image_dir = st.text_input("Image directory", placeholder=r"D:\Google_Germany")
    with col2:
        output_dir_str = st.text_input("Output directory", placeholder=r"results\my_album")

    # Algorithm documentation
    with st.expander("How the clustering algorithm works"):
        st.markdown("""
**Mutual kNN Graph + Connected Components**

The algorithm clusters faces in two steps:

1. **Build a mutual kNN graph.** For each face, find its K nearest neighbors
   by cosine distance on 512-d ArcFace embeddings. An edge between face A and
   face B is created only if **both** A is in B's top-K **and** B is in A's top-K
   (mutual requirement), **and** their distance is below `distance_threshold`.
   The mutual constraint prevents asymmetric false edges (e.g. a blurry face
   matching many sharp faces, but not vice versa).

2. **Find connected components.** Each connected component in the graph becomes
   a cluster. Components smaller than `min_cluster_size` become noise.
   This is a **single-pass** operation, not iterative.

**Known limitation — chain connections (transitive closure):**
If face A is similar to B, and B is similar to C, connected components will
group {A, B, C} together even if A and C are very different.  This is how
different people can end up in the same cluster — a "bridge" face connects
two distinct identity groups.

**Key parameters and their effect:**

| Parameter | Lower value | Higher value |
|-----------|------------|--------------|
| `K` | Fewer edges, more noise/singletons | More edges, risk of chain connections |
| `distance_threshold` | Strict — only very similar faces connect | Loose — allows more variation but more false edges |
| `min_cluster_size` | Keeps small clusters (pairs) | Forces larger groups, more faces become noise |

**Optional post-processing stages:**
- **Split** — re-clusters wide clusters (diameter > threshold) with tighter params to break chains
- **Merge** — merges cluster pairs with close exemplars using multi-evidence criteria
- **Attach** — assigns holdout (low-quality) faces to nearest cluster via majority vote
""")

    # Pipeline stage config — each section mirrors one pipeline stage
    with st.expander("Pipeline Config", expanded=True):

        # Stage 1 & 2 have no tunable config
        st.markdown("**Stage 1 · Discover** — scan image dir for .jpg/.jpeg/.png/.heic")
        st.markdown("**Stage 2 · Embed** — InsightFace buffalo_l: detect faces, ArcFace 512-d embeddings, 1k3d68 pose")
        st.divider()

        # Stage 3 · Quality gate
        st.markdown("**Stage 3 · Quality Gate** — failed faces go to holdout (not clustered)")
        c1, c2, c3 = st.columns(3)
        with c1:
            blur_min = st.slider("blur_min", 0.0, 200.0, 50.0, 5.0,
                help="Min Laplacian variance. Blurry crops score low; 50 rejects noticeably soft faces.")
        with c2:
            max_faces = st.slider("max_faces_per_image_core", 1, 10, 3,
                help="Keep only the N largest faces per source image.")
        with c3:
            min_face_area = st.number_input("min_face_area px (0=off)", min_value=0, value=0, step=500,
                help="Reject faces whose bbox area is smaller than this. 0 = disabled.")

        st.markdown("*Pose filter (yaw/pitch/roll from InsightFace 1k3d68)*")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            yaw_max = st.slider("yaw_max °", 5.0, 90.0, 30.0, 1.0,
                help="Max absolute yaw. 0° = facing camera directly. 30° allows moderate left/right turn.")
        with c2:
            pitch_max = st.slider("pitch_max °", 5.0, 90.0, 25.0, 1.0,
                help="Max absolute pitch. 25° allows slight chin-up/down.")
        with c3:
            roll_max = st.slider("roll_max °", 5.0, 90.0, 25.0, 1.0,
                help="Max absolute roll (head tilt). 25° allows casual tilt; raise to 90 to keep sideways shots.")
        with c4:
            require_pose = st.checkbox("require_pose",
                help="If checked, faces where pose detection failed are also sent to holdout.")
        st.divider()

        # Stage 4 has no config
        st.markdown("**Stage 4 · Crops** — save aligned 112x112 crops + crop_manifest.json")
        st.divider()

        # Stage 5 · Cluster
        st.markdown("**Stage 5 · Cluster** — mutual kNN graph + connected components")
        c1, c2, c3 = st.columns(3)
        with c1:
            K = st.slider("K (kNN neighbours)", 1, 20, 5,
                help="Neighbours considered per face when building the mutual kNN graph.")
        with c2:
            distance_threshold = st.slider("distance_threshold", 0.1, 0.8, 0.35, 0.01,
                help="Max cosine distance for a kNN edge. Lower = stricter separation.")
        with c3:
            min_cluster_size = st.slider("min_cluster_size", 1, 10, 2,
                help="Clusters smaller than this become noise.")
        st.divider()

        # Stage 6 · Exemplars
        st.markdown("**Stage 6 · Exemplars** — select representative faces per cluster")
        c1, c2, c3 = st.columns(3)
        with c1:
            N_exemplars_max = st.slider("N_exemplars_max", 1, 20, 10,
                help="Maximum exemplars per cluster.")
        with c2:
            exemplars_d10_threshold = st.slider("exemplars_d10_threshold", 0.1, 0.8, 0.35, 0.01,
                help="Face is exemplar candidate only if d10 distance (Kth-NN dist) is below this.")
        with c3:
            exemplar_suppression_radius = st.slider("exemplar_suppression_radius", 0.05, 0.5, 0.2, 0.01,
                help="Min distance between two exemplars — suppresses near-duplicates.")
        st.divider()

        # Stage 7 has no config
        st.markdown("**Stage 7 · Export** — write faces.csv, clusters.csv, embeddings.npy, export_summary.json")
        st.divider()

        # Optional stages
        st.markdown("**Optional Stages** — disabled by default")
        c1, c2, c3 = st.columns(3)
        with c1:
            split_enabled = st.checkbox("split_enabled",
                help="Re-cluster any cluster whose diameter exceeds split_diameter_threshold.")
        with c2:
            merge_enabled = st.checkbox("merge_enabled",
                help="Merge cluster pairs where exemplars are closer than an adaptive threshold.")
        with c3:
            attach_enabled = st.checkbox("attach_enabled",
                help="Attach holdout faces to nearest cluster via majority vote among K-NN core faces.")

    worker: _AsyncState = st.session_state.pipeline_worker

    # -- Running state: show live stage progress + visible log ---------------
    if worker is not None and worker.is_running:
        new_lines = worker.drain_logs()
        if new_lines:
            log = st.session_state.pipeline_log
            log.extend(new_lines)
            if len(log) > _LOG_MAX_STORED:
                st.session_state.pipeline_log = log[-_LOG_MAX_STORED:]

        st.info("Pipeline running — switch to other tabs freely.")
        st.subheader("Stage Execution Plan")
        _render_stage_plan(st.session_state.active_run_dir)
        st.subheader("Live Log")
        _render_live_log(st.session_state.pipeline_log)
        time.sleep(0.5)
        st.rerun()
        return

    # -- Done state ----------------------------------------------------------
    if worker is not None and worker.is_done:
        # Apply result to session_state on the render thread (not the worker thread)
        if worker.result is not None and st.session_state.pipeline_result is None:
            st.session_state.pipeline_result = worker.result
            _invalidate_run_caches()

        result = st.session_state.pipeline_result
        if result is not None:
            st.success("Pipeline complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faces",    result.summary["n_faces"])
            m2.metric("Core",     result.summary["n_core"])
            m3.metric("Clusters", result.summary["n_clusters"])
            m4.metric("Noise",    result.summary["n_noise"])
            st.subheader("Stage Execution Plan")
            _render_stage_plan(st.session_state.active_run_dir)
            st.info("Switch to **Run Overview** to analyse.")
            _render_log_expander(st.session_state.pipeline_log, label="Run log")
            with st.expander("Output Files", expanded=False):
                _render_run_files_panel(Path(result.output_dir), key_prefix="run_tab_")

    # -- Error state ---------------------------------------------------------
    if worker is not None and worker.has_error:
        st.error(f"Pipeline failed: {worker.error}")
        _render_log_expander(st.session_state.pipeline_log, label="Error log")

    # -- Run button ----------------------------------------------------------
    run_disabled = not (image_dir and output_dir_str)
    if st.button("Run Pipeline", type="primary", disabled=run_disabled):
        config = PipelineConfig(
            K=K,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
            blur_min=blur_min,
            max_faces_per_image_core=max_faces,
            min_face_area=min_face_area if min_face_area > 0 else None,
            yaw_max=yaw_max,
            pitch_max=pitch_max,
            roll_max=roll_max,
            require_pose=require_pose,
            N_exemplars_max=N_exemplars_max,
            exemplars_d10_threshold=exemplars_d10_threshold,
            exemplar_suppression_radius=exemplar_suppression_radius,
            split_enabled=split_enabled,
            merge_enabled=merge_enabled,
            attach_enabled=attach_enabled,
        )

        def _run_pipeline():
            pipeline = FaceClusteringPipeline(config)
            # Returns result — render thread applies it to session_state
            return pipeline.run(image_dir, output_dir_str)

        new_worker = _AsyncState()
        st.session_state.pipeline_worker  = new_worker
        st.session_state.pipeline_log     = []
        st.session_state.active_run_dir   = Path(output_dir_str)
        # Reset result so done-state handler applies the new one
        st.session_state.pipeline_result  = None
        new_worker.start(_run_pipeline)
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: History
# ---------------------------------------------------------------------------

def render_history_tab():
    st.header("History")

    results_root = st.text_input("Results root directory", value="results")
    root = Path(results_root)
    if not root.exists():
        st.warning(f"Directory not found: {root}")
        return

    # Collect runs, sorted by started_at descending
    runs = []
    for run_dir in root.iterdir():
        summary_path = run_dir / "pipeline_run.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                rec = json.load(f)
            runs.append({
                "run_id":        rec.get("run_id", run_dir.name),
                "output_folder": run_dir.name,
                "album":         Path(rec.get("source_album", "")).name,
                "started":       rec.get("started_at", "")[:19],
                "status":        rec.get("status", "?"),
                "faces":         rec.get("summary", {}).get("n_faces", "?"),
                "clusters":      rec.get("summary", {}).get("n_clusters", "?"),
                "noise":         rec.get("summary", {}).get("n_noise", "?"),
                "_dir":          str(run_dir),
                "_started_sort": rec.get("started_at", ""),
            })
        except Exception:
            continue

    runs.sort(key=lambda r: r["_started_sort"], reverse=True)

    if not runs:
        st.info("No completed runs found. Run a pipeline first.")
        return

    st.markdown(f"**{len(runs)} run(s)**")
    display_cols = ["run_id", "output_folder", "album", "started", "status", "faces", "clusters", "noise"]
    st.dataframe(pd.DataFrame(runs)[display_cols], hide_index=True)

    selected_id = st.selectbox("Select run to inspect", [r["run_id"] for r in runs])
    selected = next(r for r in runs if r["run_id"] == selected_id)
    selected_dir = Path(selected["_dir"])

    st.divider()
    with st.expander("Data Files", expanded=True):
        _render_run_files_panel(selected_dir, key_prefix="hist_tab_")

    st.divider()
    # Guard: check if required files exist before allowing load
    run_status = selected.get("status", "?")
    has_faces_csv = (selected_dir / "faces.csv").exists()
    is_incomplete = run_status != "complete" or not has_faces_csv

    if is_incomplete:
        missing = []
        for required in ["faces.csv", "clusters.csv", "embeddings.npy"]:
            if not (selected_dir / required).exists():
                missing.append(required)
        st.warning(
            f"Run **{selected_id}** is incomplete (status: `{run_status}`). "
            + (f"Missing files: {', '.join(missing)}. " if missing else "")
            + "The pipeline likely crashed or was interrupted before the export stage. "
            "Re-run the pipeline to generate all output files."
        )

    already_loaded = (
        st.session_state.pipeline_result is not None
        and Path(st.session_state.pipeline_result.output_dir) == selected_dir
    )
    if already_loaded:
        st.success(f"Run **{selected_id}** is currently loaded.")
    elif is_incomplete:
        st.button("Load into analysis tabs", type="primary", disabled=True,
                   help="Cannot load — required files are missing")
    else:
        if st.button("Load into analysis tabs", type="primary"):
            try:
                result = load_pipeline_result(selected_dir)
                st.session_state.pipeline_result = result
                _invalidate_run_caches()
                st.success("Loaded. Switch to **Run Overview** to analyse.")
                st.rerun()
            except Exception as exc:
                st.error(f"Load failed: {exc}")


# ---------------------------------------------------------------------------
# Tab 3: Run Overview
# ---------------------------------------------------------------------------

def render_run_overview_tab():
    st.header("Run Overview")
    _breadcrumb()

    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return

    worker: _AsyncState = st.session_state.overview_worker

    if worker is None:
        w = _AsyncState()
        st.session_state.overview_worker = w
        w.start(RunOverview.compute, result)
        st.rerun()
        return

    if worker.is_running:
        st.info("Computing run overview (UMAP may take ~10s)...")
        time.sleep(0.5)
        st.rerun()
        return

    if worker.has_error:
        st.error(f"Run overview failed: {worker.error}")
        return

    overview = worker.result

    st.subheader("Quality Gate Funnel")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images",           overview.n_images)
    c2.metric("Faces detected",   overview.n_faces_detected)
    c3.metric("Core (passed)",    overview.n_core)
    c4.metric("Holdout (failed)", overview.n_holdout)
    pass_rate = 100 * overview.n_core / max(overview.n_faces_detected, 1)
    st.caption(f"Pass rate: {pass_rate:.0f}%")

    st.subheader("Clustering Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters",        overview.n_clusters)
    c2.metric("Noise faces",     overview.n_noise)
    c3.metric("Clustered faces", overview.n_core - overview.n_noise)

    if overview.stage_timings:
        with st.expander("Stage timings"):
            tdf = pd.DataFrame([
                {"stage": s, "seconds": f"{v:.1f}s" if v else "-"}
                for s, v in overview.stage_timings.items()
            ])
            st.dataframe(tdf, hide_index=True)

    st.subheader("All Clusters")
    if overview.cluster_rows:
        rows = sorted(overview.cluster_rows, key=lambda r: -r.size)
        cdf = pd.DataFrame([{
            "cluster_id":      r.cluster_id,
            "size":            r.size,
            "diameter":        r.diameter,
            "avg_intra_dist":  r.avg_intra_dist,
            "nearest_cluster": r.nearest_cluster_id,
            "nearest_dist":    r.nearest_cluster_dist,
            "merge_candidate": "YES" if r.merge_candidate else "",
        } for r in rows])
        st.dataframe(cdf, hide_index=True)

        sel_cid = st.selectbox(
            "Select cluster to analyse",
            [r.cluster_id for r in rows],
            format_func=lambda c: f"Cluster {c} ({next(r.size for r in rows if r.cluster_id==c)} faces)",
        )
        if st.button("Open Cluster Analysis", type="primary"):
            st.session_state.selected_cluster = sel_cid
            st.session_state.cluster_worker       = None
            st.session_state.cluster_debug_worker  = None
            st.info(f"Cluster {sel_cid} selected. Switch to **Cluster Analysis**.")

    if overview.cluster_rows:
        st.subheader("Cluster Size Distribution")
        st.bar_chart(pd.Series(
            {f"C{r.cluster_id}": r.size for r in overview.cluster_rows}
        ).sort_values(ascending=False))

    st.subheader("UMAP — All Faces")
    if overview.umap_coords is not None:
        import plotly.express as px
        udf = pd.DataFrame({
            "x":       overview.umap_coords[:, 0],
            "y":       overview.umap_coords[:, 1],
            "cluster": [f"C{l}" if l >= 0 else "noise" for l in overview.umap_labels],
            "face_id": overview.umap_face_ids or list(range(len(overview.umap_coords))),
        })
        fig = px.scatter(udf, x="x", y="y", color="cluster",
                         hover_data=["face_id", "cluster"], height=500)
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig)
    else:
        st.info("UMAP not available (embeddings not present in this run).")

    # ------------------------------------------------------------------
    # Mapping Tables — face-to-cluster and face-to-source-image
    # ------------------------------------------------------------------
    st.subheader("Face Mapping Tables")
    faces_df = _load_faces_df(result.output_dir)

    with st.expander("Face -> Cluster mapping", expanded=False):
        map_df = faces_df[["face_id", "cluster_id", "image_path", "is_core"]].copy()
        map_df["image_file"] = map_df["image_path"].apply(
            lambda p: Path(str(p)).name if pd.notna(p) else ""
        )
        map_df["cluster_label"] = map_df["cluster_id"].apply(
            lambda c: f"C{c}" if c >= 0 else "noise"
        )
        st.dataframe(
            map_df[["face_id", "cluster_label", "cluster_id", "image_file", "is_core"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "face_id": st.column_config.NumberColumn("Face ID", format="%d"),
                "cluster_label": "Cluster",
                "cluster_id": st.column_config.NumberColumn("Cluster ID", format="%d"),
                "image_file": "Source Image",
                "is_core": "Core?",
            },
        )
        st.caption(f"{len(map_df)} faces total  --  "
                   f"{(map_df['cluster_id'] >= 0).sum()} clustered  --  "
                   f"{(map_df['cluster_id'] == -1).sum()} noise")

    with st.expander("Face -> Source Image mapping", expanded=False):
        img_df = faces_df[["face_id", "image_path", "crop_path", "cluster_id",
                           "blur_score", "area"]].copy()
        img_df["image_file"] = img_df["image_path"].apply(
            lambda p: Path(str(p)).name if pd.notna(p) else ""
        )
        img_df["crop_file"] = img_df["crop_path"].apply(
            lambda p: Path(str(p)).name if pd.notna(p) else ""
        )
        st.dataframe(
            img_df[["face_id", "image_file", "crop_file", "cluster_id",
                     "blur_score", "area"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "face_id": st.column_config.NumberColumn("Face ID", format="%d"),
                "image_file": "Source Image",
                "crop_file": "Crop File",
                "cluster_id": st.column_config.NumberColumn("Cluster", format="%d"),
                "blur_score": st.column_config.NumberColumn("Blur", format="%.0f"),
                "area": st.column_config.NumberColumn("Area", format="%.0f"),
            },
        )

    with st.expander("Cluster -> Faces summary", expanded=False):
        cluster_faces = (
            faces_df[faces_df["cluster_id"] >= 0]
            .groupby("cluster_id")
            .agg(
                n_faces=("face_id", "count"),
                face_ids=("face_id", lambda x: ", ".join(str(i) for i in sorted(x))),
                images=("image_path",
                        lambda x: ", ".join(sorted(set(
                            Path(str(p)).name for p in x if pd.notna(p)
                        )))),
            )
            .reset_index()
            .sort_values("n_faces", ascending=False)
        )
        st.dataframe(cluster_faces, hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Worked Algorithm Example — concrete edge + cluster formation
    # ------------------------------------------------------------------
    _render_worked_example(result, overview)


def _render_worked_example(result, overview):
    """Show how the algorithm built edges and formed a cluster, using actual run data."""
    st.subheader("Worked Algorithm Example")
    st.caption("A concrete walkthrough of edge construction and cluster formation "
               "from the loaded run.")

    faces = result.faces
    cr = result.cluster_result
    if not cr.clusters:
        st.info("No clusters in this run.")
        return

    # Pick a good example cluster: size 3-6, not too big, not trivial
    good_ids = [cid for cid, members in cr.clusters.items()
                if 3 <= len(members) <= 6]
    if not good_ids:
        # fallback: any small cluster
        good_ids = [cid for cid, members in cr.clusters.items()
                    if 2 <= len(members) <= 10]
    if not good_ids:
        st.info("No suitable small cluster found for a worked example.")
        return

    # Let user pick, default to first good one
    example_cid = st.selectbox(
        "Pick a cluster for the worked example",
        good_ids,
        format_func=lambda c: f"Cluster {c} ({len(cr.clusters[c])} faces)",
        key="worked_example_cid",
    )

    member_indices = cr.clusters[example_cid]
    n = len(member_indices)
    fid_list = [faces[i].face_id for i in member_indices]

    # Load config
    run_cfg = result.summary.get("config") or {}
    K = run_cfg.get("K", 5)
    dist_thresh = run_cfg.get("distance_threshold", 0.35)

    face_labels = ", ".join(f"face\\_{fid}" for fid in fid_list)
    st.markdown(f"**Cluster {example_cid}** has **{n} faces**: {face_labels}")
    st.markdown(f"Config: `K={K}`, `distance_threshold={dist_thresh}`")

    # Show crops
    crop_cols = st.columns(min(n, 8))
    for i, fid in enumerate(fid_list):
        with crop_cols[i % len(crop_cols)]:
            img = _crop_for_face(fid, result.output_dir)
            if img:
                st.image(img, caption=f"face_{fid:04d}", width=90)

    # Build pairwise distance matrix (use same logic as analysis_views._embeddings_matrix)
    embs = []
    missing_emb = False
    for idx in member_indices:
        f = faces[idx]
        emb = f.embedding_normalized if f.embedding_normalized is not None else f.embedding
        if emb is None:
            missing_emb = True
            break
        embs.append(emb / (np.linalg.norm(emb) + 1e-9))

    if missing_emb or len(embs) == 0:
        st.warning("Embeddings not available for this cluster — cannot compute distances.")
        return

    mat = np.stack(embs).astype(np.float32)
    from scipy.spatial.distance import cdist
    pw = cdist(mat, mat, metric="cosine")

    # Step 1: show pairwise distance table
    st.markdown("---")
    st.markdown("#### Step 1: Pairwise cosine distances")
    dist_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_rows.append({
                "Face A": f"face_{fid_list[i]:04d}",
                "Face B": f"face_{fid_list[j]:04d}",
                "Distance": round(float(pw[i, j]), 4),
            })
    st.dataframe(pd.DataFrame(dist_rows), hide_index=True)

    # Step 2: kNN per face
    st.markdown("#### Step 2: K nearest neighbors (mutual kNN)")
    k_actual = min(K, n - 1)
    neighbor_sets = []
    knn_rows = []
    for i in range(n):
        dists_i = pw[i].copy()
        dists_i[i] = np.inf
        nearest = np.argsort(dists_i)[:k_actual]
        neighbor_sets.append(set(nearest.tolist()))
        nbr_str = ", ".join(
            f"face_{fid_list[j]:04d} ({pw[i,j]:.4f})" for j in nearest
        )
        knn_rows.append({
            "Face": f"face_{fid_list[i]:04d}",
            f"Top-{k_actual} neighbors": nbr_str,
        })
    st.dataframe(pd.DataFrame(knn_rows), hide_index=True, use_container_width=True)

    # Step 3: mutual edges
    st.markdown("#### Step 3: Edge creation (mutual + threshold)")
    st.markdown(f"An edge is created between A and B only if:\n"
                f"- B is in A's top-{k_actual} **AND** A is in B's top-{k_actual} (mutual)\n"
                f"- distance <= {dist_thresh}")
    edge_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            mutual = j in neighbor_sets[i] and i in neighbor_sets[j]
            below = pw[i, j] <= dist_thresh
            created = mutual and below
            edge_rows.append({
                "Face A": f"face_{fid_list[i]:04d}",
                "Face B": f"face_{fid_list[j]:04d}",
                "Distance": round(float(pw[i, j]), 4),
                "Mutual kNN?": "yes" if mutual else "no",
                f"<= {dist_thresh}?": "yes" if below else "no",
                "Edge created": "YES" if created else "no",
            })
    edf = pd.DataFrame(edge_rows)
    # Highlight created edges
    st.dataframe(edf, hide_index=True, use_container_width=True)
    n_created = sum(1 for r in edge_rows if r["Edge created"] == "YES")
    st.markdown(f"**Result: {n_created} edges created** out of {len(edge_rows)} possible pairs.")

    # Step 4: connected components
    st.markdown("#### Step 4: Connected components -> cluster")
    st.markdown(f"All {n} faces are reachable through the {n_created} edges above, "
                f"so they form **one connected component** = **Cluster {example_cid}**.")
    st.markdown(f"If any face had zero edges, it would become **noise** "
                f"(or a cluster smaller than `min_cluster_size`).")


# ---------------------------------------------------------------------------
# Tab 4: Cluster Analysis
# ---------------------------------------------------------------------------

def render_cluster_analysis_tab():
    st.header("Cluster Analysis")
    _breadcrumb()

    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return

    cr = result.cluster_result
    cluster_ids = sorted(cr.clusters.keys())
    if not cluster_ids:
        st.warning("No clusters found.")
        return

    default_idx = 0
    if st.session_state.selected_cluster in cluster_ids:
        default_idx = cluster_ids.index(st.session_state.selected_cluster)

    selected = st.selectbox(
        "Cluster",
        cluster_ids,
        index=default_idx,
        format_func=lambda c: f"Cluster {c}  ({len(cr.clusters[c])} faces)",
    )
    if selected != st.session_state.selected_cluster:
        st.session_state.selected_cluster = selected
        st.session_state.cluster_worker       = None
        st.session_state.cluster_debug_worker  = None

    worker: _AsyncState = st.session_state.cluster_worker

    if worker is None:
        w = _AsyncState()
        st.session_state.cluster_worker = w
        w.start(ClusterView.compute, result, selected)
        st.rerun()
        return

    if worker.is_running:
        st.info(f"Analysing cluster {selected}...")
        time.sleep(0.4)
        st.rerun()
        return

    if worker.has_error:
        st.error(f"Cluster analysis failed: {worker.error}")
        return

    view = worker.result

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Faces",          view.size)
    c2.metric("Diameter",       f"{view.diameter:.3f}")
    c3.metric("Avg intra-dist", f"{view.avg_intra_dist:.3f}")
    c4.metric("Exemplars",      len(view.exemplar_face_ids))
    c5.metric("Outliers",       len(view.outlier_face_ids))

    if view.split_signal:
        st.warning("Split signal — bimodal distance distribution. This cluster may contain two people.")

    st.subheader("Exemplars (d10 — most central)")
    if view.exemplar_face_ids:
        n_show = min(len(view.exemplar_face_ids), 5)
        cols = st.columns(n_show)
        for i, fid in enumerate(view.exemplar_face_ids[:n_show]):
            with cols[i]:
                img = _crop_for_face(fid, result.output_dir)
                if img:
                    st.image(img, caption=f"face_{fid:04d}")
                else:
                    st.caption(f"face_{fid:04d}")
                if i == 0:
                    st.caption("TOP EXEMPLAR")

    st.subheader("All Faces in Cluster")

    # Visual grid — 8 faces per row, sorted: exemplars first, then by dist_to_exemplar
    _GRID_COLS = 8
    sorted_faces = sorted(
        view.faces,
        key=lambda fr: (0 if fr.face_id in view.exemplar_face_ids else 1,
                        fr.dist_to_exemplar if fr.dist_to_exemplar is not None else 9.0),
    )
    for row_start in range(0, len(sorted_faces), _GRID_COLS):
        row_faces = sorted_faces[row_start:row_start + _GRID_COLS]
        cols = st.columns(_GRID_COLS)
        for col, fr in zip(cols, row_faces):
            with col:
                img = _crop_for_face(fr.face_id, result.output_dir)
                if img:
                    st.image(img)
                else:
                    st.markdown("_(no crop)_")
                is_ex = fr.face_id in view.exemplar_face_ids
                tag = "EX " if is_ex else ("!" if fr.is_outlier else "")
                dist_str = f"{fr.dist_to_exemplar:.3f}" if fr.dist_to_exemplar is not None else "-"
                st.caption(f"{tag}face_{fr.face_id:04d}\nd={dist_str}")

    # Detail table (collapsed by default — grid is the primary view)
    with st.expander("Face detail table"):
        fdf = pd.DataFrame([{
            "face_id":       fr.face_id,
            "image":         Path(fr.image_path).name if fr.image_path else "",
            "blur":          round(fr.blur_score, 1),
            "area":          int(fr.area),
            "dist_exemplar": fr.dist_to_exemplar,
            "dist_centroid": fr.dist_to_centroid,
            "role":          fr.role,
            "outlier":       "YES" if fr.is_outlier else "",
        } for fr in view.faces])
        st.dataframe(fdf, hide_index=True)

    sel_face = st.selectbox("Select face to analyse in Face Analysis tab",
                            [fr.face_id for fr in view.faces],
                            format_func=lambda f: f"face_{f:04d}")
    if st.button("Open Face Analysis", type="primary"):
        st.session_state.selected_face = sel_face
        st.session_state.face_worker   = None
        st.info(f"face_{sel_face:04d} selected. Switch to **Face Analysis**.")

    st.subheader("Nearest Clusters")
    if view.nearest_clusters:
        for nc in view.nearest_clusters:
            # Header row: metrics
            badge = " -- MERGE CANDIDATE" if nc.merge_candidate else ""
            st.markdown(
                f"**Cluster {nc.cluster_id}** ({nc.size} faces){badge}  "
                f"&nbsp;&nbsp; min_dist `{nc.min_exemplar_dist:.3f}` "
                f"| p10 `{nc.p10_cross_dist:.3f}` "
                f"| threshold `{nc.merge_threshold:.3f}`"
            )
            # Thumbnail row: up to 4 exemplar crops + Go button
            ex_indices = cr.exemplars.get(nc.cluster_id, [])
            ex_fids = [result.faces[i].face_id for i in ex_indices[:4]]
            thumb_cols = st.columns([1, 1, 1, 1, 2])
            for col_i, fid in enumerate(ex_fids):
                with thumb_cols[col_i]:
                    img = _crop_for_face(fid, result.output_dir)
                    if img:
                        st.image(img, width=70)
                    st.caption(f"face_{fid:04d}")
            with thumb_cols[4]:
                if st.button(f"Go to C{nc.cluster_id}",
                             key=f"goto_nc_{nc.cluster_id}",
                             use_container_width=True):
                    st.session_state.selected_cluster     = nc.cluster_id
                    st.session_state.cluster_worker       = None
                    st.session_state.cluster_debug_worker = None
                    st.info(f"Cluster {nc.cluster_id} selected. Switch to **Cluster Analysis**.")
            st.divider()

    # -- Cluster Debug Diagnostics -------------------------------------------
    st.subheader("Graph Debug")
    dbg_worker: _AsyncState = st.session_state.cluster_debug_worker

    if dbg_worker is None:
        w = _AsyncState()
        st.session_state.cluster_debug_worker = w
        w.start(ClusterDebugView.compute, result, selected)
        st.rerun()
        return

    if dbg_worker.is_running:
        st.info("Computing graph diagnostics...")
        time.sleep(0.4)
        st.rerun()
        return

    if dbg_worker.has_error:
        st.error(f"Debug diagnostics failed: {dbg_worker.error}")
    else:
        dbg = dbg_worker.result
        _render_cluster_debug(dbg, result)


def _render_cluster_debug(dbg: ClusterDebugView, result):
    """Render graph-level diagnostics for a cluster."""
    # Health metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Edges", f"{dbg.n_edges} / {dbg.max_possible_edges}")
    c2.metric("Edge density", f"{dbg.edge_density:.1%}")
    c3.metric("Chain score", f"{dbg.chain_score:.2f}",
              help="diameter / (2 * median_dist). >1.5 suggests chain structure.")
    c4.metric("Bridge faces", len(dbg.bridge_face_ids))

    # Interpretation
    if dbg.chain_score > 1.5:
        st.warning(
            f"**Chain structure detected** (score {dbg.chain_score:.2f}). "
            f"Diameter ({dbg.diameter:.3f}) is much larger than median pairwise distance ({dbg.median_dist:.3f}). "
            f"This cluster likely contains faces linked through intermediaries, not a tight group."
        )
    elif dbg.edge_density < 0.15 and dbg.n_faces > 4:
        st.warning(
            f"**Sparse graph** (density {dbg.edge_density:.1%}). "
            f"Most face pairs are NOT directly connected -- they are grouped by transitive reachability only."
        )

    # Bridge faces — removing these would split the cluster
    if dbg.bridge_face_ids:
        st.markdown(
            f"**Bridge faces** (articulation points) -- removing any one splits the cluster: "
            f"`{', '.join(f'face_{fid:04d}' for fid in dbg.bridge_face_ids)}`"
        )
        cols = st.columns(min(len(dbg.bridge_face_ids), 8))
        for i, fid in enumerate(dbg.bridge_face_ids[:8]):
            with cols[i]:
                img = _crop_for_face(fid, result.output_dir)
                if img:
                    st.image(img, caption=f"face_{fid:04d}")

    # Distance heatmap
    if dbg.distance_matrix is not None and len(dbg.distance_matrix) > 1:
        with st.expander("Pairwise distance heatmap", expanded=dbg.n_faces <= 30):
            import plotly.express as px
            labels = [f"f{fid}" for fid in dbg.face_ids_order]
            fig = px.imshow(
                dbg.distance_matrix,
                x=labels, y=labels,
                color_continuous_scale="RdYlGn_r",
                zmin=0.0, zmax=min(0.6, float(dbg.distance_matrix.max()) + 0.05),
                labels=dict(color="cosine dist"),
                aspect="equal",
            )
            fig.update_layout(height=max(300, 18 * dbg.n_faces + 100))
            st.plotly_chart(fig, use_container_width=True)

    # Per-face connectivity table
    with st.expander("Per-face graph connectivity"):
        rows = []
        for fg in dbg.face_graph:
            nbr_str = ", ".join(f"f{n}({d:.3f})" for n, d in zip(fg.neighbors, fg.neighbor_dists))
            rows.append({
                "face_id": f"face_{fg.face_id:04d}",
                "edges": fg.n_edges,
                "bridge": "YES" if fg.is_bridge else "",
                "neighbors (dist)": nbr_str,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Edge list
    with st.expander("All edges"):
        if dbg.edges:
            edf = pd.DataFrame([{
                "face_a": f"face_{e.face_id_a:04d}",
                "face_b": f"face_{e.face_id_b:04d}",
                "distance": e.distance,
            } for e in dbg.edges])
            edf = edf.sort_values("distance")
            st.dataframe(edf, hide_index=True)
        else:
            st.info("No edges in this cluster (all faces are isolated).")


# ---------------------------------------------------------------------------
# Tab 5: Face Analysis
# ---------------------------------------------------------------------------

def render_face_analysis_tab():
    st.header("Face Analysis")
    _breadcrumb()

    result = st.session_state.pipeline_result
    if result is None:
        _no_result()
        return

    faces_df = _load_faces_df(result.output_dir)
    face_ids = list(faces_df["face_id"].astype(int))

    default_face = st.session_state.selected_face
    default_idx  = face_ids.index(default_face) if default_face in face_ids else 0

    selected_face = st.selectbox("Face", face_ids, index=default_idx,
                                 format_func=lambda f: f"face_{f:04d}")
    if selected_face != st.session_state.selected_face:
        st.session_state.selected_face = selected_face
        st.session_state.face_worker   = None

    worker: _AsyncState = st.session_state.face_worker

    if worker is None:
        w = _AsyncState()
        st.session_state.face_worker = w
        w.start(FaceView.compute, result, selected_face)
        st.rerun()
        return

    if worker.is_running:
        st.info(f"Analysing face_{selected_face:04d}...")
        time.sleep(0.4)
        st.rerun()
        return

    if worker.has_error:
        st.error(f"Face analysis failed: {worker.error}")
        return

    view = worker.result

    col_img, col_attrs = st.columns([1, 3])
    with col_img:
        img = _crop_for_face(view.face_id, result.output_dir)
        if img:
            st.image(img, caption=f"face_{view.face_id:04d}")
        else:
            st.caption("(no crop)")

    with col_attrs:
        gate_color = "green" if view.gate_result == "core" else "red"
        st.markdown(f"**Gate**: :{gate_color}[{view.gate_result.upper()}]")
        if view.gate_rejection_reason:
            st.caption(f"Rejection: {view.gate_rejection_reason}")
        st.markdown(f"**Cluster**: {view.cluster_id if view.cluster_id >= 0 else 'noise/holdout'}")
        st.markdown(f"**Source**: `{Path(view.image_path).name}`  rank #{view.rank_in_image}")
        st.markdown(f"**Blur**: {view.blur_score:.1f}  |  **Area**: {int(view.area):,} px²")
        if view.pose:
            yaw, pitch, roll = view.pose
            st.markdown(f"**Pose**: yaw {yaw:.1f}  pitch {pitch:.1f}  roll {roll:.1f}")

    st.subheader("Closest — Same Cluster")
    if view.closest_same_cluster:
        cols = st.columns(min(len(view.closest_same_cluster), 5))
        for i, cf in enumerate(view.closest_same_cluster):
            with cols[i]:
                img = _crop_for_face(cf.face_id, result.output_dir)
                if img:
                    st.image(img)
                st.caption(f"face_{cf.face_id:04d}\nd={cf.distance:.3f}")
    else:
        st.info("No same-cluster neighbors.")

    st.subheader("Closest — Other Clusters")
    if view.closest_other_clusters:
        cols = st.columns(min(len(view.closest_other_clusters), 5))
        for i, cf in enumerate(view.closest_other_clusters):
            with cols[i]:
                img = _crop_for_face(cf.face_id, result.output_dir)
                if img:
                    st.image(img)
                lbl = f"C{cf.cluster_id}" if cf.cluster_id >= 0 else "noise"
                st.caption(f"face_{cf.face_id:04d}\n{lbl}  d={cf.distance:.3f}")
    else:
        st.info("No cross-cluster neighbors.")

    if view.coimage_faces:
        st.subheader(f"Other Faces in {Path(view.image_path).name}")
        cols = st.columns(min(len(view.coimage_faces), 6))
        for i, fr in enumerate(view.coimage_faces):
            with cols[i]:
                img = _crop_for_face(fr.face_id, result.output_dir)
                if img:
                    st.image(img)
                lbl = f"C{fr.cluster_id}" if fr.cluster_id >= 0 else "holdout"
                st.caption(f"face_{fr.face_id:04d}\n{lbl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import numpy as np  # needed by _render_run_files_panel for .npy shape display

st.set_page_config(page_title="Face Clustering", layout="wide")
st.title("Face Clustering")

tab_run, tab_hist, tab_overview, tab_cluster, tab_face = st.tabs([
    "Run", "History", "Run Overview", "Cluster Analysis", "Face Analysis",
])

with tab_run:      render_run_tab()
with tab_hist:     render_history_tab()
with tab_overview: render_run_overview_tab()
with tab_cluster:  render_cluster_analysis_tab()
with tab_face:     render_face_analysis_tab()
