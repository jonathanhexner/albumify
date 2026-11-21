"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sim_bench.api.database.session import init_db, get_session
from sim_bench.api.routers import albums, pipeline, steps, websocket, people, results, config, events, faces, jobs
from sim_bench.logging_setup import setup_logging, get_logger
from sim_bench.api.services.config_service import ConfigService

# Setup logging (creates timestamped folder). spec-041 follow-up:
# surface name distinguishes this process's log file from the v2 FC
# App's and the CLI's, so concurrent runs don't overwrite each other.
log_dir = setup_logging("api")
logger = get_logger(__name__)

app = FastAPI(
    title="Album Organization API",
    description="API for organizing photo albums using ML-powered pipelines",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(albums.router)
app.include_router(pipeline.router)
app.include_router(steps.router)
app.include_router(websocket.router)
app.include_router(people.router)
app.include_router(results.router)
app.include_router(config.router)
app.include_router(events.router)
app.include_router(faces.router)
app.include_router(jobs.router)   # spec-112: control-plane jobs (DB rows + job store)


@app.on_event("startup")
def startup():
    """Initialize database on startup."""
    logger.info(f"Starting Album Organization API")
    logger.info(f"Logs directory: {log_dir}")
    # spec-117 R1: DATABASE_URL / R2_* come from .env, read once here (never on import of a library module)
    from sim_bench.storage.database import load_env
    load_env()
    init_db()
    _start_run_watchdog()
    logger.info("Database initialized")

    import sim_bench.pipeline.steps.all_steps
    logger.info("Pipeline steps registered")

    # Ensure default config profile exists
    session = next(get_session())
    config_service = ConfigService(session)
    config_service.ensure_default_profile()
    logger.info("Default config profile ready")


def watchdog_tick(job_service=None, session_factory=None) -> int:
    """One watchdog pass: fail detached jobs whose worker is gone, then end their runs failed so Albums shows it
    (SIGHTING-157). Returns how many runs were ended."""
    from sim_bench.api.database.session import get_session_direct
    from sim_bench.api.routers.jobs import get_job_service
    from sim_bench.api.services.pipeline_service import PipelineService

    jobs = job_service or get_job_service()
    jobs.watchdog().reconcile()
    session = (session_factory or get_session_direct)()
    try:
        return PipelineService(session, job_service=jobs).fail_runs_whose_job_failed()
    finally:
        session.close()


def _start_run_watchdog(period_s: float = 30.0) -> None:
    """spec-117 R9: detached L4 runs whose worker is gone end failed within ~2 min, even if no page asks.
    Only with DATABASE_URL set: without it there are no detached runs."""
    import os
    import threading

    if not os.environ.get("DATABASE_URL"):
        return

    def loop() -> None:
        stop = threading.Event()
        while not stop.wait(period_s):
            try:
                watchdog_tick()
            except Exception as e:      # never takes the API down
                logger.warning("run watchdog: %s", type(e).__name__)

    threading.Thread(target=loop, name="run-watchdog", daemon=True).start()
    logger.info("run watchdog started (every %.0f s)", period_s)


@app.on_event("shutdown")
def shutdown():
    """Log shutdown."""
    logger.info("Shutting down Album Organization API")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
