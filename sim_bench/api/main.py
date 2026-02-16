"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sim_bench.api.database.session import init_db, get_session
from sim_bench.api.routers import albums, pipeline, steps, websocket, people, results, config, events, faces
from sim_bench.api.logging import setup_logging, get_logger
from sim_bench.api.services.config_service import ConfigService

# Setup logging (creates timestamped folder)
log_dir = setup_logging()
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


@app.on_event("startup")
def startup():
    """Initialize database on startup."""
    logger.info(f"Starting Album Organization API")
    logger.info(f"Logs directory: {log_dir}")
    init_db()
    logger.info("Database initialized")

    import sim_bench.pipeline.steps.all_steps
    logger.info("Pipeline steps registered")

    # Ensure default config profile exists
    session = next(get_session())
    config_service = ConfigService(session)
    config_service.ensure_default_profile()
    logger.info("Default config profile ready")


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
