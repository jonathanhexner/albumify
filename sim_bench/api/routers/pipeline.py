"""Pipeline router - API endpoints for pipeline execution."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from sim_bench.api.database.session import get_session
from sim_bench.api.schemas.pipeline import PipelineRequest, PipelineStatus, PipelineResultResponse
from sim_bench.api.services.pipeline_service import PipelineService
from sim_bench.api.services.config_service import ConfigService

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


@router.post("/run")
def run_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """Start a pipeline run."""
    # Get merged config: profile defaults + runtime overrides
    config_service = ConfigService(session)
    merged_config = config_service.get_merged_config(
        profile_name=request.profile,
        overrides=request.config
    )

    pipeline_service = PipelineService(session)
    job_id = pipeline_service.start_pipeline(
        album_id=request.album_id,
        steps=request.steps,
        step_configs=merged_config,
        fail_fast=request.fail_fast
    )

    background_tasks.add_task(pipeline_service.execute_pipeline, job_id)

    return {"job_id": job_id, "status": "started", "profile": request.profile or "default"}


@router.get("/{job_id}", response_model=PipelineStatus)
def get_pipeline_status(job_id: str, session: Session = Depends(get_session)):
    """Get status of a pipeline run."""
    service = PipelineService(session)
    run = service.get_status(job_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    return PipelineStatus(
        job_id=run.id,
        album_id=run.album_id,
        status=run.status,
        current_step=run.current_step,
        progress=run.progress,
        message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at
    )


@router.get("/{job_id}/result", response_model=PipelineResultResponse)
def get_pipeline_result(job_id: str, session: Session = Depends(get_session)):
    """Get result of a completed pipeline run."""
    service = PipelineService(session)
    result = service.get_result(job_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Pipeline result not found")

    return PipelineResultResponse(
        job_id=job_id,
        album_id=result.run.album_id,
        total_images=result.total_images,
        filtered_images=result.filtered_images,
        num_clusters=result.num_clusters,
        num_selected=result.num_selected,
        scene_clusters=result.scene_clusters,
        selected_images=result.selected_images,
        step_timings=result.step_timings,
        total_duration_ms=result.total_duration_ms
    )
