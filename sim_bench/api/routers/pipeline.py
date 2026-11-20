"""Pipeline router - API endpoints for pipeline execution."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from sim_bench.api.database.session import get_session
from sim_bench.api.routers.jobs import get_job_service
from sim_bench.api.schemas.jobs import EstimateOut
from sim_bench.api.schemas.pipeline import PipelineRequest, PipelineStatus, PipelineResultResponse, RunRow
from sim_bench.api.services.pipeline_service import PipelineService
from sim_bench.api.services.config_service import ConfigService
from sim_bench.serving.job_service import JobService
from sim_bench.serving.serving_config import RUNNERS

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


def _pipeline_service(session: Session = Depends(get_session),
                      jobs: JobService = Depends(get_job_service)) -> PipelineService:
    return PipelineService(session, job_service=jobs)


def _check_runner(runner: str) -> None:
    if runner not in RUNNERS:
        raise HTTPException(422, f"runner must be one of {RUNNERS}")


@router.post("/run")
def run_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    pipeline_service: PipelineService = Depends(_pipeline_service),
):
    """Start a pipeline run on the chosen machine; an L4 run over the spend cap is refused (422)."""
    _check_runner(request.runner)
    # Get merged config: profile defaults + runtime overrides
    config_service = ConfigService(session)
    merged_config = config_service.get_merged_config(
        profile_name=request.profile,
        overrides=request.config
    )

    try:
        job_id = pipeline_service.start_pipeline(
            album_id=request.album_id,
            steps=request.steps,
            pipeline_name=request.pipeline_name or "default_pipeline",
            step_configs=merged_config,
            fail_fast=request.fail_fast,
            runner=request.runner,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    background_tasks.add_task(pipeline_service.execute_pipeline, job_id)

    return {"job_id": job_id, "status": "started", "profile": request.profile or "default",
            "runner": request.runner}


@router.get("/runs", response_model=list[RunRow])
def list_runs(service: PipelineService = Depends(_pipeline_service)):
    """spec-116 R10: every run of every album, newest first."""
    return service.list_runs()


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, pipeline_service: PipelineService = Depends(_pipeline_service)):
    """Delete a finished run: its job, its stored files and its rows (SIGHTING-158)."""
    try:
        return pipeline_service.delete_run(run_id)
    except KeyError:
        raise HTTPException(404, f"no run {run_id!r}")
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.get("/estimate", response_model=EstimateOut)
def estimate_run(album_id: str, runner: str = "local", service: PipelineService = Depends(_pipeline_service)):
    """spec-116 R1: photos, and on the L4 GPU minutes, dollars and whether the spend cap allows it."""
    _check_runner(runner)
    try:
        est = service.estimate(album_id, runner)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return EstimateOut(**{k: getattr(est, k) for k in EstimateOut.model_fields})


@router.post("/{job_id}/cancel")
def cancel_run(job_id: str, service: PipelineService = Depends(_pipeline_service)):
    """spec-116 R7: the run ends failed / "cancelled by owner"."""
    try:
        service.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "This run has no job to cancel")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id, "status": "cancelling"}


@router.get("/{job_id}", response_model=PipelineStatus)
def get_pipeline_status(job_id: str, service: PipelineService = Depends(_pipeline_service)):
    """Get status of a pipeline run."""
    run = service.get_status(job_id)

    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    total_steps = len(run.steps) if run.steps else None

    status = dict(
        job_id=run.id,
        album_id=run.album_id,
        status=run.status,
        current_step=run.current_step,
        progress=run.progress,
        message=run.error_message,
        completed_steps=run.completed_steps,
        total_steps=total_steps,
        created_at=run.created_at,
        started_at=run.started_at
    )
    return PipelineStatus(**{**status, **service.live(run)})


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
        total_duration_ms=result.total_duration_ms,
        fc_export_dir=result.fc_export_dir,
    )
