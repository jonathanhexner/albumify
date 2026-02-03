"""Steps router - API endpoints for step discovery."""

from typing import Optional
from fastapi import APIRouter, HTTPException

from sim_bench.api.schemas.step import StepInfo, StepListResponse

import sim_bench.pipeline.steps.all_steps
from sim_bench.pipeline.registry import get_registry

router = APIRouter(prefix="/api/v1/steps", tags=["steps"])


@router.get("/", response_model=StepListResponse)
def list_steps(category: Optional[str] = None):
    """List all available pipeline steps."""
    registry = get_registry()
    metadata_list = registry.list_steps(category=category)

    steps = [
        StepInfo(
            name=m.name,
            display_name=m.display_name,
            description=m.description,
            category=m.category,
            requires=list(m.requires),
            produces=list(m.produces),
            depends_on=m.depends_on,
            config_schema=m.config_schema
        )
        for m in metadata_list
    ]

    return StepListResponse(steps=steps)


@router.get("/{step_name}", response_model=StepInfo)
def get_step(step_name: str):
    """Get detailed info about a specific step."""
    registry = get_registry()

    if not registry.has_step(step_name):
        raise HTTPException(status_code=404, detail=f"Step not found: {step_name}")

    m = registry.get_metadata(step_name)

    return StepInfo(
        name=m.name,
        display_name=m.display_name,
        description=m.description,
        category=m.category,
        requires=list(m.requires),
        produces=list(m.produces),
        depends_on=m.depends_on,
        config_schema=m.config_schema
    )


@router.get("/{step_name}/schema")
def get_step_config_schema(step_name: str):
    """Get JSON Schema for step configuration."""
    registry = get_registry()

    if not registry.has_step(step_name):
        raise HTTPException(status_code=404, detail=f"Step not found: {step_name}")

    m = registry.get_metadata(step_name)
    return m.config_schema
