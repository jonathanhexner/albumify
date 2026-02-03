"""Config router - API endpoints for configuration profiles."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sim_bench.api.database.session import get_session
from sim_bench.api.schemas.config import (
    ConfigProfileCreate,
    ConfigProfileUpdate,
    ConfigProfileResponse,
    ConfigProfileListResponse,
)
from sim_bench.api.services.config_service import ConfigService, get_default_config

router = APIRouter(prefix="/api/v1/config", tags=["config"])


@router.get("/defaults")
def get_defaults():
    """Get the default configuration from pipeline.yaml."""
    return get_default_config()


@router.get("/profiles", response_model=ConfigProfileListResponse)
def list_profiles(session: Session = Depends(get_session)):
    """List all configuration profiles."""
    service = ConfigService(session)
    profiles = service.list_profiles()
    return ConfigProfileListResponse(
        profiles=[ConfigProfileResponse.model_validate(p) for p in profiles],
        total=len(profiles)
    )


@router.get("/profiles/{name}", response_model=ConfigProfileResponse)
def get_profile(name: str, session: Session = Depends(get_session)):
    """Get a configuration profile by name."""
    service = ConfigService(session)
    profile = service.get_profile(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    return ConfigProfileResponse.model_validate(profile)


@router.post("/profiles", response_model=ConfigProfileResponse, status_code=201)
def create_profile(
    request: ConfigProfileCreate,
    session: Session = Depends(get_session)
):
    """Create a new configuration profile."""
    service = ConfigService(session)
    try:
        profile = service.create_profile(
            name=request.name,
            config=request.config,
            description=request.description,
            base_profile=request.base_profile
        )
        return ConfigProfileResponse.model_validate(profile)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/profiles/{name}", response_model=ConfigProfileResponse)
def update_profile(
    name: str,
    request: ConfigProfileUpdate,
    session: Session = Depends(get_session)
):
    """Update an existing configuration profile."""
    service = ConfigService(session)
    try:
        profile = service.update_profile(
            name=name,
            config=request.config,
            description=request.description
        )
        return ConfigProfileResponse.model_validate(profile)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/profiles/{name}")
def delete_profile(name: str, session: Session = Depends(get_session)):
    """Delete a configuration profile."""
    service = ConfigService(session)
    try:
        deleted = service.delete_profile(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
        return {"deleted": True, "name": name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/profiles/{name}/reset", response_model=ConfigProfileResponse)
def reset_profile(name: str, session: Session = Depends(get_session)):
    """Reset a profile to default values from pipeline.yaml."""
    service = ConfigService(session)
    try:
        profile = service.reset_profile_to_defaults(name)
        return ConfigProfileResponse.model_validate(profile)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/profiles/{name}/duplicate", response_model=ConfigProfileResponse)
def duplicate_profile(
    name: str,
    new_name: str,
    session: Session = Depends(get_session)
):
    """Duplicate a profile with a new name."""
    service = ConfigService(session)
    try:
        profile = service.duplicate_profile(name, new_name)
        return ConfigProfileResponse.model_validate(profile)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/merged")
def get_merged_config(
    profile: str = None,
    session: Session = Depends(get_session)
):
    """Get merged configuration for a profile (or default if not specified)."""
    service = ConfigService(session)
    return service.get_merged_config(profile_name=profile)
