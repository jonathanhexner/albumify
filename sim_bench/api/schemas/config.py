"""Pydantic schemas for configuration profiles."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ConfigProfileBase(BaseModel):
    """Base schema for config profile."""
    name: str
    description: Optional[str] = None
    config: dict


class ConfigProfileCreate(BaseModel):
    """Schema for creating a config profile."""
    name: str
    description: Optional[str] = None
    config: dict = {}  # Overrides to apply on top of defaults
    base_profile: Optional[str] = None  # Profile to inherit from


class ConfigProfileUpdate(BaseModel):
    """Schema for updating a config profile."""
    config: Optional[dict] = None
    description: Optional[str] = None


class ConfigProfileResponse(BaseModel):
    """Schema for config profile response."""
    id: str
    name: str
    description: Optional[str]
    config: dict
    is_default: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConfigProfileListResponse(BaseModel):
    """Schema for listing config profiles."""
    profiles: list[ConfigProfileResponse]
    total: int
