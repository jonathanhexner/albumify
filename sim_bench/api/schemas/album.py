"""Album API schemas."""

from datetime import datetime
from pydantic import BaseModel


class AlbumCreate(BaseModel):
    name: str
    source_path: str


class AlbumResponse(BaseModel):
    id: str
    name: str
    source_path: str
    image_count: int
    created_at: datetime

    class Config:
        from_attributes = True
