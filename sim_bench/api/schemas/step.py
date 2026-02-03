"""Step API schemas."""

from pydantic import BaseModel


class StepInfo(BaseModel):
    name: str
    display_name: str
    description: str
    category: str
    requires: list[str]
    produces: list[str]
    depends_on: list[str]
    config_schema: dict


class StepListResponse(BaseModel):
    steps: list[StepInfo]
