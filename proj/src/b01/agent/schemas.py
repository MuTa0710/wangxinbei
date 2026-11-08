"""Schemas and validation for the M6 Agent boundary.

The model is allowed to produce semantic decisions and query parameters. The
submission layer only accepts validated structures from this module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


BehaviorName = Literal[
    "large_climb",
    "climb",
    "large_descent",
    "descent",
    "gentle_step_accel",
    "abrupt_step_accel",
    "long_platform_step_accel",
    "step_accel",
    "step_decel",
    "orbit",
    "speed_hold",
    "cruise",
    "low_speed",
    "approach",
    "periodic_speed",
]


class AgentBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BehaviorItem(AgentBaseModel):
    name: BehaviorName
    term: str = ""
    char_position: int | None = None

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class Task2QueryIR(AgentBaseModel):
    """Canonical query IR consumed by the deterministic retrieval tool."""

    sample_id: str
    text_query: str
    top_k: int = Field(default=10, ge=1, le=1000)
    time_start_utc: str | None = None
    time_end_utc: str | None = None
    altitude_range_m: tuple[float, float] | None = None
    speed_range_mps: tuple[float, float] | None = None
    positive_regions: list[str] = Field(default_factory=list)
    negative_regions: list[str] = Field(default_factory=list)
    behavior_sequence: list[BehaviorItem] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)

    @field_validator("time_start_utc", "time_end_utc")
    @classmethod
    def validate_time(cls, value: str | None) -> str | None:
        if value is not None:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("altitude_range_m", "speed_range_mps")
    @classmethod
    def validate_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError("range lower bound is greater than upper bound")
        return value

    def as_retrieval_dict(self) -> dict[str, Any]:
        result = self.model_dump()
        result["behavior_sequence"] = [item.model_dump() for item in self.behavior_sequence]
        return result


class Task1Decision(AgentBaseModel):
    choice: Literal["A", "B", "C", "D"]

    @field_validator("choice", mode="before")
    @classmethod
    def normalize_choice(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value


def model_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema for prompts and trace files."""

    return model.model_json_schema()
