from typing import Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["collector", "admin"]
RecState = Literal["IDLE", "RECORDING", "SAVING", "ERROR"]
RolloutMode = Literal["demonstration", "autonomous", "intervention", "recovery"]
OutcomeLabel = Literal["success", "partial_success", "failure", "aborted"]


class Template(BaseModel):
    id: str
    task_id: str
    subset: Literal["base", "dagger"]
    prompt: str
    enabled: bool = True
    note: str = ""


class StartRecordingReq(BaseModel):
    template_id: str
    operator: str


class StageOutcome(BaseModel):
    """Outcome of one named task stage; order is the execution order."""

    stage: str = Field(min_length=1)
    success: bool
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    failure_mode: Optional[str] = None


class SaveRecordingReq(BaseModel):
    # `success` remains for old clients and existing statistics. `outcome` is the
    # canonical P1 label and is derived from success when omitted.
    success: bool = True
    outcome: Optional[OutcomeLabel] = None
    rollout_mode: RolloutMode = "demonstration"
    stage_outcomes: list[StageOutcome] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    intervention_count: int = Field(default=0, ge=0)
    recovery_success: Optional[bool] = None
    unsafe_event: bool = False
    time_limit_reached: bool = False
    note: str = ""
    scene_tags: list[str] = Field(default_factory=list)

    def rollout_outcome(self) -> dict:
        """Return the versioned, storage-facing outcome contract."""
        label = self.outcome or ("success" if self.success else "failure")
        return {
            "schema_version": 1,
            "label": label,
            "rollout_mode": self.rollout_mode,
            "stage_outcomes": [stage.model_dump() for stage in self.stage_outcomes],
            "failure_modes": list(dict.fromkeys(x.strip() for x in self.failure_modes if x.strip())),
            "intervention_count": self.intervention_count,
            "recovery_success": self.recovery_success,
            "unsafe_event": self.unsafe_event,
            "time_limit_reached": self.time_limit_reached,
        }


class EpisodeMeta(BaseModel):
    episode_id: int
    chunk: str = "chunk-000"
    task_id: str
    subset: str
    prompt: str
    operator: str
    success: bool
    note: str
    duration_s: float
    size_bytes: int
    created_at: float
    parquet_path: str
    video_paths: dict[str, str]
    incomplete: bool = False
    incomplete_reason: Optional[str] = None


class StatsBucket(BaseModel):
    key: str
    count: int


class StatsResponse(BaseModel):
    total: int
    today: int
    this_week: int
    incomplete: int
    total_duration_s: float
    total_size_bytes: int
    by_task_subset: list[StatsBucket]
    by_operator: list[StatsBucket]
    by_prompt: list[StatsBucket]
    by_success: list[StatsBucket]
    last_scan_at: float
