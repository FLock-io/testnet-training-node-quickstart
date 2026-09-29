"""Validation-package manifest models.

Time convention (shared by ground truth, detector output and scoring):
frame ``i`` covers ``[i / fps, (i + 1) / fps)``. An issue spanning frames
``s..e`` inclusive has ``start_time = s / fps`` and ``end_time = (e + 1) / fps``.
A point event (``dropped_frames``) at the cut before output frame ``k`` has
``start_time == end_time == k / fps`` and ``start_frame == end_frame == k``.

Bounding boxes are normalised ``[x0, y0, x1, y1]`` in ``[0, 1]`` with
``x0 < x1`` and ``y0 < y1``, relative to the frame width/height; for a spatial
issue the box is the union of the affected region over its whole time span.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from validator.modules.video_inconsistency.issue_types import (
    ISSUE_TYPES_BY_NAME,
    SPATIAL_ISSUE_TYPES,
    SUITE_VERSION,
)


Difficulty = Literal["easy", "medium", "hard", "expert"]
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard", "expert")

# Legitimate, UNLABELLED events a detector must not flag. Recorded in manifests for
# analysis and training (hard negatives); never scored and never sent to a detector.
DecoyType = Literal[
    "scene_cut",
    "exposure_drift",
    "white_balance_drift",
    "smooth_zoom",
    "object_enters",
    "object_exits",
    "object_stops",
    "camera_stops",
    # v2.1 decoys, each a legitimate look-alike of an issue type:
    "illumination_flicker",  # periodic mains/cloud lighting vs exposure_flicker
    "auto_exposure_step",  # eased, persistent AE correction vs exposure_flicker
    "auto_white_balance_step",  # eased, persistent AWB correction vs color_grade_jump
    "fast_zoom",  # smooth, persistent zoom vs zoom_jump
    "camera_direction_change",  # pan eases into reverse vs reversed_segment
    "camera_speed_change",  # pan eases to a new speed vs dropped_frames
]


def validate_bbox(bbox: list[float] | None) -> list[float] | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        raise ValueError("bbox coordinates must be normalised to [0, 1]")
    if not (x0 < x1 and y0 < y1):
        raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
    return [x0, y0, x1, y1]


class IssueLabel(BaseModel, frozen=True):
    """One ground-truth inconsistency in a clip."""

    type: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)  # exclusive for spans; == start_frame for point events
    bbox: list[float] | None = None
    params: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in ISSUE_TYPES_BY_NAME:
            raise ValueError(f"unknown issue type {value!r}")
        return value

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(cls, value: list[float] | None) -> list[float] | None:
        return validate_bbox(value)

    @model_validator(mode="after")
    def _consistent(self) -> "IssueLabel":
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        if self.type in SPATIAL_ISSUE_TYPES and self.bbox is None:
            raise ValueError(f"{self.type} labels require a bbox")
        return self


class DecoyLabel(BaseModel, frozen=True):
    type: DecoyType
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> "DecoyLabel":
        if self.end_time < self.start_time:
            raise ValueError("decoy end_time must be >= start_time")
        return self


class ClipSpec(BaseModel, frozen=True):
    clip_id: str
    video_path: str  # relative to the package root
    fps: float = Field(gt=0.0)
    num_frames: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    difficulty: Difficulty = "medium"
    source: str = "procedural"  # "procedural" | "footage" — telemetry only
    crf: int | None = Field(default=None, ge=0, le=51)  # encode quality — telemetry only
    issues: list[IssueLabel] = Field(default_factory=list)
    decoys: list[DecoyLabel] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @model_validator(mode="after")
    def _issues_within_clip(self) -> "ClipSpec":
        tolerance = 1e-6
        for issue in self.issues:
            if issue.end_time > self.duration + tolerance:
                raise ValueError(
                    f"clip {self.clip_id}: issue {issue.type} ends after the clip"
                )
            if issue.end_frame > self.num_frames:
                raise ValueError(
                    f"clip {self.clip_id}: issue {issue.type} end_frame out of range"
                )
        return self


class VideoManifest(BaseModel, frozen=True):
    suite_version: str = SUITE_VERSION
    clips: list[ClipSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> "VideoManifest":
        ids = [clip.clip_id for clip in self.clips]
        if len(ids) != len(set(ids)):
            raise ValueError("clip_id values must be unique")
        if not self.clips:
            raise ValueError("manifest must contain at least one clip")
        return self
