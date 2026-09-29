"""Canonical catalogue of the video inconsistencies a detector must find.

This file is the single source of truth shared by the dataset builder (which
injects these edits), the validator (which scores detections against them) and
the trainer sample code. Adding or renaming a type is a breaking change to the
suite and must bump ``SUITE_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass


# v2: "expert" difficulty tier, unlabelled decoy events (legitimate cuts, exposure
# and white-balance drift, smooth zooms, objects entering/leaving), post-edit
# degradation, and rank-based (mAP) scoring. v1 packages are not comparable.
SUITE_VERSION = "video_inconsistency_v2"


@dataclass(frozen=True)
class IssueType:
    name: str
    category: str  # "temporal" | "appearance" | "geometric" | "spatial"
    spatial: bool  # True when a bounding box is part of the ground truth
    point_event: bool  # True when the issue is an instant (a cut), not a span
    description: str


ISSUE_TYPES: tuple[IssueType, ...] = (
    IssueType(
        name="frozen_frames",
        category="temporal",
        spatial=False,
        point_event=False,
        description=(
            "Motion stops: the same frame is held for a run of frames, then "
            "motion resumes with a jump."
        ),
    ),
    IssueType(
        name="dropped_frames",
        category="temporal",
        spatial=False,
        point_event=True,
        description=(
            "A run of frames was cut out mid-shot, so objects and the camera "
            "jump forward instantly. Reported at the cut."
        ),
    ),
    IssueType(
        name="reversed_segment",
        category="temporal",
        spatial=False,
        point_event=False,
        description="A segment plays backwards: motion runs in reverse, then snaps forward.",
    ),
    IssueType(
        name="spliced_footage",
        category="temporal",
        spatial=False,
        point_event=False,
        description=(
            "Frames from an unrelated shot are inserted into the middle of a "
            "continuous shot."
        ),
    ),
    IssueType(
        name="color_grade_jump",
        category="appearance",
        spatial=False,
        point_event=False,
        description=(
            "The colour grade / white balance changes abruptly for a segment and "
            "then changes back."
        ),
    ),
    IssueType(
        name="exposure_flicker",
        category="appearance",
        spatial=False,
        point_event=False,
        description="One to three frames are much brighter or darker than their neighbours.",
    ),
    IssueType(
        name="mirrored_segment",
        category="geometric",
        spatial=False,
        point_event=False,
        description="A segment is horizontally flipped, so the scene's layout swaps sides.",
    ),
    IssueType(
        name="zoom_jump",
        category="geometric",
        spatial=False,
        point_event=False,
        description=(
            "A segment is abruptly punched in (cropped and scaled up), then "
            "snaps back to the original framing."
        ),
    ),
    IssueType(
        name="inserted_object",
        category="spatial",
        spatial=True,
        point_event=False,
        description=(
            "A foreign object is pasted into the frame; it pops in and out "
            "abruptly and does not interact with the scene."
        ),
    ),
    IssueType(
        name="blurred_region",
        category="spatial",
        spatial=True,
        point_event=False,
        description=(
            "A rectangular region is blurred or pixelated for a segment, as if "
            "retouched or censored."
        ),
    ),
)

ISSUE_TYPE_NAMES: tuple[str, ...] = tuple(issue.name for issue in ISSUE_TYPES)
ISSUE_TYPES_BY_NAME: dict[str, IssueType] = {issue.name: issue for issue in ISSUE_TYPES}
SPATIAL_ISSUE_TYPES: frozenset[str] = frozenset(
    issue.name for issue in ISSUE_TYPES if issue.spatial
)
POINT_EVENT_ISSUE_TYPES: frozenset[str] = frozenset(
    issue.name for issue in ISSUE_TYPES if issue.point_event
)


def is_known_issue_type(name: str) -> bool:
    return name in ISSUE_TYPES_BY_NAME
