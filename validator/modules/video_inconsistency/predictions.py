"""Strict parsing of a trainer detector's raw output into typed predictions.

The detector runs in a sandbox and its return value crosses a process boundary as
JSON, so nothing about it can be trusted. This module is the single place that
turns that untrusted object into validated ``PredictedIssue`` values; the scorer
never sees anything that did not pass through ``parse_detector_output``.

Rules (any violation raises a *non-fatal* ``VideoSubmissionError`` with
``failure_mode="detector_output_invalid"`` naming the offending item index and
field, so the validator can retry the clip and then score it as an empty answer):

* the payload is either ``{"issues": [...]}`` or a bare list; at most 1000 items;
* each item is a dict with a known ``type`` and finite real ``start_time`` /
  ``end_time`` (``bool`` is not a number); both are clamped into
  ``[0, duration]`` and ``end_time >= start_time`` must hold after clamping
  (1e-6 tolerance);
* ``confidence`` is optional (default 1.0) and must be finite in ``[0, 1]``;
* ``bbox`` is optional (``null`` allowed) and must be a valid normalised box;
* ``description`` is optional free text (<= 500 chars) and is dropped;
* unknown extra keys are ignored.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel

from validator.modules.video_inconsistency.errors import VideoSubmissionError
from validator.modules.video_inconsistency.issue_types import is_known_issue_type
from validator.modules.video_inconsistency.manifest import validate_bbox


MAX_ISSUES_PER_OUTPUT = 1000
MAX_DESCRIPTION_CHARS = 500
_TIME_TOLERANCE = 1e-6
_FAILURE_MODE = "detector_output_invalid"


class PredictedIssue(BaseModel, frozen=True):
    """One validated detection, with times clamped into the clip."""

    type: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    bbox: list[float] | None = None


def _invalid(index: int | None, field: str | None, problem: str) -> VideoSubmissionError:
    if index is None:
        where = "detector output"
    elif field is None:
        where = f"detector output item {index}"
    else:
        where = f"detector output item {index} field {field!r}"
    return VideoSubmissionError(f"{where}: {problem}", _FAILURE_MODE, fatal=False)


def _is_real_number(value: Any) -> bool:
    # bool is an int subclass; a detector sending True/False is a schema bug.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, index: int, field: str) -> float:
    if not _is_real_number(value):
        raise _invalid(index, field, f"must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise _invalid(index, field, "must be finite")
    return number


def _parse_bbox(value: Any, index: int) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise _invalid(index, "bbox", "must be null or a list of four numbers")
    if len(value) != 4:
        raise _invalid(index, "bbox", "must be [x0, y0, x1, y1]")
    numbers = [_finite_number(v, index, "bbox") for v in value]
    try:
        return validate_bbox(numbers)
    except ValueError as exc:
        raise _invalid(index, "bbox", str(exc)) from exc


def _parse_item(item: Any, index: int, duration: float) -> PredictedIssue:
    if not isinstance(item, dict):
        raise _invalid(index, None, f"must be an object, got {type(item).__name__}")

    issue_type = item.get("type")
    if not isinstance(issue_type, str) or not is_known_issue_type(issue_type):
        raise _invalid(index, "type", f"unknown issue type {issue_type!r}")

    if "start_time" not in item:
        raise _invalid(index, "start_time", "is required")
    if "end_time" not in item:
        raise _invalid(index, "end_time", "is required")
    start = min(max(_finite_number(item["start_time"], index, "start_time"), 0.0), duration)
    end = min(max(_finite_number(item["end_time"], index, "end_time"), 0.0), duration)
    if end < start - _TIME_TOLERANCE:
        raise _invalid(index, "end_time", "must be >= start_time")
    end = max(end, start)

    confidence = 1.0
    if "confidence" in item and item["confidence"] is not None:
        confidence = _finite_number(item["confidence"], index, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise _invalid(index, "confidence", "must be within [0, 1]")

    bbox = _parse_bbox(item.get("bbox"), index)

    description = item.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise _invalid(index, "description", "must be a string")
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise _invalid(
                index, "description", f"must be at most {MAX_DESCRIPTION_CHARS} characters"
            )

    return PredictedIssue(
        type=issue_type,
        start_time=start,
        end_time=end,
        confidence=confidence,
        bbox=bbox,
    )


def parse_detector_output(raw: Any, duration: float) -> list[PredictedIssue]:
    """Validate ``raw`` and return predictions clamped to ``[0, duration]``."""
    if isinstance(raw, dict):
        if "issues" not in raw:
            raise _invalid(None, None, "object output must contain an 'issues' list")
        items = raw["issues"]
    else:
        items = raw
    if not isinstance(items, list):
        raise _invalid(None, None, "must be a list of issues or {'issues': [...]}")
    if len(items) > MAX_ISSUES_PER_OUTPUT:
        raise _invalid(None, None, f"at most {MAX_ISSUES_PER_OUTPUT} issues are allowed")
    return [_parse_item(item, index, duration) for index, item in enumerate(items)]
