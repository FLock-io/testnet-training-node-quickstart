"""Temporal model and interval decoder for the video-inconsistency baseline.

``TemporalIssueNet`` is a small non-causal dilated 1D conv net that maps the
per-frame feature matrix from ``vic_features`` to per-frame logits for the 10
issue types. ``decode_intervals`` (numpy only) turns per-frame probabilities into
the detector output schema.

The submission cannot import anything from ``validator.*`` (it is not readable
inside the sandbox), so the issue-type order is hard-coded here. ``train.py``
asserts it equals the validator's ``ISSUE_TYPE_NAMES`` at training time.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

ISSUE_TYPE_NAMES: tuple[str, ...] = (
    "frozen_frames",
    "dropped_frames",
    "reversed_segment",
    "spliced_footage",
    "color_grade_jump",
    "exposure_flicker",
    "mirrored_segment",
    "zoom_jump",
    "inserted_object",
    "blurred_region",
)
SPATIAL_TYPES = frozenset({"inserted_object", "blurred_region"})
POINT_EVENT_TYPES = frozenset({"dropped_frames"})
MODEL_VERSION = 2

# Shortest run (in frames) worth reporting per type. A flicker is 1-3 frames and a drop is
# a single boundary, everything else is a segment that lasts at least a few frames.
DEFAULT_MIN_FRAMES: dict[str, int] = {name: 3 for name in ISSUE_TYPE_NAMES}
DEFAULT_MIN_FRAMES.update({"exposure_flicker": 1, "dropped_frames": 1})

# The validator matches at most 25 predictions per clip (mAP rewards a ranked candidate list).
MAX_ISSUES = 25
# Candidate runs are extracted from a fixed probability floor, independent of the type's
# decision threshold (so tuning the threshold trades F1 / clip accuracy against confident false
# positives, and does not also change how many weak candidates the mAP gets to rank). They are
# emitted with confidence below 0.5, so they cost nothing in F1 / clip accuracy.
CANDIDATE_FLOOR = 0.1
CONFIDENCE_FLOOR = 0.05
# A type whose threshold is >= 1.0 ("switched off" by calibration) is still emitted, but only as
# candidates whose confidence stays below 0.5.
OFF_CONFIDENCE_CEILING = 0.45
CANDIDATE_CEILING = 0.49


try:  # torch is optional so the heuristic detector works in a torch-free environment.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def dilations_for_layers(layers: int) -> tuple[int, ...]:
    """Dilation schedule for ``layers`` residual blocks: 1, 2, 4, ..., 32 and then repeat.

    ``layers=6`` gives the original ``(1, 2, 4, 8, 16, 32)`` (receptive field 127 frames).
    """
    if layers < 1:
        raise ValueError("layers must be >= 1")
    return tuple(2 ** (i % 6) for i in range(layers))


if nn is not None:

    class _ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.conv = nn.Conv1d(
                channels, channels, kernel_size=3, dilation=dilation, padding=dilation
            )
            self.mix = nn.Conv1d(channels, channels, kernel_size=1)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)

        def forward(
            self, x: "torch.Tensor", mask: "torch.Tensor | None"
        ) -> "torch.Tensor":
            y = self.mix(self.drop(self.act(self.conv(self.act(x)))))
            x = x + y
            # Zero the padded positions after every block so training on padded batches sees
            # the same "zero padding at the edges" as inference on a single unpadded clip.
            return x if mask is None else x * mask

    class TemporalIssueNet(nn.Module):
        """Input ``(B, T, F)`` features -> ``(B, T, num_classes)`` per-frame logits."""

        def __init__(
            self,
            in_features: int,
            hidden: int = 64,
            dilations: Sequence[int] = dilations_for_layers(6),
            dropout: float = 0.1,
            num_classes: int = len(ISSUE_TYPE_NAMES),
        ) -> None:
            super().__init__()
            self.in_proj = nn.Conv1d(in_features, hidden, kernel_size=1)
            self.blocks = nn.ModuleList(
                _ResidualBlock(hidden, d, dropout) for d in dilations
            )
            self.head = nn.Sequential(
                nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden, num_classes, 1)
            )

        def forward(
            self, features: "torch.Tensor", mask: "torch.Tensor | None" = None
        ) -> "torch.Tensor":
            """``mask`` is ``(B, T)`` with 1 for real frames and 0 for padding (optional)."""
            channel_mask = (
                None if mask is None else mask.unsqueeze(1).to(features.dtype)
            )
            x = self.in_proj(features.transpose(1, 2))
            if channel_mask is not None:
                x = x * channel_mask
            for block in self.blocks:
                x = block(x, channel_mask)
            return self.head(x).transpose(1, 2)

    def model_hyperparameters(model: "TemporalIssueNet") -> dict[str, Any]:
        """Constructor arguments needed to rebuild ``model`` from ``vic_config.json``."""
        return {
            "in_features": int(model.in_proj.in_channels),
            "hidden": int(model.in_proj.out_channels),
            "dilations": [int(b.conv.dilation[0]) for b in model.blocks],
            "dropout": float(model.blocks[0].drop.p) if len(model.blocks) else 0.0,
            "num_classes": int(model.head[-1].out_channels),
        }


def _runs(active: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` runs of True values, merging gaps of <= ``max_gap`` frames."""
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value - prev - 1 > max_gap:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _per_type(value: Any, name: str, default: float) -> float:
    if isinstance(value, Mapping):
        return float(value.get(name, default))
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[ISSUE_TYPE_NAMES.index(name)])
    if value is None:
        return default
    return float(value)


def calibrated_confidence(prob: float, threshold: float) -> float:
    """Map a probability so that ``threshold`` lands on 0.5 (piecewise linear, monotone).

    ``[0, threshold] -> [0, 0.5]`` and ``[threshold, 1] -> [0.5, 1]``. Unlike v1 there is no
    clipping to >= 0.5: probabilities below the decision threshold give confidences below 0.5,
    which the validator ignores for F1 / clip accuracy but ranks for mAP.
    """
    threshold = float(np.clip(threshold, 0.01, 0.99))
    if prob >= threshold:
        value = 0.5 + 0.5 * (prob - threshold) / (1.0 - threshold)
    else:
        value = 0.5 * prob / threshold
    return float(np.clip(value, CONFIDENCE_FLOOR, 1.0))


def decode_intervals(
    probs: np.ndarray,
    fps: float,
    duration: float,
    thresholds: Mapping[str, float] | Sequence[float] | float | None = None,
    min_frames: Mapping[str, int] | None = None,
    max_issues: int = MAX_ISSUES,
    candidate_floor: float = CANDIDATE_FLOOR,
) -> list[dict[str, Any]]:
    """Turn ``(T, 10)`` per-frame probabilities into a ranked list of detector issues.

    Per type there are two kinds of runs:

    * **Confident runs**: frames with probability >= the type's decision threshold, merged
      across gaps of <= 1 frame, shorter runs than the per-type minimum dropped. Their
      confidence is ``calibrated_confidence(mean probability, threshold)`` >= 0.5, exactly
      as in v1 (threshold -> 0.5).
    * **Candidate runs**: frames >= ``candidate_floor`` (default 0.1; or the threshold if that is lower) that do NOT
      overlap a confident run. Their boundaries are trimmed to frames >= half of the run's peak
      (so a weak, wide plateau does not become a long interval) and their confidence is the
      same calibrated mapping, so it is below 0.5. They matter for mAP (a ranked list of
      guesses with honest confidences is rewarded); they cannot hurt F1 or clip accuracy.

    Time convention: frames ``s..e`` inclusive -> ``s/fps .. (e+1)/fps``.
    ``dropped_frames`` is a point event: the run's probability-weighted centre gives the cut
    position ``k`` (the model is trained to fire on frames ``k-1`` and ``k``, so their centre
    is the boundary), reported with ``start_time == end_time == k / fps``.

    A threshold >= 1.0 ("off") keeps the type as candidates only, with confidence capped below
    0.5, so a type the model cannot detect reliably never adds a confident false positive but
    still contributes a ranking. The result is capped to the ``max_issues`` most confident
    (the validator matches at most 25 per clip).

    Each issue also carries private ``_start_frame`` / ``_end_frame`` keys (frame indices, end
    exclusive) for ``vic_localize``; the adapter removes them before returning.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(ISSUE_TYPE_NAMES):
        raise ValueError(f"probs must be (T, {len(ISSUE_TYPE_NAMES)})")
    num_frames = probs.shape[0]
    min_table = dict(DEFAULT_MIN_FRAMES)
    if min_frames:
        min_table.update(min_frames)

    def make_issue(
        name: str,
        curve: np.ndarray,
        start: int,
        end: int,
        threshold: float,
        ceiling: float = 1.0,
    ) -> dict[str, Any]:
        length = end - start + 1
        mean_prob = float(curve[start : end + 1].mean())
        confidence = calibrated_confidence(mean_prob, threshold)
        confidence = min(confidence, ceiling)
        if name in POINT_EVENT_TYPES:
            weights = curve[start : end + 1]
            centre = float(
                (np.arange(start, end + 1) * weights).sum() / max(weights.sum(), 1e-9)
            )
            boundary = int(np.clip(round(centre + 0.5), 1, max(num_frames - 1, 1)))
            start_time = end_time = min(boundary / fps, duration)
        else:
            start_time = min(start / fps, duration)
            end_time = min((end + 1) / fps, duration)
        return {
            "type": name,
            "start_time": float(start_time),
            "end_time": float(max(end_time, start_time)),
            "confidence": float(confidence),
            "description": f"{name}: mean probability {mean_prob:.2f} over {length} frame(s)",
            "_start_frame": int(start),
            "_end_frame": int(end) + 1,
        }

    issues: list[dict[str, Any]] = []
    for index, name in enumerate(ISSUE_TYPE_NAMES):
        curve = probs[:, index]
        raw_threshold = _per_type(thresholds, name, 0.5)
        off = raw_threshold >= 1.0
        threshold = float(np.clip(0.99 if off else raw_threshold, 0.01, 0.99))
        minimum = int(min_table.get(name, 1))
        candidate = float(min(max(candidate_floor, 0.01), threshold))

        confident = [
            (s, e) for s, e in _runs(curve >= threshold) if e - s + 1 >= minimum
        ]
        covered = np.zeros(num_frames, dtype=bool)
        for start, end in confident:
            covered[start : end + 1] = True
            # A run of an "off" type is only a candidate: keep it (capped) rather than dropping it.
            issues.append(
                make_issue(
                    name,
                    curve,
                    start,
                    end,
                    threshold,
                    OFF_CONFIDENCE_CEILING if off else 1.0,
                )
            )
        if candidate >= threshold:
            continue
        for start, end in _runs(curve >= candidate):
            if covered[start : end + 1].any():
                continue
            peak = float(curve[start : end + 1].max())
            keep = curve[start : end + 1] >= max(candidate, 0.5 * peak)
            for sub_start, sub_end in _runs(keep):
                if sub_end - sub_start + 1 < minimum:
                    continue
                issues.append(
                    make_issue(
                        name,
                        curve,
                        start + sub_start,
                        start + sub_end,
                        threshold,
                        CANDIDATE_CEILING,
                    )
                )
    issues.sort(key=lambda item: -item["confidence"])
    issues = issues[:max_issues]
    issues.sort(key=lambda item: (item["start_time"], item["type"]))
    return issues
