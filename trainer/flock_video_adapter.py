"""Submission adapter for the video_inconsistency task (goes in the repo root).

The validator loads this file inside a sandbox (no network, read-only filesystem, no
access to ``validator.*``), calls ``load_detector(model_dir, device, dtype)`` once and then
``detect(video)`` for every clip. See the README for the full contract.

Two detectors live here:

* ``LearnedDetector``: ``vic_features`` -> ``TemporalIssueNet`` (weights in
  ``weights.safetensors``, thresholds and feature statistics in ``vic_config.json``).
* ``HeuristicDetector``: rule-based scores on the same features, used when no trained
  weights are present so the kit works end to end before you train anything.

Both produce a ``(T, 10)`` per-frame probability matrix that ``vic_model.decode_intervals``
turns into issues; spatial issues then get a box from ``vic_localize``.

Design rules for this file: import helpers as TOP-LEVEL modules from this directory (the
sandbox puts the repo root on ``sys.path``), never import anything from ``validator.*``,
keep torch imports lazy so the heuristic path works without torch, and never read from
the network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the helper modules importable regardless of how this file was loaded (by path, by the
# sandbox worker, or from the repo directory).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Belt and braces: nothing here talks to the Hub, and the sandbox has no network anyway.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import vic_features  # noqa: E402
import vic_localize  # noqa: E402
import vic_model  # noqa: E402

CONFIG_FILENAME = "vic_config.json"
WEIGHTS_FILENAME = "weights.safetensors"
_TYPE_INDEX = {name: i for i, name in enumerate(vic_model.ISSUE_TYPE_NAMES)}
_FEATURE_INDEX = {name: i for i, name in enumerate(vic_features.FEATURE_NAMES)}


# ---------------------------------------------------------------------------------------
# heuristic scores (rule-based, no training needed)
# ---------------------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _peaks(score: np.ndarray, threshold: float) -> list[int]:
    """Indices of local maxima of ``score`` that reach ``threshold``."""
    padded = np.concatenate([[-1.0], score, [-1.0]])
    is_peak = (padded[1:-1] >= threshold) & (padded[1:-1] >= padded[:-2]) & (padded[1:-1] > padded[2:])
    return [int(i) for i in np.flatnonzero(is_peak)]


def _pair_symmetric(score: np.ndarray, threshold: float, max_len: int) -> tuple[np.ndarray, set[int]]:
    """A segment starts at one boundary peak and ends at the next: fill [a, b) with min score.

    Returns the per-frame segment score and the set of boundary frames that were paired.
    """
    out = np.zeros_like(score)
    used: set[int] = set()
    peaks = _peaks(score, threshold)
    i = 0
    while i + 1 < len(peaks):
        a, b = peaks[i], peaks[i + 1]
        if b - a <= max_len:
            out[a:b] = min(score[a], score[b])
            used.update((a, b))
            i += 2
        else:
            i += 1
    return out, used


def _pair_directed(
    onset: np.ndarray, offset: np.ndarray, threshold: float, max_len: int
) -> tuple[np.ndarray, set[int]]:
    """Like ``_pair_symmetric`` but the start and end boundaries have separate scores."""
    out = np.zeros_like(onset)
    used: set[int] = set()
    offsets = _peaks(offset, threshold)
    for a in _peaks(onset, threshold):
        following = [b for b in offsets if 0 < b - a <= max_len]
        if following:
            b = following[0]
            out[a:b] = np.maximum(out[a:b], min(onset[a], offset[b]))
            used.update((a, b))
    return out, used


def _short_runs(active: np.ndarray, score: np.ndarray, max_len: int) -> np.ndarray:
    """Keep only runs of ``active`` that are at most ``max_len`` frames long."""
    out = np.zeros_like(score)
    t = 0
    while t < len(active):
        if not active[t]:
            t += 1
            continue
        end = t
        while end + 1 < len(active) and active[end + 1]:
            end += 1
        if end - t + 1 <= max_len:
            out[t : end + 1] = score[t : end + 1]
        t = end + 1
    return out


# Boundary peaks / runs are accepted from this score up (not 0.5): the decoder's own threshold
# (0.5) then splits them into confident detections and lower-confidence candidates, which is
# what a rank-based (mAP) score wants: weak evidence should be ranked below strong evidence,
# not discarded.
CANDIDATE_SCORE = 0.15


def heuristic_probabilities(features: np.ndarray, fps: float) -> np.ndarray:
    """Rule-based ``(T, 10)`` per-frame probabilities from raw (un-normalised) features.

    Every rule looks for the fingerprint documented in ``vic_features``. Boundary-type edits
    (mirror, zoom, colour, splice) are found as *pairs* of boundary spikes and the frames in
    between are marked; a spike that no pair explains is reported as a dropped-frames cut.
    The constants are hand-set from the feature statistics of 15 fps synthetic clips; the two
    spatial types (inserted object, blurred region) are NOT reliably detectable with
    frame-level statistics, so their rules are deliberately strict and rarely fire. The trained
    model replaces all of this.
    """
    f = {name: features[:, i] for name, i in _FEATURE_INDEX.items()}
    num_frames = features.shape[0]
    probs = np.zeros((num_frames, len(vic_model.ISSUE_TYPE_NAMES)), dtype=np.float64)
    max_len = int(round(fps * 6))

    # A boundary is a frame whose difference to its predecessor spikes above the local median.
    spike = _sigmoid(5.0 * (f["logratio_prev"] - 0.6))
    strong_spike = _sigmoid(4.0 * (f["logratio_prev"] - 1.0))

    # frozen frames: the frame is (almost) identical to its predecessor.
    probs[:, _TYPE_INDEX["frozen_frames"]] = f["dup_prev"]

    # mirror: at a boundary, comparing against the FLIPPED previous frame fits far better.
    flip_boundary = spike * _sigmoid(-3.0 * (f["flip_ratio"] + 0.9))
    probs[:, _TYPE_INDEX["mirrored_segment"]], mirror_used = _pair_symmetric(flip_boundary, CANDIDATE_SCORE, max_len)

    # splice: an unrelated shot changes the colour histogram completely.
    splice_boundary = _sigmoid(20.0 * (f["hist_prev"] - 0.25)) * strong_spike
    probs[:, _TYPE_INDEX["spliced_footage"]], splice_used = _pair_symmetric(splice_boundary, CANDIDATE_SCORE, max_len)

    # colour grade: histogram moves a lot but the frame difference barely spikes.
    color_boundary = (
        _sigmoid(60.0 * (f["hist_prev"] - 0.07)) * (1.0 - strong_spike) * (1.0 - splice_boundary)
    )
    probs[:, _TYPE_INDEX["color_grade_jump"]], _ = _pair_symmetric(color_boundary, CANDIDATE_SCORE, max_len)

    # zoom: a punched-in frame matches a zoomed-in copy of its predecessor (onset) or vice versa (offset).
    not_other = (1.0 - splice_boundary) * (1.0 - flip_boundary)
    zoom_on = spike * _sigmoid(-5.0 * (f["zin_ratio"] - 0.3)) * not_other
    zoom_off = spike * _sigmoid(-5.0 * (f["zout_ratio"] - 0.3)) * not_other
    probs[:, _TYPE_INDEX["zoom_jump"]], zoom_used = _pair_directed(zoom_on, zoom_off, CANDIDATE_SCORE, max_len)

    # exposure flicker: 1-4 frames whose luminance leaves the rolling median by a lot.
    flicker = _sigmoid(60.0 * (np.abs(f["lum_med9"]) - 0.06))
    probs[:, _TYPE_INDEX["exposure_flicker"]] = _short_runs(flicker > CANDIDATE_SCORE, flicker, 4)

    # dropped frames: an isolated difference spike that no paired edit explains.
    explained = mirror_used | splice_used | zoom_used
    drop = spike * (1.0 - f["dup_prev"]) * not_other * (1.0 - color_boundary)
    for k in _peaks(drop, CANDIDATE_SCORE):
        if k not in explained and k >= 1:
            probs[k - 1 : k + 1, _TYPE_INDEX["dropped_frames"]] = drop[k]

    # reversal: global motion runs against the clip's trend.
    moving = _sigmoid(6.0 * (f["shift_speed"] - 1.0))
    probs[:, _TYPE_INDEX["reversed_segment"]] = _sigmoid(-8.0 * (f["shift_agree31"] + 0.5)) * moving

    # spatial types: strict rules, expected to fire rarely (see docstring).
    probs[:, _TYPE_INDEX["inserted_object"]] = _sigmoid(40.0 * (f["objg_rel"] - 0.25))
    probs[:, _TYPE_INDEX["blurred_region"]] = _sigmoid(-4.0 * (f["blk_sharp_glob"] + 2.5))
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------------------
class _BaseDetector:
    """Shared plumbing: features -> probabilities -> intervals -> boxes -> plain JSON types."""

    thresholds: dict[str, float] | None = None
    min_frames: dict[str, int] | None = None
    candidate_floor: float = vic_model.CANDIDATE_FLOOR

    def probabilities(self, features: np.ndarray, fps: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def detect(self, video: dict[str, Any]) -> dict[str, Any]:
        frames = video["frames"]
        fps = float(video["fps"])
        num_frames = int(video.get("num_frames", frames.shape[0]))
        duration = float(video.get("duration", num_frames / fps))
        if frames.shape[0] < 2:
            return {"issues": []}

        features = vic_features.extract_features(frames)
        probs = self.probabilities(features, fps)
        issues = vic_model.decode_intervals(
            probs, fps, duration, self.thresholds, self.min_frames, candidate_floor=self.candidate_floor
        )
        return {"issues": [self._finalise(issue, frames) for issue in issues]}

    @staticmethod
    def _finalise(issue: dict[str, Any], frames: np.ndarray) -> dict[str, Any]:
        start_frame, end_frame = issue.pop("_start_frame"), issue.pop("_end_frame")
        box = vic_localize.localize(issue["type"], frames, start_frame, end_frame)
        if box is not None:
            issue["bbox"] = [float(v) for v in box]
        return issue


class HeuristicDetector(_BaseDetector):
    """Rule-based fallback; needs only numpy."""

    def __init__(self) -> None:
        self.thresholds = None  # decode_intervals default (0.5)
        self.min_frames = None

    def probabilities(self, features: np.ndarray, fps: float) -> np.ndarray:
        return heuristic_probabilities(features, fps)


def _resolve_device(device: str) -> Any:
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(device)
    except (RuntimeError, TypeError):
        return torch.device("cpu")


class LearnedDetector(_BaseDetector):
    """``TemporalIssueNet`` on normalised features, thresholds from ``vic_config.json``."""

    def __init__(self, model_dir: Path, device: str, dtype: str) -> None:
        import torch
        from safetensors.torch import load_file

        config = json.loads((model_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
        if list(config["issue_types"]) != list(vic_model.ISSUE_TYPE_NAMES):
            raise ValueError("vic_config.json was trained with a different issue-type order")
        if list(config["feature_names"]) != list(vic_features.FEATURE_NAMES):
            raise ValueError("vic_config.json was trained with a different feature set")
        self._torch = torch
        self.device = _resolve_device(device)
        # The network is tiny: float32 is exact and fast on CPU. Half precision is only honoured
        # on GPU, where it is harmless for a model this size.
        half = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(str(dtype))
        self.dtype = half if (half is not None and self.device.type == "cuda") else torch.float32
        self.stats = config["feature_stats"]
        self.thresholds = {k: float(v) for k, v in config["thresholds"].items()}
        self.min_frames = {k: int(v) for k, v in config.get("min_frames", {}).items()} or None
        self.candidate_floor = float(config.get("candidate_floor", vic_model.CANDIDATE_FLOOR))
        self.model = vic_model.TemporalIssueNet(**config["model"])
        self.model.load_state_dict(load_file(str(model_dir / WEIGHTS_FILENAME)))
        self.model.to(device=self.device, dtype=self.dtype).eval()

    def probabilities(self, features: np.ndarray, fps: float) -> np.ndarray:
        torch = self._torch
        normed = vic_features.normalize_features(features, self.stats)
        with torch.no_grad():
            x = torch.from_numpy(normed)[None].to(device=self.device, dtype=self.dtype)
            return torch.sigmoid(self.model(x).float())[0].cpu().numpy()


def load_detector(model_dir: str, device: str = "cpu", dtype: str = "float32") -> Any:
    """Entry point called by the validator.

    Uses the trained model when ``vic_config.json`` and ``weights.safetensors`` are present in
    ``model_dir`` (and torch imports), otherwise the heuristic detector. A present-but-broken
    weights file raises instead of silently degrading, so a bad upload is visible in local
    validation rather than scored as a weak model.
    """
    directory = Path(model_dir)
    has_weights = (directory / CONFIG_FILENAME).is_file() and (directory / WEIGHTS_FILENAME).is_file()
    if has_weights:
        return LearnedDetector(directory, device, dtype)
    return HeuristicDetector()
