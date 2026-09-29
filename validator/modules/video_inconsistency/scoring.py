"""Scoring of detected issues against the hidden ground truth (suite v2).

The headline term is a rank-based **mean average precision** (mAP); the v1 F1
computations are kept for reporting and for the localization / clip-accuracy terms.
Everything is deterministic.

Shared steps
------------
* Interval padding: any interval (ground truth or prediction) shorter than
  ``min_event_seconds`` is widened symmetrically about its centre to that length
  and then shifted to stay inside ``[0, clip duration]``. This makes point events
  (``dropped_frames``) and very short flickers matchable with a sensible
  tolerance. Temporal IoU (tIoU) is computed on the padded intervals.
* Difficulty weighting: every TP / FP / FN (and every AP increment) is weighted by
  its clip's difficulty weight (easy 1.0, medium 1.5, hard 2.0, expert 2.5).

Mean average precision (weight ``weight_map``)
----------------------------------------------
1. Per clip, ALL predictions are ranked by confidence, descending (stable, so ties
   keep the detector's order) and only the top ``max_predictions_per_clip`` are
   kept; the rest are dropped from the AP computation (AP already punishes low
   ranked junk). ``confidence_threshold`` is NOT applied: low-confidence guesses
   sit at the bottom of the ranking and cost almost nothing, while a
   well-calibrated ranking is rewarded.
2. For every issue type and every threshold ``t`` in ``tiou_thresholds``: pool the
   type's predictions over all clips and sort by confidence descending (ties: clip
   order, then the detector's prediction order). Walk the list; each prediction is
   matched to the still-unmatched ground truth of the same type IN THE SAME CLIP
   with the highest tIoU. It is a true positive iff that tIoU is ``>= t`` and, for
   spatial types, the prediction carries a bbox whose IoU with that ground truth's
   bbox is ``>= bbox_iou_threshold`` (no bbox: never a TP); otherwise it is a
   false positive and consumes nothing. TP / FP increments equal the clip's
   difficulty weight.
3. Precision / recall after each step (recall denominator: the type's total
   weighted ground truth); AP is the all-point interpolated area under the
   precision envelope (VOC2010+ / COCO style, monotone non-increasing precision).
4. A type with ground truth scores AP as computed (0 with no predictions). A type
   without ground truth is excluded when it has no predictions and scores AP = 0
   when it has any (hallucination). If no type is included at all (all clean, no
   predictions) ``mean_ap = 1``.
5. ``per_type_ap[type]`` = mean over thresholds; ``ap_by_tiou[t]`` = mean over
   included types; ``mean_ap`` = mean over included types and thresholds.

F1 / localization / clip accuracy (v1 semantics, ``tiou_threshold``)
--------------------------------------------------------------------
6. Per clip keep predictions with ``confidence >= confidence_threshold``, sort by
   confidence (stable); only the top ``max_predictions_per_clip`` are matched and
   every prediction beyond the cap is a false positive of its type.
7. Greedy matching per type in confidence order: each prediction takes the
   unmatched ground truth of the same type with the highest tIoU; TP when that
   tIoU is ``>= tiou_threshold``, else FP. Unmatched ground truth are FNs.
8. Per-type precision / recall / F1 from the weighted counts; a type with no
   ground truth and no predictions is excluded from ``macro_f1`` (mean over
   included types). ``micro_f1`` / ``precision`` / ``recall`` come from pooled
   counts. With no included type all are 1.0.
9. ``localization_score = sum(weight * q over TPs) / (sum(weight over all GT) +
   sum(weight over confident FPs))`` (1.0 when both are zero), so confident junk
   is never free, where
   ``q = tIoU`` for non-spatial types and ``q = 0.5 * tIoU + 0.5 * bbox_IoU`` for
   spatial types (a missing predicted bbox gives bbox_IoU 0).
10. ``clip_accuracy`` is the balanced accuracy of "the clip contains an issue"
    (predicted positive iff it has a prediction passing the confidence
    threshold); unweighted so clean clips always count.

Final
-----
``score = weight_map * mean_ap + weight_f1 * macro_f1 + weight_localization *
localization_score + weight_clip_accuracy * clip_accuracy``, clipped to
``[0, 1]``; ``loss = 1 - score``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from validator.modules.video_inconsistency.issue_types import (
    ISSUE_TYPE_NAMES,
    SPATIAL_ISSUE_TYPES,
)
from validator.modules.video_inconsistency.manifest import (
    DIFFICULTIES,
    ClipSpec,
    IssueLabel,
)
from validator.modules.video_inconsistency.predictions import PredictedIssue


DEFAULT_DIFFICULTY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("easy", 1.0),
    ("medium", 1.5),
    ("hard", 2.0),
    ("expert", 2.5),
)
DEFAULT_TIOU_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
_WEIGHT_SUM_TOLERANCE = 1e-6
# Absorbs float noise so a prediction exactly on a threshold (e.g. tIoU 0.5 computed
# as 0.4999999999) still counts.
_MATCH_EPSILON = 1e-9


@dataclass(frozen=True)
class ScoringSettings:
    tiou_thresholds: tuple[float, ...] = DEFAULT_TIOU_THRESHOLDS
    tiou_threshold: float = 0.3  # F1 / localization reporting only
    bbox_iou_threshold: float = 0.3
    min_event_seconds: float = 0.4
    confidence_threshold: float = 0.5
    max_predictions_per_clip: int = 25
    difficulty_weights: tuple[tuple[str, float], ...] = DEFAULT_DIFFICULTY_WEIGHTS
    weight_map: float = 0.75
    weight_f1: float = 0.0
    weight_localization: float = 0.20
    weight_clip_accuracy: float = 0.05

    def __post_init__(self) -> None:
        # Accept a mapping (convenient from JSON config) but store a hashable tuple.
        weights = self.difficulty_weights
        if isinstance(weights, dict):
            weights = tuple(weights.items())
        weights = tuple((str(name), float(value)) for name, value in weights)
        object.__setattr__(self, "difficulty_weights", weights)

        thresholds = tuple(float(t) for t in self.tiou_thresholds)
        object.__setattr__(self, "tiou_thresholds", thresholds)
        if not thresholds:
            raise ValueError("tiou_thresholds must not be empty")
        if any(not (math.isfinite(t) and 0.0 < t <= 1.0) for t in thresholds):
            raise ValueError("tiou_thresholds must all be in (0, 1]")
        if len({tiou_key(t) for t in thresholds}) != len(thresholds):
            raise ValueError("tiou_thresholds must be distinct")
        if not 0.0 < self.tiou_threshold <= 1.0:
            raise ValueError("tiou_threshold must be in (0, 1]")
        if not 0.0 < self.bbox_iou_threshold <= 1.0:
            raise ValueError("bbox_iou_threshold must be in (0, 1]")
        if not (
            math.isfinite(self.min_event_seconds) and self.min_event_seconds >= 0.0
        ):
            raise ValueError("min_event_seconds must be finite and >= 0")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_predictions_per_clip < 1:
            raise ValueError("max_predictions_per_clip must be >= 1")
        table = dict(weights)
        for difficulty in DIFFICULTIES:
            if difficulty not in table:
                raise ValueError(f"difficulty_weights must define {difficulty!r}")
        if any(not (math.isfinite(v) and v > 0.0) for v in table.values()):
            raise ValueError("difficulty weights must be finite and > 0")
        term_weights = (
            self.weight_map,
            self.weight_f1,
            self.weight_localization,
            self.weight_clip_accuracy,
        )
        if any(not (math.isfinite(w) and w >= 0.0) for w in term_weights):
            raise ValueError("score term weights must be finite and >= 0")
        if abs(sum(term_weights) - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                "weight_map + weight_f1 + weight_localization + weight_clip_accuracy "
                "must equal 1"
            )

    def weight_for(self, difficulty: str) -> float:
        return dict(self.difficulty_weights).get(difficulty, 1.0)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    loss: float
    macro_f1: float
    micro_f1: float
    precision: float
    recall: float
    localization_score: float
    clip_accuracy: float
    per_type_f1: dict[str, float]
    per_type_counts: dict[str, dict[str, float]]
    mean_ap: float
    per_type_ap: dict[str, float]
    ap_by_tiou: dict[str, float]


def tiou_key(threshold: float) -> str:
    """Stable string key for a tIoU threshold (0.3 -> "0.3")."""
    return f"{threshold:g}"


def _pad_interval(
    start: float, end: float, min_length: float, duration: float
) -> tuple[float, float]:
    if end - start >= min_length:
        return start, end
    centre = (start + end) / 2.0
    start, end = centre - min_length / 2.0, centre + min_length / 2.0
    if start < 0.0:
        end -= start
        start = 0.0
    if end > duration:
        start -= end - duration
        end = duration
    return max(start, 0.0), end


def _tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    if union <= 0.0:
        # Two zero-length intervals (only possible with min_event_seconds == 0).
        return 1.0 if a[0] == b[0] else 0.0
    return intersection / union


def _bbox_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0.0 else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return _ratio(2.0 * precision * recall, precision + recall)


class _Counts:
    __slots__ = ("tp", "fp", "fn", "support")

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.support = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "support": self.support}


def _match_clip(
    clip: ClipSpec,
    predictions: list[PredictedIssue],
    settings: ScoringSettings,
    weight: float,
    counts: dict[str, _Counts],
) -> float:
    """Match one clip, update ``counts`` and return the summed weighted quality of its TPs."""
    duration = clip.duration
    kept = [p for p in predictions if p.confidence >= settings.confidence_threshold]
    kept.sort(key=lambda p: -p.confidence)  # stable
    matched_preds = kept[: settings.max_predictions_per_clip]
    overflow = kept[settings.max_predictions_per_clip :]

    for pred in overflow:
        counts[pred.type].fp += weight

    gt_intervals = [
        _pad_interval(g.start_time, g.end_time, settings.min_event_seconds, duration)
        for g in clip.issues
    ]
    for label in clip.issues:
        counts[label.type].support += weight
    taken = [False] * len(clip.issues)

    quality_sum = 0.0
    for pred in matched_preds:
        pred_interval = _pad_interval(
            pred.start_time, pred.end_time, settings.min_event_seconds, duration
        )
        best_index = -1
        best_tiou = -1.0
        for index, label in enumerate(clip.issues):
            if taken[index] or label.type != pred.type:
                continue
            value = _tiou(pred_interval, gt_intervals[index])
            if value > best_tiou:
                best_index, best_tiou = index, value
        if best_index >= 0 and best_tiou >= settings.tiou_threshold:
            taken[best_index] = True
            counts[pred.type].tp += weight
            quality_sum += weight * _match_quality(
                pred, clip.issues[best_index], best_tiou
            )
        else:
            counts[pred.type].fp += weight

    for index, label in enumerate(clip.issues):
        if not taken[index]:
            counts[label.type].fn += weight
    return quality_sum


def _match_quality(pred: PredictedIssue, label: IssueLabel, tiou: float) -> float:
    if label.type in SPATIAL_ISSUE_TYPES:
        return 0.5 * tiou + 0.5 * _bbox_iou(pred.bbox, label.bbox)
    return tiou


def _balanced_clip_accuracy(
    clips: Sequence[ClipSpec],
    predictions: Sequence[list[PredictedIssue]],
    settings: ScoringSettings,
) -> float:
    edited_total = edited_hit = clean_total = clean_correct = 0
    for clip, preds in zip(clips, predictions):
        flagged = any(p.confidence >= settings.confidence_threshold for p in preds)
        if clip.issues:
            edited_total += 1
            edited_hit += int(flagged)
        else:
            clean_total += 1
            clean_correct += int(not flagged)
    rates = []
    if edited_total:
        rates.append(edited_hit / edited_total)
    if clean_total:
        rates.append(clean_correct / clean_total)
    return sum(rates) / len(rates)


class _RankedPrediction:
    """A prediction in the pooled per-type ranking, with its precomputed matches."""

    __slots__ = (
        "confidence",
        "clip_index",
        "order",
        "weight",
        "pred",
        "gt_indices",
        "tious",
    )

    def __init__(
        self,
        confidence: float,
        clip_index: int,
        order: int,
        weight: float,
        pred: PredictedIssue,
        gt_indices: list[int],
        tious: list[float],
    ) -> None:
        self.confidence = confidence
        self.clip_index = clip_index
        self.order = order
        self.weight = weight
        self.pred = pred
        self.gt_indices = gt_indices  # indices into clip.issues of the same type
        self.tious = tious  # padded tIoU against each of those ground truths


def _average_precision(
    ranked: Sequence[tuple[float, bool]], total_gt_weight: float
) -> float:
    """All-point interpolated AP for ``(weight, is_tp)`` rows in rank order."""
    if total_gt_weight <= 0.0:
        return 0.0
    cum_tp = cum_fp = 0.0
    recalls = [0.0]
    precisions = [0.0]
    for weight, is_tp in ranked:
        if is_tp:
            cum_tp += weight
        else:
            cum_fp += weight
        recalls.append(min(1.0, cum_tp / total_gt_weight))
        precisions.append(cum_tp / (cum_tp + cum_fp))
    recalls.append(recalls[-1])
    precisions.append(0.0)
    # Monotone (non-increasing) precision envelope, right to left.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum(
        (recalls[i] - recalls[i - 1]) * precisions[i]
        for i in range(1, len(recalls))
        if recalls[i] != recalls[i - 1]
    )


def _ranked_by_type(
    clips: Sequence[ClipSpec],
    predictions: Sequence[list[PredictedIssue]],
    settings: ScoringSettings,
) -> dict[str, list[_RankedPrediction]]:
    """Apply the per-clip cap by confidence and pool the survivors per type."""
    pooled: dict[str, list[_RankedPrediction]] = {name: [] for name in ISSUE_TYPE_NAMES}
    for clip_index, (clip, preds) in enumerate(zip(clips, predictions)):
        duration = clip.duration
        indexed = sorted(
            enumerate(preds), key=lambda item: -item[1].confidence
        )  # stable
        weight = settings.weight_for(clip.difficulty)
        gt_intervals = [
            _pad_interval(
                g.start_time, g.end_time, settings.min_event_seconds, duration
            )
            for g in clip.issues
        ]
        for order, pred in indexed[: settings.max_predictions_per_clip]:
            pred_interval = _pad_interval(
                pred.start_time, pred.end_time, settings.min_event_seconds, duration
            )
            gt_indices = [i for i, g in enumerate(clip.issues) if g.type == pred.type]
            tious = [_tiou(pred_interval, gt_intervals[i]) for i in gt_indices]
            pooled[pred.type].append(
                _RankedPrediction(
                    pred.confidence, clip_index, order, weight, pred, gt_indices, tious
                )
            )
    for rows in pooled.values():
        rows.sort(key=lambda r: (-r.confidence, r.clip_index, r.order))
    return pooled


def _ap_at_threshold(
    clips: Sequence[ClipSpec],
    rows: Sequence[_RankedPrediction],
    total_gt_weight: float,
    tiou_threshold: float,
    settings: ScoringSettings,
) -> float:
    taken: set[tuple[int, int]] = set()  # (clip index, ground-truth index)
    outcomes: list[tuple[float, bool]] = []
    for row in rows:
        best_gt = -1
        best_tiou = -1.0
        for gt_index, value in zip(row.gt_indices, row.tious):
            if (row.clip_index, gt_index) in taken:
                continue
            if value > best_tiou:
                best_gt, best_tiou = gt_index, value
        is_tp = best_gt >= 0 and best_tiou >= tiou_threshold - _MATCH_EPSILON
        if is_tp and row.pred.type in SPATIAL_ISSUE_TYPES:
            gt_bbox = clips[row.clip_index].issues[best_gt].bbox
            is_tp = (
                row.pred.bbox is not None
                and _bbox_iou(row.pred.bbox, gt_bbox)
                >= settings.bbox_iou_threshold - _MATCH_EPSILON
            )
        if is_tp:
            taken.add((row.clip_index, best_gt))
        outcomes.append((row.weight, is_tp))
    return _average_precision(outcomes, total_gt_weight)


def _mean_average_precision(
    clips: Sequence[ClipSpec],
    predictions: Sequence[list[PredictedIssue]],
    settings: ScoringSettings,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return ``(mean_ap, per_type_ap, ap_by_tiou)``."""
    gt_weight = {name: 0.0 for name in ISSUE_TYPE_NAMES}
    for clip in clips:
        weight = settings.weight_for(clip.difficulty)
        for issue in clip.issues:
            gt_weight[issue.type] += weight
    pooled = _ranked_by_type(clips, predictions, settings)

    per_type_at: dict[str, list[float]] = {}
    for name in ISSUE_TYPE_NAMES:
        if gt_weight[name] <= 0.0 and not pooled[name]:
            continue  # nothing to detect, nothing hallucinated: not informative
        if gt_weight[name] <= 0.0:
            per_type_at[name] = [0.0] * len(settings.tiou_thresholds)
        else:
            per_type_at[name] = [
                _ap_at_threshold(clips, pooled[name], gt_weight[name], t, settings)
                for t in settings.tiou_thresholds
            ]

    if not per_type_at:
        return 1.0, {}, {tiou_key(t): 1.0 for t in settings.tiou_thresholds}
    per_type_ap = {name: sum(v) / len(v) for name, v in per_type_at.items()}
    ap_by_tiou = {
        tiou_key(t): sum(v[i] for v in per_type_at.values()) / len(per_type_at)
        for i, t in enumerate(settings.tiou_thresholds)
    }
    mean_ap = sum(per_type_ap.values()) / len(per_type_ap)
    return mean_ap, per_type_ap, ap_by_tiou


def score_predictions(
    clips: Sequence[ClipSpec],
    predictions: Sequence[list[PredictedIssue]],
    settings: ScoringSettings = ScoringSettings(),
) -> ScoreResult:
    if len(clips) != len(predictions):
        raise ValueError("clips and predictions must have the same length")
    if not clips:
        raise ValueError("at least one clip is required")

    counts: dict[str, _Counts] = {name: _Counts() for name in ISSUE_TYPE_NAMES}
    quality_sum = 0.0
    total_gt_weight = 0.0
    for clip, preds in zip(clips, predictions):
        weight = settings.weight_for(clip.difficulty)
        total_gt_weight += weight * len(clip.issues)
        quality_sum += _match_clip(clip, list(preds), settings, weight, counts)

    per_type_f1: dict[str, float] = {}
    pooled_tp = pooled_fp = pooled_fn = 0.0
    for name, c in counts.items():
        if c.tp + c.fp + c.fn <= 0.0:
            continue  # nothing to detect and nothing hallucinated: not informative
        precision = _ratio(c.tp, c.tp + c.fp)
        recall = _ratio(c.tp, c.tp + c.fn)
        per_type_f1[name] = _f1(precision, recall)
        pooled_tp += c.tp
        pooled_fp += c.fp
        pooled_fn += c.fn

    if per_type_f1:
        macro_f1 = sum(per_type_f1.values()) / len(per_type_f1)
        precision = _ratio(pooled_tp, pooled_tp + pooled_fp)
        recall = _ratio(pooled_tp, pooled_tp + pooled_fn)
        micro_f1 = _f1(precision, recall)
    else:
        macro_f1 = micro_f1 = precision = recall = 1.0

    # Confident false positives join the denominator (a quality-weighted Jaccard):
    # normalising by ground truth alone let a detector fire confident junk on every
    # clip to fish for matches at no cost to this term.
    localization_denominator = total_gt_weight + pooled_fp
    if localization_denominator > 0.0:
        localization = quality_sum / localization_denominator
    else:
        # No ground truth and no confident predictions: nothing to localise.
        localization = 1.0
    clip_accuracy = _balanced_clip_accuracy(clips, predictions, settings)

    mean_ap, per_type_ap, ap_by_tiou = _mean_average_precision(
        clips, predictions, settings
    )

    score = (
        settings.weight_map * mean_ap
        + settings.weight_f1 * macro_f1
        + settings.weight_localization * localization
        + settings.weight_clip_accuracy * clip_accuracy
    )
    score = min(1.0, max(0.0, score))
    return ScoreResult(
        score=score,
        loss=1.0 - score,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        precision=precision,
        recall=recall,
        localization_score=localization,
        clip_accuracy=clip_accuracy,
        per_type_f1=per_type_f1,
        per_type_counts={name: c.as_dict() for name, c in counts.items()},
        mean_ap=mean_ap,
        per_type_ap=per_type_ap,
        ap_by_tiou=ap_by_tiou,
    )
