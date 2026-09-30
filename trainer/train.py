"""Train the baseline temporal detector on a ``generate_data.py`` or Hugging Face dataset split.

    python train.py --data-dir data/train --out-dir runs/v1 --epochs 40 --device cpu
    python train.py --data-dir hf_dataset/train --out-dir runs/big --hidden 128 --layers 12 \
        --device cuda --batch-size 64 --workers 16

``--data-dir`` may be a ``generate_data.py`` output folder (``labels.jsonl`` + ``videos/``), a
split folder of the Hugging Face dataset (``metadata.jsonl`` + ``<clip_id>.mp4``), or the
dataset root (with ``train/`` and, optionally, ``validation/``); the layout is detected.

Pipeline: decode every clip -> per-frame features (cached as .npz keyed by the file's hash)
-> per-frame multi-hot targets -> ``TemporalIssueNet`` trained with masked BCE on random
temporal crops (frames around unlabelled decoys are up-weighted: hard negatives) ->
per-type decision thresholds calibrated on the validation split by maximising the validator's
final score (mean AP dominated) -> ``weights.safetensors`` + ``vic_config.json`` in ``--out-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn


def _find_repo_root(start: Path) -> Path:
    """The nearest ancestor holding ``validator/modules/video_inconsistency``.

    Works both inside the FLock-validator checkout and in the trainer quickstart repo,
    which vendors the needed validator modules at its root.
    """
    for candidate in (start, *start.parents):
        if (
            candidate
            / "validator"
            / "modules"
            / "video_inconsistency"
            / "issue_types.py"
        ).is_file():
            return candidate
    return start.parents[3] if len(start.parents) > 3 else start


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_HERE)
for _path in (str(_HERE), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import vic_features  # noqa: E402
from vic_model import (  # noqa: E402
    CANDIDATE_FLOOR,
    DEFAULT_MIN_FRAMES,
    ISSUE_TYPE_NAMES,
    MODEL_VERSION,
    SPATIAL_TYPES,
    TemporalIssueNet,
    decode_intervals,
    dilations_for_layers,
    model_hyperparameters,
)

from validator.modules.video_inconsistency.issue_types import (  # noqa: E402
    ISSUE_TYPE_NAMES as VALIDATOR_ISSUE_TYPE_NAMES,
)
from validator.modules.video_inconsistency.video_io import decode_video  # noqa: E402

# The submission cannot import the validator, so vic_model hard-codes the type order. Make sure
# the copy never silently diverges: a mismatch would permute every label.
assert tuple(ISSUE_TYPE_NAMES) == tuple(VALIDATOR_ISSUE_TYPE_NAMES), (
    "issue type order drifted"
)

CACHE_DIRNAME = "feature_cache"
THRESHOLD_GRID = [round(float(v), 2) for v in np.arange(0.10, 0.951, 0.05)]
TIOU_THRESHOLD = 0.3
MIN_EVENT_SECONDS = 0.4


# ---------------------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------------------
def _normalise_record(record: dict[str, Any], root: Path) -> dict[str, Any]:
    """Give a label row of either layout the keys the rest of this file uses."""
    record = dict(record)
    record.setdefault("clip_file", record.get("file_name"))
    record.setdefault("decoys", [])
    record.setdefault("difficulty", "medium")
    record["_root"] = str(root)
    return record


def read_labels(data_dir: Path) -> list[dict[str, Any]]:
    """Read ``labels.jsonl`` (``generate_data.py`` layout) or ``metadata.jsonl`` (Hugging Face
    dataset split layout, ``file_name`` relative to the file). Detected automatically."""
    for name in ("labels.jsonl", "metadata.jsonl"):
        path = data_dir / name
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                return [
                    _normalise_record(json.loads(line), data_dir)
                    for line in handle
                    if line.strip()
                ]
    raise FileNotFoundError(f"no labels.jsonl or metadata.jsonl in {data_dir}")


def resolve_splits(data_dir: Path, val_dir: str | None) -> tuple[Path, Path | None]:
    """Training folder and optional explicit validation folder.

    Pointing ``--data-dir`` at the dataset root uses ``train/`` for training and ``validation/``
    (when present) for validation; ``--val-dir`` overrides the latter.
    """
    train_dir, explicit_val = data_dir, Path(val_dir) if val_dir else None
    has_labels = (data_dir / "labels.jsonl").is_file() or (
        data_dir / "metadata.jsonl"
    ).is_file()
    if not has_labels and (data_dir / "train" / "metadata.jsonl").is_file():
        train_dir = data_dir / "train"
        if (
            explicit_val is None
            and (data_dir / "validation" / "metadata.jsonl").is_file()
        ):
            explicit_val = data_dir / "validation"
    return train_dir, explicit_val


def _file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_features(cache_dir: Path, record: dict[str, Any]) -> np.ndarray:
    """Features of one clip, from the .npz cache when present (key = file hash + version)."""
    video_path = Path(record["_root"]) / record["clip_file"]
    cache_path = (
        cache_dir / f"{_file_hash(video_path)}_v{vic_features.FEATURE_VERSION}.npz"
    )
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                return data["features"]
        except Exception:  # noqa: BLE001 - corrupt cache entry: recompute
            pass
    features = vic_features.extract_features(decode_video(video_path))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.stem + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, features=features)
    tmp.replace(cache_path)
    return features


def _features_job(job: tuple[str, dict[str, Any]]) -> np.ndarray:
    return cached_features(Path(job[0]), job[1])


def load_all_features(
    cache_dir: Path, records: list[dict[str, Any]], workers: int
) -> list[np.ndarray]:
    """Decode + featurise every clip, in parallel (spawned processes) for ``workers`` > 1."""
    jobs = [(str(cache_dir), record) for record in records]
    if workers <= 1 or len(jobs) < 4:
        return [_features_job(job) for job in jobs]
    out: list[np.ndarray] = []
    started = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for count, features in enumerate(
            pool.imap(_features_job, jobs, chunksize=4), start=1
        ):
            out.append(features)
            if count % 500 == 0:
                print(
                    f"[train] features {count}/{len(jobs)} ({time.time() - started:.0f}s)",
                    flush=True,
                )
    return out


def build_targets(record: dict[str, Any], num_frames: int) -> np.ndarray:
    """``(T, 10)`` multi-hot targets. Spans cover [start_frame, end_frame); a dropped-frames
    point event at cut ``k`` marks the two frames around the cut, ``k-1`` and ``k``."""
    targets = np.zeros((num_frames, len(ISSUE_TYPE_NAMES)), dtype=np.float32)
    for issue in record["issues"]:
        column = ISSUE_TYPE_NAMES.index(issue["type"])
        start, end = int(issue["start_frame"]), int(issue["end_frame"])
        if issue["type"] == "dropped_frames":
            targets[max(0, start - 1) : min(num_frames, start + 1), column] = 1.0
        else:
            targets[max(0, start) : min(num_frames, max(end, start + 1)), column] = 1.0
    return targets


def build_frame_weights(
    record: dict[str, Any], num_frames: int, decoy_weight: float
) -> np.ndarray:
    """Per-frame loss weights: 1 everywhere, ``decoy_weight`` on and around unlabelled decoys.

    Decoys (legitimate cuts, drifts, zooms, objects entering...) look like edits but are not;
    every frame of one is a hard negative, and the boundaries deserve extra attention.
    """
    weights = np.ones(num_frames, dtype=np.float32)
    if decoy_weight == 1.0:
        return weights
    fps = float(record["fps"])
    for decoy in record.get("decoys", []) or []:
        start = int(np.floor(decoy["start_time"] * fps)) - 2
        end = int(np.ceil(decoy["end_time"] * fps)) + 2
        weights[max(0, start) : min(num_frames, max(end, start + 1))] = decoy_weight
    return weights


# ---------------------------------------------------------------------------------------
# threshold calibration with the validator's matching rule
# ---------------------------------------------------------------------------------------
def _pad(start: float, end: float, duration: float) -> tuple[float, float]:
    if end - start >= MIN_EVENT_SECONDS:
        return start, end
    centre = (start + end) / 2
    start, end = centre - MIN_EVENT_SECONDS / 2, centre + MIN_EVENT_SECONDS / 2
    if start < 0:
        end -= start
        start = 0.0
    if end > duration:
        start -= end - duration
        end = duration
    return max(start, 0.0), end


def _tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def local_type_f1(
    records: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    issue_type: str,
) -> tuple[float, dict[str, int]]:
    """Per-type F1 with greedy confidence-ordered tIoU matching (unweighted by difficulty)."""
    tp = fp = fn = 0
    for record, preds in zip(records, predictions):
        duration = record["num_frames"] / record["fps"]
        truths = [
            _pad(i["start_time"], i["end_time"], duration)
            for i in record["issues"]
            if i["type"] == issue_type
        ]
        taken = [False] * len(truths)
        mine = sorted(
            (p for p in preds if p["type"] == issue_type and p["confidence"] >= 0.5),
            key=lambda p: -p["confidence"],
        )
        for pred in mine:
            interval = _pad(pred["start_time"], pred["end_time"], duration)
            best, best_iou = -1, -1.0
            for index, truth in enumerate(truths):
                if not taken[index]:
                    value = _tiou(interval, truth)
                    if value > best_iou:
                        best, best_iou = index, value
            if best >= 0 and best_iou >= TIOU_THRESHOLD:
                taken[best] = True
                tp += 1
            else:
                fp += 1
        fn += taken.count(False)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator if denominator else 0.0), {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _validator_pieces() -> Any | None:
    try:
        from validator.modules.video_inconsistency.manifest import (
            ClipSpec,
            DecoyLabel,
            IssueLabel,
        )
        from validator.modules.video_inconsistency.predictions import PredictedIssue
        from validator.modules.video_inconsistency.scoring import score_predictions
    except ImportError:
        return None
    return ClipSpec, IssueLabel, DecoyLabel, PredictedIssue, score_predictions


def to_clip_spec(index: int, record: dict[str, Any]) -> Any:
    ClipSpec, IssueLabel, DecoyLabel, _, _ = _validator_pieces()  # type: ignore[misc]
    return ClipSpec(
        clip_id=f"c{index}",
        video_path=record["clip_file"],
        fps=record["fps"],
        num_frames=record["num_frames"],
        width=record["width"],
        height=record["height"],
        difficulty=record["difficulty"],
        issues=[IssueLabel(**i) for i in record["issues"]],
        decoys=[DecoyLabel(**d) for d in record.get("decoys", []) or []],
    )


def _to_predicted(
    PredictedIssue: Any, issue: dict[str, Any], bbox: list[float] | None = None
) -> Any:
    return PredictedIssue(
        type=issue["type"],
        start_time=issue["start_time"],
        end_time=issue["end_time"],
        confidence=issue["confidence"],
        bbox=bbox,
    )


def _localize_job(
    job: tuple[str, list[tuple[str, int, int]]],
) -> list[list[float] | None]:
    """Decode one clip and box each ``(type, start_frame, end_frame)`` candidate."""
    from vic_localize import localize

    path, keys = job
    frames = decode_video(path)
    return [localize(name, frames, start, end) for name, start, end in keys]


def localize_candidates(
    records: list[dict[str, Any]],
    decoded: dict[float, list[list[dict[str, Any]]]],
    workers: int,
) -> dict[tuple[int, str, int, int], list[float] | None]:
    """Boxes for every distinct spatial candidate across all calibration thresholds.

    The validator only counts a spatial detection when its box overlaps the truth (IoU >= 0.3),
    so calibrating a spatial type on time alone would greatly overestimate its worth.
    """
    wanted: dict[int, set[tuple[str, int, int]]] = {}
    for per_clip in decoded.values():
        for c, issues in enumerate(per_clip):
            for issue in issues:
                if issue["type"] in SPATIAL_TYPES:
                    wanted.setdefault(c, set()).add(
                        (issue["type"], issue["_start_frame"], issue["_end_frame"])
                    )
    clips = sorted(wanted)
    jobs = [
        (str(Path(records[c]["_root"]) / records[c]["clip_file"]), sorted(wanted[c]))
        for c in clips
    ]
    if workers > 1 and len(jobs) > 3:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_localize_job, jobs, chunksize=4)
    else:
        results = [_localize_job(job) for job in jobs]
    boxes: dict[tuple[int, str, int, int], list[float] | None] = {}
    for c, job, result in zip(clips, jobs, results):
        for (name, start, end), box in zip(job[1], result):
            boxes[(c, name, start, end)] = box
    return boxes


def calibrate_thresholds(
    records: list[dict[str, Any]],
    probs: list[np.ndarray],
    max_clips: int = 500,
    workers: int = 1,
    f1_weight: float = 0.1,
) -> tuple[dict[str, float], dict[str, float]]:
    """Choose per-type decision thresholds on the validation split.

    Step 1 picks, per type, the threshold maximising that type's F1 under the validator's
    matching rule (ties -> closest to 0.5). Step 2 (only when the validator scorer is
    importable) refines them by coordinate ascent on the validator's final ``score`` (mean AP
    dominated, plus F1-free localisation and clip accuracy), which also accounts for false
    positives on clean clips and decoys. A threshold of 1.0 makes a type "candidates only"
    (confidence below 0.5, ranked but never a confident false positive).

    ``f1_weight`` adds a small ``f1_weight * macro_f1`` regulariser to that objective. The
    validator's localisation term is normalised by the ground truth only, so it never penalises
    a confident false positive; without the regulariser the search happily fires low-threshold
    junk on every clip (clip accuracy 0.5). A little F1 keeps confidence >= 0.5 meaningful, at
    almost no cost in the final score. 0 gives the pure validator score.

    At most ``max_clips`` clips are used (a fixed subset) to keep the search fast.
    """
    if len(records) > max_clips:
        keep = np.random.default_rng(0).permutation(len(records))[:max_clips]
        records = [records[i] for i in keep]
        probs = [probs[i] for i in keep]
    decoded: dict[float, list[list[dict[str, Any]]]] = {}
    for threshold in THRESHOLD_GRID + [1.0]:
        decoded[threshold] = [
            decode_intervals(p, r["fps"], r["num_frames"] / r["fps"], threshold)
            for r, p in zip(records, probs)
        ]
    thresholds: dict[str, float] = {}
    f1s: dict[str, float] = {}
    for name in ISSUE_TYPE_NAMES:
        support = sum(1 for r in records for i in r["issues"] if i["type"] == name)
        best_threshold, best_f1 = 0.5, -1.0
        if support > 0:
            for threshold in THRESHOLD_GRID:
                f1, _ = local_type_f1(records, decoded[threshold], name)
                better = f1 > best_f1 + 1e-9
                tie = abs(f1 - best_f1) <= 1e-9 and abs(threshold - 0.5) < abs(
                    best_threshold - 0.5
                )
                if better or tie:
                    best_threshold, best_f1 = threshold, f1
        thresholds[name] = float(best_threshold)
        f1s[name] = float(max(best_f1, 0.0))

    pieces = _validator_pieces()
    if pieces is None:
        return thresholds, f1s
    _, _, _, PredictedIssue, score_predictions = pieces
    clips = [to_clip_spec(i, r) for i, r in enumerate(records)]
    boxes = localize_candidates(records, decoded, workers)
    # cache[type][threshold][clip] -> predictions of that type only
    cache: dict[str, dict[float, list[list[Any]]]] = {}
    for name in ISSUE_TYPE_NAMES:
        cache[name] = {
            threshold: [
                [
                    _to_predicted(
                        PredictedIssue,
                        i,
                        boxes.get((c, i["type"], i["_start_frame"], i["_end_frame"])),
                    )
                    for i in decoded[threshold][c]
                    if i["type"] == name
                ]
                for c in range(len(records))
            ]
            for threshold in THRESHOLD_GRID + [1.0]
        }

    def total_score(current: dict[str, float]) -> float:
        merged = [
            [p for name in ISSUE_TYPE_NAMES for p in cache[name][current[name]][c]]
            for c in range(len(records))
        ]
        result = score_predictions(clips, merged)
        return float(result.score) + f1_weight * float(result.macro_f1)

    def ascend(start: dict[str, float]) -> tuple[float, dict[str, float]]:
        current, value = dict(start), total_score(start)
        for sweep in range(2):
            for name in ISSUE_TYPE_NAMES:
                grid = THRESHOLD_GRID + [1.0]
                if sweep == 1:  # second sweep: only refine near the current value
                    here = grid.index(current[name])
                    grid = grid[max(0, here - 2) : here + 3]
                for threshold in grid:
                    trial = dict(current, **{name: threshold})
                    trial_value = total_score(trial)
                    if trial_value > value + 1e-4:  # ignore noise-level gains
                        value, current = trial_value, trial
        return value, current

    # A single start can get stuck: if two weak types both fire on every clean clip, moving
    # just one of them changes nothing. So also start with all weak types "candidates only".
    weak_off = {name: (1.0 if f1s[name] < 0.3 else t) for name, t in thresholds.items()}
    best_value, best = max(
        (ascend(thresholds), ascend(weak_off)), key=lambda pair: pair[0]
    )
    return best, f1s


# ---------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------
def make_batch(
    items: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    crop_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random temporal crops padded to a common length: features, targets, mask, frame weights."""
    crops = []
    for features, targets, frame_weights in items:
        length = features.shape[0]
        crop = int(
            rng.integers(min(crop_range[0], length), min(crop_range[1], length) + 1)
        )
        start = int(rng.integers(0, length - crop + 1))
        sl = slice(start, start + crop)
        crops.append((features[sl], targets[sl], frame_weights[sl]))
    longest = max(c[0].shape[0] for c in crops)
    x = np.zeros((len(crops), longest, crops[0][0].shape[1]), np.float32)
    y = np.zeros((len(crops), longest, crops[0][1].shape[1]), np.float32)
    w = np.ones((len(crops), longest), np.float32)
    mask = np.zeros((len(crops), longest), np.float32)
    for i, (f, t, fw) in enumerate(crops):
        n = len(f)
        x[i, :n], y[i, :n], w[i, :n], mask[i, :n] = f, t, fw, 1.0
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(mask),
        torch.from_numpy(w),
    )


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    frame_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    weights = mask if frame_weights is None else mask * frame_weights
    normaliser = (mask.unsqueeze(-1).sum() * logits.shape[-1]).clamp_min(1.0)
    return (loss * weights.unsqueeze(-1)).sum() / normaliser


def _padded_batches(
    arrays: list[np.ndarray], batch_size: int
) -> list[tuple[list[int], torch.Tensor, torch.Tensor]]:
    """Length-sorted padded batches ``(indices, features, mask)`` for fast batched inference."""
    order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[0])
    batches = []
    for begin in range(0, len(order), batch_size):
        idx = order[begin : begin + batch_size]
        longest = arrays[idx[-1]].shape[0]
        x = np.zeros((len(idx), longest, arrays[idx[0]].shape[1]), np.float32)
        mask = np.zeros((len(idx), longest), np.float32)
        for row, i in enumerate(idx):
            n = arrays[i].shape[0]
            x[row, :n], mask[row, :n] = arrays[i], 1.0
        batches.append((idx, torch.from_numpy(x), torch.from_numpy(mask)))
    return batches


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    features: list[np.ndarray],
    device: torch.device,
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Per-clip ``(T, 10)`` logits via padded batches (the mask makes it equal to per-clip runs)."""
    model.eval()
    out: list[np.ndarray | None] = [None] * len(features)
    for idx, x, mask in _padded_batches(features, batch_size):
        logits = model(x.to(device), mask.to(device)).float().cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = logits[row, : features[i].shape[0]]
    return out  # type: ignore[return-value]


def predict_probs(
    model: nn.Module, features: list[np.ndarray], device: torch.device
) -> list[np.ndarray]:
    return [
        1.0 / (1.0 + np.exp(-logits))
        for logits in predict_logits(model, features, device)
    ]


def compute_pos_weight(targets: list[np.ndarray]) -> torch.Tensor:
    stacked = np.concatenate(targets, axis=0)
    positives = stacked.sum(axis=0)
    negatives = stacked.shape[0] - positives
    # sqrt keeps rare types from dominating; the threshold calibration fixes the operating point.
    weight = np.clip(np.sqrt(negatives / np.maximum(positives, 1.0)), 1.0, 10.0)
    return torch.tensor(weight, dtype=torch.float32)


def fit(
    train_x: list[np.ndarray],
    train_y: list[np.ndarray],
    train_w: list[np.ndarray],
    val_x: list[np.ndarray],
    val_y: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = TemporalIssueNet(
        in_features=train_x[0].shape[1],
        hidden=args.hidden,
        dilations=dilations_for_layers(args.layers),
        dropout=args.dropout,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] model: hidden {args.hidden}, {args.layers} blocks, {params / 1e3:.0f}k parameters",
        flush=True,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, int(np.ceil(len(train_x) / args.batch_size)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        total_steps=steps_per_epoch * args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    pos_weight = compute_pos_weight(train_y).to(device)
    non_blocking = device.type == "cuda"

    def val_loss() -> float:
        if not val_x:
            return float("nan")
        logits_list = predict_logits(model, val_x, device)
        total, count = 0.0, 0
        for logits, target in zip(logits_list, val_y):
            loss = masked_bce(
                torch.from_numpy(logits)[None],
                torch.from_numpy(target)[None],
                torch.ones(1, logits.shape[0]),
                pos_weight.cpu(),
            )
            total += float(loss) * logits.shape[0]
            count += logits.shape[0]
        return total / max(count, 1)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        running = 0.0
        for begin in range(0, len(order), args.batch_size):
            batch = [
                (train_x[i], train_y[i], train_w[i])
                for i in order[begin : begin + args.batch_size]
            ]
            x, y, mask, frame_w = make_batch(batch, (64, 10_000), rng)
            if (
                args.noise > 0
            ):  # feature-space jitter: cheap regularisation for small datasets
                x = x + args.noise * torch.randn_like(x) * mask.unsqueeze(-1)
            x, y, mask, frame_w = (
                t.to(device, non_blocking=non_blocking) for t in (x, y, mask, frame_w)
            )
            loss = masked_bce(model(x, mask), y, mask, pos_weight, frame_w)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch)
        current = val_loss()
        history.append(
            {"epoch": epoch, "train_loss": running / len(train_x), "val_loss": current}
        )
        monitored = current if val_x else running / len(train_x)
        if monitored < best_loss - 1e-4:
            best_loss, best_epoch, stale = monitored, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[train] epoch {epoch:3d}  train {running / len(train_x):.4f}  val {current:.4f}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"[train] early stop at epoch {epoch} (best epoch {best_epoch})")
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
    }


# ---------------------------------------------------------------------------------------
# reporting with the validator's scorer
# ---------------------------------------------------------------------------------------
def score_with_validator(
    records: list[dict[str, Any]],
    predictions: list[list[dict[str, Any]]],
    with_boxes: bool = True,
) -> Any | None:
    """Full validator score (mean AP, F1, localisation, clip accuracy). ``None`` if unavailable."""
    pieces = _validator_pieces()
    if pieces is None:
        return None
    _, _, _, PredictedIssue, score_predictions = pieces
    from vic_localize import localize

    clips, preds = [], []
    for index, (record, issues) in enumerate(zip(records, predictions)):
        clips.append(to_clip_spec(index, record))
        frames = None
        converted = []
        for issue in issues:
            bbox = None
            if with_boxes and issue["type"] in SPATIAL_TYPES:
                if frames is None:
                    frames = decode_video(Path(record["_root"]) / record["clip_file"])
                bbox = localize(
                    issue["type"], frames, issue["_start_frame"], issue["_end_frame"]
                )
            converted.append(_to_predicted(PredictedIssue, issue, bbox))
        preds.append(converted)
    return score_predictions(clips, preds)


def result_summary(result: Any) -> dict[str, Any]:
    """The headline numbers of a ``ScoreResult`` as plain JSON (mean AP when the scorer has it)."""
    summary: dict[str, Any] = {
        "score": result.score,
        "macro_f1": result.macro_f1,
        "localization": result.localization_score,
        "clip_accuracy": result.clip_accuracy,
        "per_type_f1": result.per_type_f1,
    }
    if getattr(result, "mean_ap", None) is not None:
        summary["mean_ap"] = result.mean_ap
        summary["per_type_ap"] = dict(getattr(result, "per_type_ap", {}) or {})
    return summary


def print_report(title: str, result: Any) -> None:
    print(f"\n== {title} ==")
    mean_ap = getattr(result, "mean_ap", None)
    ap_text = f"mean AP {mean_ap:.3f} | " if mean_ap is not None else ""
    print(
        f"score {result.score:.3f} | {ap_text}macro F1 {result.macro_f1:.3f} | localisation "
        f"{result.localization_score:.3f} | clip accuracy {result.clip_accuracy:.3f}"
    )
    per_type_ap = getattr(result, "per_type_ap", {}) or {}
    for name in ISSUE_TYPE_NAMES:
        if name in result.per_type_f1:
            counts = result.per_type_counts[name]
            ap_part = f"AP {per_type_ap[name]:.3f}  " if name in per_type_ap else ""
            print(
                f"  {name:18s} {ap_part}F1 {result.per_type_f1[name]:.3f}  "
                f"(tp {counts['tp']:.0f} fp {counts['fp']:.0f} fn {counts['fn']:.0f})"
            )


def _describe(records: list[dict[str, Any]], title: str) -> None:
    difficulty: dict[str, int] = {}
    for r in records:
        difficulty[r["difficulty"]] = difficulty.get(r["difficulty"], 0) + 1
    decoys = sum(len(r.get("decoys", []) or []) for r in records)
    edited = sum(1 for r in records if r["issues"])
    print(
        f"[train] {title}: {len(records)} clips ({edited} edited, {decoys} decoys), difficulty {difficulty}"
    )


# ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="generate_data.py output, a Hugging Face split folder, or the dataset root",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="explicit validation folder (same layouts); default: hold out --val-fraction",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="feature cache folder (default: <data-dir>/feature_cache)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hidden", type=int, default=64, help="channels of the temporal net"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="residual blocks (dilations 1,2,4,..,32 repeating); 6 = 127-frame receptive field",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.15,
        help="std of Gaussian jitter on normalised features",
    )
    parser.add_argument(
        "--decoy-weight",
        type=float,
        default=2.0,
        help="loss weight of frames on/around unlabelled decoys (hard negatives); 1 disables",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument(
        "--patience", type=int, default=12, help="early-stopping patience (epochs)"
    )
    parser.add_argument(
        "--calib-f1-weight",
        type=float,
        default=0.1,
        help="weight of macro F1 added to the validator score when tuning thresholds "
        "(keeps confident detections meaningful); 0 = pure validator score",
    )
    parser.add_argument(
        "--calib-max-clips",
        type=int,
        default=500,
        help="validation clips used for threshold calibration (all are used for the report)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, mp.cpu_count())),
        help="processes for decoding + feature extraction",
    )
    args = parser.parse_args(argv)

    train_dir, val_dir = resolve_splits(Path(args.data_dir), args.val_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else train_dir / CACHE_DIRNAME
    torch.set_num_threads(max(1, min(8, mp.cpu_count())))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[train] CUDA requested but not available; falling back to CPU", flush=True
        )
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )

    started = time.time()
    train_records = read_labels(train_dir)
    val_records_all = read_labels(val_dir) if val_dir is not None else []
    _describe(train_records, "train data")
    if val_records_all:
        _describe(val_records_all, "validation data")
    print(f"[train] extracting features ({args.workers} workers)...", flush=True)
    all_records = train_records + val_records_all
    features = load_all_features(cache_dir, all_records, args.workers)
    targets = [build_targets(r, f.shape[0]) for r, f in zip(all_records, features)]
    weights = [
        build_frame_weights(r, f.shape[0], args.decoy_weight)
        for r, f in zip(all_records, features)
    ]
    print(f"[train] features ready in {time.time() - started:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    if val_records_all:
        train_idx = np.arange(len(train_records))
        val_idx = np.arange(len(train_records), len(all_records))
    else:
        order = rng.permutation(len(train_records))
        num_val = int(round(len(train_records) * args.val_fraction))
        if len(train_records) - num_val < 2:
            num_val = 0
        val_idx, train_idx = order[:num_val], order[num_val:]
    stats = vic_features.compute_feature_stats([features[i] for i in train_idx])
    normed = [vic_features.normalize_features(f, stats) for f in features]
    train_x, train_y, train_w = (
        [a[i] for i in train_idx] for a in (normed, targets, weights)
    )
    val_x, val_y = [normed[i] for i in val_idx], [targets[i] for i in val_idx]
    val_records = [all_records[i] for i in val_idx]

    model, info = fit(train_x, train_y, train_w, val_x, val_y, args, device)

    thresholds = {name: 0.5 for name in ISSUE_TYPE_NAMES}
    calibrated_f1: dict[str, float] = {}
    report: dict[str, Any] = {}
    if val_x:
        val_probs = predict_probs(model, val_x, device)
        print(
            f"[train] calibrating thresholds on up to {args.calib_max_clips} validation clips...",
            flush=True,
        )
        thresholds, calibrated_f1 = calibrate_thresholds(
            val_records,
            val_probs,
            args.calib_max_clips,
            args.workers,
            args.calib_f1_weight,
        )
        decoded = [
            decode_intervals(p, r["fps"], r["num_frames"] / r["fps"], thresholds)
            for r, p in zip(val_records, val_probs)
        ]
        result = score_with_validator(val_records, decoded)
        if result is not None:
            print_report("validation (validator scorer, calibrated thresholds)", result)
            report = result_summary(result)
        else:
            print(
                "\n== validation per-type F1 (local matcher; validator scoring not importable) =="
            )
            for name in ISSUE_TYPE_NAMES:
                print(
                    f"  {name:18s} F1 {calibrated_f1[name]:.3f}  threshold {thresholds[name]:.2f}"
                )
            report = {"per_type_f1": calibrated_f1}
        print("thresholds:", {k: v for k, v in thresholds.items()})

    # Save. safetensors needs contiguous CPU tensors.
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out_dir / "weights.safetensors"))
    fps_values = [r["fps"] for r in all_records]
    config = {
        "version": MODEL_VERSION,
        "feature_version": vic_features.FEATURE_VERSION,
        "feature_names": list(vic_features.FEATURE_NAMES),
        "issue_types": list(ISSUE_TYPE_NAMES),
        "feature_stats": stats,
        "thresholds": thresholds,
        "min_frames": DEFAULT_MIN_FRAMES,
        "candidate_floor": CANDIDATE_FLOOR,
        "model": model_hyperparameters(model),
        "fps_assumption": float(np.median(fps_values)),
        "train": {
            "num_train_clips": int(len(train_idx)),
            "num_val_clips": int(len(val_idx)),
            "epochs_run": len(info["history"]),
            "best_epoch": info["best_epoch"],
            "seed": args.seed,
            "decoy_weight": args.decoy_weight,
        },
        "validation": report,
    }
    with open(out_dir / "vic_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(
        f"\n[train] saved weights.safetensors + vic_config.json to {out_dir} ({time.time() - started:.0f}s total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
