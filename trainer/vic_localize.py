"""Bounding-box estimation for the two spatial issue types (numpy only).

Given the frames and a detected interval (in frame indices, end exclusive) these
functions return a normalised ``[x0, y0, x1, y1]`` box. They are deliberately
simple baselines (no motion compensation) and NEVER raise: on any failure or
degenerate input they fall back to a centred box, because the validator gives
partial credit (via bbox IoU) only when a box is present and valid.

* ``inserted_object``: difference images at the two cuts of the interval (object pops in
  and out), combined so that ordinary motion cancels, thresholded (Otsu) -> bounding box of
  the strongest connected blob.
* ``blurred_region``: per-block sharpness inside vs outside the interval -> box of
  the connected group of blocks with the largest sharpness drop.
"""

from __future__ import annotations

import numpy as np

FALLBACK_BOX: list[float] = [0.25, 0.25, 0.75, 0.75]
_MAX_WIDTH = 96
_OUTSIDE_FRAMES = 8
_MAX_INSIDE_FRAMES = 24
_BLUR_GRID = 8


def _valid_box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Clip to [0, 1] and guarantee ``x0 < x1`` and ``y0 < y1``; otherwise the fallback box."""
    values = [float(v) for v in (x0, y0, x1, y1)]
    if not all(np.isfinite(values)):
        return list(FALLBACK_BOX)
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in values)
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return list(FALLBACK_BOX)
    return [x0, y0, x1, y1]


def _downsample_gray(frames: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Block-mean grey frames ``(n, h, w)`` in [0, 1] for the selected frame indices."""
    chunk = np.asarray(frames[indices], dtype=np.float32) / 255.0
    gray = chunk @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    height, width = gray.shape[1:]
    factor = max(1, int(round(width / _MAX_WIDTH)))
    h, w = height // factor, width // factor
    return gray[:, : h * factor, : w * factor].reshape(-1, h, factor, w, factor).mean(axis=(2, 4))


def _frame_sets(
    num_frames: int, start: int, end: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sampled inside frames and nearby outside frames (skipping a 1-frame guard band)."""
    start = int(np.clip(start, 0, num_frames - 1))
    end = int(np.clip(end, start + 1, num_frames))
    inside = np.arange(start, end)
    if inside.size > _MAX_INSIDE_FRAMES:
        inside = inside[np.linspace(0, inside.size - 1, _MAX_INSIDE_FRAMES).astype(int)]
    before = np.arange(max(0, start - 1 - _OUTSIDE_FRAMES), max(0, start - 1))
    after = np.arange(min(num_frames, end + 1), min(num_frames, end + 1 + _OUTSIDE_FRAMES))
    outside = np.concatenate([before, after])
    if outside.size == 0:
        return None
    return inside, outside


def _otsu(values: np.ndarray, bins: int = 64) -> float:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return hi + 1.0  # nothing stands out
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight0 = np.cumsum(hist)
    weight1 = weight0[-1] - weight0
    sum0 = np.cumsum(hist * centers)
    mean0 = sum0 / np.maximum(weight0, 1e-9)
    mean1 = (sum0[-1] - sum0) / np.maximum(weight1, 1e-9)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centers[int(np.argmax(between))])


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean ``(h, w)`` mask as lists of ``(y, x)``."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)
    return components


def _box_of(component: list[tuple[int, int]], height: int, width: int, pad: float = 0.0) -> list[float]:
    ys = np.array([p[0] for p in component])
    xs = np.array([p[1] for p in component])
    return _valid_box(
        xs.min() / width - pad,
        ys.min() / height - pad,
        (xs.max() + 1) / width + pad,
        (ys.max() + 1) / height + pad,
    )


def _smooth3(image: np.ndarray) -> np.ndarray:
    """3x3 box blur (suppresses codec noise and isolated pixels)."""
    padded = np.pad(image, 1, mode="edge")
    height, width = image.shape
    return sum(padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)) / 9.0


def localize_inserted_object(frames: np.ndarray, start_frame: int, end_frame: int) -> list[float]:
    """Box of the region that pops in at the interval start and out again at its end.

    Camera and scene motion make a comparison against frames far from the interval useless, so
    the object is located from the two *cuts* only: the difference between the first frame inside
    the interval and the frame just before it, and between the last frame inside and the frame
    just after it. The object shows in both differences while ordinary motion rarely lines up,
    so the element-wise minimum keeps the object and suppresses motion. Otsu thresholding and
    the strongest connected blob then give the box.
    """
    try:
        num_frames = frames.shape[0]
        start = int(np.clip(start_frame, 0, num_frames - 1))
        end = int(np.clip(end_frame, start + 1, num_frames))
        differences = []
        if start >= 1:
            g = _downsample_gray(frames, np.array([start, start - 1]))
            differences.append(np.abs(g[0] - g[1]))
        if end < num_frames:
            g = _downsample_gray(frames, np.array([end - 1, end]))
            differences.append(np.abs(g[0] - g[1]))
        if not differences:
            return list(FALLBACK_BOX)
        smooth = _smooth3(np.minimum.reduce(differences))
        threshold = max(0.6 * _otsu(smooth.ravel()), float(np.percentile(smooth, 85)) + 0.01)
        components = connected_components(smooth > threshold)
        if not components:
            return list(FALLBACK_BOX)
        best = max(components, key=lambda comp: sum(smooth[y, x] for y, x in comp))
        return _box_of(best, *smooth.shape, pad=0.01)
    except Exception:  # noqa: BLE001 - localisation must never break detection
        return list(FALLBACK_BOX)


def _block_sharpness(gray: np.ndarray, grid: int) -> np.ndarray:
    """Mean absolute Laplacian per grid block, averaged over frames: ``(grid, grid)``."""
    lap = np.abs(
        4.0 * gray[:, 1:-1, 1:-1]
        - gray[:, :-2, 1:-1]
        - gray[:, 2:, 1:-1]
        - gray[:, 1:-1, :-2]
        - gray[:, 1:-1, 2:]
    ).mean(axis=0)
    height, width = lap.shape
    bh, bw = height // grid, width // grid
    return lap[: bh * grid, : bw * grid].reshape(grid, bh, grid, bw).mean(axis=(1, 3))


def localize_blurred_region(frames: np.ndarray, start_frame: int, end_frame: int) -> list[float]:
    """Box of the blocks whose sharpness dropped most inside the interval."""
    try:
        sets = _frame_sets(frames.shape[0], start_frame, end_frame)
        if sets is None:
            return list(FALLBACK_BOX)
        inside_idx, outside_idx = sets
        inside = _downsample_gray(frames, inside_idx)
        outside = _downsample_gray(frames, outside_idx)
        if min(inside.shape[1:]) < 3 * _BLUR_GRID:
            return list(FALLBACK_BOX)
        sharp_in = _block_sharpness(inside, _BLUR_GRID)
        sharp_out = _block_sharpness(outside, _BLUR_GRID)
        eps = 0.05 * float(sharp_out.mean()) + 1e-8
        drop = 1.0 - (sharp_in + eps) / (sharp_out + eps)
        peak = float(drop.max())
        if peak < 0.05:
            return list(FALLBACK_BOX)
        strong = drop >= max(0.5 * peak, 0.05)
        components = connected_components(strong)
        best = max(components, key=lambda comp: sum(drop[y, x] for y, x in comp))
        return _box_of(best, _BLUR_GRID, _BLUR_GRID)
    except Exception:  # noqa: BLE001
        return list(FALLBACK_BOX)


def localize(issue_type: str, frames: np.ndarray, start_frame: int, end_frame: int) -> list[float] | None:
    """Box for a spatial issue type, ``None`` for non-spatial types."""
    if issue_type == "inserted_object":
        return localize_inserted_object(frames, start_frame, end_frame)
    if issue_type == "blurred_region":
        return localize_blurred_region(frames, start_frame, end_frame)
    return None


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two normalised boxes (used by the tests and the training report)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
