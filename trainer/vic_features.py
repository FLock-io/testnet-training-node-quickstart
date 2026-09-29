"""Per-frame feature extraction for the video-inconsistency baseline.

Self-contained (numpy only, torch NOT required). The same file is used by
``train.py`` and by the submission adapter (it is copied into the submission
repo), so training and inference can never drift apart.

``extract_features(frames)`` turns a ``(T, H, W, 3)`` uint8 clip into a
``(T, F)`` float32 matrix. Row ``t`` describes frame ``t`` and mostly looks at
the step ``t-1 -> t`` (and a little at ``t -> t+1``). Every feature is built so
that one kind of edit leaves a distinctive fingerprint:

    frozen frames   -> frame-to-frame difference collapses to ~0
    dropped frames  -> one difference spikes above the local median
    splice          -> difference and colour-histogram distance spike at both ends
    colour jump     -> per-channel means step and deviate from their rolling median
    exposure flicker-> luminance of 1-3 frames deviates from its neighbours
    mirror          -> at the boundary, ``diff(f_t, flip(f_{t-1}))`` beats ``diff(f_t, f_{t-1})``
    zoom            -> at the boundary, ``diff(f_t, zoom(f_{t-1}))`` beats the plain diff
    reversal        -> global translation flips sign against the rolling median velocity
    blurred region  -> one grid cell loses sharpness relative to its own history
    inserted object -> one grid cell deviates from its rolling-median appearance

All window sizes are in frames and assume clips of roughly 10-30 fps (the
validator's synthetic clips are 15 fps). All divisions are guarded, so any
``T >= 1`` (``T == 1`` yields all-zero temporal features), odd or even sizes and
constant frames give finite output.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


FEATURE_VERSION = 3

# Working resolutions (see ``_downsample``): a small RGB proxy for most features and a
# somewhat larger grey proxy for sharpness, because blur is invisible after a 4x block mean.
_TARGET_WIDTH = 80
_SHARP_WIDTH = 160
_OBJECT_WIDTH = 40
_GRID = 4
_SHARP_GRID = 8
_CHUNK = 16
_ZOOM_SCALES = (1.1, 1.25, 1.5)
_HIST_BINS = 16

FEATURE_NAMES: tuple[str, ...] = (
    # --- frame difference ------------------------------------------------------------
    "diff_prev",          # mean |f_t - f_{t-1}| (0..1); ~0 when frozen
    "diff_next",          # mean |f_{t+1} - f_t|
    "logratio_prev",      # log((diff_prev+e)/(rolling median diff+e)): spike -> cut/drop/splice
    "logratio_next",
    "dup_prev",           # soft near-duplicate indicator for f_t ~= f_{t-1} (freeze)
    "dup_next",
    # --- colour histogram ------------------------------------------------------------
    "hist_prev",          # L1/2 distance of the RGB histogram to the previous frame
    "hist_med",           # ... to the rolling-median histogram (window 15)
    # --- channel means ---------------------------------------------------------------
    "dev9_r", "dev9_g", "dev9_b",       # channel mean minus its rolling median (window 9)
    "dev31_r", "dev31_g", "dev31_b",    # ... window 31 (catches longer colour segments)
    "step_r", "step_g", "step_b",       # channel mean step from the previous frame
    # --- luminance -------------------------------------------------------------------
    "lum_neigh",          # luminance minus mean of its two neighbours (flicker)
    "lum_med9",           # luminance minus rolling median (window 9)
    # --- mirror ----------------------------------------------------------------------
    "flip_ratio",         # log((diff(f_t, fliplr f_{t-1})+e)/(diff_prev+e))
    "flip_diff",          # diff(f_t, fliplr f_{t-1})
    # --- zoom ------------------------------------------------------------------------
    "zin_ratio",          # log((min_s diff(f_t, zoomin_s f_{t-1})+e)/(diff_prev+e))
    "zout_ratio",         # log((min_s diff(zoomin_s f_t, f_{t-1})+e)/(diff_prev+e))
    "zin_diff",           # min_s diff(f_t, zoomin_s f_{t-1})
    "zout_diff",          # min_s diff(zoomin_s f_t, f_{t-1})
    # --- global translation (phase correlation) --------------------------------------
    "shift_x",            # horizontal shift t-1 -> t, percent of frame width
    "shift_y",            # vertical shift, percent of frame height
    "shift_peak",         # phase-correlation peak height (1 = pure translation)
    "shift_speed",        # sqrt(shift_x^2 + shift_y^2)
    "shift_agree15",      # cosine between velocity and rolling median velocity (window 15)
    "shift_agree31",      # ... window 31: -1 means motion runs against the trend (reversal)
    # --- sharpness / blur ------------------------------------------------------------
    "sharp_log",          # log mean |Laplacian| (global sharpness)
    "sharp_dev",          # sharp_log minus its rolling median (window 31)
    "blk_sharp_glob",     # min over 8x8 cells of log(sharpness / cell's whole-clip median)
    "blk_sharp_roll",     # min over 8x8 cells of log(sharpness / cell's rolling median, window 31)
    "blk_sharp_drop",     # min over cells of the frame-to-frame log sharpness change (blur onset)
    "blk_sharp_rise",     # max over cells of the frame-to-frame log sharpness change (blur offset)
    # --- appearance deviation (inserted object) --------------------------------------
    "obj_max",            # max over 4x4 cells of mean |frame - rolling median frame| (window 31)
    "obj_rel",            # obj_max minus the median cell deviation
    "obj_frac",           # fraction of pixels deviating strongly from the rolling median
    "obj_mean",           # mean deviation over the whole frame
    "objg_max",           # max over 4x4 cells of mean |frame - whole-clip median frame| (RGB)
    "objg_rel",           # objg_max minus the median cell deviation
    "cell_spike",         # max over 8x8 cells of log(cell frame-diff / its rolling median): pop in/out
    "cell_conc",          # log(max cell frame-diff / mean cell frame-diff): change is localised
)
NUM_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------
def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median along axis 0 with edge replication (window is made odd)."""
    if window % 2 == 0:
        window += 1
    half = window // 2
    pad = [(half, half)] + [(0, 0)] * (x.ndim - 1)
    padded = np.pad(x, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    # sliding_window_view puts the window axis last.
    return np.median(view, axis=-1).astype(x.dtype, copy=False)


def _log_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.log((numerator + eps) / (denominator + eps)), -5.0, 5.0)


def _shift_prev(x: np.ndarray) -> np.ndarray:
    """``out[t] = x[t-1]`` with the first row replicated."""
    return np.concatenate([x[:1], x[:-1]], axis=0)


def _pair_to_prev_next(pair_values: np.ndarray, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``d[i] = dist(f_i, f_{i+1})`` (length T-1) into per-frame prev/next arrays."""
    if num_frames < 2:
        zeros = np.zeros(num_frames, dtype=np.float32)
        return zeros, zeros.copy()
    prev = np.concatenate([pair_values[:1], pair_values])
    nxt = np.concatenate([pair_values, pair_values[-1:]])
    return prev.astype(np.float32), nxt.astype(np.float32)


def _block_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Block mean over the two spatial axes of ``(n, H, W, ...)`` (crops the remainder)."""
    n, height, width = x.shape[:3]
    rest = x.shape[3:]
    h, w = height // factor, width // factor
    cropped = x[:, : h * factor, : w * factor]
    return cropped.reshape((n, h, factor, w, factor) + rest).mean(axis=(2, 4))


def _factor(width: int, height: int, target_width: int) -> int:
    return max(1, min(int(round(width / target_width)), width, height))


def _grid_reduce(values: np.ndarray, grid: int, fn: Any) -> np.ndarray:
    """Reduce ``(n, H, W)`` over a ``grid x grid`` partition -> ``(n, gy, gx)``."""
    n, height, width = values.shape
    gy, gx = min(grid, height), min(grid, width)
    bh, bw = height // gy, width // gx
    blocks = values[:, : gy * bh, : gx * bw].reshape(n, gy, bh, gx, bw)
    return fn(blocks, axis=(2, 4))


# ---------------------------------------------------------------------------------------
# downsampling
# ---------------------------------------------------------------------------------------
def _downsample(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb proxy ``(T,h,w,3)`` in [0,1], sharpness ``(T, cells + 1)``).

    The sharpness matrix holds the per-cell Laplacian variances followed by the global one.

    Works chunk by chunk so a large memmap is never fully converted to float.
    """
    num_frames, height, width, _ = frames.shape
    factor = _factor(width, height, _TARGET_WIDTH)
    sharp_factor = _factor(width, height, _SHARP_WIDTH)
    luma_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    rgb_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    global_parts: list[np.ndarray] = []
    for start in range(0, num_frames, _CHUNK):
        chunk = np.asarray(frames[start : start + _CHUNK], dtype=np.float32) / 255.0
        rgb_parts.append(_block_mean(chunk, factor).astype(np.float32))
        luma = chunk @ luma_weights  # (n, H, W)
        gray = _block_mean(luma, sharp_factor)
        cells, global_var = _laplacian_stats(gray)
        cell_parts.append(cells)
        global_parts.append(global_var)
    rgb = np.concatenate(rgb_parts, axis=0)
    cells = np.concatenate(cell_parts, axis=0)
    global_var = np.concatenate(global_parts, axis=0)
    return rgb, np.concatenate([cells.reshape(num_frames, -1), global_var[:, None]], axis=1)


def _laplacian_stats(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell and global mean |4-neighbour Laplacian| of ``(n, h, w)`` grey.

    Mean absolute response is markedly more stable than variance for telling a blurred region
    (no texture, no sensor noise) from a naturally smooth one.
    """
    n, height, width = gray.shape
    if height < 3 or width < 3:
        return np.zeros((n, 1, 1), np.float32), np.zeros(n, np.float32)
    lap = (
        4.0 * gray[:, 1:-1, 1:-1]
        - gray[:, :-2, 1:-1]
        - gray[:, 2:, 1:-1]
        - gray[:, 1:-1, :-2]
        - gray[:, 1:-1, 2:]
    )
    magnitude = np.abs(lap)
    cells = _grid_reduce(magnitude, _SHARP_GRID, np.mean)
    return cells.astype(np.float32), magnitude.reshape(n, -1).mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------------------
# feature groups
# ---------------------------------------------------------------------------------------
def _zoom_in(gray: np.ndarray, scale: float) -> np.ndarray:
    """Centre crop by ``1/scale`` and bilinearly rescale back: ``(T,h,w) -> (T,h,w)``."""
    _, height, width = gray.shape

    def coords(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centre = (size - 1) / 2.0
        pos = np.clip(centre + (np.arange(size) - centre) / scale, 0, size - 1)
        low = np.floor(pos).astype(int)
        high = np.minimum(low + 1, size - 1)
        return low, high, (pos - low).astype(np.float32)

    y0, y1, wy = coords(height)
    x0, x1, wx = coords(width)
    top = gray[:, y0][:, :, x0] * (1 - wx) + gray[:, y0][:, :, x1] * wx
    bottom = gray[:, y1][:, :, x0] * (1 - wx) + gray[:, y1][:, :, x1] * wx
    return top * (1 - wy)[None, :, None] + bottom * wy[None, :, None]


def _phase_correlation(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel global translation between consecutive frames: ``(dx, dy, peak)`` per pair."""
    num_frames, height, width = gray.shape
    if num_frames < 2:
        empty = np.zeros(0, np.float32)
        return empty, empty.copy(), empty.copy()
    # np.hanning(n + 2)[1:-1] is strictly positive for any n >= 1 (np.hanning(1..2) is degenerate).
    window = np.outer(np.hanning(height + 2)[1:-1], np.hanning(width + 2)[1:-1]).astype(np.float32)
    spectrum = np.fft.rfft2((gray - gray.mean(axis=(1, 2), keepdims=True)) * window)
    cross = spectrum[1:] * np.conj(spectrum[:-1])
    cross /= np.abs(cross) + 1e-6
    corr = np.fft.irfft2(cross, s=(height, width))  # (T-1, h, w)
    pairs = corr.shape[0]
    flat = corr.reshape(pairs, -1)
    peak_index = flat.argmax(axis=1)
    peak = flat[np.arange(pairs), peak_index]
    py, px = np.divmod(peak_index, width)

    def refine(axis_size: int, idx: np.ndarray, sample: Any) -> np.ndarray:
        """Parabolic interpolation around the peak along one axis (circular)."""
        left = sample((idx - 1) % axis_size)
        centre = sample(idx)
        right = sample((idx + 1) % axis_size)
        denom = left - 2.0 * centre + right
        safe = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        offset = np.where(np.abs(denom) < 1e-9, 0.0, 0.5 * (left - right) / safe)
        return np.clip(offset, -0.5, 0.5)

    rows = np.arange(pairs)
    oy = refine(height, py, lambda i: corr[rows, i, px])
    ox = refine(width, px, lambda i: corr[rows, py, i])
    dy = np.where(py > height // 2, py - height, py) + oy
    dx = np.where(px > width // 2, px - width, px) + ox
    return dx.astype(np.float32), dy.astype(np.float32), peak.astype(np.float32)


def _histograms(rgb: np.ndarray) -> np.ndarray:
    """Per-frame concatenated per-channel histograms, each channel normalised to sum 1."""
    num_frames, height, width, _ = rgb.shape
    bins = np.clip((rgb * _HIST_BINS).astype(np.int64), 0, _HIST_BINS - 1)
    offsets = np.arange(3)[None, None, None, :] * _HIST_BINS + np.arange(num_frames)[:, None, None, None] * (
        3 * _HIST_BINS
    )
    counts = np.bincount((bins + offsets).ravel(), minlength=num_frames * 3 * _HIST_BINS)
    return (counts.reshape(num_frames, 3 * _HIST_BINS) / float(height * width)).astype(np.float32)


def _velocity_agreement(vx: np.ndarray, vy: np.ndarray, window: int) -> np.ndarray:
    mx, my = _rolling_median(vx, window), _rolling_median(vy, window)
    dot = vx * mx + vy * my
    norm = np.sqrt(vx**2 + vy**2) * np.sqrt(mx**2 + my**2)
    # eps (in squared percent-per-frame) keeps near-static clips at 0 instead of noisy +-1.
    return np.clip(dot / (norm + 0.02), -1.0, 1.0)


def _sharpness_features(sharp: np.ndarray) -> tuple[np.ndarray, ...]:
    global_var = sharp[:, -1]
    cells = sharp[:, :-1]
    log_global = np.log(global_var + 1e-5)
    sharp_dev = log_global - _rolling_median(log_global, 31)

    cell_median_whole = np.median(cells, axis=0, keepdims=True)
    cell_median_roll = _rolling_median(cells, 31)
    # eps relative to the typical cell sharpness so textureless cells cannot produce huge ratios.
    eps = 0.02 * float(cell_median_whole.mean()) + 1e-8
    glob = np.log((cells + eps) / (cell_median_whole + eps)).min(axis=1)
    roll = np.log((cells + eps) / (cell_median_roll + eps)).min(axis=1)
    log_cells = np.log(cells + eps)
    step = log_cells - _shift_prev(log_cells)
    return (
        log_global,
        sharp_dev,
        np.clip(glob, -6, 6),
        np.clip(roll, -6, 6),
        np.clip(step.min(axis=1), -6, 6),
        np.clip(step.max(axis=1), -6, 6),
    )


def _object_features(gray_small: np.ndarray) -> tuple[np.ndarray, ...]:
    reference = _rolling_median(gray_small, 31)
    dev = np.abs(gray_small - reference)  # (T, h, w)
    cells = _grid_reduce(dev, _GRID, np.mean).reshape(dev.shape[0], -1)
    cell_max = cells.max(axis=1)
    cell_rel = cell_max - np.median(cells, axis=1)
    frac = (dev > 0.1).reshape(dev.shape[0], -1).mean(axis=1)
    return cell_max, cell_rel, frac, dev.reshape(dev.shape[0], -1).mean(axis=1)


def _global_object_features(rgb_small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deviation from the whole-clip median frame (an object present for < half the clip stands out)."""
    dev = np.abs(rgb_small - np.median(rgb_small, axis=0, keepdims=True)).mean(axis=-1)
    cells = _grid_reduce(dev, _GRID, np.mean).reshape(dev.shape[0], -1)
    top = cells.max(axis=1)
    return top, top - np.median(cells, axis=1)


def _cell_spike_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Localised frame-difference spikes: a pop-in/out changes one cell, a camera move changes all."""
    num_frames = gray.shape[0]
    if num_frames < 2:
        return np.zeros(num_frames, np.float32), np.zeros(num_frames, np.float32)
    cell = _grid_reduce(np.abs(gray[1:] - gray[:-1]), _SHARP_GRID, np.mean).reshape(num_frames - 1, -1)
    spike = _log_ratio(cell, _rolling_median(cell, 31), 1e-3).max(axis=1)
    conc = np.log((cell.max(axis=1) + 1e-3) / (cell.mean(axis=1) + 1e-3))
    return _pair_to_prev_next(spike, num_frames)[0], _pair_to_prev_next(conc, num_frames)[0]


# ---------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------
def extract_features(frames: np.ndarray) -> np.ndarray:
    """Compute the ``(T, NUM_FEATURES)`` float32 feature matrix of a ``(T,H,W,3)`` uint8 clip."""
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.shape[0] < 1:
        raise ValueError("frames must be a non-empty (T, H, W, 3) array")
    num_frames = frames.shape[0]
    rgb, sharp = _downsample(frames)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # (T, h, w)
    cols: dict[str, np.ndarray] = {}

    # frame difference -------------------------------------------------------------
    if num_frames >= 2:
        pair = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    else:
        pair = np.zeros(0, np.float32)
    d_prev, d_next = _pair_to_prev_next(pair, num_frames)
    med = _rolling_median(d_prev, 9) if num_frames else d_prev
    dup_scale = 0.15 * float(np.median(pair)) + 3e-4 if pair.size else 3e-4
    cols["diff_prev"], cols["diff_next"] = d_prev, d_next
    cols["logratio_prev"] = _log_ratio(d_prev, med, 1e-3)
    cols["logratio_next"] = _log_ratio(d_next, med, 1e-3)
    cols["dup_prev"] = 1.0 / (1.0 + (d_prev / dup_scale) ** 4)
    cols["dup_next"] = 1.0 / (1.0 + (d_next / dup_scale) ** 4)

    # colour histogram -------------------------------------------------------------
    hists = _histograms(rgb)
    hist_pair = 0.5 * np.abs(hists[1:] - hists[:-1]).sum(axis=1) / 3.0 if num_frames >= 2 else np.zeros(0)
    cols["hist_prev"], _ = _pair_to_prev_next(hist_pair, num_frames)
    cols["hist_med"] = (0.5 * np.abs(hists - _rolling_median(hists, 15)).sum(axis=1) / 3.0).astype(np.float32)

    # channel means and luminance --------------------------------------------------
    means = rgb.mean(axis=(1, 2))  # (T, 3)
    dev9 = means - _rolling_median(means, 9)
    dev31 = means - _rolling_median(means, 31)
    step = means - _shift_prev(means)
    for index, name in enumerate("rgb"):
        cols[f"dev9_{name}"] = dev9[:, index]
        cols[f"dev31_{name}"] = dev31[:, index]
        cols[f"step_{name}"] = step[:, index]
    lum = gray.mean(axis=(1, 2))
    lum_next = np.concatenate([lum[1:], lum[-1:]])
    cols["lum_neigh"] = lum - 0.5 * (_shift_prev(lum) + lum_next)
    cols["lum_med9"] = lum - _rolling_median(lum, 9)

    # mirror -----------------------------------------------------------------------
    if num_frames >= 2:
        flip_pair = np.abs(gray[1:] - gray[:-1, :, ::-1]).mean(axis=(1, 2))
    else:
        flip_pair = np.zeros(0, np.float32)
    flip_prev, _ = _pair_to_prev_next(flip_pair, num_frames)
    # For pair index t (frames t, t+1) the flipped comparison uses the *previous* frame flipped,
    # which is exactly what ``flip_pair`` computes; ``_pair_to_prev_next`` aligns it with diff_prev.
    cols["flip_diff"] = flip_prev
    cols["flip_ratio"] = _log_ratio(flip_prev, d_prev, 1e-3)

    # zoom -------------------------------------------------------------------------
    zin = np.full(num_frames, 1.0, np.float32)
    zout = np.full(num_frames, 1.0, np.float32)
    if num_frames >= 2:
        best_in = np.full(num_frames - 1, np.inf, np.float32)
        best_out = np.full(num_frames - 1, np.inf, np.float32)
        for scale in _ZOOM_SCALES:
            zoomed = _zoom_in(gray, scale)
            best_in = np.minimum(best_in, np.abs(gray[1:] - zoomed[:-1]).mean(axis=(1, 2)))
            best_out = np.minimum(best_out, np.abs(zoomed[1:] - gray[:-1]).mean(axis=(1, 2)))
        zin, _ = _pair_to_prev_next(best_in, num_frames)
        zout, _ = _pair_to_prev_next(best_out, num_frames)
    cols["zin_diff"], cols["zout_diff"] = zin, zout
    cols["zin_ratio"] = _log_ratio(zin, d_prev, 1e-3)
    cols["zout_ratio"] = _log_ratio(zout, d_prev, 1e-3)

    # global translation -----------------------------------------------------------
    dx, dy, peak = _phase_correlation(gray)
    h, w = gray.shape[1], gray.shape[2]
    vx, _ = _pair_to_prev_next(dx * (100.0 / w), num_frames)
    vy, _ = _pair_to_prev_next(dy * (100.0 / h), num_frames)
    peak_prev, _ = _pair_to_prev_next(peak, num_frames)
    cols["shift_x"], cols["shift_y"], cols["shift_peak"] = vx, vy, peak_prev
    cols["shift_speed"] = np.sqrt(vx**2 + vy**2)
    cols["shift_agree15"] = _velocity_agreement(vx, vy, 15)
    cols["shift_agree31"] = _velocity_agreement(vx, vy, 31)

    # sharpness --------------------------------------------------------------------
    (
        cols["sharp_log"],
        cols["sharp_dev"],
        cols["blk_sharp_glob"],
        cols["blk_sharp_roll"],
        cols["blk_sharp_drop"],
        cols["blk_sharp_rise"],
    ) = _sharpness_features(sharp)

    # appearance deviation ---------------------------------------------------------
    small_factor = max(1, int(round(gray.shape[2] / _OBJECT_WIDTH)))
    small = _block_mean(gray[..., None], small_factor)[..., 0]
    cols["obj_max"], cols["obj_rel"], cols["obj_frac"], cols["obj_mean"] = _object_features(small)
    rgb_small = _block_mean(rgb, small_factor)
    cols["objg_max"], cols["objg_rel"] = _global_object_features(rgb_small)
    cols["cell_spike"], cols["cell_conc"] = _cell_spike_features(gray)

    matrix = np.stack([np.asarray(cols[name], dtype=np.float32) for name in FEATURE_NAMES], axis=1)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def compute_feature_stats(features: Sequence[np.ndarray]) -> dict[str, list[float]]:
    """Per-feature mean / std over a list of ``(T_i, F)`` matrices (for ``normalize_features``)."""
    if not features:
        raise ValueError("need at least one feature matrix")
    stacked = np.concatenate([np.asarray(f, dtype=np.float64) for f in features], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return {"mean": mean.tolist(), "std": np.maximum(std, 1e-6).tolist()}


def normalize_features(features: np.ndarray, stats: dict[str, Sequence[float]] | None) -> np.ndarray:
    """Standardise with ``stats`` (from ``compute_feature_stats``) and clip to +-10 sigma."""
    if stats is None:
        return np.clip(features, -10.0, 10.0).astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(stats["std"], dtype=np.float32), 1e-6)
    return np.clip((features - mean) / std, -10.0, 10.0).astype(np.float32)
