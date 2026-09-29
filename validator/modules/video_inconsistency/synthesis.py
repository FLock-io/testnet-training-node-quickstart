"""Synthetic clips with ground-truth-labelled inconsistencies (v2).

A clip is built in five steps:

1. render a continuous *source* shot (a procedural scene, or real footage) with
   optional **decoys**: legitimate, unlabelled events a detector must not flag
   (a scene cut, exposure / white-balance drift, a smooth zoom, an object
   entering / leaving / stopping, the camera coming to rest);
2. plan non-overlapping edit windows on the source timeline and apply one of
   the ten editors in each window (each editor is followed by a detectability
   guard that is evaluated *through the clip's post-edit degradation*);
3. assemble the output frames from untouched chunks and edited chunks, computing
   every label in the *output* timeline (``dropped_frames`` shortens the clip,
   so the source is rendered longer by exactly the number of dropped frames);
4. remap the decoys onto the output timeline;
5. degrade the *whole* clip (per-frame sensor noise, optional blur / sharpen,
   optional down-up rescale) so degradation never marks where the edits are.

Everything is deterministic from ``(seed, config)``: all randomness flows from
``np.random.default_rng`` generators seeded through :func:`_derive_seed`, with
no global RNG and no time dependence.

Every editor is followed by a detectability guard. An edit that would be
invisible (e.g. freezing a static stretch, mirroring a symmetric one, or a
1 % colour shift on a dim scene) is resampled (other parameters, then other
window positions, then a whole new plan); a label is never emitted for an edit
that did not visibly change the clip.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, get_args

import numpy as np
from loguru import logger
from PIL import Image, ImageFilter

from validator.modules.video_inconsistency.issue_types import (
    ISSUE_TYPE_NAMES,
    POINT_EVENT_ISSUE_TYPES,
)
from validator.modules.video_inconsistency.manifest import (
    DIFFICULTIES,
    DecoyLabel,
    DecoyType,
    IssueLabel,
)
from validator.modules.video_inconsistency.video_io import decode_video, probe_stream


DECOY_TYPES: tuple[str, ...] = tuple(get_args(DecoyType))

# Timing rules for multi-issue clips (seconds).
MIN_ISSUE_GAP_SECONDS = 1.0
MIN_END_MARGIN_SECONDS = 0.5
# A scene cut must stay this far from every edit window.
CUT_CLEARANCE_SECONDS = 0.5

# Retry budgets for the detectability guard.
_EDIT_RETRIES = 3  # stochastic re-draws of an editor at one window position
_SLIDE_TRIES = 8  # alternative window positions per edit
_PLAN_ATTEMPTS = 6  # per rendered source
_SOURCE_ROUNDS = 4  # fresh scenes tried when no plan fits (e.g. a static scene cannot show a freeze)

# Mean absolute pixel difference (0-255) between two frames that we consider
# "real motion" (sensor noise alone gives ~1-4, so this needs real displacement).
_MOTION_MIN = 5.0
# Extra guard margin per unit of sensor-noise sigma: statistics are computed on the
# noise-free (blurred / rescaled) signal, so demand a little more than the threshold.
_NOISE_MARGIN = 0.3

# Optional telemetry sink used by measurement scripts: set to a ``Counter`` to count
# ``expert_edits`` / ``expert_resampled`` / ``plan_fallback`` events. Never touched in
# production (``None``), and never affects the generated clip.
_TELEMETRY: Optional[Counter] = None


def _bump(key: str, amount: int = 1) -> None:
    if _TELEMETRY is not None:
        _TELEMETRY[key] += amount


@dataclass(frozen=True)
class SynthesisConfig:
    width: int = 320
    height: int = 240
    fps: float = 15.0
    min_duration: float = 6.0
    max_duration: float = 10.0
    clean_fraction: float = 0.2  # clips with zero issues
    difficulty_weights: tuple[tuple[str, float], ...] = (
        ("easy", 0.10),
        ("medium", 0.25),
        ("hard", 0.35),
        ("expert", 0.30),
    )
    issue_types: tuple[str, ...] = ISSUE_TYPE_NAMES  # allowed types to inject
    footage_paths: tuple[str, ...] = ()  # optional real source videos
    footage_fraction: float = 0.0  # share of clips drawn from footage_paths
    # Expected decoys per clip is ~ decoy_rate * duration / 8 s (Poisson), for clean
    # and edited clips alike.
    decoy_rate: float = 1.2
    # Post-edit degradation of the whole clip (sensor noise, blur/sharpen, rescale)
    # and a per-clip encode quality. False disables all of it (crf fixed at 18).
    degradation: bool = True
    crf_range: tuple[int, int] = (18, 28)

    def __post_init__(self) -> None:
        if self.width < 32 or self.height < 32 or self.width % 2 or self.height % 2:
            raise ValueError("width/height must be even and >= 32")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not (2.0 <= self.min_duration <= self.max_duration):
            raise ValueError("need 2.0 <= min_duration <= max_duration")
        if not (0.0 <= self.clean_fraction <= 1.0):
            raise ValueError("clean_fraction must be in [0, 1]")
        if not (0.0 <= self.footage_fraction <= 1.0):
            raise ValueError("footage_fraction must be in [0, 1]")
        if not self.issue_types:
            raise ValueError("issue_types must not be empty")
        unknown = [name for name in self.issue_types if name not in ISSUE_TYPE_NAMES]
        if unknown:
            raise ValueError(f"unknown issue types: {unknown}")
        weights = dict(self.difficulty_weights)
        if not weights or any(k not in DIFFICULTIES or v < 0 for k, v in weights.items()):
            raise ValueError(f"difficulty_weights must map {DIFFICULTIES} to weights >= 0")
        if sum(weights.values()) <= 0:
            raise ValueError("difficulty_weights must not all be zero")
        if self.decoy_rate < 0:
            raise ValueError("decoy_rate must be >= 0")
        low, high = self.crf_range
        if not (0 <= low <= high <= 51):
            raise ValueError("crf_range must satisfy 0 <= low <= high <= 51")


@dataclass(frozen=True)
class SyntheticClip:
    frames: np.ndarray  # (T, H, W, 3) uint8 -- the EDITED, degraded clip
    fps: float
    issues: list[IssueLabel]  # ground truth in OUTPUT timeline, sorted by start_frame
    difficulty: str  # "easy" | "medium" | "hard" | "expert"
    source: str  # "procedural" | "footage"
    decoys: list[DecoyLabel] = field(default_factory=list)  # legitimate, unlabelled events
    crf: int = 18  # per-clip encode quality (pass to ``encode_video``)


def _derive_seed(*parts: object) -> int:
    """Stable child seed from a tuple of parts (no dependence on hash randomisation)."""
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(_derive_seed(*parts))


def _smoothstep(u: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(u, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


# ---------------------------------------------------------------------------
# Procedural scene rendering
# ---------------------------------------------------------------------------


def _hsv(h: float, s: float, v: float) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0))
    return np.array([r, g, b], dtype=np.float32) * 255.0


def _resize_float(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bicubic-resize an (h, w, c) float array via PIL's float mode."""
    channels = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(array[..., c], dtype=np.float32)).resize(
                (width, height), Image.BICUBIC
            ),
            dtype=np.float32,
        )
        for c in range(array.shape[2])
    ]
    return np.stack(channels, axis=2)


# Shape extents (left, top, right, bottom) relative to the shape centre, as a
# multiple of (hw, hh). The triangle points up and has circumradius 1.2 * hw.
_TRI_R = 1.2


def _shape_extent(kind: str, hw: float, hh: float) -> tuple[float, float, float, float]:
    if kind == "triangle":
        radius = _TRI_R * hw
        return 0.866 * radius, radius, 0.866 * radius, 0.5 * radius
    return hw, hh, hw, hh


def _shape_sdf(kind: str, u: np.ndarray, v: np.ndarray, hw: float, hh: float) -> np.ndarray:
    """Signed distance (px, negative inside) of a shape in its local frame."""
    if kind == "circle":
        return np.sqrt(u * u + v * v) - hw
    if kind == "ellipse":
        return (np.sqrt((u / hw) ** 2 + (v / hh) ** 2) - 1.0) * min(hw, hh)
    if kind == "rect":
        qx = np.abs(u) - hw
        qy = np.abs(v) - hh
        outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        return outside + np.minimum(np.maximum(qx, qy), 0.0)
    if kind == "triangle":
        rho = _TRI_R * hw * 0.5
        bottom = v - rho
        right = 0.866 * u - 0.5 * v - rho
        left = -0.866 * u - 0.5 * v - rho
        return np.maximum(np.maximum(bottom, right), left)
    if kind == "ring":
        return np.abs(np.sqrt(u * u + v * v) - 0.72 * hw) - 0.28 * hw
    raise ValueError(f"unknown shape kind {kind!r}")


def _draw_shape(
    canvas: np.ndarray,
    offset: tuple[int, int],
    center: tuple[float, float],
    kind: str,
    hw: float,
    hh: float,
    angle: float,
    color_a: np.ndarray,
    color_b: np.ndarray,
    freq: float,
    phase: float,
    feather: float = 1.0,
    stripe_depth: float = 0.12,
) -> None:
    """Alpha-composite a shaded, textured shape into a float32 canvas in place.

    ``offset`` is the canvas' top-left in scene coordinates, so the same call
    draws into a full background or into a camera sub-window.
    """
    height, width = canvas.shape[:2]
    ox, oy = offset
    cx, cy = center
    reach = math.hypot(hw, hh) * (_TRI_R if kind == "triangle" else 1.0) + feather + 2.0
    x0 = max(int(math.floor(cx - reach)) - ox, 0)
    x1 = min(int(math.ceil(cx + reach)) - ox + 1, width)
    y0 = max(int(math.floor(cy - reach)) - oy, 0)
    y1 = min(int(math.ceil(cy + reach)) - oy + 1, height)
    if x1 <= x0 or y1 <= y0:
        return
    dx = (np.arange(x0, x1, dtype=np.float32) + (ox + 0.5 - cx))[None, :]
    dy = (np.arange(y0, y1, dtype=np.float32) + (oy + 0.5 - cy))[:, None]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u = cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy
    sdf = _shape_sdf(kind, u, v, hw, hh)
    alpha = np.clip(0.5 - sdf / max(feather, 1e-3), 0.0, 1.0)[..., None]
    grad = np.clip(0.5 + 0.5 * v / max(hh, 1.0), 0.0, 1.0)[..., None]
    stripe = (1.0 - stripe_depth) + stripe_depth * np.sin(freq * u + phase)[..., None]
    color = (color_a * (1.0 - grad) + color_b * grad) * stripe
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = region * (1.0 - alpha) + color * alpha


@dataclass
class _MovingObject:
    kind: str
    hw: float
    hh: float
    color_a: np.ndarray
    color_b: np.ndarray
    freq: float
    phase: float
    angle0: float
    omega: float  # rad / s
    pos0: np.ndarray
    vel: np.ndarray  # px / s
    radius: float
    behavior: str = "bounce"  # "bounce" off the canvas walls | "wrap" (leaves, re-enters)
    pauses: tuple[tuple[float, float], ...] = ()  # short natural pauses, seconds


@dataclass(frozen=True)
class _Style:
    """Per-scene look: some scenes are dim, low-contrast, muted or near mirror-symmetric."""

    contrast: float = 1.0  # < 1 flattens the background around its mean
    brightness: float = 1.0  # < 1 is a dim scene
    saturation: float = 1.0
    symmetry: float = 0.0  # 0 = asymmetric layout .. 1 = mirror-symmetric background


@dataclass(frozen=True)
class _SceneHints:
    """What the planned edits need from the scene, so it is built to be able to show them."""

    symmetric: bool = False  # near mirror-symmetric background (subtle-tier mirrors)
    camera_modes: Optional[tuple[str, ...]] = None  # restrict the camera mode
    fast_objects: bool = False  # no slow movers or pauses: visible motion everywhere


# Edits that are only visible when the picture actually moves across their window.
MOTION_DEPENDENT_ISSUES = frozenset({"frozen_frames", "dropped_frames", "reversed_segment"})


def _scene_hints(names: Sequence[str], difficulty: str, symmetric: bool = False) -> _SceneHints:
    """Scene constraints implied by the planned issue types (chosen before rendering)."""
    motion = any(name in MOTION_DEPENDENT_ISSUES for name in names)
    modes: Optional[tuple[str, ...]] = _MOVING_CAMERA_MODES if motion else None
    if difficulty == "expert" and "mirrored_segment" in names:
        # A strong pan moves the picture away from its mirror axis and makes a flip obvious:
        # keep the camera calm when possible (a freeze / drop / reversal in the same plan still
        # needs some motion, so only handheld remains for those).
        modes = ("handheld",) if motion else ("static", "drift", "handheld")
    return _SceneHints(symmetric=symmetric, camera_modes=modes, fast_objects=motion)


def _sample_style(rng: np.random.Generator, symmetric: bool) -> _Style:
    # The look never depends on the planned issues (it would be a learnable shortcut); a
    # dim or low-contrast scene that cannot show an edit is re-rendered by the fallback.
    look = str(rng.choice(["normal", "low_contrast", "dim", "muted"], p=[0.5, 0.18, 0.14, 0.18]))
    contrast, brightness, saturation = 1.0, 1.0, 1.0
    if look == "low_contrast":
        contrast, saturation = float(rng.uniform(0.4, 0.65)), 0.75
    elif look == "dim":
        brightness, contrast = float(rng.uniform(0.5, 0.75)), 0.85
    elif look == "muted":
        saturation = float(rng.uniform(0.35, 0.6))
    if symmetric:
        symmetry = float(rng.uniform(0.7, 0.95))
    elif rng.random() < 0.12:
        symmetry = float(rng.uniform(0.5, 0.9))
    else:
        symmetry = 0.0
    return _Style(contrast=contrast, brightness=brightness, saturation=saturation, symmetry=symmetry)


_CAMERA_MODES = ("pan", "handheld", "static", "drift")
_CAMERA_WEIGHTS = (0.40, 0.25, 0.20, 0.15)
_MOVING_CAMERA_MODES = ("pan", "handheld")


@dataclass
class _SceneSpec:
    """Every random choice of a procedural shot, drawn before any frame-count-dependent step."""

    width: int
    height: int
    fps: float
    cw: int  # canvas size (the camera sees a window of it)
    ch: int
    palette_seed: int  # colour palette / texture family
    layout_seed: int  # spatial layout of cells, landmarks and horizon
    noise_seed: int
    jitter_seed: int
    style: _Style
    camera_mode: str
    cam: dict[str, float]
    objects: list[_MovingObject]
    light: tuple[float, float, float, np.ndarray]


def _make_background(
    prng: np.random.Generator, lrng: np.random.Generator, cw: int, ch: int, style: _Style
) -> np.ndarray:
    """Float32 (ch, cw, 3) background. ``prng`` draws the palette / texture family,
    ``lrng`` the spatial layout, so two scenes sharing ``prng`` look alike but differ in layout."""
    hue0 = float(prng.random())
    sat_lo = float(prng.uniform(0.15, 0.45)) * style.saturation
    sat_span = 0.4 * style.saturation
    val_lo = float(prng.uniform(0.30, 0.55))
    palette = np.stack(
        [
            _hsv(
                hue0 + float(prng.uniform(-0.15, 0.15)),
                float(prng.uniform(sat_lo, sat_lo + sat_span)),
                float(prng.uniform(val_lo, val_lo + 0.35)),
            )
            for _ in range(4)
        ]
    )
    fine_amp = float(prng.uniform(9.0, 18.0))
    fine_div = int(prng.integers(3, 7))
    mid_amp = float(prng.uniform(10.0, 22.0))
    ground_scale = float(prng.uniform(0.45, 0.75))
    landmark_colors = [
        (float(prng.random()), float(prng.uniform(0.3, 0.8)) * style.saturation, float(prng.uniform(0.35, 0.9)), float(prng.uniform(0.5, 0.85)))
        for _ in range(4)
    ]

    grid_h, grid_w = int(lrng.integers(3, 6)), int(lrng.integers(4, 8))
    cells = palette[lrng.integers(0, len(palette), size=(grid_h, grid_w))]
    background = _resize_float(cells + lrng.normal(0.0, 10.0, cells.shape), cw, ch)

    # Textures at two scales give real gradients for motion to be visible.
    fine = lrng.uniform(-1.0, 1.0, (ch // fine_div + 2, cw // fine_div + 2, 3)).astype(np.float32)
    fine_full = _resize_float(fine, cw, ch)
    background += fine_amp * fine_full
    mid = lrng.uniform(-1.0, 1.0, (ch // 20 + 2, cw // 20 + 2, 3)).astype(np.float32)
    background += mid_amp * _resize_float(mid, cw, ch)

    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    theta = float(lrng.uniform(0.0, 2.0 * math.pi))
    ramp = ((xx - cw / 2) * math.cos(theta) + (yy - ch / 2) * math.sin(theta)) / max(cw, ch)
    background += ramp[..., None] * lrng.uniform(-45.0, 45.0, 3).astype(np.float32)

    # Tilted horizon with a striped ground band underneath: breaks mirror symmetry.
    horizon0 = float(lrng.uniform(0.45, 0.68)) * ch
    slope = float(lrng.choice([-1.0, 1.0])) * float(lrng.uniform(0.08, 0.32))
    horizon = horizon0 + slope * (xx - cw / 2)
    ground_mask = np.clip((yy - horizon) / 2.0 + 0.5, 0.0, 1.0)[..., None]
    ground_base = palette[int(lrng.integers(0, len(palette)))] * ground_scale
    stripes = 0.85 + 0.15 * np.sin((yy - horizon) * float(lrng.uniform(0.12, 0.35)) + float(lrng.uniform(0, 6)))
    ground = ground_base[None, None, :] * stripes[..., None] + 0.6 * fine_amp * fine_full
    background = background * (1.0 - ground_mask) + ground * ground_mask

    # Static landmarks: a tall block on one side, a triangle on the other, a ring and a disc.
    side = 1.0 if lrng.random() < 0.5 else -1.0
    tower_x = cw * (0.5 + side * float(lrng.uniform(0.18, 0.38)))
    tower_hw = float(lrng.uniform(0.045, 0.075)) * cw
    tower_hh = float(lrng.uniform(0.13, 0.22)) * ch
    tower_base = horizon0 + slope * (tower_x - cw / 2) + 4.0
    landmarks: list[tuple[str, float, float, float, float]] = [
        ("rect", tower_x, tower_base - tower_hh, tower_hw, tower_hh),
        (
            "triangle",
            cw * (0.5 - side * float(lrng.uniform(0.15, 0.35))),
            horizon0 - float(lrng.uniform(0.02, 0.06)) * ch,
            float(lrng.uniform(0.04, 0.07)) * cw,
            0.0,
        ),
        (
            "ring",
            cw * float(lrng.uniform(0.1, 0.9)),
            ch * float(lrng.uniform(0.08, 0.3)),
            float(lrng.uniform(0.03, 0.05)) * cw,
            0.0,
        ),
    ]
    if lrng.random() < 0.7:
        landmarks.append(
            (
                "circle",
                cw * float(lrng.uniform(0.1, 0.9)),
                ch * float(lrng.uniform(0.08, 0.35)),
                float(lrng.uniform(0.02, 0.04)) * cw,
                0.0,
            )
        )
    for (kind, lx, ly, hw, hh), (lh, ls, lv, shade) in zip(landmarks, landmark_colors):
        color_a = _hsv(lh, ls, lv)
        _draw_shape(
            background,
            (0, 0),
            (lx, ly),
            kind,
            hw,
            hh if hh > 0 else hw,
            0.0,
            color_a,
            color_a * shade,
            float(lrng.uniform(0.15, 0.5)),
            float(lrng.uniform(0.0, 6.0)),
        )

    if style.symmetry > 0.0:
        # Blend the right part towards the mirrored left part: a near-symmetric layout
        # whose mirrored segments are hard (but, with objects and camera offset, not
        # impossible) to see.
        mirrored = background[:, ::-1].copy()
        half = cw // 2
        s = style.symmetry
        background[:, half:] = (1.0 - s) * background[:, half:] + s * mirrored[:, half:]
    mean = background.mean(axis=(0, 1), keepdims=True)
    background = mean + (background - mean) * style.contrast
    background = background * style.brightness
    return background.astype(np.float32)


_OBJECT_KINDS = ("circle", "rect", "triangle", "ring", "ellipse")


def _new_object(
    rng: np.random.Generator, cw: int, ch: int, speed_range: tuple[float, float], behavior: str
) -> _MovingObject:
    kind = str(_OBJECT_KINDS[int(rng.integers(0, len(_OBJECT_KINDS)))])
    hw = float(rng.uniform(0.032, 0.075)) * min(cw, ch)
    hh = hw * (float(rng.uniform(0.5, 1.2)) if kind in ("rect", "ellipse") else 1.0)
    radius = math.hypot(hw, hh) * (_TRI_R if kind == "triangle" else 1.0)
    color_a = _hsv(float(rng.random()), float(rng.uniform(0.45, 1.0)), float(rng.uniform(0.55, 1.0)))
    color_b = _hsv(float(rng.random()), float(rng.uniform(0.35, 0.9)), float(rng.uniform(0.3, 0.8)))
    speed = float(rng.uniform(*speed_range))
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    return _MovingObject(
        kind=kind,
        hw=hw,
        hh=hh,
        color_a=color_a,
        color_b=color_b,
        freq=float(rng.uniform(0.2, 0.6)),
        phase=float(rng.uniform(0.0, 6.0)),
        angle0=float(rng.uniform(0.0, 2.0 * math.pi)),
        omega=float(rng.uniform(-2.5, 2.5)) if rng.random() < 0.55 else 0.0,
        pos0=np.array(
            [rng.uniform(radius + 2, cw - radius - 2), rng.uniform(radius + 2, ch - radius - 2)]
        ),
        vel=np.array([math.cos(heading), math.sin(heading)]) * speed,
        radius=radius,
        behavior=behavior,
    )


def _make_moving_objects(
    rng: np.random.Generator, cw: int, ch: int, camera_mode: str, fast: bool = False
) -> list[_MovingObject]:
    count = int(rng.integers(4, 10)) if camera_mode == "static" else int(rng.integers(3, 8))
    objects: list[_MovingObject] = []
    for _ in range(count):
        # A mix of slow and fast movers; ~40 % wander off the canvas and come back
        # from the other side (they cross the frame edges), the rest bounce.
        slow = rng.random() < 0.3 and not fast
        behavior = "wrap" if rng.random() < 0.4 else "bounce"
        obj = _new_object(rng, cw, ch, (6.0, 28.0) if slow else (30.0, 120.0), behavior)
        pauses: list[tuple[float, float]] = []
        if rng.random() < 0.35 and not fast:  # pause, then resume (short: longer stops are labelled decoys)
            for _ in range(int(rng.integers(1, 3))):
                t0 = float(rng.uniform(0.5, 30.0))
                pauses.append((t0, t0 + float(rng.uniform(0.2, 0.45))))
        obj.pauses = tuple(pauses)
        objects.append(obj)
    return objects


def _sample_scene(
    rng: np.random.Generator,
    width: int,
    height: int,
    fps: float,
    *,
    camera_modes: Sequence[str] | None = None,
    symmetric: bool = False,
    fast_objects: bool = False,
    like: _SceneSpec | None = None,
) -> _SceneSpec:
    """Draw every parameter of a shot. ``like`` keeps the palette / texture family,
    style and camera mode of another shot but redraws layout, objects and phases."""
    cw, ch = int(round(width * 1.5)), int(round(height * 1.5))
    palette_seed = int(rng.integers(0, 2**62))
    layout_seed = int(rng.integers(0, 2**62))
    noise_seed = int(rng.integers(0, 2**62))
    jitter_seed = int(rng.integers(0, 2**62))
    style = _sample_style(rng, symmetric)
    modes = list(camera_modes) if camera_modes is not None else list(_CAMERA_MODES)
    weights = np.array([_CAMERA_WEIGHTS[_CAMERA_MODES.index(m)] for m in modes], dtype=float)
    camera_mode = str(modes[int(rng.choice(len(modes), p=weights / weights.sum()))])
    if like is not None:
        palette_seed, style, camera_mode = like.palette_seed, like.style, like.camera_mode

    margin_x, margin_y = (cw - width) / 2.0, (ch - height) / 2.0
    amp_x = float(rng.uniform(0.45, 0.8)) * margin_x
    amp_y = float(rng.uniform(0.3, 0.65)) * margin_y
    cam: dict[str, float] = {
        "ax": amp_x,
        "ay": amp_y,
        "fx": 1.0 / float(rng.uniform(3.5, 7.0)),
        "fy": 1.0 / float(rng.uniform(4.0, 8.0)),
        "px": float(rng.uniform(0, 2 * math.pi)),
        "py": float(rng.uniform(0, 2 * math.pi)),
        "fx2": 0.0,
        "px2": float(rng.uniform(0, 2 * math.pi)),
        "zoom_amp": float(rng.uniform(0.0, 0.03)),
        "zoom_freq": 1.0 / float(rng.uniform(12.0, 24.0)),
        "zoom_phase": float(rng.uniform(0, 2 * math.pi)),
        "jitter": 0.0,
        "wobble_freq": float(rng.uniform(2.0, 3.5)),
        "wobble_phase": float(rng.uniform(0, 2 * math.pi)),
    }
    cam["fx2"] = cam["fx"] * float(rng.uniform(1.7, 2.6))
    if camera_mode == "handheld":
        scale = float(rng.uniform(0.3, 0.6))
        cam["ax"], cam["ay"] = amp_x * scale, amp_y * scale
        cam["jitter"] = float(rng.uniform(0.7, 1.3))  # AR(1) innovation std, px per frame
    elif camera_mode == "static":
        cam["ax"] = cam["ay"] = 0.0
        cam["zoom_amp"] = 0.0
    elif camera_mode == "drift":
        cam["ax"], cam["ay"] = amp_x * 0.9, amp_y * 0.9
        cam["fx"] = 1.0 / float(rng.uniform(28.0, 55.0))
        cam["fy"] = 1.0 / float(rng.uniform(30.0, 60.0))
        cam["fx2"] = cam["fx"] * 1.5
        cam["zoom_amp"] *= 0.5
    light = (
        float(rng.uniform(0.01, 0.03)),
        1.0 / float(rng.uniform(9.0, 18.0)),
        float(rng.uniform(0, 2 * math.pi)),
        1.0 + rng.uniform(-0.01, 0.01, 3).astype(np.float32),
    )
    objects = _make_moving_objects(rng, cw, ch, camera_mode, fast_objects)
    return _SceneSpec(
        width=width, height=height, fps=fps, cw=cw, ch=ch,
        palette_seed=palette_seed, layout_seed=layout_seed, noise_seed=noise_seed,
        jitter_seed=jitter_seed, style=style, camera_mode=camera_mode, cam=cam,
        objects=objects, light=light,
    )


def _object_tracks(
    objects: Sequence[_MovingObject],
    num_frames: int,
    cw: int,
    ch: int,
    fps: float,
    stops: dict[int, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, N, 2)`` positions and ``(T, N)`` motion clocks (frames of own motion).

    Bouncing objects reflect off the canvas walls; wrapping objects leave and come back
    from the opposite side. Natural pauses and forced ``stops`` freeze the motion (and the
    spin, through the clock) for a while.
    """
    count = len(objects)
    tracks = np.empty((num_frames, count, 2), dtype=np.float64)
    clocks = np.empty((num_frames, count), dtype=np.float64)
    pos = np.stack([obj.pos0 for obj in objects]).astype(np.float64)
    vel = np.stack([obj.vel for obj in objects]).astype(np.float64) / fps
    radius = np.array([obj.radius for obj in objects])[:, None]
    wrap = np.array([obj.behavior == "wrap" for obj in objects])[:, None]
    lo = radius + 1.0
    hi = np.array([cw, ch], dtype=np.float64)[None, :] - radius - 1.0
    wrap_lo = -radius - 2.0
    wrap_span = np.array([cw, ch], dtype=np.float64)[None, :] + 2.0 * radius + 4.0
    motion = np.ones((num_frames, count))
    for index, obj in enumerate(objects):
        for start, end in obj.pauses:
            motion[int(start * fps) : int(end * fps), index] = 0.0
    for index, (start, end) in (stops or {}).items():
        motion[start:end, index] = 0.0
        if start > 0:
            motion[start - 1, index] = min(motion[start - 1, index], 0.5)
        if end < num_frames:
            motion[end, index] = min(motion[end, index], 0.5)
    clock = np.zeros(count)
    for index in range(num_frames):
        tracks[index] = pos
        clocks[index] = clock
        step = motion[index][:, None]
        pos = pos + vel * step
        clock = clock + motion[index]
        under, over = (pos < lo) & ~wrap, (pos > hi) & ~wrap
        pos = np.where(under, 2 * lo - pos, pos)
        pos = np.where(over, 2 * hi - pos, pos)
        vel = np.where(under | over, -vel, vel)
        pos = np.where(wrap, wrap_lo + (pos - wrap_lo) % wrap_span, pos)
    return tracks, clocks


def _camera_arrays(
    spec: _SceneSpec, num_frames: int, events: Sequence[tuple[str, int, int, float]] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame camera centre (canvas px) and zoom.

    ``events`` are ``(kind, start, end, factor)`` camera decoys; the camera path is evaluated
    on a warped clock whose speed they shape:

    * ``camera_stops``: speed eases to zero, stays there, and eases back (handheld shake is
      switched off too);
    * ``camera_direction_change``: speed eases from +1 through 0 to -1 and stays reversed;
    * ``camera_speed_change``: speed eases from 1 to ``factor`` and stays there.
    """
    fps, cam = spec.fps, spec.cam
    index = np.arange(num_frames)
    speed = np.ones(num_frames)
    for kind, start, end, factor in events:
        start, end = max(start, 0), min(end, num_frames)
        if end <= start:
            continue
        if kind == "camera_stops":
            ease = max(1, min(4, (end - start) // 4))
            envelope = np.ones(num_frames)
            envelope[start:end] = 0.0
            for i in range(ease):
                value = 0.5 + 0.5 * math.cos(math.pi * (i + 1) / (ease + 1))
                envelope[start + i] = value
                envelope[end - 1 - i] = value
            speed = speed * envelope
        else:
            progress = _smoothstep((index - start + 1) / max(end - start, 1))
            speed = speed * (1.0 - 2.0 * progress if kind == "camera_direction_change" else 1.0 + (factor - 1.0) * progress)
    tau = np.concatenate([[0.0], np.cumsum(speed[:-1])]) / fps
    two_pi = 2.0 * math.pi
    center_x = spec.cw / 2 + 0.75 * cam["ax"] * np.sin(two_pi * cam["fx"] * tau + cam["px"])
    center_x = center_x + 0.25 * cam["ax"] * np.sin(two_pi * cam["fx2"] * tau + cam["px2"])
    center_y = spec.ch / 2 + cam["ay"] * np.sin(two_pi * cam["fy"] * tau + cam["py"])
    zoom = 1.0 + cam["zoom_amp"] * (0.5 + 0.5 * np.sin(two_pi * cam["zoom_freq"] * tau + cam["zoom_phase"]))
    if cam["jitter"] > 0.0:
        # Handheld shake: a low-frequency AR(1) random walk (a few px), a small wobble and
        # a +-1 % scale jitter. Drawn frame by frame so a longer render keeps the same prefix.
        jrng = np.random.default_rng(spec.jitter_seed)
        state = np.zeros(3)
        jitter = np.empty((num_frames, 3))
        for frame in range(num_frames):
            noise = jrng.standard_normal(3)
            state[:2] = 0.92 * state[:2] + cam["jitter"] * noise[:2]
            state[2] = 0.97 * state[2] + 0.0015 * noise[2]
            jitter[frame] = state
        wobble = 0.6 * np.sin(two_pi * cam["wobble_freq"] * np.arange(num_frames) / fps + cam["wobble_phase"])
        shake = np.minimum(np.abs(speed), 1.0)
        center_x = center_x + (jitter[:, 0] + wobble) * shake
        center_y = center_y + (jitter[:, 1] - 0.7 * wobble) * shake
        zoom = zoom * (1.0 + np.clip(jitter[:, 2], -0.01, 0.01) * shake)
    return center_x, center_y, zoom


def _view_rects(spec: _SceneSpec, center_x: np.ndarray, center_y: np.ndarray, zoom: np.ndarray) -> np.ndarray:
    crop_w, crop_h = spec.width / zoom, spec.height / zoom
    bx0 = np.minimum(np.maximum(center_x - crop_w / 2, 0.0), spec.cw - crop_w)
    by0 = np.minimum(np.maximum(center_y - crop_h / 2, 0.0), spec.ch - crop_h)
    return np.stack([bx0, by0, bx0 + crop_w, by0 + crop_h], axis=1)


def _visibility(track: np.ndarray, radius: float, rects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (any part visible, fully visible) flags of a circle-bounded object."""
    x, y = track[:, 0], track[:, 1]
    reach = 0.85 * radius
    seen = (
        (x + reach > rects[:, 0]) & (x - reach < rects[:, 2])
        & (y + reach > rects[:, 1]) & (y - reach < rects[:, 3])
    )
    full = (
        (x - radius >= rects[:, 0]) & (x + radius <= rects[:, 2])
        & (y - radius >= rects[:, 1]) & (y + radius <= rects[:, 3])
    )
    return seen, full


def _realize_transit(
    rng: np.random.Generator,
    spec: _SceneSpec,
    rects: np.ndarray,
    num_frames: int,
    direction: str,
    frame: int,
) -> tuple[_MovingObject, np.ndarray, int, int] | None:
    """An extra object that crosses a frame edge in a straight line (canvas space).

    Returns ``(object, track, start, end)``; the interval is measured from the frames
    (first pixel visible to fully inside for an entry, last fully inside to gone for an
    exit), so the label is exactly what a viewer sees. None if no clean crossing was found.
    """
    fps = spec.fps
    for _ in range(16):
        obj = _new_object(rng, spec.cw, spec.ch, (45.0, 110.0), "bounce")
        speed = float(np.hypot(*obj.vel)) / fps
        edge = int(rng.integers(0, 4))
        rect = rects[min(max(frame, 0), num_frames - 1)]
        frac = float(rng.uniform(0.25, 0.75))
        if edge < 2:  # left / right edge
            y = rect[1] + frac * (rect[3] - rect[1])
            x = rect[0] if edge == 0 else rect[2]
            inward = np.array([1.0 if edge == 0 else -1.0, 0.0])
        else:
            x = rect[0] + frac * (rect[2] - rect[0])
            y = rect[1] if edge == 2 else rect[3]
            inward = np.array([0.0, 1.0 if edge == 2 else -1.0])
        angle = float(rng.uniform(-0.35, 0.35))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        heading = np.array([inward[0] * cos_a - inward[1] * sin_a, inward[0] * sin_a + inward[1] * cos_a])
        step = heading * speed * (1.0 if direction == "enter" else -1.0)
        index = np.arange(num_frames, dtype=np.float64)[:, None]
        track = np.array([x, y])[None, :] + step[None, :] * (index - frame)
        seen, full = _visibility(track, obj.radius, rects)
        if direction == "enter":
            if not seen.any():
                continue
            first = int(np.argmax(seen))
            if first < 2 or abs(first - frame) > 8 or not full[first:].any():
                continue
            settled = first + int(np.argmax(full[first:]))
            if not 2 <= settled - first <= 25 or not full[settled : settled + 8].all():
                continue
            return obj, track, first, settled + 1
        if not seen.any():
            continue
        last = int(num_frames - 1 - np.argmax(seen[::-1]))
        left_full = np.nonzero(full[: last + 1])[0]
        if last > num_frames - 3 or len(left_full) == 0 or abs(last - frame) > 8:
            continue
        settled = int(left_full[-1])
        if settled < 6 or not 2 <= last - settled <= 25 or not full[settled - 6 : settled + 1].all():
            continue
        return obj, track, settled + 1, last + 1
    return None


def _style_colors(obj: _MovingObject, style: _Style, bg_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Objects follow the scene look: low-contrast scenes get less contrasty objects."""
    strength = 0.5 + 0.5 * style.contrast
    def adjust(color: np.ndarray) -> np.ndarray:
        return (bg_mean + (color - bg_mean) * strength) * style.brightness
    return adjust(obj.color_a), adjust(obj.color_b)


def _render_scene(
    spec: _SceneSpec, num_frames: int, events: Sequence[tuple[str, int, int]] = ()
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Render ``num_frames`` frames of a shot.

    ``events`` are requested shot-level decoys ``(type, start, end)`` in frames of this
    shot; the returned list holds the ones that could be realised, with measured
    intervals (an entering object's interval is what the frames actually show).
    The first ``n`` frames are identical regardless of ``num_frames`` when no events
    are requested, because every scene parameter was drawn in :func:`_sample_scene`.
    """
    width, height, fps, cw, ch = spec.width, spec.height, spec.fps, spec.cw, spec.ch
    background = _make_background(
        np.random.default_rng(spec.palette_seed), np.random.default_rng(spec.layout_seed), cw, ch, spec.style
    )
    bg_mean = background.mean(axis=(0, 1))
    background_u8 = np.clip(background, 0, 255).astype(np.uint8)

    realized: list[tuple[str, int, int]] = []
    erng = _rng(spec.jitter_seed, "events")
    camera_events: list[tuple[str, int, int, float]] = []
    if spec.camera_mode in _MOVING_CAMERA_MODES:
        base_x, base_y, _ = _camera_arrays(spec, num_frames)
        pan_speed = np.concatenate([[0.0], np.hypot(np.diff(base_x), np.diff(base_y))])
        for kind, start, end in events:
            if kind not in _CAMERA_DECOYS:
                continue
            end = min(end, num_frames)
            if kind != "camera_stops":
                # A reversal / speed change is only visible while the camera is really panning:
                # pick the fastest nearby start (the pan is sinusoidal, so it slows at its turns).
                span = end - start
                options = list(range(max(start - 20, 2), min(start + 21, num_frames - span - 2)))
                if not options:
                    continue
                start = max(options, key=lambda t: float(pan_speed[t : t + span].mean()))
                if float(pan_speed[start : start + span].mean()) < 0.006 * width:
                    continue
                end = start + span
            if any(s0 < end and start < e0 for _k, s0, e0, _f in camera_events):
                continue  # camera decoys never overlap each other
            factor = 1.0
            if kind == "camera_speed_change":
                factor = float(erng.uniform(1.5, 2.5))
                factor = factor if erng.random() < 0.5 else 1.0 / factor
            camera_events.append((kind, start, end, factor))
            realized.append((kind, start, end))
    center_x, center_y, zoom = _camera_arrays(spec, num_frames, camera_events)
    rects = _view_rects(spec, center_x, center_y, zoom)

    objects = list(spec.objects)
    tracks, clocks = _object_tracks(objects, num_frames, cw, ch, fps)
    stops: dict[int, tuple[int, int]] = {}
    for kind, start, end in events:
        if kind != "object_stops":
            continue
        order = [i for i in erng.permutation(len(objects)) if float(np.hypot(*objects[i].vel)) >= 35.0 and i not in stops]
        for index in order[:6]:
            centre = tracks[min(start, num_frames - 1), index]
            rect = rects[min(start, num_frames - 1)]
            r = objects[index].radius
            if not (rect[0] + r < centre[0] < rect[2] - r and rect[1] + r < centre[1] < rect[3] - r):
                continue
            stops[int(index)] = (start, end)
            realized.append(("object_stops", start, end))
            break
    if stops:
        tracks, clocks = _object_tracks(objects, num_frames, cw, ch, fps, stops)

    extra_objects: list[_MovingObject] = []
    extra_tracks: list[np.ndarray] = []
    for kind, start, _end in events:
        if kind not in ("object_enters", "object_exits"):
            continue
        found = _realize_transit(erng, spec, rects, num_frames, "enter" if kind == "object_enters" else "exit", start)
        if found is not None:
            obj, track, first, last = found
            extra_objects.append(obj)
            extra_tracks.append(track)
            realized.append((kind, first, last))
    draw_objects = objects + extra_objects
    draw_tracks = np.concatenate([tracks] + [t[:, None, :] for t in extra_tracks], axis=1) if extra_tracks else tracks
    draw_clocks = (
        np.concatenate([clocks] + [np.arange(num_frames, dtype=np.float64)[:, None]] * len(extra_tracks), axis=1)
        if extra_tracks
        else clocks
    )
    colors = [_style_colors(obj, spec.style, bg_mean) for obj in draw_objects]

    light_amp, light_freq, light_phase, light_tint = spec.light
    noise_rng = np.random.default_rng(spec.noise_seed)
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for index in range(num_frames):
        t = index / fps
        bx0, by0, bx1, by1 = rects[index]
        crop_w, crop_h = bx1 - bx0, by1 - by0
        rx0, ry0 = int(math.floor(bx0)), int(math.floor(by0))
        rx1, ry1 = min(int(math.ceil(bx1)) + 1, cw), min(int(math.ceil(by1)) + 1, ch)

        region = background_u8[ry0:ry1, rx0:rx1].astype(np.float32)
        for obj_index, obj in enumerate(draw_objects):
            px, py = draw_tracks[index, obj_index]
            if (
                px + obj.radius < rx0
                or px - obj.radius > rx1
                or py + obj.radius < ry0
                or py - obj.radius > ry1
            ):
                continue
            _draw_shape(
                region,
                (rx0, ry0),
                (float(px), float(py)),
                obj.kind,
                obj.hw,
                obj.hh,
                obj.angle0 + obj.omega * draw_clocks[index, obj_index] / fps,
                colors[obj_index][0],
                colors[obj_index][1],
                obj.freq,
                obj.phase,
            )
        window = Image.fromarray(np.clip(region, 0, 255).astype(np.uint8)).resize(
            (width, height),
            Image.BILINEAR,
            box=(bx0 - rx0, by0 - ry0, bx0 - rx0 + crop_w, by0 - ry0 + crop_h),
        )
        gain = (1.0 + light_amp * math.sin(2 * math.pi * light_freq * t + light_phase)) * light_tint
        noise = noise_rng.integers(-1, 2, size=(height, width, 3), dtype=np.int8)  # sigma ~ 0.8
        frame = np.asarray(window, dtype=np.float32) * gain + noise
        frames[index] = np.clip(frame, 0, 255).astype(np.uint8)
    return frames, realized


def render_procedural_scene(
    rng: np.random.Generator, num_frames: int, width: int, height: int, fps: float
) -> np.ndarray:
    """Render ``num_frames`` frames of a procedural shot (no decoys, no edits).

    Camera mode (smooth pan, handheld, static tripod or slow drift), object behaviour
    (fast and slow movers, pauses, objects crossing the frame edges) and look (dim,
    low-contrast, muted, near-symmetric) are sampled per call from ``rng``. The first
    ``n`` frames are identical regardless of ``num_frames``.
    """
    spec = _sample_scene(rng, width, height, fps)
    return _render_scene(spec, num_frames)[0]


# ---------------------------------------------------------------------------
# Real footage sources
# ---------------------------------------------------------------------------


def _load_footage_frames(path: str, config: SynthesisConfig, num_frames: int) -> np.ndarray | None:
    """Decode, fps-resample (by index), centre-crop and resize footage; None if unusable."""
    try:
        _width, _height, src_fps = probe_stream(path)
        needed = int(math.floor((num_frames - 1) * src_fps / config.fps)) + 1
        raw = decode_video(path, max_frames=needed)
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("unusable footage {}: {}", path, exc)
        return None
    if raw.shape[0] < needed:
        return None
    indices = np.minimum(
        np.floor(np.arange(num_frames) * src_fps / config.fps).astype(np.int64), raw.shape[0] - 1
    )
    src_h, src_w = raw.shape[1:3]
    target_aspect = config.width / config.height
    if src_w / src_h > target_aspect:
        crop_w, crop_h = int(round(src_h * target_aspect)), src_h
    else:
        crop_w, crop_h = src_w, int(round(src_w / target_aspect))
    x0, y0 = (src_w - crop_w) // 2, (src_h - crop_h) // 2
    out = np.empty((num_frames, config.height, config.width, 3), dtype=np.uint8)
    cache: dict[int, np.ndarray] = {}
    for out_index, src_index in enumerate(indices):
        key = int(src_index)
        if key not in cache:
            crop = Image.fromarray(raw[key, y0 : y0 + crop_h, x0 : x0 + crop_w])
            cache[key] = np.asarray(crop.resize((config.width, config.height), Image.BICUBIC))
        out[out_index] = cache[key]
    return out


# ---------------------------------------------------------------------------
# Post-edit degradation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Degradation:
    """Whole-clip degradation, applied after every edit so it never marks their location."""

    sigma: float  # per-frame independent Gaussian sensor noise, in 8-bit levels
    blur_sigma: float = 0.0  # mild Gaussian blur (0 = off)
    unsharp: int = 0  # unsharp-mask strength in percent (0 = off)
    scale: float = 1.0  # down-then-up rescale factor (1 = off)
    seed: int = 0

    @property
    def has_deterministic_part(self) -> bool:
        return self.blur_sigma > 0.0 or self.unsharp > 0 or self.scale < 0.98


def _sample_degradation(seed: int, config: SynthesisConfig) -> tuple[Optional[_Degradation], int]:
    """Per-clip degradation parameters and encode crf (None / 18 when disabled)."""
    if not config.degradation:
        return None, 18
    rng = _rng(seed, "degradation")
    sigma = float(rng.uniform(1.0, 4.0))
    roll = float(rng.random())
    blur_sigma, unsharp = 0.0, 0
    if roll < 0.30:
        blur_sigma = float(rng.uniform(0.4, 0.8))
    elif roll < 0.45:
        unsharp = int(rng.integers(60, 141))
    scale = float(rng.uniform(0.6, 1.0)) if rng.random() < 0.4 else 1.0
    low, high = config.crf_range
    crf = int(rng.integers(low, high + 1))
    return _Degradation(sigma, blur_sigma, unsharp, scale, int(rng.integers(0, 2**62))), crf


def _degrade_frame_det(frame: np.ndarray, deg: _Degradation) -> np.ndarray:
    image = Image.fromarray(frame)
    if deg.scale < 0.98:
        width, height = image.size
        small = (max(16, int(round(width * deg.scale))), max(16, int(round(height * deg.scale))))
        image = image.resize(small, Image.LANCZOS).resize((width, height), Image.BICUBIC)
    if deg.blur_sigma > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(deg.blur_sigma))
    elif deg.unsharp > 0:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=deg.unsharp, threshold=2))
    return np.asarray(image)


def _degrade_det(frames: np.ndarray, deg: Optional[_Degradation]) -> np.ndarray:
    """The noise-free part of the degradation (rescale, blur / sharpen); ``(H,W,3)`` or ``(T,H,W,3)``."""
    if deg is None or not deg.has_deterministic_part:
        return frames
    if frames.ndim == 3:
        return _degrade_frame_det(frames, deg)
    return np.stack([_degrade_frame_det(frame, deg) for frame in frames])


def _degrade(frames: np.ndarray, deg: Optional[_Degradation]) -> np.ndarray:
    """Full degradation: rescale, blur / sharpen, then independent noise on every frame."""
    if deg is None:
        return frames
    rng = np.random.default_rng(deg.seed)
    out = np.empty_like(frames)
    for index in range(len(frames)):
        frame = _degrade_frame_det(frames[index], deg) if deg.has_deterministic_part else frames[index]
        noise = rng.standard_normal(frame.shape, dtype=np.float32) * np.float32(deg.sigma)
        out[index] = np.clip(np.rint(frame.astype(np.float32) + noise), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Decoys: legitimate, unlabelled events
# ---------------------------------------------------------------------------

_CAMERA_DECOYS = frozenset({"camera_stops", "camera_direction_change", "camera_speed_change"})
_SHOT_LEVEL_DECOYS = frozenset({"object_stops", "object_enters", "object_exits"}) | _CAMERA_DECOYS
_GLOBAL_DECOYS = frozenset(
    {"exposure_drift", "white_balance_drift", "smooth_zoom", "illumination_flicker",
     "auto_exposure_step", "auto_white_balance_step", "fast_zoom"}
)
_ZOOM_DECOYS = frozenset({"smooth_zoom", "fast_zoom"})  # both persist: at most one per clip
_DECOY_SECONDS: dict[str, tuple[float, float]] = {
    "exposure_drift": (1.0, 3.0),
    "white_balance_drift": (1.5, 3.5),
    "smooth_zoom": (1.5, 3.5),
    "camera_stops": (1.0, 2.5),
    "object_stops": (0.5, 1.4),
    "object_enters": (0.6, 0.6),  # placement window only; the label is measured
    "object_exits": (0.6, 0.6),
    "illumination_flicker": (2.0, 4.0),
    "fast_zoom": (0.4, 0.8),
    "camera_direction_change": (0.3, 0.6),
    "camera_speed_change": (0.2, 0.4),
}
_DECOY_STEP_FRAMES = (5, 10)  # auto exposure / white-balance steps ease over 5-10 frames

# Decoy -> {edit type: clearance in seconds}. A decoy must never make a labelled edit
# ambiguous, so the edit windows keep this distance from the decoy's transition. The scene
# cut (any edit) is handled separately with CUT_CLEARANCE_SECONDS.
_DECOY_EDIT_CLEARANCE: dict[str, dict[str, float]] = {
    "illumination_flicker": {"exposure_flicker": 0.5},
    "auto_exposure_step": {"exposure_flicker": 0.5, "color_grade_jump": 0.3},
    "auto_white_balance_step": {"color_grade_jump": 0.3},
    "fast_zoom": {"zoom_jump": 0.3},
    "camera_direction_change": {"reversed_segment": 0.5, "dropped_frames": 0.3, "frozen_frames": 0.3},
    "camera_speed_change": {"dropped_frames": 0.5, "frozen_frames": 0.3, "reversed_segment": 0.3},
}


@dataclass(frozen=True)
class _DecoyPlan:
    type: str
    start: int  # frame on the source timeline
    end: int  # exclusive; == start for a scene cut
    shot: int = 0  # 0 = first shot, 1 = after the scene cut


def _choose_decoy_types(
    rng: np.random.Generator, config: SynthesisConfig, num_out: int, footage: bool, cut_available: bool
) -> list[str]:
    pool = [t for t in DECOY_TYPES if not (footage and t in _SHOT_LEVEL_DECOYS)]
    if not cut_available:
        pool = [t for t in pool if t != "scene_cut"]
    lam = config.decoy_rate * (num_out / config.fps) / 8.0
    count = min(int(rng.poisson(lam)), len(pool))
    if count == 0:
        return []
    return [str(pool[i]) for i in rng.choice(len(pool), size=count, replace=False)]


def _place_decoys(
    rng: np.random.Generator,
    types: Sequence[str],
    num_out: int,
    fps: float,
    *,
    moving_camera: Callable[[int], bool],
) -> tuple[list[_DecoyPlan], Optional[int]]:
    """Choose time windows on the source timeline; unplaceable decoys are dropped.

    A scene cut is a permanent switch to a second shot. Shot-level decoys (objects, camera)
    live wholly inside one shot, away from the cut; whole-clip decoys (drifts, zoom) may
    span it, which does not change their meaning.
    """
    plans: list[_DecoyPlan] = []
    cut: Optional[int] = None
    if "scene_cut" in types:
        low = int(math.ceil(1.0 * fps))
        high = num_out - low
        if high >= low:
            cut = int(rng.integers(low, high + 1))
            plans.append(_DecoyPlan("scene_cut", cut, cut))
    margin = int(math.ceil(0.3 * fps))
    shot_margin = int(math.ceil(0.5 * fps))
    regions = [(0, 0, num_out)] if cut is None else [(0, 0, cut - shot_margin), (1, cut + shot_margin, num_out)]
    for decoy in types:
        if decoy == "scene_cut":
            continue
        if decoy in ("auto_exposure_step", "auto_white_balance_step"):
            low_s, high_s = _DECOY_STEP_FRAMES[0] / fps, _DECOY_STEP_FRAMES[1] / fps
        else:
            low_s, high_s = _DECOY_SECONDS[decoy]
        if decoy in _ZOOM_DECOYS and any(p.type in _ZOOM_DECOYS for p in plans):
            continue  # one persistent zoom per clip
        floor = int(math.ceil(low_s * fps - 1e-9))
        for _ in range(8):
            duration = max(floor, int(round(float(rng.uniform(low_s, high_s)) * fps)))
            if decoy in _SHOT_LEVEL_DECOYS:
                fits = [
                    (shot, r0, r1)
                    for shot, r0, r1 in regions
                    if r1 - r0 >= duration + 2 * margin and (decoy not in _CAMERA_DECOYS or moving_camera(shot))
                ]
                if not fits:
                    break
                shot, r0, r1 = fits[int(rng.integers(0, len(fits)))]
                start = int(rng.integers(r0 + margin, r1 - margin - duration + 1))
            else:
                shot = 0
                duration = min(duration, num_out - 2 * margin)
                if duration < floor:
                    break
                start = int(rng.integers(margin, num_out - margin - duration + 1))
            group = _CAMERA_DECOYS if decoy in _CAMERA_DECOYS else {decoy}
            if any(p.type in group and p.start < start + duration and start < p.end for p in plans):
                continue
            plans.append(_DecoyPlan(decoy, start, start + duration, shot))
            break
    return plans, cut


def _ramp_profile(num_frames: int, start: int, end: int, shape: str) -> np.ndarray:
    """0 before ``start``; then a smooth 0 -> 1 ramp that holds ("ramp") or returns ("bump")."""
    u = (np.arange(num_frames) - start) / max(end - start, 1)
    if shape == "bump":
        return np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 2
    return _smoothstep(u)


def _apply_gain_decoy(frames: np.ndarray, profile: np.ndarray, gain: np.ndarray) -> None:
    """In place: multiply frame ``t`` by ``1 + profile[t] * gain`` (gain is per channel)."""
    active = np.nonzero(np.abs(profile) > 1e-3)[0]
    for lo in range(0, len(active), 24):
        chunk = active[lo : lo + 24]
        factors = (1.0 + profile[chunk, None] * gain[None, :]).astype(np.float32)
        block = frames[chunk].astype(np.float32) * factors[:, None, None, :]
        frames[chunk] = np.clip(np.rint(block), 0, 255).astype(np.uint8)


def _apply_zoom_decoy(frames: np.ndarray, zoom: np.ndarray, center: tuple[float, float]) -> None:
    """In place: continuous digital zoom about ``center`` (fractions of the frame)."""
    height, width = frames.shape[1:3]
    for index in np.nonzero(zoom > 1.0005)[0]:
        crop_w, crop_h = width / zoom[index], height / zoom[index]
        x0 = center[0] * (width - crop_w)
        y0 = center[1] * (height - crop_h)
        frames[index] = np.asarray(
            Image.fromarray(frames[index]).resize(
                (width, height), Image.BILINEAR, box=(x0, y0, x0 + crop_w, y0 + crop_h)
            )
        )


def _apply_global_decoys(frames: np.ndarray, plans: Sequence[_DecoyPlan], seed: int, fps: float) -> None:
    """Exposure / white-balance drift and smooth zoom, applied to the whole source in place."""
    for index, plan in enumerate(plans):
        if plan.type not in _GLOBAL_DECOYS:
            continue
        prng = _rng(seed, "decoy_params", index)
        shape = "bump" if prng.random() < 0.4 else "ramp"
        profile = _ramp_profile(len(frames), plan.start, plan.end, shape)
        if plan.type == "illumination_flicker":
            # Mains / cloud flicker: a smooth periodic +-2-4 % brightness modulation, faded in and out.
            amplitude = float(prng.uniform(0.02, 0.04))
            freq = float(prng.uniform(0.5, 3.0))
            t = np.arange(len(frames))
            fade = int(math.ceil(0.3 * fps))
            envelope = _smoothstep((t - plan.start) / fade) * _smoothstep((plan.end - 1 - t) / fade)
            wave = np.sin(2.0 * math.pi * freq * (t - plan.start) / fps + float(prng.uniform(0, 6.28)))
            _apply_gain_decoy(frames, envelope * wave, np.full(3, amplitude))
        elif plan.type == "auto_exposure_step":
            # The camera's auto exposure settles: a persistent, eased brightness step.
            step = float(prng.uniform(0.05, 0.15)) * float(prng.choice([-1.0, 1.0]))
            _apply_gain_decoy(frames, _ramp_profile(len(frames), plan.start, plan.end, "ramp"), np.full(3, step))
        elif plan.type == "auto_white_balance_step":
            step = float(prng.uniform(0.04, 0.12)) * float(prng.choice([-1.0, 1.0]))
            gain = step * np.array([1.0, float(prng.uniform(-0.2, 0.2)), -1.0])
            _apply_gain_decoy(frames, _ramp_profile(len(frames), plan.start, plan.end, "ramp"), gain)
        elif plan.type == "fast_zoom":
            top = float(prng.uniform(1.1, 1.3))
            center = (float(prng.uniform(0.35, 0.65)), float(prng.uniform(0.35, 0.65)))
            _apply_zoom_decoy(frames, 1.0 + (top - 1.0) * _ramp_profile(len(frames), plan.start, plan.end, "ramp"), center)
        elif plan.type == "exposure_drift":
            delta = float(prng.uniform(0.08, 0.25)) * float(prng.choice([-1.0, 1.0]))
            _apply_gain_decoy(frames, profile, np.full(3, delta))
        elif plan.type == "white_balance_drift":
            strength = float(prng.uniform(0.06, 0.18)) * float(prng.choice([-1.0, 1.0]))
            gain = strength * np.array([1.0, float(prng.uniform(-0.2, 0.2)), -1.0])
            _apply_gain_decoy(frames, profile, gain)
        else:
            top = float(prng.uniform(1.1, 1.3))
            center = (float(prng.uniform(0.35, 0.65)), float(prng.uniform(0.35, 0.65)))
            _apply_zoom_decoy(frames, 1.0 + (top - 1.0) * profile, center)


@dataclass
class _Source:
    frames: np.ndarray  # source timeline (num_out + room for dropped frames), decoys applied
    kind: str  # "procedural" | "footage"
    shots: list[tuple[int, Optional[_SceneSpec]]]  # (first frame, scene) of each shot
    decoys: list[_DecoyPlan]  # realised decoys on the source timeline
    cut: Optional[int]


def _build_source_once(
    seed: int,
    config: SynthesisConfig,
    num_src: int,
    num_out: int,
    attempt: int,
    *,
    hints: _SceneHints,
    forced: Sequence[str] | None,
    salt: int = 0,
) -> _Source:
    fps = config.fps
    rng = _rng(seed, "source_choice")
    footage: np.ndarray | None = None
    footage_path = ""
    if (
        config.footage_paths
        and config.footage_fraction > 0
        and float(rng.random()) < config.footage_fraction
    ):
        footage_path = config.footage_paths[int(rng.integers(0, len(config.footage_paths)))]
        footage = _load_footage_frames(footage_path, config, num_src)
        if footage is None:
            logger.warning("falling back to a procedural source for seed {}", seed)

    drng = _rng(seed, "decoys", attempt)
    others = [p for p in config.footage_paths if p != footage_path]
    second_footage: np.ndarray | None = None
    if forced is not None:
        types = list(dict.fromkeys(forced))
    else:
        types = _choose_decoy_types(drng, config, num_out, footage is not None, footage is None or bool(others))

    if footage is not None:
        plans, cut = _place_decoys(drng, types, num_out, fps, moving_camera=lambda _shot: False)
        frames = np.ascontiguousarray(footage[:num_src])
        if cut is not None:
            second_path = others[int(drng.integers(0, len(others)))] if others else ""
            second_footage = _load_footage_frames(second_path, config, num_src - cut) if second_path else None
            if second_footage is None:
                plans = [p for p in plans if p.type != "scene_cut"]
                cut = None
            else:
                frames[cut:] = second_footage
        shots: list[tuple[int, Optional[_SceneSpec]]] = [(0, None)] if cut is None else [(0, None), (cut, None)]
        realized = plans
    else:
        modes = hints.camera_modes
        if forced is not None and _CAMERA_DECOYS & set(forced):
            modes = _MOVING_CAMERA_MODES
        scene_kwargs = dict(
            camera_modes=modes, symmetric=hints.symmetric,
            fast_objects=hints.fast_objects,
        )
        # A forced decoy that cannot be realised on a scene (e.g. a weak pan) retries on a fresh one.
        scene_parts = ("scene",) if salt == 0 and attempt == 0 else ("scene", salt, attempt)
        spec_a = _sample_scene(_rng(seed, *scene_parts), config.width, config.height, fps, **scene_kwargs)
        spec_b = (
            _sample_scene(_rng(seed, "scene_b", salt, attempt), config.width, config.height, fps, **scene_kwargs)
            if "scene_cut" in types
            else None
        )
        specs = [spec_a, spec_b]

        def moving(shot: int) -> bool:
            spec = specs[shot]
            return spec is not None and spec.camera_mode in _MOVING_CAMERA_MODES

        plans, cut = _place_decoys(drng, types, num_out, fps, moving_camera=moving)
        offsets = [0, cut if cut is not None else num_src]
        lengths = [offsets[1], num_src - offsets[1]]
        parts: list[np.ndarray] = []
        realized = [p for p in plans if p.type not in _SHOT_LEVEL_DECOYS]
        for shot in (0, 1):
            if lengths[shot] <= 0 or specs[shot] is None:
                continue
            events = [
                (p.type, p.start - offsets[shot], p.end - offsets[shot])
                for p in plans
                if p.type in _SHOT_LEVEL_DECOYS and p.shot == shot
            ]
            shot_frames, done = _render_scene(specs[shot], lengths[shot], events)
            parts.append(shot_frames)
            realized += [_DecoyPlan(kind, a + offsets[shot], b + offsets[shot], shot) for kind, a, b in done]
        frames = np.ascontiguousarray(np.concatenate(parts, axis=0))
        shots = [(0, spec_a)] if cut is None else [(0, spec_a), (cut, spec_b)]
    _apply_global_decoys(frames, realized, seed, fps)
    return _Source(frames=frames, kind="footage" if footage is not None else "procedural", shots=shots, decoys=realized, cut=cut)


def _build_source(
    seed: int,
    config: SynthesisConfig,
    num_src: int,
    num_out: int,
    *,
    hints: Optional[_SceneHints] = None,
    forced: Sequence[str] | None,
    salt: int = 0,
) -> _Source:
    """Render the source shot(s) with decoys. Forced decoys are retried until all are realised."""
    attempts = 6 if forced else 1
    for attempt in range(attempts):
        source = _build_source_once(
            seed, config, num_src, num_out, attempt, hints=hints or _SceneHints(), forced=forced, salt=salt
        )
        if not forced or set(dict.fromkeys(forced)) <= {p.type for p in source.decoys}:
            return source
    missing = sorted(set(forced or ()) - {p.type for p in source.decoys})
    raise RuntimeError(f"could not realise forced decoys {missing} for seed {seed}")


def _source_to_output(drops: Sequence[tuple[int, int]]) -> Callable[[int], int]:
    def to_output(frame: int) -> int:
        shift = 0
        for start, length in drops:
            if frame >= start + length:
                shift += length
            elif frame > start:
                shift += frame - start
        return frame - shift

    return to_output


# ---------------------------------------------------------------------------
# Editors
# ---------------------------------------------------------------------------


@dataclass
class _EditOutcome:
    frames: np.ndarray  # replacement frames for the window (0 frames for dropped_frames)
    params: dict[str, float | int | str]
    bbox: Optional[list[float]] = None


@dataclass(frozen=True)
class _EditContext:
    config: SynthesisConfig
    # donor(rng, length, position=0, similar=False) -> frames of an inserted shot
    donor: Callable[..., np.ndarray]
    deg: Optional[_Degradation] = None

    @property
    def sigma(self) -> float:
        return self.deg.sigma if self.deg is not None else 0.0

    def det(self, frames: np.ndarray) -> np.ndarray:
        """Frames as the detector will see them, minus the (noise-free-comparable) sensor noise."""
        return _degrade_det(frames, self.deg)

    def margin(self) -> float:
        return _NOISE_MARGIN * self.sigma


EditorFn = Callable[
    [np.random.Generator, np.ndarray, int, int, str, _EditContext], Optional[_EditOutcome]
]


def _mad(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute difference (0-255), subsampled for speed."""
    a = first[..., ::2, ::2, :].astype(np.int16)
    b = second[..., ::2, ::2, :].astype(np.int16)
    return float(np.mean(np.abs(a - b)))


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _sample_indices(length: int) -> list[int]:
    return sorted({0, length // 2, length - 1})


# Guard thresholds. Statistics are computed on the noise-free degraded signal; every
# threshold below also gets ``ctx.margin()`` (0.3 x sensor sigma) added.
# Tiers: v2.1 shifted every v2 tier down one step (easy = old medium, medium = old hard,
# hard = old expert) and added a new, subtler expert tier. "Subtle" tiers (hard, expert) use
# frame-count windows, similar-shot splices, scene-following sprites and relative guards.
SUBTLE_TIERS = ("hard", "expert")
_FROZEN_MIN = {"easy": 5.0, "medium": 5.0, "hard": 3.5, "expert": 2.5}
_DROPPED_MIN = {"easy": 5.0, "medium": 5.0, "hard": 4.0, "expert": 3.5}
_DROPPED_RATIO = {"hard": 1.3, "expert": 1.25}  # jump across the cut vs a normal step, at coarse scale
_SPLICE_MIN = {"easy": 12.0, "medium": 12.0, "hard": 8.0, "expert": 7.0}
_MIRROR_MIN = {"easy": 8.0, "medium": 8.0, "hard": 6.0, "expert": 4.5}
_ZOOM_MIN = {"easy": 8.0, "medium": 8.0, "hard": 4.0, "expert": 2.5}
_FLICKER_MIN = {"easy": 8.0, "medium": 8.0, "hard": 5.0, "expert": 3.0}
_GRADE_MIN_MAD = {"easy": 4.0, "medium": 4.0}
_GRADE_MIN_SHIFT = {"hard": 2.5, "expert": 1.6}  # mean channel shift, levels
_REVERSE_ASYM_MIN = {"easy": 5.0, "medium": 5.0, "hard": 3.0, "expert": 2.5}
_REVERSE_MOTION = {"easy": (5.0, 1e9), "medium": (5.0, 1e9), "hard": (3.5, 10.0), "expert": (3.0, 9.0)}


def _edit_frozen(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    # Freezing only shows if the picture would have moved: the jump when motion resumes.
    if b >= len(src):
        return None
    pair = ctx.det(src[[a, b]])
    if _mad(pair[0], pair[1]) < _FROZEN_MIN[difficulty] + ctx.margin():
        return None
    return _EditOutcome(np.repeat(src[a : a + 1], b - a, axis=0), {"hold_frames": b - a})


def _pool(frames: np.ndarray, factor: int) -> np.ndarray:
    """Box-average ``(..., H, W, 3)`` frames by ``factor`` (float32)."""
    height, width = frames.shape[-3:-1]
    h, w = height // factor * factor, width // factor * factor
    view = frames[..., :h, :w, :].astype(np.float32)
    shape = view.shape[:-3] + (h // factor, factor, w // factor, factor, 3)
    return view.reshape(shape).mean(axis=(-4, -2))


def _edit_dropped(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    # The jump across the cut must be clearly larger than normal frame-to-frame change.
    if a < 1 or b >= len(src):
        return None
    lead = max(a - 3, 0)
    seen = ctx.det(src[[*range(lead, a), b]])  # the frames either side of the cut, and a run-up
    scale = 0.5 if difficulty in SUBTLE_TIERS else 1.0  # sub-2-frame gaps: the ratio test below carries the weight
    if _mad(seen[-2], seen[-1]) < _DROPPED_MIN[difficulty] + scale * ctx.margin():
        return None
    if difficulty in SUBTLE_TIERS and len(seen) >= 3:
        # A 1-2 frame gap in fast, high-contrast motion decorrelates the picture either way
        # and is not perceptible: compare the jump with normal steps at a coarse scale
        # where image difference still grows with displacement.
        pooled = _pool(seen, max(2, seen.shape[2] // 40))
        normal = float(np.mean([np.abs(pooled[i + 1] - pooled[i]).mean() for i in range(len(pooled) - 2)]))
        jump = float(np.abs(pooled[-1] - pooled[-2]).mean())
        if jump < _DROPPED_RATIO[difficulty] * normal or jump - normal < 0.6 + 0.5 * ctx.margin():
            return None
    empty = np.empty((0,) + src.shape[1:], dtype=np.uint8)
    return _EditOutcome(empty, {"dropped_frames": b - a})


def _edit_reversed(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    window = src[a:b]
    length = len(window)
    det = ctx.det(window)
    probes = sorted({0, length // 4, length // 2 - 1 if length > 2 else 0})
    asym = np.mean([_mad(det[i], det[length - 1 - i]) for i in probes])
    motion = _mad(det[0], det[-1])
    low, high = _REVERSE_MOTION[difficulty]
    margin = ctx.margin()
    if asym < _REVERSE_ASYM_MIN[difficulty] + margin or not (low + margin <= motion <= high):
        return None
    return _EditOutcome(np.ascontiguousarray(window[::-1]), {"frames": length})


def _edit_spliced(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    # Expert splices come from a *similar* shot (same palette / texture family, new layout).
    donor = ctx.donor(rng, b - a, a, difficulty in SUBTLE_TIERS)
    before, after = src[a - 1], src[min(b, len(src) - 1)]
    edges = ctx.det(np.stack([donor[0], donor[-1], before, after]))
    threshold = _SPLICE_MIN[difficulty] + ctx.margin()
    if _mad(edges[0], edges[2]) < threshold or _mad(edges[1], edges[3]) < threshold:
        return None
    return _EditOutcome(donor, {"frames": b - a})


_GRADE_STRENGTH = {  # (gain spread range, hue-rotation range in degrees)
    "easy": ((0.15, 0.30), (20.0, 40.0)),
    "medium": ((0.08, 0.15), (10.0, 20.0)),
    "hard": ((0.03, 0.06), (4.0, 8.0)),
    "expert": ((0.02, 0.04), (3.0, 5.0)),
}


def _hue_matrix(degrees: float) -> np.ndarray:
    """RGB hue-rotation matrix (same construction as the CSS ``hue-rotate`` filter)."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
            [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
            [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
        ],
        dtype=np.float32,
    )


def _edit_color_grade(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    spread_range, hue_range = _GRADE_STRENGTH[difficulty]
    # Subtle grades are a pure gain change OR a pure hue rotation: the two would add up.
    mode = str(rng.choice(["gain", "hue"] if difficulty in SUBTLE_TIERS else ["gain", "hue", "both"]))
    gains = np.ones(3, dtype=np.float32)
    hue = 0.0
    params: dict[str, float | int | str] = {"mode": mode}
    if mode in ("gain", "both"):
        spread = _uniform(rng, spread_range)
        share = float(rng.uniform(0.3, 0.7))
        order = rng.permutation(3)
        gains[order[0]] = 1.0 + spread * share
        gains[order[1]] = 1.0 - spread * (1.0 - share)
        gains[order[2]] = 1.0 + spread * float(rng.uniform(-0.5, 0.5)) * 0.5
        params.update(
            gain_r=round(float(gains[0]), 4), gain_g=round(float(gains[1]), 4), gain_b=round(float(gains[2]), 4)
        )
    if mode in ("hue", "both"):
        hue = _uniform(rng, hue_range) * float(rng.choice([-1.0, 1.0]))
        params["hue_deg"] = round(hue, 2)

    def grade(frames: np.ndarray) -> np.ndarray:
        flat = frames.astype(np.float32).reshape(-1, 3)
        if hue:
            flat = flat @ _hue_matrix(hue).T
        return np.clip(flat * gains, 0, 255).astype(np.uint8).reshape(frames.shape)

    window = src[a:b]
    sample = window[_sample_indices(len(window))]
    graded_sample, original = ctx.det(grade(sample)), ctx.det(sample)
    if difficulty in SUBTLE_TIERS:
        shift = np.abs(graded_sample.reshape(-1, 3).mean(axis=0) - original.reshape(-1, 3).mean(axis=0))
        if float(shift.max()) < _GRADE_MIN_SHIFT[difficulty]:
            return None
    elif _mad(graded_sample, original) < _GRADE_MIN_MAD[difficulty] + ctx.margin():
        return None
    return _EditOutcome(grade(window), params)


_FLICKER_FACTORS = {
    "easy": {"bright": (1.4, 1.8), "dark": (0.5, 0.7)},
    "medium": {"bright": (1.2, 1.4), "dark": (0.7, 0.82)},
    "hard": {"bright": (1.08, 1.12), "dark": (0.88, 0.92)},
    "expert": {"bright": (1.04, 1.07), "dark": (0.93, 0.96)},
}


def _edit_flicker(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    direction = "bright" if rng.random() < 0.5 else "dark"
    factor = _uniform(rng, _FLICKER_FACTORS[difficulty][direction])
    window = src[a:b]
    scaled = np.clip(window.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if abs(float(scaled.mean()) - float(window.mean())) < _FLICKER_MIN[difficulty]:
        return None
    return _EditOutcome(scaled, {"factor": round(factor, 4), "direction": direction})


def _edit_mirrored(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    window = src[a:b]
    step = max(1, len(window) // 3)
    sample = ctx.det(window[::step])
    if _mad(sample[:, :, ::-1], sample) < _MIRROR_MIN[difficulty] + ctx.margin():
        return None
    return _EditOutcome(np.ascontiguousarray(window[:, :, ::-1]), {"frames": b - a})


_ZOOM_SCALES = {"easy": (1.2, 1.4), "medium": (1.1, 1.2), "hard": (1.03, 1.08), "expert": (1.02, 1.05)}


def _zoom_frames(frames: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frames.shape[1:3]
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR, box=box)) for frame in frames]
    )


def _edit_zoom(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    window = src[a:b]
    height, width = window.shape[1:3]
    scale = _uniform(rng, _ZOOM_SCALES[difficulty])
    crop_w, crop_h = width / scale, height / scale
    x0 = float(rng.uniform(0.0, width - crop_w))
    y0 = float(rng.uniform(0.0, height - crop_h))
    box = (x0, y0, x0 + crop_w, y0 + crop_h)
    step = max(1, len(window) // 3)
    original = ctx.det(window[::step])
    if _mad(ctx.det(_zoom_frames(window[::step], box)), original) < _ZOOM_MIN[difficulty] + ctx.margin():
        return None
    return _EditOutcome(_zoom_frames(window, box), {"scale": round(scale, 4)})


_INSERT_SIZE = {"easy": (0.10, 0.15), "medium": (0.06, 0.10), "hard": (0.04, 0.06), "expert": (0.03, 0.05)}
_INSERT_CONTRAST = {"easy": 10.0, "medium": 5.0, "hard": 5.0, "expert": 4.0}  # median painted |diff|
_INSERT_CORE = {"easy": 0.0, "medium": 0.0, "hard": 11.0, "expert": 9.0}  # 90th percentile painted |diff|
_INSERT_FOLLOW = {"easy": 0.4, "medium": 0.6, "hard": 1.0, "expert": 1.0}
_INSERT_SPREAD = {"easy": 22.0, "medium": 14.0, "hard": 13.0, "expert": 11.0}  # colour match tolerance


def _global_shifts(window: np.ndarray) -> np.ndarray:
    """``(T, 2)`` cumulative (dx, dy) image motion (px, full resolution) relative to frame 0.

    Phase correlation of consecutive frames at half resolution. Weak or ambiguous peaks
    count as "no motion", so a static camera yields a still sprite instead of a trembling one.
    """
    count = len(window)
    shifts = np.zeros((count, 2))
    if count < 2:
        return shifts
    gray = window[:, ::2, ::2].astype(np.float32).mean(axis=3)
    gray -= gray.mean(axis=(1, 2), keepdims=True)
    height, width = gray.shape[1:]
    taper = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    spectra = np.fft.rfft2(gray * taper)
    for index in range(1, count):
        cross = spectra[index] * np.conj(spectra[index - 1])
        cross = cross / (np.abs(cross) + 0.05 * float(np.abs(cross).mean()) + 1e-6)
        corr = np.fft.irfft2(cross, s=(height, width))
        peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)
        peak = float(corr[peak_y, peak_x])
        step = np.zeros(2)
        if peak > 8.0 * float(np.abs(corr).mean()):
            offsets = []
            for axis, size, at in ((1, width, peak_x), (0, height, peak_y)):
                left = corr[peak_y, (at - 1) % size] if axis == 1 else corr[(at - 1) % size, peak_x]
                right = corr[peak_y, (at + 1) % size] if axis == 1 else corr[(at + 1) % size, peak_x]
                denom = left - 2.0 * peak + right
                frac = 0.5 * (left - right) / denom if abs(denom) > 1e-9 else 0.0
                signed = at - size if at > size // 2 else at
                offsets.append(2.0 * (signed + float(np.clip(frac, -0.5, 0.5))))
            step = np.clip(np.array(offsets), -24.0, 24.0)
        shifts[index] = shifts[index - 1] + step
    return shifts


def _edit_inserted(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    window = src[a:b]
    length = len(window)
    height, width = window.shape[1:3]
    size = max(8.0 if difficulty in SUBTLE_TIERS else 10.0, _uniform(rng, _INSERT_SIZE[difficulty]) * max(width, height))
    kind = str(rng.choice(["circle", "rect", "triangle", "ellipse"]))
    hw = size / 2.0
    hh = hw * (float(rng.uniform(0.7, 1.0)) if kind in ("rect", "ellipse") else 1.0)
    if difficulty == "easy":
        feather = 2.0
    elif difficulty == "medium":
        feather = max(3.0, 0.3 * size)
    else:
        # Soft, pasted-looking edge (subtle sprites are tiny, so a wide feather would eat the core).
        feather = max(2.0, 0.14 * size)
    left, top, right, bottom = _shape_extent(kind, hw, hh)
    pad = feather / 2.0 + 1.0
    lo_x, hi_x = left + pad, width - right - pad
    lo_y, hi_y = top + pad, height - bottom - pad
    if hi_x <= lo_x or hi_y <= lo_y:
        return None

    # Path of the sprite: it either rides along with the scene / camera motion (so it does
    # not sit still against a panning background) or is static / drifts linearly.
    follow = float(rng.random()) < _INSERT_FOLLOW[difficulty]
    if follow:
        relative = _global_shifts(window)
        if difficulty in SUBTLE_TIERS:
            relative = relative + rng.uniform(-3.0, 3.0, 2)[None, :] * (np.arange(length)[:, None] / ctx.config.fps)
        low = np.array([lo_x, lo_y]) - relative.min(axis=0)
        high = np.array([hi_x, hi_y]) - relative.max(axis=0)
        if np.any(high <= low):
            return None
        start = np.array([rng.uniform(low[0], high[0]), rng.uniform(low[1], high[1])])
        path = start[None, :] + relative
    else:
        start = np.array([rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)])
        if rng.random() < 0.4:
            end = start.copy()  # static
        else:
            drift = rng.uniform(-0.15, 0.15, 2) * np.array([width, height])
            end = np.clip(start + drift, [lo_x, lo_y], [hi_x, hi_y])
        mix = (np.arange(length) / max(length - 1, 1))[:, None]
        path = start[None, :] * (1.0 - mix) + end[None, :] * mix

    # Sample the local mean colour under the sprite's start position.
    px0, px1 = int(max(path[0, 0] - left, 0)), int(min(path[0, 0] + right, width))
    py0, py1 = int(max(path[0, 1] - top, 0)), int(min(path[0, 1] + bottom, height))
    local_mean = window[:, py0:py1, px0:px1].reshape(-1, 3).mean(axis=0).astype(np.float32)
    # Colour matched to the surroundings, within a tolerance that shrinks with the tier.
    spread = _INSERT_SPREAD[difficulty]
    color_a = np.clip(local_mean + rng.normal(0.0, spread, 3), 0, 255).astype(np.float32)
    color_b = np.clip(color_a + rng.normal(0.0, spread, 3), 0, 255).astype(np.float32)
    stripe_depth = 0.1 if difficulty in SUBTLE_TIERS else 0.2
    freq = float(rng.uniform(0.25, 0.7))
    phase = float(rng.uniform(0.0, 6.0))

    def paint(index: int) -> np.ndarray:
        canvas = window[index].astype(np.float32)
        _draw_shape(
            canvas, (0, 0), (float(path[index, 0]), float(path[index, 1])), kind, hw, hh, 0.0,
            color_a, color_b, freq, phase, feather=feather, stripe_depth=stripe_depth,
        )
        return np.clip(canvas, 0, 255).astype(np.uint8)

    indices = _sample_indices(length)
    before = ctx.det(window[indices])
    after = ctx.det(np.stack([paint(i) for i in indices]))
    diff = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(axis=3)
    contrast = float(np.mean([float(np.median(d[d > 0.5])) if (d > 0.5).any() else 0.0 for d in diff]))
    core = float(np.mean([float(np.percentile(d[d > 0.5], 90)) if (d > 0.5).any() else 0.0 for d in diff]))
    if contrast < _INSERT_CONTRAST[difficulty] or core < _INSERT_CORE[difficulty]:
        return None
    out = np.stack([paint(i) for i in range(length)])
    boxes = np.concatenate([path - np.array([left + pad, top + pad]), path + np.array([right + pad, bottom + pad])], axis=1)
    x0 = max(float(boxes[:, 0].min()), 0.0) / width
    y0 = max(float(boxes[:, 1].min()), 0.0) / height
    x1 = min(float(boxes[:, 2].max()), float(width)) / width
    y1 = min(float(boxes[:, 3].max()), float(height)) / height
    return _EditOutcome(
        out,
        {
            "shape": kind,
            "size_px": round(size, 2),
            "drift": "follows_scene" if follow else ("static" if np.allclose(path[0], path[-1]) else "linear"),
        },
        bbox=[float(x0), float(y0), float(x1), float(y1)],
    )


_BLUR_FRACTION = {"easy": (0.15, 0.25), "medium": (0.10, 0.15), "hard": (0.06, 0.10), "expert": (0.05, 0.08)}
_BLUR_SIGMA = {"easy": (3.0, 6.0), "medium": (2.0, 3.0), "hard": (1.2, 2.0), "expert": (1.0, 1.5)}
_PIXEL_BLOCK = {"easy": (5, 8), "medium": (3, 5), "hard": (2, 3), "expert": (2, 2)}
# (min energy drop, max after/before ratio, min patch MAD)
_BLUR_GUARD = {
    "easy": (1.0, 0.7, 3.0),
    "medium": (1.0, 0.7, 3.0),
    "hard": (0.6, 0.85, 2.0),
    "expert": (0.4, 0.9, 1.5),
}


def _gradient_energy(patch: np.ndarray) -> float:
    gray = patch.astype(np.float32).mean(axis=2)
    return float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())


def _blur_patch(patch: np.ndarray, pixelate: bool, block: int, sigma: float) -> np.ndarray:
    image = Image.fromarray(patch)
    if pixelate:
        rect_h, rect_w = patch.shape[:2]
        small = image.resize((max(1, rect_w // block), max(1, rect_h // block)), Image.BOX)
        image = small.resize((rect_w, rect_h), Image.NEAREST)
    else:
        image = image.filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(image)


def _edit_blurred(rng, src, a, b, difficulty, ctx):  # noqa: ANN001
    window = src[a:b]
    height, width = window.shape[1:3]
    rect_w = max(12, int(round(_uniform(rng, _BLUR_FRACTION[difficulty]) * width)))
    rect_h = max(12, int(round(_uniform(rng, _BLUR_FRACTION[difficulty]) * height)))
    if rect_w >= width or rect_h >= height:
        return None
    mid = len(window) // 2
    tries = 4 if difficulty in SUBTLE_TIERS else 1  # small subtle boxes go where there is texture to lose
    best: tuple[float, int, int] | None = None
    for _ in range(tries):
        cx = int(rng.integers(0, width - rect_w + 1))
        cy = int(rng.integers(0, height - rect_h + 1))
        energy = _gradient_energy(window[mid, cy : cy + rect_h, cx : cx + rect_w])
        if best is None or energy > best[0]:
            best = (energy, cx, cy)
    assert best is not None
    _, x0, y0 = best
    x1, y1 = x0 + rect_w, y0 + rect_h
    pixelate = rng.random() < 0.5
    block, sigma = 1, 0.0
    if pixelate:
        lo, hi = _PIXEL_BLOCK[difficulty]
        block = int(rng.integers(lo, hi + 1))
        params: dict[str, float | int | str] = {"mode": "pixelate", "block": block}
    else:
        sigma = _uniform(rng, _BLUR_SIGMA[difficulty])
        params = {"mode": "gaussian", "sigma": round(sigma, 3)}

    mid_frame = window[mid].copy()
    mid_frame[y0:y1, x0:x1] = _blur_patch(window[mid, y0:y1, x0:x1], pixelate, block, sigma)
    seen_before = ctx.det(window[mid])[y0:y1, x0:x1]
    seen_after = ctx.det(mid_frame)[y0:y1, x0:x1]
    before, after = _gradient_energy(seen_before), _gradient_energy(seen_after)
    min_drop, max_ratio, min_mad = _BLUR_GUARD[difficulty]
    if before - after < min_drop + 0.1 * ctx.sigma or after > max_ratio * before:
        return None
    if _mad(seen_after, seen_before) < min_mad + 0.15 * ctx.sigma:
        return None
    out = window.copy()
    for index in range(len(window)):
        out[index, y0:y1, x0:x1] = _blur_patch(window[index, y0:y1, x0:x1], pixelate, block, sigma)
    return _EditOutcome(out, params, bbox=[x0 / width, y0 / height, x1 / width, y1 / height])


EDITORS: dict[str, EditorFn] = {
    "frozen_frames": _edit_frozen,
    "dropped_frames": _edit_dropped,
    "reversed_segment": _edit_reversed,
    "spliced_footage": _edit_spliced,
    "color_grade_jump": _edit_color_grade,
    "exposure_flicker": _edit_flicker,
    "mirrored_segment": _edit_mirrored,
    "zoom_jump": _edit_zoom,
    "inserted_object": _edit_inserted,
    "blurred_region": _edit_blurred,
}

# Window length in seconds per difficulty. Subtle tiers use frame-count windows for the
# types in ``_FRAME_WINDOWS``.
_LENGTH_SECONDS: dict[str, dict[str, tuple[float, float]]] = {
    "frozen_frames": {"easy": (0.6, 1.0), "medium": (0.3, 0.6)},
    "dropped_frames": {"easy": (0.4, 0.8), "medium": (0.2, 0.4)},
    "reversed_segment": {"easy": (0.8, 1.2), "medium": (0.5, 0.8), "hard": (0.3, 0.5), "expert": (0.2, 0.35)},
    "spliced_footage": {"easy": (0.5, 0.8), "medium": (0.27, 0.5)},
    "color_grade_jump": {d: (0.8, 2.5) for d in DIFFICULTIES},
    "mirrored_segment": {"easy": (0.8, 1.2), "medium": (0.4, 0.8), "hard": (0.3, 0.6), "expert": (0.3, 0.5)},
    "zoom_jump": {d: (0.8, 2.0) for d in DIFFICULTIES},
    "inserted_object": {d: (1.0, 3.0) for d in DIFFICULTIES},
    "blurred_region": {d: (1.0, 3.0) for d in DIFFICULTIES},
}
_FRAME_WINDOWS: dict[tuple[str, str], tuple[int, int]] = {
    ("frozen_frames", "hard"): (2, 3),
    ("frozen_frames", "expert"): (2, 2),
    ("dropped_frames", "hard"): (1, 2),
    ("dropped_frames", "expert"): (1, 1),
    ("spliced_footage", "hard"): (2, 4),
    ("spliced_footage", "expert"): (2, 3),
}
_MIN_FRAMES = {"dropped_frames": 3, "spliced_footage": 4}
assert set(_LENGTH_SECONDS) | {"exposure_flicker"} == set(EDITORS)


def _sample_length(
    name: str, rng: np.random.Generator, difficulty: str, fps: float, *, shrink: bool
) -> int:
    """Window length in frames; ``shrink`` picks the lower end of the range."""
    if name == "exposure_flicker":
        return 1 if (shrink or difficulty in SUBTLE_TIERS) else int(rng.integers(1, 4))
    if (name, difficulty) in _FRAME_WINDOWS:
        low, high = _FRAME_WINDOWS[(name, difficulty)]
        return low if shrink else int(rng.integers(low, high + 1))
    lo, hi = _LENGTH_SECONDS[name][difficulty]
    seconds = lo if shrink else float(rng.uniform(lo, hi))
    return max(_MIN_FRAMES.get(name, 3), int(round(seconds * fps)))


def _max_dropped_frames(difficulty: str, fps: float) -> int:
    if ("dropped_frames", difficulty) in _FRAME_WINDOWS:
        return _FRAME_WINDOWS[("dropped_frames", difficulty)][1]
    return max(_MIN_FRAMES["dropped_frames"], int(math.ceil(_LENGTH_SECONDS["dropped_frames"][difficulty][1] * fps)))


# ---------------------------------------------------------------------------
# Clip composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlannedEdit:
    name: str
    start: int  # source-timeline window [start, start + length)
    length: int


def _plan_edits(
    rng: np.random.Generator, names: Sequence[str], num_out: int, difficulty: str, fps: float
) -> list[_PlannedEdit] | None:
    """Place non-overlapping windows on the source timeline, or None if they cannot fit."""
    count = len(names)
    gap = int(math.ceil(MIN_ISSUE_GAP_SECONDS * fps))
    margin = int(math.ceil(MIN_END_MARGIN_SECONDS * fps))
    for shrink in (False, True):
        lengths = [_sample_length(name, rng, difficulty, fps, shrink=shrink) for name in names]
        if count == 1:
            lengths = [min(lengths[0], num_out - 2 * margin)]
        # Dropped frames extend the source by exactly their length, so they do not
        # consume output time.
        kept = sum(n for name, n in zip(names, lengths) if name not in POINT_EVENT_ISSUE_TYPES)
        slack = num_out - 2 * margin - kept - (count - 1) * gap
        if slack >= 0 and min(lengths) >= 1:
            break
    else:
        return None
    cuts = np.sort(rng.integers(0, slack + 1, size=count))
    plan: list[_PlannedEdit] = []
    cursor = margin
    for index, (name, length) in enumerate(zip(names, lengths)):
        start = margin + int(cuts[index]) + sum(lengths[:index]) + index * gap
        plan.append(_PlannedEdit(name, start, length))
        cursor = start + length
    assert cursor <= num_out + sum(
        n for name, n in zip(names, lengths) if name in POINT_EVENT_ISSUE_TYPES
    ) - margin
    return plan


def _edit_clear(decoys: Sequence[_DecoyPlan], name: str, start: int, length: int, fps: float) -> bool:
    """True if an edit window keeps its distance from every decoy that could make it ambiguous.

    A scene cut must not fall inside an edit window nor within 0.5 s of it; other decoys keep the
    clearances of ``_DECOY_EDIT_CLEARANCE`` from the edit types they resemble.
    """
    for decoy in decoys:
        if decoy.type == "scene_cut":
            seconds = CUT_CLEARANCE_SECONDS
        else:
            seconds = _DECOY_EDIT_CLEARANCE.get(decoy.type, {}).get(name)
            if seconds is None:
                continue
        clearance = int(math.ceil(seconds * fps))
        if decoy.start - clearance < start + length and start < decoy.end + clearance:
            return False
    return True


def _make_donor(
    seed: int, config: SynthesisConfig, shots: Sequence[tuple[int, Optional[_SceneSpec]]] = ()
) -> Callable[..., np.ndarray]:
    def donor(rng: np.random.Generator, length: int, position: int = 0, similar: bool = False) -> np.ndarray:
        if config.footage_paths and len(config.footage_paths) > 1 and rng.random() < config.footage_fraction:
            path = config.footage_paths[int(rng.integers(0, len(config.footage_paths)))]
            frames = _load_footage_frames(path, config, length)
            if frames is not None:
                return frames
        child = np.random.default_rng(int(rng.integers(0, 2**62)))
        if similar:
            # Same palette / texture family / camera style as the shot being edited,
            # different layout and objects: a plausible "other take" of the same place.
            base = None
            for start, spec in shots:
                if start <= position and spec is not None:
                    base = spec
            if base is not None:
                spec = _sample_scene(child, config.width, config.height, config.fps, like=base)
                return _render_scene(spec, length)[0]
        return render_procedural_scene(child, length, config.width, config.height, config.fps)

    return donor


def _sample_difficulty(rng: np.random.Generator, config: SynthesisConfig) -> str:
    names = [name for name, weight in config.difficulty_weights if weight > 0]
    weights = np.array([weight for _, weight in config.difficulty_weights if weight > 0], dtype=float)
    return str(names[int(rng.choice(len(names), p=weights / weights.sum()))])


_ISSUE_COUNT = {"easy": (1, 1), "medium": (1, 2), "hard": (2, 3), "expert": (2, 4)}


def _sample_issue_count(rng: np.random.Generator, difficulty: str, pool: int) -> int:
    low, high = _ISSUE_COUNT[difficulty]
    return int(min(rng.integers(low, high + 1), pool))


def _assemble(
    src: np.ndarray,
    plan: Sequence[_PlannedEdit],
    difficulty: str,
    seed: int,
    attempt: int,
    ctx: _EditContext,
    decoys: Sequence[_DecoyPlan],
    num_out: int,
) -> tuple[np.ndarray, list[IssueLabel], list[tuple[int, int]]] | None:
    """Apply the planned edits; returns (frames, labels, dropped windows on the source timeline).

    An edit that fails its guard is redrawn at the planned position, then slid to other
    positions inside the slack that keeps the spacing rules (and the scene-cut clearance).
    """
    fps = ctx.config.fps
    gap = int(math.ceil(MIN_ISSUE_GAP_SECONDS * fps))
    margin = int(math.ceil(MIN_END_MARGIN_SECONDS * fps))
    dropped_total = sum(e.length for e in plan if e.name in POINT_EVENT_ISSUE_TYPES)
    limit = num_out + dropped_total  # end of the used source
    chunks: list[np.ndarray] = []
    labels: list[IssueLabel] = []
    drops: list[tuple[int, int]] = []
    cursor = 0
    out_len = 0
    prev_end = 0
    for index, edit in enumerate(plan):
        low = margin if index == 0 else prev_end + gap
        high = (
            plan[index + 1].start - gap - edit.length
            if index + 1 < len(plan)
            else limit - margin - edit.length
        )
        srng = _rng(seed, "slide", attempt, index)
        alternatives = [
            s for s in range(low, high + 1) if s != edit.start and _edit_clear(decoys, edit.name, s, edit.length, fps)
        ]
        if len(alternatives) > _SLIDE_TRIES:
            alternatives = [alternatives[int(i)] for i in srng.choice(len(alternatives), _SLIDE_TRIES, replace=False)]
        outcome: _EditOutcome | None = None
        start = edit.start
        tries = 0
        for candidate in [edit.start] + alternatives:
            for retry in range(_EDIT_RETRIES):
                tries += 1
                erng = _rng(seed, "edit", attempt, index, candidate, retry)
                outcome = EDITORS[edit.name](erng, src, candidate, candidate + edit.length, difficulty, ctx)
                if outcome is not None:
                    break
            if outcome is not None:
                start = candidate
                break
        if difficulty == "expert":
            _bump("expert_edits")
            _bump(f"expert_evals:{edit.name}")
            _bump(f"expert_first_fail:{edit.name}", int(tries > 1))
            _bump("expert_resampled", int(tries > 1))  # first draw failed its guard
            _bump("expert_edit_failed", int(outcome is None))  # every draw and position failed
            _bump("expert_accepted", int(outcome is not None))
            _bump("expert_accepted_resampled", int(outcome is not None and tries > 1))
        if outcome is None:
            return None
        end = start + edit.length
        chunks.append(src[cursor:start])
        out_len += start - cursor
        chunks.append(outcome.frames)
        start_frame = out_len
        out_len += len(outcome.frames)
        # Manifest convention: spans use exclusive end_frame; a point event sits
        # at the first output frame after the cut with start == end.
        is_point = edit.name in POINT_EVENT_ISSUE_TYPES
        end_frame = start_frame if is_point else out_len
        if is_point:
            drops.append((start, edit.length))
        labels.append(
            IssueLabel(
                type=edit.name,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                start_frame=start_frame,
                end_frame=end_frame,
                bbox=outcome.bbox,
                params=outcome.params,
            )
        )
        cursor = end
        prev_end = end
    chunks.append(src[cursor:limit])
    out_len += limit - cursor
    frames = np.ascontiguousarray(np.concatenate(chunks, axis=0))
    return frames, labels, drops


def _decoy_labels(
    source: _Source, drops: Sequence[tuple[int, int]], num_out: int, fps: float
) -> list[DecoyLabel]:
    """Decoys on the output timeline (dropped frames shift everything after them)."""
    to_output = _source_to_output(drops)
    labels: list[DecoyLabel] = []
    for plan in sorted(source.decoys, key=lambda p: (p.start, p.end, p.type)):
        start, end = to_output(plan.start), to_output(plan.end)
        if plan.type != "scene_cut" and end <= start:
            continue  # swallowed by a dropped-frames window
        if start >= num_out:
            continue
        labels.append(DecoyLabel(type=plan.type, start_time=start / fps, end_time=min(end, num_out) / fps))  # type: ignore[arg-type]
    return labels


def generate_clip(
    seed: int,
    config: SynthesisConfig = SynthesisConfig(),
    *,
    issue_types: Sequence[str] | None = None,
    difficulty: str | None = None,
    decoys: Sequence[str] | None = None,
) -> SyntheticClip:
    """Generate one edited clip with ground-truth labels, deterministically from ``seed``.

    ``issue_types`` forces exactly those edits; ``difficulty`` forces the tier. ``decoys=None``
    samples decoys from ``config.decoy_rate``, ``decoys=[]`` disables them, and a list forces
    exactly those decoy types (each once).
    """
    if issue_types is not None:
        unknown = [name for name in issue_types if name not in EDITORS]
        if unknown:
            raise ValueError(f"unknown issue types: {unknown}")
        if not issue_types:
            raise ValueError("issue_types must not be empty when given")
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if decoys is not None:
        unknown = [name for name in decoys if name not in DECOY_TYPES]
        if unknown:
            raise ValueError(f"unknown decoy types: {unknown}")

    rng = _rng(seed, "plan")
    fps = config.fps
    num_out = int(round(float(rng.uniform(config.min_duration, config.max_duration)) * fps))
    clip_difficulty = difficulty or _sample_difficulty(rng, config)
    clean = issue_types is None and float(rng.random()) < config.clean_fraction
    pool = list(issue_types) if issue_types is not None else list(config.issue_types)

    deg, crf = _sample_degradation(seed, config)
    max_extra = 0
    if not clean and "dropped_frames" in pool:
        max_extra = _max_dropped_frames(clip_difficulty, fps)
    forced = None if decoys is None else list(decoys)
    if forced == []:
        forced = None
        config_for_source = replace(config, decoy_rate=0.0)
    else:
        config_for_source = config

    # The issue types are chosen BEFORE the scene is rendered, so the scene can be built to
    # show them (moving camera and fast objects for freezes / drops / reversals) instead of
    # types that need motion being dropped after the fact from static scenes.
    name_sets: list[list[str]] = []
    if not clean:
        if issue_types is not None:
            name_sets = [list(pool)]
        else:
            nrng = _rng(seed, "names")
            count = _sample_issue_count(nrng, clip_difficulty, len(pool))
            names = [pool[i] for i in nrng.choice(len(pool), size=count, replace=False)]
            while len(names) > 1 and _plan_edits(_rng(seed, "feasible"), names, num_out, clip_difficulty, fps) is None:
                names = names[:-1]  # no room for all of them: keep the first that fit
            # Last resort, after every scene round failed: drop the trailing type.
            name_sets = [names[:k] for k in range(len(names), 0, -1)]
    share = {"hard": 0.5, "expert": 0.8}.get(clip_difficulty, 0.0)  # near-symmetric mirror scenes
    symmetric = bool(
        name_sets and "mirrored_segment" in name_sets[0] and float(rng.random()) < share
    )

    source: _Source | None = None
    for set_index, current in enumerate(name_sets or [[]]):
        hints = _scene_hints(current, clip_difficulty, symmetric)
        rounds = 1 if clean else (_SOURCE_ROUNDS if set_index == 0 else 2)
        for round_index in range(rounds):
            salt = set_index * _SOURCE_ROUNDS + round_index
            source = _build_source(
                seed, config_for_source, num_out + max_extra, num_out, hints=hints, forced=forced, salt=salt
            )
            if clean:
                break
            ctx = _EditContext(config=config, donor=_make_donor(seed, config, source.shots), deg=deg)
            for attempt in range(_PLAN_ATTEMPTS):
                arng = _rng(seed, "attempt", salt, attempt)
                names = [pool[i] for i in arng.permutation(len(pool))] if issue_types is not None else list(current)
                plan = _plan_clear_of_decoys(arng, names, num_out, clip_difficulty, fps, source.decoys)
                if plan is None and issue_types is None:
                    # Cut clearance left no room: keep the first that fit.
                    while plan is None and len(names) > 1:
                        names = names[:-1]
                        plan = _plan_clear_of_decoys(arng, names, num_out, clip_difficulty, fps, source.decoys)
                if plan is None:
                    continue
                dropped = sum(e.length for e in plan if e.name in POINT_EVENT_ISSUE_TYPES)
                if dropped > max_extra or len(source.frames) < num_out + dropped:
                    continue
                assembled = _assemble(
                    source.frames, plan, clip_difficulty, seed, salt * 100 + attempt, ctx, source.decoys, num_out
                )
                if assembled is None:
                    continue
                frames, labels, drops = assembled
                if len(frames) != num_out:
                    raise AssertionError("assembled clip has the wrong frame count")
                labels.sort(key=lambda label: (label.start_frame, label.end_frame))
                return SyntheticClip(
                    frames=_degrade(frames, deg),
                    fps=fps,
                    issues=labels,
                    difficulty=clip_difficulty,
                    source=source.kind,
                    decoys=_decoy_labels(source, drops, num_out, fps),
                    crf=crf,
                )
    assert source is not None
    if not clean:
        if issue_types is not None:
            raise RuntimeError(
                f"could not place a detectable {list(issue_types)} edit for seed {seed}"
            )
        _bump("plan_fallback")
        logger.warning("seed {}: no detectable edit plan found, emitting a clean clip", seed)

    frames = np.ascontiguousarray(source.frames[:num_out])
    if len(frames) != num_out:
        # Footage shorter than the clip cannot happen (checked at load), but stay safe.
        raise AssertionError("source clip has the wrong frame count")
    return SyntheticClip(
        frames=_degrade(frames, deg),
        fps=fps,
        issues=[],
        difficulty=clip_difficulty,
        source=source.kind,
        decoys=_decoy_labels(source, [], num_out, fps),
        crf=crf,
    )


def _plan_clear_of_decoys(
    rng: np.random.Generator,
    names: Sequence[str],
    num_out: int,
    difficulty: str,
    fps: float,
    decoys: Sequence[_DecoyPlan],
    tries: int = 6,
) -> list[_PlannedEdit] | None:
    """A plan whose windows keep their clearance from the decoys (a few random draws)."""
    for _ in range(tries):
        plan = _plan_edits(rng, names, num_out, difficulty, fps)
        if plan is None:
            return None
        if all(_edit_clear(decoys, e.name, e.start, e.length, fps) for e in plan):
            return plan
    return None


def collect_footage(directory: str | Path) -> tuple[str, ...]:
    """Sorted list of video files (mp4/mov/mkv/webm) under ``directory``."""
    suffixes = {".mp4", ".mov", ".mkv", ".webm"}
    root = Path(directory)
    return tuple(
        sorted(str(path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    )
