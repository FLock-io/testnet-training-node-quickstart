"""Video encode/decode/probe helpers built on the ffmpeg binary bundled with imageio-ffmpeg.

Only ``imageio_ffmpeg`` (no system ffmpeg, no opencv) is required on the host.
The encoder settings are chosen so that the *bitstream itself* never leaks where
an edit was injected:

* fixed GOP (``-g 30 -keyint_min 30``) with scene-cut detection disabled, so
  keyframes land on a regular grid instead of at cuts / splices;
* container metadata stripped and bitexact flags set, so the file carries no
  encoder timestamps or tags (and rebuilding a package is byte-reproducible);
* a fixed x264 thread count, because x264 output depends on the thread count.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np


_X264_THREADS = 4
_STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _tail(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")[-_STDERR_TAIL_CHARS:].strip()


def encode_video(
    frames: np.ndarray, path: str | Path, fps: float, *, crf: int = 18
) -> Path:
    """Encode ``(T, H, W, 3)`` uint8 RGB frames to an H.264/yuv420p mp4 at quality ``crf``."""
    if (
        not isinstance(crf, (int, np.integer))
        or isinstance(crf, bool)
        or not 0 <= crf <= 51
    ):
        raise ValueError("crf must be an integer in [0, 51]")
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.uint8:
        raise ValueError("frames must be a (T, H, W, 3) uint8 array")
    num_frames, height, width, _ = frames.shape
    if num_frames == 0:
        raise ValueError("cannot encode an empty video")
    if height % 2 or width % 2:
        raise ValueError(f"yuv420p needs even dimensions, got {width}x{height}")
    if fps <= 0:
        raise ValueError("fps must be positive")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        repr(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(crf)),
        "-preset",
        "medium",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-x264-params",
        f"scenecut=0:threads={_X264_THREADS}",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        str(output),
    ]
    contiguous = np.ascontiguousarray(frames)
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
        assert process.stdin is not None
        try:
            for index in range(num_frames):
                process.stdin.write(contiguous[index].tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg died early; the return code below carries the reason
        finally:
            return_code = process.wait()
        if return_code != 0:
            stderr_file.seek(0)
            raise RuntimeError(f"ffmpeg encode failed: {_tail(stderr_file.read())}")
    return output


def probe_stream(path: str | Path) -> tuple[int, int, float]:
    """Cheap header probe returning ``(width, height, fps)`` without decoding."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"video file not found: {source}")
    return _probe_stream(source)


def probe_video(path: str | Path) -> VideoInfo:
    """Return fps / frame count / size. The frame count comes from a real decode."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"video file not found: {source}")
    width, height, fps = _probe_stream(source)
    return VideoInfo(
        fps=fps, num_frames=_count_frames(source), width=width, height=height
    )


def _probe_stream(source: Path) -> tuple[int, int, float]:
    result = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", str(source)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    text = result.stderr.decode("utf-8", errors="replace")
    video_line = next(
        (line for line in text.splitlines() if "Video:" in line and "Stream #" in line),
        None,
    )
    if video_line is None:
        raise ValueError(f"no decodable video stream in {source}")
    size_match = re.search(r",\s*(\d{2,5})x(\d{2,5})[,\s\[]", video_line)
    if size_match is None:
        raise ValueError(f"cannot read frame size of {source}")
    fps_match = re.search(r"([\d.]+)\s*fps", video_line) or re.search(
        r"([\d.]+)\s*tbr", video_line
    )
    if fps_match is None:
        raise ValueError(f"cannot read frame rate of {source}")
    fps = float(fps_match.group(1))
    if fps <= 0:
        raise ValueError(f"invalid frame rate in {source}")
    return int(size_match.group(1)), int(size_match.group(2)), fps


def _count_frames(source: Path) -> int:
    result = subprocess.run(
        [
            _ffmpeg_exe(),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot decode {source}: {_tail(result.stderr)}")
    counts = re.findall(rb"frame=\s*(\d+)", result.stderr)
    if not counts:
        raise ValueError(f"cannot count frames of {source}")
    return int(counts[-1])


def decode_video(path: str | Path, *, max_frames: int | None = None) -> np.ndarray:
    """Decode to a C-contiguous ``(T, H, W, 3)`` uint8 RGB array."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"video file not found: {source}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    width, height, _ = _probe_stream(source)
    frame_bytes = width * height * 3

    command = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
    ]
    if max_frames is not None:
        command += ["-frames:v", str(max_frames)]
    command.append("-")

    chunks: list[bytes] = []
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        assert process.stdout is not None
        try:
            while True:
                chunk = process.stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                chunks.append(chunk)
        finally:
            process.stdout.close()
            return_code = process.wait()
        if return_code != 0 and not chunks:
            stderr_file.seek(0)
            raise ValueError(f"cannot decode {source}: {_tail(stderr_file.read())}")
    if not chunks:
        raise ValueError(f"no frames decoded from {source}")
    frames = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    return np.ascontiguousarray(frames.reshape(len(chunks), height, width, 3))
