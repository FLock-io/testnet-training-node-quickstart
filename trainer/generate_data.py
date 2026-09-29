"""Generate a labelled training set of edited clips.

    python generate_data.py --out-dir data/train --num-clips 300 --seed 0 --workers 4

Writes ``<out-dir>/videos/clip_XXXXX.mp4`` and ``<out-dir>/labels.jsonl``; each
line of the latter is::

    {"clip_file": "videos/clip_00000.mp4", "clip_seed": 1000000, "fps": 15.0,
     "num_frames": 120, "width": 320, "height": 240, "difficulty": "medium",
     "source": "procedural", "crf": 23, "issues": [<IssueLabel.model_dump()>, ...],
     "decoys": [<DecoyLabel.model_dump()>, ...]}

``decoys`` lists the legitimate, unlabelled events in the clip (scene cuts, exposure / white
balance drift, smooth zooms, objects entering / leaving, ...). They are never scored; they are
hard negatives: things that look like edits but are not.

The clips come from the validator's own synthesiser
(``validator.modules.video_inconsistency.synthesis``), so their statistics are close to the
validation clips. Optionally pass ``--footage-dir`` with real videos: a fraction of the clips
(``--footage-fraction``) is then edited from that footage, which is the single most useful
thing you can do to make a detector robust to content it has not seen.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

# The scripts in this folder are run from anywhere; make the repo root importable so we can use
# the validator's synthesiser / video I/O. (The *submission* never imports validator.*.)
def _find_repo_root(start: Path) -> Path:
    """The nearest ancestor holding ``validator/modules/video_inconsistency``.

    Works both inside the FLock-validator checkout and in the trainer quickstart repo,
    which vendors the needed validator modules at its root.
    """
    for candidate in (start, *start.parents):
        if (candidate / "validator" / "modules" / "video_inconsistency" / "issue_types.py").is_file():
            return candidate
    return start.parents[3] if len(start.parents) > 3 else start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validator.modules.video_inconsistency.synthesis import (  # noqa: E402
    SynthesisConfig,
    generate_clip,
)
from validator.modules.video_inconsistency.video_io import encode_video  # noqa: E402

# Training clip seeds start at 1_000_000 + run_seed * 100_000. The documented dev package
# (``build_package --seed 7`` etc.) and the validators' packages use small seeds (package clip
# seeds are hash-derived from the package seed, and the Hugging Face dataset builder derives its
# own from a separate namespace), so keeping training seeds far away from them means you never
# train on your own dev/validation clips (which would make local validation scores meaningless).
SEED_OFFSET = 1_000_000
SEED_STRIDE = 100_000
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def clip_seed(run_seed: int, index: int) -> int:
    return SEED_OFFSET + run_seed * SEED_STRIDE + index


def _render_one(job: tuple[int, int, str, dict[str, Any]]) -> dict[str, Any]:
    index, seed, out_dir, config_kwargs = job
    config = SynthesisConfig(**config_kwargs)
    clip = generate_clip(seed, config)
    relative = f"videos/clip_{index:05d}.mp4"
    # Encode to mp4 and train on the DECODED frames (train.py decodes the file): the validator
    # feeds the detector frames decoded from H.264, so codec artefacts (blocking, softened
    # edges, chroma bleeding) are part of what the detector sees. Training on raw synthetic
    # frames would give a model that has never seen them.
    # Each clip carries its own encode quality (``crf``) so the detector sees a spread of codec
    # damage, as it will in the validator's packages.
    crf = int(getattr(clip, "crf", 18))
    try:
        encode_video(clip.frames, Path(out_dir) / relative, clip.fps, crf=crf)
    except TypeError:  # an older video_io without the crf argument
        encode_video(clip.frames, Path(out_dir) / relative, clip.fps)
    return {
        "clip_file": relative,
        "clip_seed": seed,
        "fps": float(clip.fps),
        "num_frames": int(clip.frames.shape[0]),
        "width": int(clip.frames.shape[2]),
        "height": int(clip.frames.shape[1]),
        "difficulty": clip.difficulty,
        "source": clip.source,
        "crf": crf,
        "issues": [issue.model_dump() for issue in clip.issues],
        "decoys": [decoy.model_dump() for decoy in getattr(clip, "decoys", [])],
    }


def _footage_paths(footage_dir: str | None) -> tuple[str, ...]:
    if not footage_dir:
        return ()
    root = Path(footage_dir)
    return tuple(
        sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in _VIDEO_SUFFIXES)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-clips", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0, help="run seed; different seeds give disjoint clips")
    parser.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count())))
    parser.add_argument("--footage-dir", default=None, help="folder of real videos to edit (optional)")
    parser.add_argument("--footage-fraction", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args(argv)

    footage = _footage_paths(args.footage_dir)
    if args.footage_dir and not footage:
        parser.error(f"no video files found under {args.footage_dir}")
    config_kwargs: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "footage_paths": footage,
        "footage_fraction": args.footage_fraction if footage else 0.0,
    }

    out_dir = Path(args.out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    jobs = [
        (i, clip_seed(args.seed, i), str(out_dir), config_kwargs) for i in range(args.num_clips)
    ]
    started = time.time()
    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        results = map(_render_one, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(args.workers)
        results = pool.imap(_render_one, jobs, chunksize=1)
    try:
        for count, record in enumerate(results, start=1):
            records.append(record)
            if count % 25 == 0 or count == len(jobs):
                print(f"[generate] {count}/{len(jobs)} clips ({time.time() - started:.0f}s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    edited = sum(1 for r in records if r["issues"])
    decoys = sum(len(r["decoys"]) for r in records)
    print(f"[generate] wrote {len(records)} clips ({edited} edited, {decoys} decoys) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
