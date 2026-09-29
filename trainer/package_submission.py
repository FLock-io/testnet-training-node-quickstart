"""Assemble (and optionally smoke-test / upload) a submission folder.

    python package_submission.py --weights-dir runs/v1 --out-dir my_submission --smoke-test
    HF_TOKEN=hf_xxx python package_submission.py --out-dir my_submission --push-to-hub you/my-vic-detector

The submission folder is a self-contained repo: the adapter, its helper modules, the trained
weights and config, and a README. Nothing in it may need the network or the validator code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    """The nearest ancestor holding ``validator/modules/video_inconsistency``.

    Works both inside the FLock-validator checkout and in the trainer quickstart repo,
    which vendors the needed validator modules at its root.
    """
    for candidate in (start, *start.parents):
        if (candidate / "validator" / "modules" / "video_inconsistency" / "issue_types.py").is_file():
            return candidate
    return start.parents[3] if len(start.parents) > 3 else start


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_HERE)

ADAPTER_FILENAME = "flock_video_adapter.py"
CODE_FILES = (ADAPTER_FILENAME, "vic_features.py", "vic_model.py", "vic_localize.py")
WEIGHT_FILES = ("weights.safetensors", "vic_config.json")

# Runs in a fresh interpreter with a scrubbed environment: this mimics the sandbox closely
# enough to catch the classic mistakes (an import that only works from the dev machine, a helper
# module left out of the folder, a path that is not relative to the adapter).
_SMOKE_SCRIPT = r"""
import importlib.util, json, sys
import numpy as np
adapter_path, frames_path, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
spec = importlib.util.spec_from_file_location("flock_video_adapter", adapter_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frames = np.load(frames_path, mmap_mode="r")
detector = module.load_detector(sys.argv[4], "cpu", "float32")
result = detector.detect({
    "frames": frames, "frames_path": frames_path, "video_path": "", "fps": fps,
    "num_frames": int(frames.shape[0]), "width": int(frames.shape[2]), "height": int(frames.shape[1]),
    "duration": frames.shape[0] / fps, "issue_types": [],
})
print("RESULT:" + json.dumps(result))
"""


def build_submission(weights_dir: Path | None, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(_HERE / name, out_dir / name)
    config: dict = {}
    if weights_dir is not None:
        for name in WEIGHT_FILES:
            source = weights_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"{source} not found; run train.py first")
            shutil.copy2(source, out_dir / name)
        config = json.loads((out_dir / "vic_config.json").read_text(encoding="utf-8"))
    (out_dir / "README.md").write_text(_submission_readme(config), encoding="utf-8")
    return out_dir


def _submission_readme(config: dict) -> str:
    if config:
        train = config.get("train", {})
        validation = config.get("validation", {})
        model_line = (
            f"Trained temporal conv net (`TemporalIssueNet`, hidden={config['model']['hidden']}) on "
            f"{train.get('num_train_clips', '?')} clips; validation mean AP "
            f"{validation.get('mean_ap', float('nan')):.3f}, macro F1 "
            f"{validation.get('macro_f1', float('nan')):.3f}."
        )
    else:
        model_line = "No trained weights: this submission runs the rule-based heuristic detector."
    return (
        "# video_inconsistency submission\n\n"
        f"{model_line}\n\n"
        "Detector: per-frame features (`vic_features.py`) -> temporal conv net (`vic_model.py`) -> "
        "interval decoding; boxes for spatial issues from `vic_localize.py`. Entry point: "
        f"`{ADAPTER_FILENAME}` (`load_detector` / `Detector.detect`).\n\n"
        "The repo is fully self-contained (no network access, no validator imports); only "
        "numpy, torch and safetensors are needed.\n"
    )


def smoke_test(out_dir: Path) -> None:
    """Run the packaged adapter in a clean subprocess on one synthetic clip and validate its output."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import numpy as np

    fps = 15.0
    try:
        from validator.modules.video_inconsistency.synthesis import generate_clip

        # Force two edits (one spatial) so the feature, decoding and box paths are all exercised.
        clip = generate_clip(424242, issue_types=["frozen_frames", "inserted_object"])
        frames, fps = clip.frames, float(clip.fps)
        print(f"[smoke] synthetic clip: {frames.shape[0]} frames, {len(clip.issues)} injected issue(s)")
    except ImportError:
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (60, 120, 160, 3), dtype=np.uint8)
        print("[smoke] synthesiser unavailable, using random frames")

    with tempfile.TemporaryDirectory() as tmp:
        frames_path = Path(tmp) / "frames.npy"
        np.save(frames_path, frames)
        # Scrubbed environment: no inherited PYTHONPATH, no HF cache, no tokens.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp,
            "TMPDIR": tmp,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",  # keep __pycache__ out of the folder that gets uploaded
        }
        adapter = (out_dir / ADAPTER_FILENAME).resolve()
        completed = subprocess.run(
            [sys.executable, "-c", _SMOKE_SCRIPT, str(adapter), str(frames_path), str(fps), str(out_dir.resolve())],
            env=env, cwd=tmp, capture_output=True, text=True, timeout=300,
        )
    if completed.returncode != 0:
        raise SystemExit(f"[smoke] adapter failed:\n{completed.stderr[-3000:]}")
    line = next((ln for ln in completed.stdout.splitlines() if ln.startswith("RESULT:")), None)
    if line is None:
        raise SystemExit(f"[smoke] adapter printed no result:\n{completed.stdout[-2000:]}")
    raw = json.loads(line[len("RESULT:"):])
    duration = frames.shape[0] / fps
    try:
        from validator.modules.video_inconsistency.predictions import parse_detector_output
    except ImportError:
        print("[smoke] validator parser unavailable; output not schema-checked")
        print(f"[smoke] adapter returned {len(raw.get('issues', []))} issue(s)")
        return
    parsed = parse_detector_output(raw, duration)
    print(f"[smoke] OK: output is schema-valid, {len(parsed)} issue(s):")
    for issue in parsed:
        print(f"    {issue.type:18s} {issue.start_time:5.2f}-{issue.end_time:5.2f}s conf {issue.confidence:.2f}")


def push_to_hub(out_dir: Path, repo_id: str) -> None:
    """Create a PRIVATE repo and upload the folder. The token comes only from $HF_TOKEN."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set the HF_TOKEN environment variable (never pass a token on the command line)")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="model").private:
        raise SystemExit(f"{repo_id} already exists and is PUBLIC; make it private or pick another name")
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(out_dir),
        ignore_patterns=["__pycache__", "*.pyc"],
    )
    print(f"[hub] uploaded {out_dir} to {repo_id}; the repo is PRIVATE (created/kept private=True)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights-dir", default=None, help="train.py output dir (omit for the heuristic detector)")
    parser.add_argument("--out-dir", required=True, help="submission folder to create")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--push-to-hub", metavar="REPO_ID", default=None)
    args = parser.parse_args(argv)

    out_dir = build_submission(Path(args.weights_dir) if args.weights_dir else None, Path(args.out_dir))
    print(f"[package] wrote {out_dir}: {sorted(p.name for p in out_dir.iterdir())}")
    if args.smoke_test:
        smoke_test(out_dir)
    if args.push_to_hub:
        push_to_hub(out_dir, args.push_to_hub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
