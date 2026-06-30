from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def member_by_suffix(zf: zipfile.ZipFile, suffix: str) -> str:
    suffix = suffix.lstrip("/")
    for name in zf.namelist():
        if name.endswith(suffix):
            return name
    raise FileNotFoundError(f"Could not find {suffix} in {zf.filename}")


def read_json(zf: zipfile.ZipFile, suffix: str):
    return json.loads(zf.read(member_by_suffix(zf, suffix)).decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the Robotics VLA training traces zip."
    )
    parser.add_argument("--data", default="data/robotics_vla_training_traces_200.zip")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    zip_path = Path(args.data).expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(f"Training zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        manifest = read_json(zf, "dataset_manifest.json")
        trajectories = manifest["trajectories"]
        task_counts = Counter(entry["task"] for entry in trajectories)
        difficulty_counts = Counter(entry["difficulty"] for entry in trajectories)

        print(f"zip: {zip_path}")
        print(f"schema_version: {manifest.get('schema_version')}")
        print(
            f"trajectory_count: {manifest.get('trajectory_count', len(trajectories))}"
        )
        print(f"tasks: {dict(sorted(task_counts.items()))}")
        print(f"difficulties: {dict(sorted(difficulty_counts.items()))}")
        print()

        for entry in trajectories[: max(0, args.examples)]:
            meta = read_json(zf, entry["metadata"])
            episode_meta = meta.get("episode", {}) if isinstance(meta, dict) else {}
            with np.load(
                io.BytesIO(zf.read(member_by_suffix(zf, entry["trajectory_npz"])))
            ) as npz:
                shapes = {key: tuple(npz[key].shape) for key in npz.files}
                dtypes = {key: str(npz[key].dtype) for key in npz.files}
            print(f"- {entry['episode_id']}")
            print(
                f"  task: {entry['task']} | difficulty: {entry['difficulty']} | success: {entry['success']}"
            )
            print(
                f"  instruction: {episode_meta.get('instruction', entry.get('instruction', ''))}"
            )
            print(f"  arrays: {shapes}")
            print(f"  dtypes: {dtypes}")


if __name__ == "__main__":
    main()
