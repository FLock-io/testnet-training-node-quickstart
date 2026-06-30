from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm


TEXT_DIM = 128
PROPRIO_DIM = 32
ACTION_DIM = 7

DEFAULT_HF_DATASET = "random-sequence/flock-robotics-vla-training-v2"


ADAPTER_SOURCE = r"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicVLAPolicyNet(nn.Module):
    def __init__(self, text_dim: int = 128, proprio_dim: int = 32, action_dim: int = 7):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.SiLU(),
        )
        self.proprio_encoder = nn.Sequential(nn.Linear(proprio_dim, 96), nn.SiLU())
        self.text_encoder = nn.Sequential(nn.Linear(text_dim, 96), nn.SiLU())
        self.action_head = nn.Sequential(
            nn.Linear(448, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, images, proprio, text_features):
        if images.ndim != 4:
            raise ValueError(f"expected image batch shaped [B,H,W,3] or [B,3,H,W], got {tuple(images.shape)}")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0
        if images.shape[-2:] != (96, 96):
            images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
        image_emb = self.image_encoder(images)
        proprio_emb = self.proprio_encoder(proprio.float())
        text_emb = self.text_encoder(text_features.float())
        return self.action_head(torch.cat([image_emb, proprio_emb, text_emb], dim=-1))


def _text_vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        vec[value % dim] += 1.0 if value & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _proprio_vector(obs: dict, dim: int) -> np.ndarray:
    raw = np.asarray(obs.get("proprio", np.zeros(25, dtype=np.float32)), dtype=np.float32).reshape(-1)
    step = float(obs.get("step", 0))
    horizon = float(obs.get("horizon", 320) or 320)
    step_feature = np.asarray([step / max(horizon, 1.0)], dtype=np.float32)
    combined = np.concatenate([raw, step_feature], axis=0)
    if combined.size < dim:
        combined = np.pad(combined, (0, dim - combined.size))
    return combined[:dim].astype(np.float32)


class BasicVLAPolicy:
    def __init__(self, model_dir: str, device: str, dtype: str):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "vla_config.json").read_text())
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = BasicVLAPolicyNet(
            text_dim=int(self.config.get("text_dim", 128)),
            proprio_dim=int(self.config.get("proprio_dim", 32)),
            action_dim=int(self.config.get("action_dim", 7)),
        ).to(self.device)
        checkpoint = torch.load(self.model_dir / "model.pt", map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs: dict) -> np.ndarray:
        image = np.asarray(obs.get("image", np.zeros((96, 96, 3), dtype=np.uint8)), dtype=np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        task = str(obs.get("task", ""))
        difficulty = str(obs.get("difficulty", ""))
        instruction = str(obs.get("instruction", ""))
        text = f"task {task} difficulty {difficulty} instruction {instruction}"
        text_features = _text_vector(text, int(self.config.get("text_dim", 128)))
        proprio = _proprio_vector(obs, int(self.config.get("proprio_dim", 32)))
        with torch.no_grad():
            action = self.model(
                torch.from_numpy(image).unsqueeze(0).to(self.device),
                torch.from_numpy(proprio).unsqueeze(0).to(self.device),
                torch.from_numpy(text_features).unsqueeze(0).to(self.device),
            )
        return np.clip(action.squeeze(0).detach().cpu().numpy(), -1.0, 1.0).astype(np.float32)


def load_policy(model_dir: str, device: str, dtype: str):
    return BasicVLAPolicy(model_dir=model_dir, device=device, dtype=dtype)
"""


class BasicVLAPolicyNet(nn.Module):
    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        proprio_dim: int = PROPRIO_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.SiLU(),
        )
        self.proprio_encoder = nn.Sequential(nn.Linear(proprio_dim, 96), nn.SiLU())
        self.text_encoder = nn.Sequential(nn.Linear(text_dim, 96), nn.SiLU())
        self.action_head = nn.Sequential(
            nn.Linear(448, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(
        self, images: torch.Tensor, proprio: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"expected image batch shaped [B,H,W,3] or [B,3,H,W], got {tuple(images.shape)}"
            )
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0
        if images.shape[-2:] != (96, 96):
            images = F.interpolate(
                images, size=(96, 96), mode="bilinear", align_corners=False
            )
        image_emb = self.image_encoder(images)
        proprio_emb = self.proprio_encoder(proprio.float())
        text_emb = self.text_encoder(text_features.float())
        return self.action_head(torch.cat([image_emb, proprio_emb, text_emb], dim=-1))


@dataclass(frozen=True)
class LoadedSamples:
    images: np.ndarray
    proprio: np.ndarray
    text_features: np.ndarray
    actions: np.ndarray
    manifest_summary: dict[str, Any]


def zip_member_by_suffix(zf: zipfile.ZipFile, suffix: str) -> str:
    suffix = suffix.lstrip("/")
    for name in zf.namelist():
        if name.endswith(suffix):
            return name
    raise FileNotFoundError(f"Could not find {suffix} in {zf.filename}")


def read_json_by_suffix(zf: zipfile.ZipFile, suffix: str) -> Any:
    return json.loads(zf.read(zip_member_by_suffix(zf, suffix)).decode("utf-8"))


def text_vector(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        vec[value % dim] += 1.0 if value & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def proprio_vector(
    raw: np.ndarray, step: int, horizon: int, dim: int = PROPRIO_DIM
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    step_feature = np.asarray(
        [float(step) / max(float(horizon), 1.0)], dtype=np.float32
    )
    combined = np.concatenate([raw, step_feature], axis=0)
    if combined.size < dim:
        combined = np.pad(combined, (0, dim - combined.size))
    return combined[:dim].astype(np.float32)


def load_samples_from_hf(
    dataset_id: str,
    split: str = "train",
    max_samples: int = 0,
    step_stride: int = 1,
    seed: int = 7,
    text_dim: int = TEXT_DIM,
    proprio_dim: int = PROPRIO_DIM,
) -> LoadedSamples:
    """Load training samples from a HuggingFace Parquet dataset.

    Each row is one timestep.  Rows are grouped by episode_id and
    step_stride is applied within each episode before shuffling.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HF loading: pip install datasets"
        ) from exc

    print(f"Loading training data from HuggingFace: {dataset_id}", flush=True)
    ds = hf_load_dataset(dataset_id, split=split)

    rng = random.Random(seed)
    episode_row_indices: dict[str, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        ep_id = str(ds[i]["episode_id"])
        episode_row_indices[ep_id].append(i)

    selected: list[int] = []
    for row_ids in episode_row_indices.values():
        selected.extend(row_ids[:: max(1, step_stride)])

    rng.shuffle(selected)
    if max_samples > 0:
        selected = selected[:max_samples]

    images: list[np.ndarray] = []
    proprio_list: list[np.ndarray] = []
    text_features_list: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    tasks: set[str] = set()
    difficulties: set[str] = set()

    subset = ds.select(selected)
    for row in tqdm(subset, desc="Loading from HF dataset", total=len(selected)):
        img = row["image"]
        if hasattr(img, "convert"):
            img_arr = np.array(img.convert("RGB"), dtype=np.uint8)
        else:
            img_arr = np.asarray(img, dtype=np.uint8)

        task_name = str(row.get("task", ""))
        difficulty_val = str(row.get("difficulty", "") or "")
        instruction = str(row.get("instruction", ""))
        step = int(row.get("step", 0))
        horizon = int(row.get("horizon", 200) or 200)

        text = f"task {task_name} difficulty {difficulty_val} instruction {instruction}"
        text_feat = text_vector(text, dim=text_dim)
        prop = np.asarray(row["proprio"], dtype=np.float32)
        act = np.asarray(row["action"], dtype=np.float32)

        images.append(img_arr)
        proprio_list.append(proprio_vector(prop, step, horizon, dim=proprio_dim))
        text_features_list.append(text_feat)
        actions.append(act)
        tasks.add(task_name)
        difficulties.add(difficulty_val)

    if not actions:
        raise ValueError("No training samples were loaded from the HF dataset.")

    manifest_summary = {
        "trajectory_count": len(episode_row_indices),
        "loaded_frame_count": len(actions),
        "step_stride": step_stride,
        "tasks": sorted(tasks),
        "difficulties": sorted(d for d in difficulties if d),
        "source": f"hf:{dataset_id}",
    }
    return LoadedSamples(
        images=np.stack(images, axis=0),
        proprio=np.stack(proprio_list, axis=0).astype(np.float32),
        text_features=np.stack(text_features_list, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        manifest_summary=manifest_summary,
    )


def load_samples_from_zip(
    zip_path: Path,
    max_samples: int,
    step_stride: int,
    seed: int,
    text_dim: int = TEXT_DIM,
    proprio_dim: int = PROPRIO_DIM,
) -> LoadedSamples:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Training zip not found: {zip_path}. "
            f"Pass --hf-dataset to load from HuggingFace instead, "
            f"or place a training zip at {zip_path} and pass --data."
        )
    rng = random.Random(seed)
    with zipfile.ZipFile(zip_path) as zf:
        manifest = read_json_by_suffix(zf, "dataset_manifest.json")
        trajectories = manifest["trajectories"]
        refs: list[tuple[int, int]] = []
        for entry_index, entry in enumerate(trajectories):
            length = int(entry["length"])
            refs.extend(
                (entry_index, step) for step in range(0, length, max(1, step_stride))
            )
        rng.shuffle(refs)
        if max_samples > 0:
            refs = refs[:max_samples]

        refs_by_entry: dict[int, list[int]] = defaultdict(list)
        for entry_index, step in refs:
            refs_by_entry[entry_index].append(step)

        images: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        text_features: list[np.ndarray] = []
        actions: list[np.ndarray] = []

        for entry_index, steps in tqdm(
            sorted(refs_by_entry.items()), desc="Loading trajectories"
        ):
            entry = trajectories[entry_index]
            meta = read_json_by_suffix(zf, entry["metadata"])
            episode_meta = meta.get("episode", {}) if isinstance(meta, dict) else {}
            trajectory_member = zip_member_by_suffix(zf, entry["trajectory_npz"])
            with np.load(io.BytesIO(zf.read(trajectory_member))) as npz:
                image_arr = npz["images"]
                proprio_arr = npz["proprio"]
                action_arr = npz["actions"]
                steps_arr = (
                    npz["steps"]
                    if "steps" in npz.files
                    else np.arange(len(action_arr), dtype=np.int32)
                )
                horizon = int(len(action_arr))
                text = (
                    f"task {episode_meta.get('task', entry.get('task', ''))} "
                    f"difficulty {episode_meta.get('difficulty', entry.get('difficulty', ''))} "
                    f"instruction {episode_meta.get('instruction', entry.get('instruction', ''))}"
                )
                text_feat = text_vector(text, dim=text_dim)
                for step in steps:
                    step = min(step, len(action_arr) - 1)
                    images.append(np.asarray(image_arr[step], dtype=np.uint8))
                    proprio.append(
                        proprio_vector(
                            proprio_arr[step],
                            int(steps_arr[step]),
                            horizon,
                            dim=proprio_dim,
                        )
                    )
                    text_features.append(text_feat)
                    actions.append(np.asarray(action_arr[step], dtype=np.float32))

    if not actions:
        raise ValueError(
            "No training samples were loaded. Check --step-stride and --max-samples."
        )

    manifest_summary = {
        "trajectory_count": int(manifest.get("trajectory_count", len(trajectories))),
        "loaded_frame_count": len(actions),
        "step_stride": step_stride,
        "tasks": sorted({entry["task"] for entry in trajectories}),
        "difficulties": sorted({entry["difficulty"] for entry in trajectories}),
    }
    return LoadedSamples(
        images=np.stack(images, axis=0),
        proprio=np.stack(proprio, axis=0).astype(np.float32),
        text_features=np.stack(text_features, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        manifest_summary=manifest_summary,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "CUDA was requested but is not available; falling back to CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(
            "MPS was requested but is not available; falling back to CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    return torch.device(requested)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, proprio, text_features, actions in loader:
            images = images.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            text_features = text_features.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            pred = model(images, proprio, text_features)
            # Accumulate validation metrics on CPU. This avoids backend-specific
            # reduction quirks on Apple MPS while keeping training on the GPU.
            diff = pred.detach().float().cpu() - actions.detach().float().cpu()
            total_loss += float(torch.square(diff).sum().item())
            total_count += int(actions.numel())
    model.train()
    return total_loss / max(total_count, 1)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = choose_device(args.device)

    zip_path = Path(args.data).expanduser()
    if zip_path.exists():
        samples = load_samples_from_zip(
            zip_path=zip_path,
            max_samples=args.max_samples,
            step_stride=args.step_stride,
            seed=args.seed,
            text_dim=args.text_dim,
            proprio_dim=args.proprio_dim,
        )
    else:
        print(
            f"Zip not found at {zip_path}; loading from HuggingFace ({args.hf_dataset}).",
            flush=True,
        )
        samples = load_samples_from_hf(
            dataset_id=args.hf_dataset,
            max_samples=args.max_samples,
            step_stride=args.step_stride,
            seed=args.seed,
            text_dim=args.text_dim,
            proprio_dim=args.proprio_dim,
        )

    dataset = TensorDataset(
        torch.from_numpy(samples.images),
        torch.from_numpy(samples.proprio),
        torch.from_numpy(samples.text_features),
        torch.from_numpy(samples.actions),
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_count = (
        max(1, int(len(indices) * args.val_fraction)) if len(indices) >= 20 else 0
    )
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_indices
        else None
    )

    model = BasicVLAPolicyNet(
        text_dim=args.text_dim,
        proprio_dim=args.proprio_dim,
        action_dim=ACTION_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()
    scaler_enabled = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    history: list[dict[str, float]] = []
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, proprio, text_features, actions in progress:
            images = images.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            text_features = text_features.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=scaler_enabled
            ):
                pred = model(images, proprio, text_features)
                loss = loss_fn(pred, actions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            batch_count = int(actions.shape[0])
            running_loss += float(loss.item()) * batch_count
            running_count += batch_count
            progress.set_postfix(train_mse=running_loss / max(running_count, 1))

        train_mse = running_loss / max(running_count, 1)
        val_mse = (
            evaluate(model, val_loader, device) if val_loader is not None else train_mse
        )
        val_is_finite = math.isfinite(float(val_mse))
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse) if val_is_finite else None,
            }
        )
        val_display = f"{val_mse:.6f}" if val_is_finite else "nonfinite"
        print(f"epoch={epoch} train_mse={train_mse:.6f} val_mse={val_display}")
        if val_is_finite and val_mse < best_val:
            best_val = val_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "robotics_vla_basic_policy_v1",
        "model_type": "basic_cnn_text_proprio_bc",
        "text_dim": args.text_dim,
        "proprio_dim": args.proprio_dim,
        "action_dim": ACTION_DIM,
        "image_size": [96, 96],
        "adapter_inputs": [
            "image",
            "proprio",
            "instruction",
            "task",
            "difficulty",
            "step",
            "horizon",
        ],
        "forbidden_adapter_inputs": [
            "raw_obs",
            "object_state",
            "sim_state",
            "mujoco_data",
        ],
        "training_data": str(Path(args.data).expanduser()),
    }
    torch.save(
        {"state_dict": model.state_dict(), "config": config}, out_dir / "model.pt"
    )
    (out_dir / "vla_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "flock_robotics_adapter.py").write_text(ADAPTER_SOURCE.strip() + "\n")
    report = {
        "samples": samples.manifest_summary,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "history": history,
        "best_val_mse": best_val if math.isfinite(float(best_val)) else None,
        "global_steps": global_step,
        "output_files": [
            "flock_robotics_adapter.py",
            "model.pt",
            "vla_config.json",
            "training_report.json",
        ],
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# Basic Robotics VLA Policy\n\n"
        "This folder was produced by `scripts/train_basic_vla.py` from the Robotics VLA training traces zip.\n\n"
        "It contains a submission-compatible `flock_robotics_adapter.py`, `model.pt`, and `vla_config.json`.\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a basic submission-compatible Robotics VLA policy from the training zip."
    )
    parser.add_argument(
        "--data",
        default="data/robotics_vla_training_traces.zip",
        help="Path to a local training zip (optional; ignored if the file does not exist)",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="HuggingFace dataset id used when --data zip is absent",
    )
    parser.add_argument(
        "--out", default="outputs/basic_vla_policy", help="Output model directory"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--step-stride",
        type=int,
        default=4,
        help="Use every Nth frame from each trajectory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12000,
        help="Maximum frame/action pairs to load; 0 means all sampled frames",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--text-dim", type=int, default=TEXT_DIM)
    parser.add_argument("--proprio-dim", type=int, default=PROPRIO_DIM)
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--amp", action="store_true", help="Use bfloat16 autocast on CUDA"
    )
    return parser.parse_args()


if __name__ == "__main__":
    training_report = train(parse_args())
    print(
        json.dumps(
            {
                "status": "ok",
                "output_files": training_report["output_files"],
                "best_val_mse": training_report["best_val_mse"],
            },
            indent=2,
        )
    )
