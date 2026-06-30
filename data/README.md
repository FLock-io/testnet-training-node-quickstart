# Data Directory

## Training Dataset

The official training dataset is published on HuggingFace and is loaded automatically by the training scripts:

```
random-sequence/flock-robotics-vla-training-v2
```

You do not need to download anything manually. Running the trainer without `--data` will pull the dataset directly:

```bash
python3 scripts/train_basic_vla.py --out outputs/basic_vla_policy
```

## Dataset Contents

103 successful demonstration trajectories across 13 tabletop manipulation tasks. Each row in the Parquet dataset is one timestep:

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | string | Unique trajectory identifier |
| `task` | string | Task name (e.g. `lift_cube`, `pick_place_can`) |
| `instruction` | string | Natural-language instruction for this episode |
| `difficulty` | string | `low`, `medium`, `hard`, or `very_high` |
| `seed` | int | Environment seed used during recording |
| `step` | int | Timestep index within the episode |
| `horizon` | int | Total episode length in steps |
| `image` | PIL Image | RGB observation frame, 96×96 pixels |
| `proprio` | float32[25] | Joint positions/velocities, end-effector pose, gripper state |
| `action` | float32[7] | Demonstrated action clipped to `[-1, 1]` |
| `reward` | float32 | Shaped task reward at this step |
| `done` | bool | Episode termination flag |

**Total timesteps:** 25,328 across 103 episodes.
**All trajectories are 100% successful** (task completed).

## Using a Local Zip (Optional)

If you have a local zip from FedLedger or a previous release, place it here and pass `--data`:

```bash
python3 scripts/train_basic_vla.py --data data/robotics_vla_training_traces.zip --out outputs/basic_vla_policy
```

The trainer will use the zip if it exists at the given path, otherwise falls back to the HF dataset automatically.
