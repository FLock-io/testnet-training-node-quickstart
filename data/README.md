# Data Directory

Downloaded and generated training data goes here. Everything under `data/` except this file is
gitignored.

## Training dataset

The official training dataset is a Hugging Face dataset repo:

```
random-sequence/flock-video-inconsistency
```

It is public: <https://huggingface.co/datasets/random-sequence/flock-video-inconsistency>.
`full_automation.py` downloads it to `data/random-sequence__flock-video-inconsistency/`. If the
download fails (for example, when offline), the script generates clips locally with the same synthesiser instead.

| Split | Clips | Clean | Size |
|-------|-------|-------|------|
| `train/` | 8000 | 1611 | 1.43 GB |
| `validation/` | 1000 | 190 | 0.18 GB |
| `dev_package/video_inconsistency_dev_package.zip` | 200 | 46 | 34 MB |

Each split is a Hugging Face `videofolder`: `<clip_id>.mp4` files plus a `metadata.jsonl` with one
row per clip:

| Field | Type | Description |
|-------|------|-------------|
| `file_name` | string | The mp4 next to the metadata file (H.264, yuv420p, 320x240, 15 fps) |
| `clip_id` | string | 12 hex characters; carries no label information |
| `fps`, `num_frames`, `width`, `height`, `duration` | numbers | Video geometry; `duration = num_frames / fps` |
| `difficulty` | string | `easy`, `medium`, `hard` or `expert` |
| `source` | string | `procedural` (all official clips are synthetic) |
| `crf` | int | H.264 quality the clip was encoded at (18 to 28) |
| `issues` | list | Ground truth: `type`, `start_time`, `end_time`, `start_frame`, `end_frame` (exclusive), `bbox` (normalised, spatial types only), `params` (edit magnitude) |
| `decoys` | list | Legitimate look-alike events (`type`, `start_time`, `end_time`). Never scored; use them as hard negatives |

`stats.json` holds per-split counts of issues per type, decoys per type and difficulty tiers.
`issue_types.json` is the canonical catalogue.

The dev package is a standard validation package (`manifest.json` + `videos/`). Pass it to the
validator's `--local-validation` to score a submission exactly as validators do. The validators'
own evaluation set comes from the same generator with a different, secret seed.

## Generating more data

```bash
python trainer/generate_data.py --out-dir data/train_extra --num-clips 5000 --seed 1 --workers 8
python trainer/train.py --data-dir data/train_extra --out-dir runs/extra --device cuda
```

Different `--seed` values give disjoint clips. `--footage-dir <folder of videos> --footage-fraction 0.5`
edits real footage as well as procedural scenes, for robustness.
