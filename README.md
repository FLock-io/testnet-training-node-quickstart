# Video Inconsistency Training Node Quickstart

This quickstart shows how to train and submit a detector for the [FLock AI Arena](https://flock.io) **Video Inconsistency** task.

Validators hold short videos (320x240 at 15 fps, 6 to 10 s). Each one is a continuous shot into which 0 to 4 known edits were injected: frozen or dropped frames, reversed, mirrored or spliced segments, colour and exposure jumps, zoom jumps, inserted objects and blurred regions. The same clips also contain legitimate look-alike events (decoys). Your detector gets one video at a time and returns a ranked list of the inconsistencies it finds, each with its type, time span, confidence and, for two types, a bounding box. You can submit **any** code and model, in any framework, behind a small adapter file. The validator runs it in a sandbox and scores it against hidden labels.

This repository contains a complete, working baseline: a data generator, a feature extractor, a small temporal model, a trainer with threshold calibration, the adapter the validator loads, and packaging, upload and submission scripts. The baseline scores about 0.57 to 0.65 on the dev package; the ceiling is 1.0.

---

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

export TASK_ID=<task id>  FLOCK_API_KEY=<your FLock API key>
export HF_USERNAME=<your HF username>  HF_TOKEN=<HF token with write access>
python3 full_automation.py
```

`full_automation.py` does the following:

1. Fetches the task.
2. Downloads the training dataset. If you cannot access the dataset, it generates 3000 clips locally with the same generator instead.
3. Trains the baseline and calibrates its thresholds against the validator's scorer.
4. Packages the submission and smoke-tests it in a clean subprocess.
5. Uploads it to a **private** Hugging Face model repo, `<HF_USERNAME>/video-inconsistency-task-<TASK_ID>`.
6. Submits the repo and its commit revision to FLock.

The same flow runs in Docker: `docker build -t vic-trainer . && docker run --gpus all -e TASK_ID -e FLOCK_API_KEY -e HF_USERNAME -e HF_TOKEN vic-trainer`.

Optional environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_REPO_ID` | `<HF_USERNAME>/video-inconsistency-task-<TASK_ID>` | Model repo to upload to (always created private) |
| `VIC_DATASET` | from the task, else `random-sequence/flock-video-inconsistency` | Hugging Face dataset repo to train on |
| `VIC_GENERATE_CLIPS` | 3000 | Clips to generate when no dataset is accessible |
| `VIC_EPOCHS` | 40 | Training epochs |
| `VIC_HIDDEN` / `VIC_LAYERS` | 64 / 6 | Width and depth of the temporal net (192 / 8 is the stronger reference) |
| `VIC_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `VIC_WORKERS` | all cores | CPU processes for data generation and feature extraction |
| `VIC_SKIP_SUBMIT` | unset | `1` trains, packages and uploads without submitting |

---

## Setup

Use Python 3.10 or 3.11. A CUDA GPU speeds up training, but the default model also trains on a CPU in minutes. Feature extraction runs on CPU processes, so more cores help.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If your platform needs a specific PyTorch CUDA wheel, install PyTorch with the official command for your CUDA version first, then run `pip install -r requirements.txt`. Video encoding and decoding use the ffmpeg binary bundled with `imageio-ffmpeg`, so you do not need a system ffmpeg.

## Repository layout

| Path | Purpose |
|------|---------|
| `full_automation.py` | One command: data, then train, package, upload and submit |
| `trainer/generate_data.py` | Generates labelled training clips with the validator's synthesiser |
| `trainer/train.py` | Trains `TemporalIssueNet`, calibrates per-type thresholds, prints mean AP, per-type AP and F1 |
| `trainer/package_submission.py` | Builds the submission folder, smoke-tests it, optionally pushes it to a private HF repo |
| `trainer/flock_video_adapter.py` | **The adapter the validator loads.** Copied into your submission root |
| `trainer/vic_features.py`, `trainer/vic_model.py`, `trainer/vic_localize.py` | Features, model and decoding, and box estimation. Imported by the adapter, so they ship with the submission |
| `validator/modules/video_inconsistency/` | Vendored, unmodified validator code: generator, video I/O, output parser, scorer (see `validator/VENDORED.md`). Used for training only, never by a submission |
| `utils/flock_api.py`, `utils/gpu_utils.py` | FLock task lookup and submission |
| `data/`, `outputs/` | Downloaded or generated data and training outputs (gitignored) |

---

## The task

Validators hold a package of short videos. Each one is a continuous shot into which 0 to 4 known inconsistencies were injected (about 20% of clips are clean). Your detector receives one video at a time and returns a ranked list of the inconsistencies it finds, with times, confidences and, for two of the types, a bounding box. Detections are scored against hidden ground truth.

Two things make the task hard on purpose:

- **Decoys.** Clips also contain legitimate, unlabelled events that look like edits but are not, about 1.1 per clip. There are 14 kinds, most of them a deliberate look-alike of one issue type:
  - a real scene cut, as opposed to `spliced_footage`, which returns to the original shot;
  - lighting flicker and auto-exposure steps, as opposed to `exposure_flicker`;
  - auto-white-balance steps and slow colour drift, as opposed to `color_grade_jump`;
  - smooth or fast zooms that never snap back, as opposed to `zoom_jump`;
  - the camera changing direction or speed, as opposed to `reversed_segment` and `dropped_frames`;
  - the camera or an object coming to rest, as opposed to `frozen_frames`;
  - objects entering or leaving the frame, as opposed to `inserted_object`.

  Decoys are never scored and never shown to the detector; reporting one is a false positive. Their labels are in the training data (`decoys`), so you can use them as hard negatives.
- **Degradation.** After editing, clips are degraded (sensor noise, optional blur or sharpening and down-up rescaling, and a per-clip H.264 quality `crf`), and every edit is still visible to a careful human looking at the frames.

### The 10 issue types

| Type | What to look for | Labelled as |
|------|------------------|-------------|
| `frozen_frames` | The picture stops moving for a run of frames (identical consecutive frames), then motion resumes with a jump. | span |
| `dropped_frames` | Frames were cut out mid-shot: objects and camera jump forward between two adjacent frames. | point event at the cut |
| `reversed_segment` | A stretch plays backwards (motion runs in reverse), then snaps forward again. | span |
| `spliced_footage` | A few frames from an unrelated shot appear in the middle of the shot: colours and content change completely, then return. | span |
| `color_grade_jump` | The colour grade / white balance shifts abruptly for a stretch, then jumps back. Structure is unchanged. | span |
| `exposure_flicker` | One to three frames are much brighter or darker than their neighbours. | span (very short) |
| `mirrored_segment` | A stretch is flipped left-right, so the scene layout swaps sides, then flips back. | span |
| `zoom_jump` | A stretch is abruptly punched in (cropped and scaled up), then snaps back to the original framing. | span |
| `inserted_object` | A foreign shape is pasted in; it pops in and out abruptly and does not interact with the scene. | span + bbox |
| `blurred_region` | A rectangle is blurred or pixelated for a stretch, as if censored or retouched. | span + bbox |

Difficulty comes in four tiers. "easy" clips have one long, strong edit; "medium" 1 to 2; "hard" 2 to 3 short, subtle edits; "expert" 2 to 4 of the shortest and subtlest edits, together with the most decoys. Clips are weighted easy 1.0, medium 1.5, hard 2.0, expert 2.5 in the score.

### Time convention

Frame `i` covers `[i / fps, (i + 1) / fps)`.

- A span over frames `s..e` inclusive has `start_time = s / fps` and `end_time = (e + 1) / fps`.
- A point event (`dropped_frames`) at the cut before frame `k` has `start_time == end_time == k / fps`.
- Bounding boxes are normalised `[x0, y0, x1, y1]` in `[0, 1]` with `x0 < x1`, `y0 < y1`. For a spatial issue the ground-truth box is the union of the affected region over the whole span.

---

---

## The adapter contract

Your repo root must contain `flock_video_adapter.py` defining:

```python
def load_detector(model_dir: str, device: str, dtype: str) -> Detector: ...

class Detector:                       # any object with this method
    def detect(self, video: dict) -> dict | list: ...
```

`load_detector` is called once (`model_dir` is your repo, `device` is `"cpu"` or `"cuda"`, `dtype` e.g. `"float32"` or `"bfloat16"`). `detect` is called once per clip. Validators run with `device="cuda"` and `dtype="bfloat16"` by default. An operator may run on CPU instead, so always honour the `device` argument.

### The `video` dict

| Key | Type | Meaning |
|-----|------|---------|
| `frames` | `np.ndarray` (T, H, W, 3) uint8 RGB | Decoded frames, a **read-only memmap** (copy before modifying) |
| `frames_path` | `str` | Path to the same frames as a `.npy` file |
| `video_path` | `str` | Path to the original `.mp4` (decode it yourself if you prefer) |
| `fps` | `float` | Frames per second |
| `num_frames` | `int` | T |
| `width`, `height` | `int` | Frame size |
| `duration` | `float` | `num_frames / fps` |
| `issue_types` | `list[str]` | The canonical issue type names |

### Return value

`{"issues": [...]}` or a bare list. Each issue:

```json
{"type": "zoom_jump", "start_time": 2.0, "end_time": 3.4,
 "confidence": 0.87,
 "bbox": [0.10, 0.20, 0.50, 0.60],
 "description": "free text, not scored"}
```

Full example for one clip with a freeze, a cut and an inserted object:

```json
{"issues": [
  {"type": "frozen_frames",   "start_time": 1.20, "end_time": 2.07, "confidence": 0.96},
  {"type": "dropped_frames",  "start_time": 4.00, "end_time": 4.00, "confidence": 0.81},
  {"type": "inserted_object", "start_time": 5.13, "end_time": 7.00, "confidence": 0.62,
   "bbox": [0.55, 0.10, 0.72, 0.31], "description": "bright blob pops in"}
]}
```

Rules: `type` must be one of the ten names; times are finite numbers in seconds and are clamped into `[0, duration]`; `confidence` is optional (default 1.0) and must be in `[0, 1]`; `bbox` is optional and must be valid if present; `description` is at most 500 characters and ignored; unknown keys are ignored; at most 1000 items. Anything else marks that clip's output invalid (it is retried, then scored as an empty answer; too many such clips invalidates the submission).

---

---

## Dataset

A ready-made training set is published as a Hugging Face **dataset repo**, `random-sequence/flock-video-inconsistency` (private: ask the task organisers for access, then log in with `huggingface-cli login` or set `HF_TOKEN`). It was produced by the validator's own synthesiser (`build_hf_dataset.py`), with labels for both issues and decoys, and comes in the Hugging Face `videofolder` layout:

```
README.md                dataset card (issue types, decoys, tiers, counts, scoring)
issue_types.json         canonical issue and decoy catalogue
train/metadata.jsonl     one row per clip (file_name, clip_id, fps, num_frames, width, height,
train/<clip_id>.mp4      duration, difficulty, source, crf, issues[], decoys[])
validation/metadata.jsonl
validation/<clip_id>.mp4
dev_package/video_inconsistency_dev_package.zip   package for local validation (see [Local validation](#local-validation))
stats.json               clip / issue / decoy / difficulty counts per split
```

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download("random-sequence/flock-video-inconsistency", repo_type="dataset"))
rows = [json.loads(line) for line in (root / "train" / "metadata.jsonl").read_text().splitlines()]
print(rows[0]["difficulty"], rows[0]["issues"], rows[0]["decoys"])
```

`full_automation.py` downloads it for you, and `trainer/train.py` reads this layout directly. It is detected automatically (`metadata.jsonl` with `file_name`, or its own `labels.jsonl`):

```bash
python trainer/train.py --data-dir "$ROOT/train" --val-dir "$ROOT/validation" --out-dir runs/hf --device cuda
python trainer/train.py --data-dir "$ROOT"       --out-dir runs/hf --device cuda   # same thing: train/ + validation/
```

The `dev_package/` zip is the file to pass as `--validation-data-url` for local validation. The validator's own evaluation set is generated with a different, secret seed: it shares the generator and the label semantics with this dataset but no clip.

---

---

## What the validator sandbox allows

Your code runs in a separate, locked-down process. Design for these rules from the start:

- **No network.** Nothing can be downloaded at runtime: not weights, not tokenizers, not base models, not `pip install`. Everything you use must be inside your repo. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set, so `from_pretrained("some/hub-id")` fails; use `from_pretrained(<path inside your repo>)`. If you fine-tune a pretrained model, save its full weights (or LoRA weights plus the base) into your repo.
- **Only the module environment is importable.** Python 3.10 or 3.11 with: `numpy`, `scipy`, `scikit-learn`, `pillow`, `opencv-python-headless`, `av`, `imageio-ffmpeg`, `torch`, `torchvision`, `transformers`, `timm`, `einops`, `safetensors`, `accelerate`, `peft`, `onnxruntime`, `huggingface-hub`. Anything else must be **vendored** into your repo as pure-Python source (the baseline does exactly that with `trainer/vic_*.py`). Native wheels cannot be vendored.
- **Do not import `validator.*`.** The validator's code is not readable inside the sandbox. Import your own helper files as top-level modules (`import vic_features`); the repo root is on `sys.path`. Keep the adapter importable without side effects.
- **Memory: 18 GiB by default, GPU plus host RAM together.** GPU memory is capped through the CUDA allocator fraction and host RSS is monitored; exceeding the ceiling ends the evaluation. Parameter count is only reported as telemetry.
- **GPU.** Detectors run on a CUDA GPU by default (`device="cuda"`). Honour the `device` argument, so the same submission also works on a CPU-only validator.
- **Time.** `load_detector` (including the adapter import) must finish within 600 s. Each `detect` call has a **60 s wall-time limit**. A clip is about 6 to 10 s at 15 fps (about 100 to 150 frames at 320x240), but do not hard-code that: hidden packages may use other sizes or real footage. There is also a cumulative CPU-time budget for the whole run (7200 s by default), so do not burn CPU on many threads for no gain.
- **Read-only filesystem.** Your repo and system libraries are read-only. The only writable place is the private scratch directory (`$TMPDIR`). Do not write caches next to your code.
- **No secrets, no labels.** No tokens or credentials are visible, and the worker sees only pixels, fps and the canonical type list: no clip id, difficulty or label.

If `detect` raises, or returns something invalid, that clip is retried and then counts as a failed clip. If more than 25% of clips fail, the whole submission is invalid and scores 0. A crash, timeout or memory breach ends the evaluation immediately.

---

---

## How you are scored

The headline term is a rank-based **mean average precision (mAP)**, so what matters is a good *ranked list* of candidates, not a hard yes/no per detection.

1. **Ranking and cap.** Per clip, your predictions are sorted by confidence and only the top 25 are kept. Low-confidence guesses are **not** discarded for the mAP: they sit at the bottom of the ranking and cost almost nothing, while a well-ordered ranking is rewarded. Confidence must therefore be an honest ordering (and, for F1 / clip accuracy below, calibrated around 0.5).
2. **Matching.** For each issue type and for each temporal-IoU threshold in **0.3, 0.4, 0.5, 0.6, 0.7**, the type's predictions from all clips are pooled and walked in decreasing confidence; each prediction takes the still-unmatched ground truth of the same type in the same clip with the best tIoU. It is a true positive when tIoU is at least the threshold, and for the spatial types (`inserted_object`, `blurred_region`) **also** when the predicted box has **bbox IoU >= 0.3** with the truth (no box means never a match). Any interval shorter than **0.4 s** (ground truth or prediction) is first widened to 0.4 s about its centre. A duplicate of an already matched issue is a false positive. Every increment is weighted by the clip's difficulty (easy 1.0, medium 1.5, hard 2.0, expert 2.5).
3. **mean AP** (weight **0.75**): the area under the precision-recall envelope of each (type, threshold), averaged over thresholds and types. Because the average includes tIoU 0.7, **precise boundaries matter**. A type with no ground truth in the package but with predictions scores AP 0 (hallucination).
4. **Localization**, weight **0.20**: over matched issues (tIoU 0.3, confidence >= 0.5), the sum of the temporal IoU (or `0.5 * tIoU + 0.5 * bbox IoU` for spatial types), divided by the number of ground-truth issues **plus the number of confident false positives**. Every confident false alarm lowers this term.
5. **Clip-level balanced accuracy**, weight **0.05**: "does this clip contain an issue?", predicted positive iff some prediction has confidence >= 0.5; averaged over edited and clean clips.

`score = 0.75 * mean_AP + 0.20 * localization + 0.05 * clip_accuracy`; `loss = 1 - score`. F1 is still reported (`macro_f1`, per-type F1 at tIoU 0.3 for predictions with confidence >= 0.5) but no longer part of the score.

What this means for a trainer:

- **Output ranked candidates with honest confidences.** Return every plausible detection (up to 25 per clip) with a confidence that orders them by how likely they are to be real. `vic_model.decode_intervals` does this: confident runs get confidence >= 0.5, weaker candidates get 0.05 to 0.5.
- **Boundaries matter.** The mAP averages over tIoU up to 0.7, so an interval that is right but ragged loses a lot; a 0.4 s event tolerates about 0.1 s of error at 0.7.
- **Spatial types need a box.** No box, or a box with IoU below 0.3, is a miss however good the timing is.
- **Do not report decoys.** They rank as false positives and lower the precision of everything below them.
- **Do not predict types that do not occur.** A type with predictions but no ground truth scores AP 0 (this cannot happen in the private set: every type occurs).

---

## Step by step

Use this path when you want to control each stage, or when you bring your own model.

```bash
# 1. Training data: the Hugging Face dataset (8000 training clips) ...
python -c "from huggingface_hub import snapshot_download as d; print(d('random-sequence/flock-video-inconsistency', repo_type='dataset', local_dir='data/hf'))"
ROOT=data/hf
#    ... or generate your own (about 1.5 CPU-seconds per clip; 900 clips take about 3 minutes on 8 cores)
python trainer/generate_data.py --out-dir data/train --num-clips 900 --seed 0 --workers 8

# 2. Train. Prints mean AP, per-type AP and F1 on the validation split
python trainer/train.py --data-dir "$ROOT" --out-dir runs/v1 --device cuda --workers 8
#    stronger reference model:
python trainer/train.py --data-dir "$ROOT" --out-dir runs/big --hidden 192 --layers 8 --epochs 80 --patience 16 --device cuda

# 3. Package the submission folder and smoke-test it in a clean subprocess
python trainer/package_submission.py --weights-dir runs/v1 --out-dir my_submission --smoke-test

# 4. Upload to the Hugging Face Hub as a PRIVATE repo (the token is read from HF_TOKEN only)
python trainer/package_submission.py --out-dir my_submission --push-to-hub <your-user>/my-vic-detector
```

Submit the repo id and its commit revision to the task, as `full_automation.py` does (`utils/flock_api.submit_task`).

### Bring your own model

A submission is just a folder, or HF repo, with `flock_video_adapter.py` at the root plus whatever code and weights it needs. Replace `trainer/flock_video_adapter.py` and the `vic_*.py` helpers with your own detector, keep the `load_detector(model_dir, device, dtype)` and `detect(video)` contract below, and keep every file your adapter imports inside the submission folder. `package_submission.py --smoke-test` still works as a quick check of the output format.

### Local validation

To score a submission exactly as validators do, with the same sandbox and scorer, use the validator repository and the dev package that ships with the dataset:

```bash
git clone -b feat/video-inconsistency-task https://github.com/FLock-io/FLock-validator.git
cd FLock-validator && pip install -r requirements.txt
python run.py video_inconsistency --local-validation \
    --hf-model-repo /path/to/my_submission \
    --validation-data-url /path/to/data/hf/dev_package/video_inconsistency_dev_package.zip
```

`run.py` creates the validator's conda environment on first use. Add `--max-clips 10` for a quick check and `--output-json out.json` to save the metrics (`score`, `mean_ap`, `per_type_ap`, ...). If `detect` raises or returns invalid output on too many clips, the result shows `invalid_submission: true` and a `diagnostics.failure_mode`; see [Troubleshooting](#troubleshooting).

Training options worth knowing (`python trainer/train.py --help`):

| Option | Meaning |
|--------|---------|
| `--data-dir`, `--val-dir` | A `generate_data.py` folder, a Hugging Face split folder or the dataset root. With `--val-dir` (or a dataset root that has `validation/`) that split is the validation set; otherwise `--val-fraction` of the training clips is held out |
| `--hidden`, `--layers` | Width and depth of the temporal net (`--layers 6` is the 127-frame receptive field of the baseline; more layers repeat the dilation cycle 1..32) |
| `--device cuda` | Training on a GPU (batched inference for validation too). Feature extraction is numpy and runs on CPU processes: give it `--workers N` |
| `--decoy-weight` | Loss multiplier (default 2) for frames on and around decoys: hard negatives |
| `--calib-f1-weight` | Weight of macro F1 added to the validator score when tuning thresholds (default 0.1; 0 = pure validator score) |
| `--calib-max-clips` | How many validation clips the threshold search uses (it runs the box localiser for spatial candidates, which is the slow part) |
| `--cache-dir` | Where per-clip features are cached (default `<data-dir>/feature_cache`); re-running is fast |

### Baseline numbers (what to expect)

Measured with the suite `video_inconsistency_v2` generator on the 200-clip dev package
(`build_package --num-clips 200 --seed 7`: 23 easy / 39 medium / 79 hard / 59 expert clips,
about 20% clean, about 1.1 decoys per clip). The trained models used `generate_data.py` clips with
default settings and `train.py` defaults, except for the size flags shown. The big model was
cross-checked through the full sandboxed validator (`--local-validation` on Linux), which gave
the same score to three decimals.

| Detector | score | mean AP | macro F1 | localization | clip accuracy | easy / medium / hard / expert score |
|----------|-------|---------|----------|--------------|---------------|-------------------------------------|
| No detections | 0.03 | 0.00 | 0.00 | 0.00 | 0.50 | 0.03 / 0.03 / 0.03 / 0.03 |
| Heuristic (no training) | 0.26 | 0.26 | 0.29 | 0.16 | 0.73 | 0.56 / 0.56 / 0.27 / 0.19 |
| Trained on 2000 clips (100k parameters) | 0.57 | 0.56 | 0.58 | 0.52 | 0.93 | 0.64 / 0.72 / 0.61 / 0.50 |
| Trained on 10000 clips, `--hidden 192 --layers 8` (0.9M parameters) | 0.65 | 0.64 | 0.66 | 0.62 | 0.96 | 0.80 / 0.77 / 0.70 / 0.58 |

Per-type AP of the 10k-clip model on the dev package:

| Type | AP | Type | AP |
|---|---|---|---|
| exposure_flicker | 1.00 | spliced_footage | 0.62 |
| mirrored_segment | 1.00 | frozen_frames | 0.27 |
| reversed_segment | 0.98 | inserted_object | 0.00 |
| zoom_jump | 0.91 | blurred_region | 0.00 |
| color_grade_jump | 0.88 | | |
| dropped_frames | 0.70 | | |

Where the headroom is:
- **The spatial types.** `inserted_object` and `blurred_region` score 0 for this baseline. Its
  handcrafted boxes almost never reach IoU 0.3, and a spatial detection without a matching box is
  never a hit. Solving them is worth up to 0.15 of score on its own.
- **Frozen frames and splices at the hard and expert tiers.** Two held frames, or a 2 to 3 frame
  splice from a similar scene, under sensor noise and compression, next to decoys such as a
  camera that stops or a legitimate scene cut. Per-frame summary features cannot resolve these;
  motion-aware models (optical flow, 3D convolutions, video transformers) can.
- **Precise boundaries.** AP is averaged over tIoU 0.3 to 0.7, so tight intervals are worth real
  points on every type.

More data helps: 2000 to 10000 clips with a bigger model is +0.08. Most of the remaining gap is
modelling, not data. Numbers move by a point or two between runs and seeds.

---

## How the baseline works

1. **Features** (`vic_features.py`, numpy only, about 0.5 s per 150-frame clip). Each frame gets a vector of 45 numbers describing the step from its predecessor: frame-difference statistics (and their ratio to a rolling median: a freeze drives it to about 0, a cut or splice makes it spike), colour-histogram distance, per-channel mean steps and deviations, luminance deviation, mirror consistency (`diff(f_t, fliplr(f_{t-1}))` versus `diff(f_t, f_{t-1})`), zoom consistency (the same with a centre-crop rescale at three scales), global translation by phase correlation and its agreement with the local trend (a reversal breaks it), global and per-cell sharpness (blur), and per-cell deviation from a rolling/clip median appearance (inserted objects). See the comments in `FEATURE_NAMES`.
2. **Model** (`vic_model.py`). `TemporalIssueNet`: a 1x1 input projection, `--layers` residual dilated 1D conv blocks (dilations 1, 2, 4, ..., 32, repeating; six blocks give a receptive field of 127 frames), dropout, and a per-frame head with 10 logits. The default (`--hidden 64 --layers 6`) has about 100k parameters and trains on CPU.
3. **Training** (`train.py`). Videos are decoded from the mp4 files (so the model sees the codec artefacts the validator's clips have), features are cached as `.npz`, targets are per-frame multi-hot (spans cover `[start_frame, end_frame)`; a dropped-frames cut at `k` marks frames `k-1` and `k`), loss is BCE with a per-type `pos_weight` and a higher weight on frames around decoys, on random temporal crops, AdamW with a one-cycle cosine schedule, early stopping on validation loss.
4. **Threshold calibration.** Per-type thresholds are first chosen to maximise each type's F1, then refined by coordinate ascent on the validator's final `score` (mean AP dominated) plus a small `--calib-f1-weight` times macro F1, with real boxes for the spatial candidates. The F1 term keeps thresholds from drifting towards firing on every clip when the validation split is small. Thresholds are saved in `vic_config.json`. A threshold of 1.0 turns a type into "candidates only" (confidence below 0.5).
5. **Decoding** (`vic_model.decode_intervals`). Per type there are two kinds of runs. *Confident* runs: frames at or above the type's threshold, merged across gaps of at most one frame, runs shorter than a per-type minimum dropped; confidence is the run's mean probability mapped so that the threshold lands on 0.5. *Candidate* runs: frames with probability of at least 0.1 (`CANDIDATE_FLOOR`) that do not overlap a confident run, trimmed to at least half of their peak, with confidence below 0.5 by the same mapping. There is no clipping to 0.5, so confidences are an honest ranking, which is what the mAP rewards; F1 and clip accuracy only look at confidence >= 0.5. `dropped_frames` becomes a point event at the run's centre. Output is capped to the 25 most confident issues (the validator matches at most 25 per clip).
6. **Boxes** (`vic_localize.py`). `inserted_object`: difference images at the two cuts of the detected interval (the object pops in and out; ordinary motion does not line up between the two), thresholded, largest blob. `blurred_region`: per-block sharpness drop inside versus outside the interval. Always returns a valid box; never raises. The validator needs a box IoU of at least 0.3 for a spatial detection to count towards AP; the handcrafted boxes rarely reach it.
7. **The heuristic detector** (`trainer/flock_video_adapter.py`, `HeuristicDetector`) uses hand-set rules on the same features and also emits graded confidences: boundary pairs are accepted from a low score (`CANDIDATE_SCORE`), and `decode_intervals` splits them into confident detections and lower-ranked candidates.

Training seeds start at `1_000_000` (`generate_data.py`), the Hugging Face dataset derives its own from a separate hash namespace, and the documented dev packages use small seeds. That keeps training clips disjoint from your dev package, so local validation scores are honest. Never train on the package you validate on.

---

---

## Ideas for a stronger model

- **Spatial types first.** A small 2D CNN or a pretrained backbone (`timm` is available) run on frame differences or on stacked frames, with a spatial head that predicts the box directly, will beat the handcrafted localiser; `blurred_region` and `inserted_object` are 2 of the 10 per-type APs and most of the localization gain, and they only count when the box overlaps the truth (IoU >= 0.3): a detector that finds them in time but boxes them badly scores 0 on both.
- **Optical flow** (OpenCV Farneback or DIS, both available via `opencv-python-headless`) gives far better cues for reversal, drops and freezes than global phase correlation: motion direction consistency, flow magnitude spikes, flow-warped residuals for inserted objects.
- **Video models.** A 3D CNN or a video transformer over short clips, or a frozen image encoder per frame feeding the temporal net. Keep it inside the 60 s per clip budget; run it on the GPU when `device == "cuda"`.
- **VLM fine-tuning** with LoRA (`peft`, `transformers`): describe the edit type and span. Remember that weights must be in the repo and the whole detector must fit in 18 GiB.
- **Real footage for robustness.** The validator's clips can be edited from real videos. Generate training data with `--footage-dir <folder of videos> --footage-fraction 0.5` so the model does not overfit to the procedural scenes.
- **More and harder data.** Use the Hugging Face dataset (8000 training clips) or generate several thousand more; the temporal types keep improving with data. Use `--seed` to make disjoint sets. Expert clips and decoys are the ones that separate strong detectors from the baseline.
- **Decoys as hard negatives.** Every decoy in the labels is a stretch that looks like an edit but is not (a real scene cut is not `spliced_footage`, a smooth zoom is not `zoom_jump`, a pan that stops is not `frozen_frames`). Up-weight them (`--decoy-weight`), add a decoy head, or mine the false positives your detector makes on them.
- **Confidence and boundaries.** mAP averages tIoU thresholds up to 0.7, so refine interval ends (e.g. a boundary regression head or snapping to the frame-difference peaks) and make confidences rank-worthy: a well-ordered list beats a well-thresholded one. Calibrate on a validation set that is as close to the real one as you can build, and re-check with `--local-validation`.
- **Test-time consistency.** Post-process: pair boundary events (an onset needs an offset), suppress overlapping predictions of different types on the same frames, and keep the guesses on clips you believe are clean at the bottom of the ranking.
- **Multi-scale windows.** The net sees 127 frames of context; longer or multi-resolution context helps for slow edits such as long colour segments.

---

---

## Troubleshooting

- **`detector_output_invalid`**: check that times are numbers in seconds (not frame indices), `confidence` is in `[0, 1]`, `bbox` has `x0 < x1`, `y0 < y1` and is normalised. `package_submission.py --smoke-test` validates your output with the validator's parser.
- **`adapter_import_failed` / `model_load_failed`**: the smoke test runs the adapter in a clean subprocess with an empty environment; if it works there but fails locally under `--local-validation`, look for an import of something outside the module environment, or a file you forgot to copy into the submission folder.
- **`detector_timeout`**: more than 60 s on a clip. Profile `detect`; downsample frames before heavy work; do not decode the mp4 again when `frames` is already provided.
- **`detector_memory_exceeded`**: the GPU plus host memory ceiling is 18 GiB. Load in `bfloat16`, avoid keeping the full clip as float32 on the GPU.
- **A perfect-looking detector scores low**: look at `diagnostics.per_type_counts` in the local validation output. Duplicate predictions and predictions on clean clips are false positives.
- **macOS local runs refuse the sandbox**: for local debugging only, `VIDEO_INCONSISTENCY_ALLOW_UNSAFE_LOCAL_ADAPTER=1`. Never on a production validator.
