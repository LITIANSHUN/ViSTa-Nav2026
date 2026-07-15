<u>*Part A includes dataset introduction and Part B includes quick start for VisTaNav algorithm.*</u>

# A. AeroNet Dataset

![Statistical analysis of the AeroNet.]https://github.com/user-attachments/assets/051d8569-2081-4b3a-8ef9-ce68ed6be6f7

**AeroNet** is a video-language UAV trajectory benchmark designed for studying
long-horizon navigation under degraded or unavailable GNSS measurements. The
release format separates numerical trajectory states, synchronized video,
natural-language annotations, and split metadata to support reproducible
training and evaluation.

![Representative UAV trajectory samples under different GNSS observability
conditions.]https://github.com/user-attachments/assets/de9e4e86-dd6e-42b9-a731-cc61b314f42e)

> **Repository notice:** the `videos/`  and `training_frames/` directories
> are intentionally partial in this package. They will be available upon acceptance.

## 1. Directory structure

```text
AeroNet/
├── metadata/
│   ├── train.json
│   ├── val.json
│   ├── seen_test.json
│   └── unseen_test.json
├── trajectories/
│   └── train/
├── videos/
│   └── scene_01/
├── annotations/
│   └── instructions.json
├── scripts/
│   ├── download_dataset.py
│   ├── validate_dataset.py
│   └── example_loader.py
└── README.md
```

## 2. Data organization

Each trajectory is stored as an independent compressed NumPy archive:

```text
trajectories/train/trajectory_000001.npz
```

Each synchronized first-person video is stored separately:

```text
videos/scene_01/trajectory_000001.mp4
```

The association among trajectories, videos, scenes, data splits, and language
instructions is defined in `metadata/*.json`.

## 3. NPZ schema

Every trajectory archive must contain the following arrays.

| Key                   | Shape    | Recommended dtype | Description                                                  |
| --------------------- | --------:| ----------------- | ------------------------------------------------------------ |
| `trajectory`          | `[N, 3]` | `float32`         | Complete ground-truth ENU trajectory in meters               |
| `degraded_trajectory` | `[N, 3]` | `float32`         | GNSS-degraded observations; unavailable entries may be `NaN` |
| `timestamps`          | `[N]`    | `float64`         | Strictly increasing timestamps in seconds                    |
| `missing_mask`        | `[N]`    | `bool` or `uint8` | `1`: observed; `0`: unavailable                              |
| `reliability`         | `[N]`    | `float32`         | Localization confidence in `[0, 1]`                          |
| `velocity`            | `[N, 3]` | `float32`         | ENU velocity in m/s                                          |
| `acceleration`        | `[N, 3]` | `float32`         | ENU acceleration in m/s²                                     |

Optional arrays may be added, but the required keys should remain unchanged
across all public releases.

## 4. Coordinate and synchronization conventions

- **Coordinate frame:** local East-North-Up (ENU).
- **Position unit:** meter.
- **Time unit:** second.
- **Sampling frequency:** 10 Hz.
- **Video synchronization:** trajectory state `i` should correspond to the
  video frame or timestamp at `timestamps[i]`.
- **Mask convention:** `missing_mask[i] = 1` indicates that a localization
  observation is available. Availability does not imply correctness; corrupted
  observations can still have a low `reliability[i]`.
- **Degradation labels:** `point_dropout`, `block_outage`,
  `multipath_noise`, `nlos_bias`, `drift`, `false_fix`, and `mixed`.

## 5. Metadata format

Each split file contains a `samples` array. A typical entry is:

```json
{
  "trajectory_id": "trajectory_000001",
  "scene_id": "scene_01",
  "split": "train",
  "trajectory_file": "trajectories/train/trajectory_000001.npz",
  "video_file": "videos/scene_01/trajectory_000001.mp4",
  "instruction_ids": ["instruction_000001"],
  "coordinate_system": "ENU",
  "sampling_rate_hz": 10,
  "num_steps": 300,
  "degradation_types": ["point_dropout", "drift"],
  "start_position_enu_m": [0.0, 0.0, 20.0],
  "goal_position_enu_m": [120.0, 85.0, 38.0],
  "environment_type": "urban_canyon"
}
```

After adding entries, update `num_samples` so that it equals the length of
`samples`.

## 6. Language annotations

`annotations/instructions.json` stores language annotations independently from
the numerical trajectory archives. A typical entry is:

```json
{
  "instruction_id": "instruction_000001",
  "trajectory_id": "trajectory_000001",
  "scene_id": "scene_01",
  "instruction": "Ascend above the parking structure and continue toward the open deck.",
  "granularity": "global",
  "language": "en",
  "annotator_status": "manually_verified",
  "verification_count": 3
}
```

This design avoids storing variable-length text inside NPZ files and makes the
annotations easy to inspect, revise, and extend.

## 7. Loading an example

Install the minimal dependency:

```bash
pip install numpy
```

Then run:

```bash
python scripts/example_loader.py --root . --split train
```

The template currently reports zero samples because the trajectory and video
directories are empty.

## 8. Validation

Validate the repository with:

```bash
python scripts/validate_dataset.py --root . 
```

```bash
python scripts/validate_dataset.py --root .
```

The validator checks:

- required directories and JSON files;
- consistency between `num_samples` and metadata entries;
- trajectory/video file references;
- annotation references;
- required NPZ keys, shapes, timestamps, masks, and reliability ranges;
- unreferenced trajectory archives.

As the dataset capacity is larger than GITHUB limit, please visit BaiduNet disk Link for raw video data download. 

traj_data.zip: https://pan.baidu.com/s/1tTpgKyWUCIoQwDjoqw8_3g?pwd=cw5x 

code: cw5x 

Raw_videos & Trainging RGB frames: https://pan.baidu.com/s/1XEQXPOuiRWyNLAHWwlwAww?pwd=mk97 

code: mk97 

**The complete dataset will be available upon acceptance.**

## 9. Download script

Before release, replace the placeholder URL and checksum in
`scripts/download_dataset.py` with the official archival location and SHA-256
checksum. Users can then run:

```bash
python scripts/download_dataset.py --output data
```

## B. Quick start for ViSTaNav

Recommended setup: Ubuntu 20.04.4 LTS (GNU/Linux 5.15.0-70-generic x86_64)

![Overview of the proposed ViSTa-Nav framework for GNSS-degraded UAV long-horizon navigation.]https://github.com/user-attachments/assets/a4d27e16-c0c5-459d-ae10-8b2db95f7e55)

```bash
conda env create -f environment.yml
conda activate vista-nav-repro

pip install open_clip_torch diffusers transformers accelerate


# 1) Validate the fixed path and infer the dataset schema
python scripts/inspect_dataset.py \
  --data-root /home/tianshun/Downloads/aeronet_repro_code/data/data

# 2) Train on AeroNet
bash run_train.sh

# 3) Evaluate 
bash run_evaluate.sh

# 4) For comparison with baselines:
cd /home/tianshun/Downloads/aeronet_repro_code

# inspect keys in several NPZ files
python scripts/inspect_npz.py --data /home/tianshun/Downloads/aeronet_repro_code/data/data --num 5

# train the proposed world-model method
python train.py --config configs/default.yaml --model vista_nav \
  --data-root /home/tianshun/Downloads/aeronet_repro_code/data/data

# evaluate one method with NE / SR / OSR / SPL
python eval.py --config configs/default.yaml --model vista_nav \
  --split seen_test \
  --data-root /home/tianshun/Downloads/aeronet_repro_code/data/data \
  --out runs/vln_experiments/vista_nav_seen_test.csv

# evaluate all baselines and ViSTa-Nav
python eval.py --config configs/default.yaml --model all \
  --split seen_test \
  --data-root /home/tianshun/Downloads/aeronet_repro_code/data/data \
  --out runs/vln_experiments/main_seen_test.csv
```

All outputs are written to `outputs/`.

## Citation

If you use the AeroNet dataset, benchmark protocol, or associated code in your research, please cite the following paper.

The citation information will be updated after publication.

```bibtex
@misc{aeronet2026,
  title        = {ViSTa-Nav: Visual-Semantic Trajectory Completion with Physical
                  Consistency for GNSS-Degraded UAV Long-Horizon Navigation},
  author       = {Li, Tianshun and Lu, Hongliang and Zhu, Zengle and Huai, Tianyi
                  and Li, Haoang and Zheng, Xinhu},
  year         = {2026},
  note         = {Manuscript under review}
}
```

## License and contact

Source code is released under the MIT License.

- AeroNet annotations and derived trajectory data are released under the
  CC BY-NC 4.0 License.
- Third-party visual assets remain subject to their original licenses.

For questions, please use the issue tracker of the official repository or
contact the corresponding author listed in the paper.
