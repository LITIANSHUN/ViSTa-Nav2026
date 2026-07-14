#!/usr/bin/env python3
"""
Validate the public AeroNet directory, metadata, annotations, and NPZ samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_FILES = ("train.json", "val.json", "seen_test.json", "unseen_test.json")
REQUIRED_NPZ_KEYS = {
    "trajectory",
    "degraded_trajectory",
    "timestamps",
    "missing_mask",
    "reliability",
    "velocity",
    "acceleration",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_npz(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as sample:
            missing = REQUIRED_NPZ_KEYS.difference(sample.files)
            if missing:
                errors.append(f"{path}: missing keys {sorted(missing)}")
                return errors

            trajectory = sample["trajectory"]
            degraded = sample["degraded_trajectory"]
            timestamps = sample["timestamps"]
            mask = sample["missing_mask"]
            reliability = sample["reliability"]
            velocity = sample["velocity"]
            acceleration = sample["acceleration"]

            if trajectory.ndim != 2 or trajectory.shape[1] != 3:
                errors.append(f"{path}: trajectory must have shape [N, 3]")
            n = trajectory.shape[0]
            if degraded.shape != (n, 3):
                errors.append(f"{path}: degraded_trajectory must have shape [N, 3]")
            if timestamps.shape != (n,):
                errors.append(f"{path}: timestamps must have shape [N]")
            if mask.shape != (n,):
                errors.append(f"{path}: missing_mask must have shape [N]")
            if reliability.shape != (n,):
                errors.append(f"{path}: reliability must have shape [N]")
            if velocity.shape != (n, 3):
                errors.append(f"{path}: velocity must have shape [N, 3]")
            if acceleration.shape != (n, 3):
                errors.append(f"{path}: acceleration must have shape [N, 3]")

            if not np.all(np.diff(timestamps) > 0):
                errors.append(f"{path}: timestamps must be strictly increasing")
            if not np.all(np.isin(mask, [0, 1, False, True])):
                errors.append(f"{path}: missing_mask must contain only 0/1 values")
            if np.any((reliability < 0) | (reliability > 1)):
                errors.append(f"{path}: reliability values must lie in [0, 1]")

    except Exception as exc:
        errors.append(f"{path}: unable to read NPZ ({exc})")
    return errors


def validate_dataset(root: Path, allow_empty: bool) -> list[str]:
    errors: list[str] = []

    required_dirs = [
        root / "metadata",
        root / "trajectories",
        root / "videos",
        root / "annotations",
        root / "scripts",
    ]
    for directory in required_dirs:
        if not directory.is_dir():
            errors.append(f"Missing directory: {directory}")

    annotation_path = root / "annotations" / "instructions.json"
    if not annotation_path.is_file():
        errors.append(f"Missing annotation file: {annotation_path}")
        annotation_ids: set[str] = set()
    else:
        try:
            annotation_data = load_json(annotation_path)
            annotations = annotation_data.get("annotations", [])
            annotation_ids = {
                item["instruction_id"]
                for item in annotations
                if isinstance(item, dict) and "instruction_id" in item
            }
        except Exception as exc:
            errors.append(f"{annotation_path}: invalid JSON ({exc})")
            annotation_ids = set()

    referenced_npz: set[Path] = set()
    for split_file in SPLIT_FILES:
        path = root / "metadata" / split_file
        if not path.is_file():
            errors.append(f"Missing split metadata: {path}")
            continue
        try:
            data = load_json(path)
        except Exception as exc:
            errors.append(f"{path}: invalid JSON ({exc})")
            continue

        samples = data.get("samples", [])
        if data.get("num_samples") != len(samples):
            errors.append(
                f"{path}: num_samples={data.get('num_samples')} "
                f"but samples contains {len(samples)} entries"
            )

        for index, item in enumerate(samples):
            prefix = f"{path}: sample {index}"
            required_fields = {
                "trajectory_id",
                "scene_id",
                "split",
                "trajectory_file",
                "video_file",
                "instruction_ids",
            }
            missing_fields = required_fields.difference(item)
            if missing_fields:
                errors.append(f"{prefix}: missing fields {sorted(missing_fields)}")
                continue

            trajectory_path = root / item["trajectory_file"]
            video_path = root / item["video_file"]
            referenced_npz.add(trajectory_path)

            if not trajectory_path.is_file():
                errors.append(f"{prefix}: missing trajectory file {trajectory_path}")
            if not video_path.is_file():
                errors.append(f"{prefix}: missing video file {video_path}")

            for instruction_id in item["instruction_ids"]:
                if instruction_id not in annotation_ids:
                    errors.append(
                        f"{prefix}: unknown instruction_id {instruction_id!r}"
                    )

    npz_files = sorted((root / "trajectories").rglob("*.npz"))
    if not npz_files and not allow_empty:
        errors.append(
            "No .npz trajectory files were found. "
            "Use --allow-empty only for the repository template."
        )

    for path in npz_files:
        errors.extend(validate_npz(path))

    unreferenced = set(npz_files).difference(referenced_npz)
    for path in sorted(unreferenced):
        errors.append(f"Unreferenced trajectory file: {path}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an AeroNet release.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="AeroNet root directory",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permit empty trajectories/videos directories for the template release",
    )
    args = parser.parse_args()

    errors = validate_dataset(args.root, args.allow_empty)
    if errors:
        print(f"Validation failed with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("AeroNet validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
