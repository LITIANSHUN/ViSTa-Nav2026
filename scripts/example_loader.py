#!/usr/bin/env python3
"""
Minimal AeroNet loader example for NumPy and PyTorch workflows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class AeroNetSample:
    trajectory_id: str
    instruction: str
    trajectory: np.ndarray
    degraded_trajectory: np.ndarray
    timestamps: np.ndarray
    missing_mask: np.ndarray
    reliability: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    video_path: Path
    metadata: dict[str, Any]


class AeroNetDataset:
    def __init__(self, root: Path, split: str = "train") -> None:
        self.root = root
        split_map = {
            "train": "train.json",
            "val": "val.json",
            "seen_test": "seen_test.json",
            "unseen_test": "unseen_test.json",
        }
        if split not in split_map:
            raise ValueError(f"Unsupported split: {split}")

        metadata_path = root / "metadata" / split_map[split]
        annotations_path = root / "annotations" / "instructions.json"

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        with annotations_path.open("r", encoding="utf-8") as file:
            annotation_data = json.load(file)

        self.samples: list[dict[str, Any]] = metadata.get("samples", [])
        self.instructions = {
            item["instruction_id"]: item["instruction"]
            for item in annotation_data.get("annotations", [])
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AeroNetSample:
        item = self.samples[index]
        trajectory_path = self.root / item["trajectory_file"]
        video_path = self.root / item["video_file"]

        with np.load(trajectory_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}

        instruction_ids = item.get("instruction_ids", [])
        instruction = (
            self.instructions[instruction_ids[0]]
            if instruction_ids
            else ""
        )

        return AeroNetSample(
            trajectory_id=item["trajectory_id"],
            instruction=instruction,
            trajectory=arrays["trajectory"],
            degraded_trajectory=arrays["degraded_trajectory"],
            timestamps=arrays["timestamps"],
            missing_mask=arrays["missing_mask"],
            reliability=arrays["reliability"],
            velocity=arrays["velocity"],
            acceleration=arrays["acceleration"],
            video_path=video_path,
            metadata=item,
        )

    def __iter__(self) -> Iterator[AeroNetSample]:
        for index in range(len(self)):
            yield self[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one AeroNet sample.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "seen_test", "unseen_test"],
        default="train",
    )
    args = parser.parse_args()

    dataset = AeroNetDataset(args.root, args.split)
    print(f"Loaded split '{args.split}' with {len(dataset)} sample(s).")

    if len(dataset) == 0:
        print(
            "The repository template contains no trajectory/video samples. "
            "Populate the metadata and data directories before loading."
        )
        return

    sample = dataset[0]
    print(f"Trajectory ID: {sample.trajectory_id}")
    print(f"Instruction: {sample.instruction}")
    print(f"Trajectory shape: {sample.trajectory.shape}")
    print(f"Video path: {sample.video_path}")


if __name__ == "__main__":
    main()
