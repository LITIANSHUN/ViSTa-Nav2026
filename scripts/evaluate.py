#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vista_nav.data.dataset import AeroNetDataset, collate_trajectories
from vista_nav.data.synthetic import SyntheticAeroNet
from vista_nav.models.vista_nav import ViSTaNav
from vista_nav.utils.inference import ode_sample
from vista_nav.utils.metrics import navigation_metrics
from vista_nav.utils.reproducibility import load_yaml, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="outputs/train/last.pt")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")
    dataset = SyntheticAeroNet(size=32, length=96, seed=cfg["seed"] + 1000) if args.synthetic else AeroNetDataset(cfg["data"]["root"], cfg["data"]["test_split"], cfg["data"]["max_len"])
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_trajectories)
    model = ViSTaNav(**{k: cfg["model"][k] for k in ["hidden_dim", "num_heads", "num_layers", "dropout", "use_language", "use_visual_foresight"]}).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, gts = [], []
    for batch in tqdm(loader, desc="evaluate"):
        lengths = (~batch["padding_mask"]).sum(dim=1).tolist()
        for k in ["trajectory", "observed_trajectory", "mask", "reliability", "timestamps", "padding_mask"]:
            batch[k] = batch[k].to(device)
        pred = ode_sample(model, batch, cfg["inference"]["ode_steps"]).cpu().numpy()
        gt = batch["trajectory"].cpu().numpy()
        for i, length in enumerate(lengths):
            preds.append(pred[i, :length])
            gts.append(gt[i, :length])
    metrics = navigation_metrics(preds, gts, cfg["inference"]["success_radius_m"])
    out = Path("outputs/evaluation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
