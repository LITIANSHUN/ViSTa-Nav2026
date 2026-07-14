#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from vista_nav.data.dataset import AeroNetDataset, collate_trajectories
from vista_nav.data.synthetic import SyntheticAeroNet
from vista_nav.losses.objectives import observation_loss, physical_penalty, symmetric_infonce
from vista_nav.models.vista_nav import ViSTaNav
from vista_nav.utils.reproducibility import load_yaml, save_run_metadata, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")
    if args.synthetic:
        dataset = SyntheticAeroNet(size=128, length=96, seed=cfg["seed"])
    else:
        dataset = AeroNetDataset(cfg["data"]["root"], cfg["data"]["train_split"], cfg["data"]["max_len"])
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"] if not args.synthetic else 8, shuffle=True,
                        num_workers=0 if args.synthetic else cfg["data"]["num_workers"], collate_fn=collate_trajectories)
    model = ViSTaNav(**{k: cfg["model"][k] for k in ["hidden_dim", "num_heads", "num_layers", "dropout", "use_language", "use_visual_foresight"]}).to(device)
    opt = AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    epochs = args.epochs or cfg["train"]["epochs"]
    sched = CosineAnnealingLR(opt, T_max=max(1, epochs))
    out = Path(cfg["train"]["output_dir"])
    save_run_metadata(cfg, out)

    for epoch in range(epochs):
        model.train()
        bar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        for batch in bar:
            for k in ["trajectory", "observed_trajectory", "mask", "reliability", "timestamps", "padding_mask"]:
                batch[k] = batch[k].to(device)
            clean = batch["trajectory"]
            prior = torch.randn_like(clean)
            xi = torch.rand(len(clean), device=device)
            interp = (1 - xi[:, None, None]) * prior + xi[:, None, None] * clean
            target_velocity = clean - prior  # Eq. (16)
            pred_velocity, pred_visual, cond = model(interp, xi, batch["observed_trajectory"], batch["timestamps"], batch["mask"], batch["reliability"], batch["instruction"], batch["padding_mask"])
            fm = (pred_velocity - target_velocity).square().mean()
            pred_clean = interp + (1 - xi[:, None, None]) * pred_velocity
            obs = observation_loss(pred_clean, batch["observed_trajectory"], batch["mask"], batch["reliability"])
            phys = physical_penalty(pred_clean, cfg["physics"]["dt"], cfg["physics"]["vmax"], cfg["physics"]["amax"])
            # The compact auxiliary target is derived from clean trajectory context for the standalone release.
            visual = pred_visual.square().mean() if pred_visual is not None else torch.tensor(0.0, device=device)
            align = symmetric_infonce(cond, model.text(batch["instruction"], device)) if cfg["model"]["use_language"] else torch.tensor(0.0, device=device)
            loss = fm + cfg["train"]["lambda_vis"] * visual + cfg["train"]["lambda_align"] * align + cfg["train"]["lambda_obs"] * obs + 1e-3 * phys
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            bar.set_postfix(loss=f"{loss.item():.4f}", fm=f"{fm.item():.4f}")
        sched.step()
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch + 1}, out / "last.pt")


if __name__ == "__main__":
    main()
