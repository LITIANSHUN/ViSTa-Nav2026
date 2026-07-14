from __future__ import annotations

import torch


@torch.no_grad()
def ode_sample(model, batch, steps: int = 20):
    """Fixed-step Euler probability-flow sampler matching Table III's 20 steps."""
    shape = batch["trajectory"].shape
    x = torch.randn(shape, device=batch["trajectory"].device)
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((shape[0],), k / steps, device=x.device)
        velocity, _, _ = model(
            x, t,
            batch["observed_trajectory"], batch["timestamps"], batch["mask"], batch["reliability"],
            batch["instruction"], batch.get("padding_mask"),
        )
        x = x + dt * velocity
    return x
