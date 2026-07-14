from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Eq. (17)-(22): symmetric cross-modal InfoNCE alignment."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def observation_loss(pred: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    """Reliability-weighted observed-anchor term corresponding to Eq. (6)."""
    weights = (mask * reliability).unsqueeze(-1)
    return ((pred - observed).square() * weights).sum() / weights.sum().clamp_min(1.0)


def physical_penalty(pred: torch.Tensor, dt: float, vmax: float, amax: float) -> torch.Tensor:
    """Soft implementation of the velocity/acceleration constraints in Eq. (4),(6)."""
    velocity = torch.diff(pred, dim=1) / dt
    acceleration = torch.diff(velocity, dim=1) / dt
    v_pen = F.relu(torch.linalg.vector_norm(velocity, dim=-1) - vmax).square().mean()
    a_pen = F.relu(torch.linalg.vector_norm(acceleration, dim=-1) - amax).square().mean()
    return v_pen + a_pen
