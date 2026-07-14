from __future__ import annotations

import numpy as np


def navigation_metrics(predictions: list[np.ndarray], targets: list[np.ndarray], success_radius_m: float = 20.0) -> dict[str, float]:
    """Compute the four metrics used in the manuscript.

    NE: final-position distance to target.
    SR: percentage with final distance <= 20 m.
    OSR: percentage whose minimum distance to target <= 20 m.
    SPL: success weighted by shortest-path / executed-path length.
    """
    ne, success, oracle, spl = [], [], [], []
    for pred, gt in zip(predictions, targets):
        target = gt[-1]
        final_dist = float(np.linalg.norm(pred[-1] - target))
        min_dist = float(np.linalg.norm(pred - target, axis=1).min())
        s = float(final_dist <= success_radius_m)
        shortest = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
        executed = float(np.linalg.norm(np.diff(pred, axis=0), axis=1).sum())
        ne.append(final_dist)
        success.append(s)
        oracle.append(float(min_dist <= success_radius_m))
        spl.append(s * shortest / max(shortest, executed, 1e-8))
    return {
        "NE": float(np.mean(ne)),
        "SR": 100.0 * float(np.mean(success)),
        "OSR": 100.0 * float(np.mean(oracle)),
        "SPL": 100.0 * float(np.mean(spl)),
    }
