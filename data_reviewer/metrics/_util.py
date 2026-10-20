from __future__ import annotations

import numpy as np

from ..types import EpisodeArrays


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_mean(xs: list[float], default: float = 0.0) -> float:
    return float(np.mean(xs)) if xs else default


def finite_ratio(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 1.0
    return float(np.isfinite(x).mean())


def motion_energy(x: np.ndarray) -> np.ndarray:
    """Per-timestep L2 of first difference. x: (T, D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < 2:
        return np.zeros((x.shape[0],), dtype=np.float64)
    d = np.diff(x, axis=0)
    e = np.linalg.norm(d, axis=1)
    return np.concatenate([[e[0]], e])


def idle_fraction(x: np.ndarray, thr: float | None = None) -> float:
    e = motion_energy(x)
    if e.size == 0:
        return 1.0
    if thr is None:
        thr = float(np.percentile(e, 20) + 1e-8)
        thr = max(thr, 1e-4 * float(np.percentile(e, 90) + 1e-8))
    return float((e < thr).mean())


def has_role(eps: list[EpisodeArrays], role: str) -> bool:
    return any(role in ep.arrays for ep in eps)


def collect(eps: list[EpisodeArrays], role: str) -> list[np.ndarray]:
    return [ep.arrays[role] for ep in eps if role in ep.arrays]
