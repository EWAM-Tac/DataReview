from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, collect, idle_fraction, motion_energy, safe_mean


def _jerk_penalty(x: np.ndarray) -> float:
    e = motion_energy(x)
    if e.size < 3:
        return 0.0
    d2 = np.diff(e)
    # high relative jerk → penalty
    return float(np.mean(np.abs(d2)) / (np.mean(e) + 1e-6))


def score_dynamics(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    **_: Any,
) -> DimensionScore:
    action_list = collect(episodes, "action") or collect(episodes, "action_tcp")
    state_list = collect(episodes, "state") or collect(episodes, "state_tcp")
    pose_list = collect(episodes, "hand_pose")
    if not action_list and not state_list and not pose_list:
        return DimensionScore(
            "dynamics", "Dynamics", None, status="na", notes=["no state/action/hand_pose"]
        )

    if action_list:
        series, label = action_list, "action"
    elif state_list:
        series, label = state_list, "state"
    else:
        series, label = pose_list, "hand_pose"

    idle = [idle_fraction(x) for x in series]
    active = [1.0 - i for i in idle]
    jerks = [_jerk_penalty(x) for x in series]
    # diversity across episodes: mean std of episode-mean vectors
    means = np.stack([x.mean(axis=0) for x in series], axis=0)
    if means.shape[0] >= 2:
        diversity = float(means.std(axis=0).mean() / (np.abs(means).mean() + 1e-6))
        diversity_s = clip01(diversity * 5.0)  # soft saturation
    else:
        diversity_s = 0.25

    active_s = safe_mean(active)
    # mild jerk: prefer some smoothness; huge jerk penalized
    jerk_s = clip01(1.0 - max(0.0, safe_mean(jerks) - 1.0) / 5.0)

    score = float(0.5 * active_s + 0.25 * jerk_s + 0.25 * diversity_s)
    status = "ok" if score >= 0.75 else ("warn" if score >= 0.55 else "fail")
    if active_s < 0.5:
        status = "fail"
    notes = []
    if active_s < 0.4:
        notes.append(f"high idle fraction on {label} (active={active_s:.2f})")
    if diversity_s < 0.2:
        notes.append("low cross-episode diversity")

    return DimensionScore(
        id="dynamics",
        name="Dynamics / richness",
        score=score,
        confidence=1.0,
        status=status,
        evidence={
            "signal": label,
            "mean_active_fraction": active_s,
            "mean_idle_fraction": safe_mean(idle),
            "mean_jerk_ratio": safe_mean(jerks),
            "diversity_score": diversity_s,
            "n_episodes": len(series),
        },
        notes=notes,
    )
