from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, collect, idle_fraction, safe_mean


def score_trainability(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    dataset_meta: dict[str, Any] | None = None,
    **_: Any,
) -> DimensionScore:
    dataset_meta = dataset_meta or {}
    notes: list[str] = []
    evidence: dict[str, Any] = {}

    # split presence
    splits = dataset_meta.get("splits") or {}
    has_split = bool(splits)
    evidence["has_splits"] = has_split
    split_s = 1.0 if has_split else 0.6
    if not has_split:
        notes.append("no splits declared in info.json")

    series = collect(episodes, "action") or collect(episodes, "state")
    idle_s = 1.0
    outlier_s = 1.0
    if series:
        idles = [idle_fraction(x) for x in series]
        idle_s = clip01(1.0 - max(0.0, safe_mean(idles) - 0.5) / 0.5)
        evidence["mean_idle_fraction"] = safe_mean(idles)
        # outlier: fraction of frames with |z| > 8 on action/state
        zs = []
        for x in series:
            mu = x.mean(axis=0)
            sd = x.std(axis=0) + 1e-6
            z = np.abs((x - mu) / sd)
            zs.append(float((z > 8).mean()))
        outlier_frac = safe_mean(zs)
        outlier_s = clip01(1.0 - outlier_frac * 20)
        evidence["outlier_frame_frac"] = outlier_frac
        if safe_mean(idles) > 0.7:
            notes.append("very high idle ratio — low sample efficiency")
        if outlier_frac > 0.01:
            notes.append(f"elevated outlier frames ({outlier_frac:.3%})")

    # length usability
    lengths = [ep.n_frames for ep in episodes]
    if lengths:
        med = float(np.median(lengths))
        evidence["median_length"] = med
        # prefer not tiny
        len_s = clip01(med / 100.0)
    else:
        len_s = 0.0

    score = float(0.3 * split_s + 0.35 * idle_s + 0.2 * outlier_s + 0.15 * len_s)
    status = "ok" if score >= 0.55 else ("warn" if score >= 0.35 else "fail")
    return DimensionScore(
        id="trainability",
        name="Trainability / leakage risk",
        score=score,
        confidence=1.0,
        status=status,
        evidence=evidence,
        notes=notes,
    )
