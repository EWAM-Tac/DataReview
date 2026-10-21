from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, motion_energy, safe_mean


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    n = min(a.size, b.size)
    if n < 5:
        return 0.0
    a = a[:n]
    b = b[:n]
    if a.std() < 1e-10 or b.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def score_consistency(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    **_: Any,
) -> DimensionScore:
    if not episodes:
        return DimensionScore("consistency", "Cross-modal consistency", 0.0, status="fail")

    a_s_corrs = []
    touch_motion_corrs = []
    subtask_change_align = []

    for ep in episodes:
        arr = ep.arrays
        act = arr.get("action")
        st = arr.get("state")
        if act is not None and st is not None:
            # same dim slice if possible
            d = min(act.shape[-1], st.shape[-1])
            ea = motion_energy(act[:, :d])
            es = motion_energy(st[:, :d])
            # delayed state often tracks action; also try lag 1
            c0 = _corr(ea, es)
            c1 = _corr(ea[:-1], es[1:]) if ea.size > 2 else c0
            a_s_corrs.append(max(c0, c1))

        tac = arr.get("tactile")
        if tac is not None:
            et = motion_energy(tac)
            # compare to action or tcp motion
            ref = arr.get("action_tcp")
            if ref is None:
                ref = arr.get("action")
            if ref is None:
                ref = arr.get("state_tcp")
            if ref is None:
                ref = arr.get("state")
            if ref is not None:
                er = motion_energy(ref)
                touch_motion_corrs.append(_corr(et, er))

        if "subtask_index" in arr and act is not None:
            sub = arr["subtask_index"].reshape(-1)
            if sub.size >= 3:
                changes = np.where(np.diff(sub) != 0)[0]
                if changes.size:
                    ea = motion_energy(act)
                    # window energy around boundary vs random
                    win = 5
                    b_e = []
                    for c in changes:
                        lo, hi = max(0, c - win), min(ea.size, c + win)
                        b_e.append(float(ea[lo:hi].mean()))
                    rng = np.random.default_rng(ep.episode_index)
                    r_idx = rng.integers(0, max(ea.size - 1, 1), size=min(20, ea.size))
                    r_e = [float(ea[max(0, i - win) : min(ea.size, i + win)].mean()) for i in r_idx]
                    if r_e and np.mean(r_e) > 1e-8:
                        subtask_change_align.append(clip01(float(np.mean(b_e) / (np.mean(r_e) + 1e-8)) / 2.0))

    parts = []
    notes = []
    evidence: dict[str, Any] = {}

    if a_s_corrs:
        # map corr [-1,1] → score; expect positive
        s = clip01(0.5 + 0.5 * safe_mean(a_s_corrs))
        parts.append(s)
        evidence["action_state_motion_corr"] = safe_mean(a_s_corrs)
        if safe_mean(a_s_corrs) < 0.1:
            notes.append("weak action↔state motion correlation")
    if touch_motion_corrs:
        s = clip01(0.5 + 0.5 * safe_mean(touch_motion_corrs))
        parts.append(s)
        evidence["tactile_vs_motion_corr"] = safe_mean(touch_motion_corrs)
    if subtask_change_align:
        parts.append(safe_mean(subtask_change_align))
        evidence["subtask_boundary_motion_ratio"] = safe_mean(subtask_change_align)

    if not parts:
        return DimensionScore(
            "consistency",
            "Cross-modal consistency",
            None,
            status="na",
            notes=["need at least action+state or tactile+motion"],
        )

    score = float(np.mean(parts))
    status = "ok" if score >= 0.75 else ("warn" if score >= 0.55 else "fail")
    return DimensionScore(
        id="consistency",
        name="Cross-modal consistency",
        score=score,
        confidence=0.8 if len(parts) == 1 else 1.0,
        status=status,
        evidence=evidence,
        notes=notes,
    )
