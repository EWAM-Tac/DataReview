from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, collect, motion_energy, safe_mean


def _contact_mask(tac: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Heuristic contact: energy above adaptive baseline."""
    if tac.ndim == 1:
        inten = np.abs(tac)
    else:
        inten = np.linalg.norm(tac, axis=1)
    base = float(np.percentile(inten, 20))
    thr = base + 0.5 * (float(np.percentile(inten, 80)) - base + 1e-8)
    thr = max(thr, base + 1e-5)
    return inten > thr, inten, thr


def score_tactile(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    **_: Any,
) -> DimensionScore:
    has_vec = "tactile" in role_map
    has_force = "force" in role_map
    has_img = "tactile_image" in role_map
    if not has_vec and not has_img and not has_force:
        return DimensionScore(
            "tactile",
            "Tactile usefulness",
            None,
            status="na",
            notes=["no tactile/force roles mapped"],
        )

    tacs = collect(episodes, "tactile") or collect(episodes, "force")
    notes: list[str] = []
    evidence: dict[str, Any] = {
        "has_tactile_vector": has_vec,
        "has_force_proxy": has_force,
        "has_tactile_image": has_img,
        "n_tactile_image_keys": len(role_map.get("tactile_image", [])),
        "signal_source": "tactile" if collect(episodes, "tactile") else ("force" if tacs else None),
    }
    if has_force and not has_vec:
        notes.append("scoring force/wrench as tactile proxy")

    parts: list[float] = []

    if tacs:
        contact_fracs = []
        snrs = []
        dead = 0
        temporal_struct = []
        for tac in tacs:
            mask, inten, thr = _contact_mask(tac)
            cf = float(mask.mean())
            contact_fracs.append(cf)
            if inten.std() < 1e-8:
                dead += 1
                snrs.append(0.0)
                temporal_struct.append(0.0)
                continue
            on = inten[mask]
            off = inten[~mask]
            if off.size == 0 or on.size == 0:
                snrs.append(0.3 if on.size else 0.0)
            else:
                snr = (float(on.mean()) - float(off.mean())) / (float(off.std()) + 1e-6)
                snrs.append(clip01(snr / 5.0))
            # temporal: contact should form contiguous runs, not salt-pepper
            if mask.size >= 4:
                switches = float(np.mean(mask[1:] != mask[:-1]))
                # prefer moderate switches (some contacts) not 0.5 random
                temporal_struct.append(clip01(1.0 - abs(switches - 0.05) / 0.25))
            else:
                temporal_struct.append(0.5)

        # ideal contact fraction roughly 5%–70%
        cf = safe_mean(contact_fracs)
        cf_score = clip01(1.0 - abs(cf - 0.25) / 0.35)
        snr_s = safe_mean(snrs)
        temp_s = safe_mean(temporal_struct)
        dead_frac = dead / max(len(tacs), 1)
        alive_s = 1.0 - dead_frac

        parts.extend([0.35 * snr_s + 0.25 * cf_score + 0.2 * temp_s + 0.2 * alive_s])
        # actually append as one combined already — fix: append components weighted later
        parts = [0.35 * snr_s + 0.25 * cf_score + 0.2 * temp_s + 0.2 * alive_s]
        evidence.update(
            {
                "mean_contact_fraction": cf,
                "mean_snr_score": snr_s,
                "temporal_structure_score": temp_s,
                "dead_sensor_episode_frac": dead_frac,
            }
        )
        if dead_frac > 0.05:
            notes.append(f"dead/constant tactile in {dead_frac:.1%} episodes")
        if cf < 0.01:
            notes.append("almost no contact-like tactile activity")
        if cf > 0.9:
            notes.append("tactile almost always 'on' — check baseline")

        # redundancy vs state/action: if tactile energy ~ linearly explained by motion, lower uniqueness
        reduns = []
        for ep in episodes:
            if "tactile" not in ep.arrays:
                continue
            et = motion_energy(ep.arrays["tactile"])
            ref = ep.arrays.get("action", ep.arrays.get("state"))
            if ref is None:
                continue
            er = motion_energy(ref)
            n = min(et.size, er.size)
            if n < 10 or et[:n].std() < 1e-10 or er[:n].std() < 1e-10:
                continue
            # R^2 of linear fit et ~ a*er + b
            A = np.stack([er[:n], np.ones(n)], axis=1)
            coef, _, _, _ = np.linalg.lstsq(A, et[:n], rcond=None)
            pred = A @ coef
            ss_res = float(((et[:n] - pred) ** 2).sum())
            ss_tot = float(((et[:n] - et[:n].mean()) ** 2).sum())
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            reduns.append(clip01(r2))
        if reduns:
            # high R2 → low unique info
            uniq = 1.0 - safe_mean(reduns)
            parts.append(0.5 + 0.5 * uniq)  # soft
            evidence["tactile_explained_by_motion_r2"] = safe_mean(reduns)
            evidence["tactile_uniqueness_score"] = uniq

    # tactile images: presence already in completeness; light check via path existence ratio
    if has_img:
        cover = []
        for ep in episodes:
            keys = role_map.get("tactile_image", [])
            if not keys:
                continue
            cover.append(sum(1 for k in keys if k in ep.video_paths) / len(keys))
        if cover:
            parts.append(safe_mean(cover))
            evidence["tactile_image_path_coverage"] = safe_mean(cover)

    if not parts:
        return DimensionScore(
            "tactile",
            "Tactile usefulness",
            0.3,
            status="warn",
            evidence=evidence,
            notes=["tactile role mapped but no vector arrays loaded"],
        )

    score = float(np.mean(parts))
    status = "ok" if score >= 0.55 else ("warn" if score >= 0.35 else "fail")
    return DimensionScore(
        id="tactile",
        name="Tactile usefulness",
        score=score,
        confidence=1.0 if tacs else 0.5,
        status=status,
        evidence=evidence,
        notes=notes,
    )
