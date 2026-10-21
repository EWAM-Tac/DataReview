from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, safe_mean


def score_completeness(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    coverage: dict[str, Any],
    fps: float,
    **_: Any,
) -> DimensionScore:
    if not episodes:
        return DimensionScore("completeness", "Completeness", 0.0, status="fail", notes=["no episodes loaded"])

    lengths = [ep.n_frames for ep in episodes]
    med = float(np.median(lengths))
    # length sanity: too short (<0.5s) or wildly uneven
    short = sum(1 for L in lengths if L < max(5, 0.5 * fps))
    short_frac = short / len(lengths)
    cv = float(np.std(lengths) / max(med, 1.0))

    # modality co-presence per episode
    core_roles = [r for r in ("state", "action", "tactile") if r in role_map]
    image_ok = []
    tactile_img_ok = []
    aligned = []
    for ep in episodes:
        present = [r for r in core_roles if r in ep.arrays and ep.arrays[r].shape[0] == ep.n_frames]
        aligned.append(len(present) / max(len(core_roles), 1))
        # videos exist if declared
        n_img = len(role_map.get("image", []))
        if n_img:
            have = sum(1 for k in role_map["image"] if k in ep.video_paths)
            image_ok.append(have / n_img)
        n_ti = len(role_map.get("tactile_image", []))
        if n_ti:
            have = sum(1 for k in role_map["tactile_image"] if k in ep.video_paths)
            tactile_img_ok.append(have / n_ti)

    align_score = safe_mean(aligned, 1.0)
    img_score = safe_mean(image_ok, 1.0) if image_ok else None
    ti_score = safe_mean(tactile_img_ok, 1.0) if tactile_img_ok else None

    length_score = clip01(1.0 - short_frac) * clip01(1.0 - max(0.0, cv - 1.0) / 2.0)

    parts = [align_score, length_score]
    if img_score is not None:
        parts.append(img_score)
    if ti_score is not None:
        parts.append(ti_score)
    score = float(np.mean(parts))

    notes = []
    if short_frac > 0.05:
        notes.append(f"{short_frac:.1%} episodes shorter than ~0.5s")
    if img_score is not None and img_score < 0.95:
        notes.append(f"image video coverage {img_score:.2f}")

    status = "ok" if score >= 0.9 else ("warn" if score >= 0.75 else "fail")
    if short_frac > 0 or align_score < 1.0:
        status = "fail"
    if img_score is not None and img_score < 1.0:
        status = "fail"
    if ti_score is not None and ti_score < 1.0:
        status = "fail"
    return DimensionScore(
        id="completeness",
        name="Completeness",
        score=score,
        confidence=1.0,
        status=status,
        evidence={
            "n_episodes": len(episodes),
            "length_median": med,
            "length_cv": cv,
            "short_frac": short_frac,
            "core_align_score": align_score,
            "image_video_coverage": img_score,
            "tactile_image_coverage": ti_score,
            "roles_present": sorted(role_map.keys()),
        },
        notes=notes,
    )
