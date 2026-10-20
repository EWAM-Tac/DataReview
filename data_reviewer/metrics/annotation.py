from __future__ import annotations

from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, safe_mean


def score_annotation(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    **_: Any,
) -> DimensionScore:
    has_task = "task_index" in role_map
    has_sub = "subtask_index" in role_map
    has_lang = any(
        isinstance(ep.meta.get("tasks"), list) and len(ep.meta["tasks"]) > 0 for ep in episodes
    ) or any(isinstance(ep.meta.get("task"), str) for ep in episodes)

    if not (has_task or has_sub or has_lang):
        return DimensionScore(
            "annotation",
            "Annotation / semantics",
            None,
            status="na",
            notes=["no task/subtask/language fields detected"],
        )

    lang_ok = []
    sub_cover = []
    sub_holes = []
    for ep in episodes:
        tasks = ep.meta.get("tasks") or ep.meta.get("task")
        if isinstance(tasks, list):
            text = " ".join(str(t) for t in tasks)
            lang_ok.append(1.0 if len(text.strip()) > 8 else 0.0)
        elif isinstance(tasks, str):
            lang_ok.append(1.0 if len(tasks.strip()) > 8 else 0.0)

        if "subtask_index" in ep.arrays:
            sub = ep.arrays["subtask_index"].reshape(-1)
            # coverage: fraction of frames with a defined subtask (not all -1)
            valid = float(np.isfinite(sub).mean())
            # unique count
            n_u = len(np.unique(sub[np.isfinite(sub)])) if valid > 0 else 0
            sub_cover.append(valid * clip01(n_u / 3.0))
            # holes: long runs of same value at start/end only is OK; empty middle jumps
            if sub.size > 10:
                # missing values
                sub_holes.append(1.0 - valid)

    parts = []
    evidence: dict[str, Any] = {}
    if lang_ok:
        parts.append(safe_mean(lang_ok))
        evidence["language_nonempty_rate"] = safe_mean(lang_ok)
    if sub_cover:
        parts.append(safe_mean(sub_cover))
        evidence["subtask_coverage_score"] = safe_mean(sub_cover)
        evidence["subtask_hole_frac"] = safe_mean(sub_holes)
    if has_task and not lang_ok:
        # task_index alone
        parts.append(0.7)
        evidence["task_index_only"] = True

    score = float(np.mean(parts)) if parts else 0.5
    notes = []
    if lang_ok and safe_mean(lang_ok) < 0.9:
        notes.append("some episodes lack language instructions")
    status = "ok" if score >= 0.6 else ("warn" if score >= 0.4 else "fail")
    return DimensionScore(
        id="annotation",
        name="Annotation / semantics",
        score=score,
        confidence=1.0,
        status=status,
        evidence=evidence,
        notes=notes,
    )
