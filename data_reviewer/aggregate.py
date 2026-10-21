from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import open_dataset
from .catalog import load_catalog, suggest_local_roots
from .metrics import SCORERS
from .profile import DatasetProfile, infer_profile_from_text
from .types import DimensionScore, ReviewReport

GATE_DIMENSIONS = {"integrity", "completeness"}


def aggregate_scores(dimensions: list[DimensionScore]) -> tuple[float | None, str, list[str]]:
    gates_failed = [d.id for d in dimensions if d.id in GATE_DIMENSIONS and d.status == "fail"]
    usable = [d for d in dimensions if d.score is not None]
    if not usable:
        return None, "fail", gates_failed
    weights = np.array([max(d.confidence, 1e-6) for d in usable], dtype=np.float64)
    scores = np.array([float(d.score) for d in usable], dtype=np.float64)
    overall = float(np.average(scores, weights=weights))

    if gates_failed or any(d.status == "fail" for d in usable):
        status = "fail"
    elif any(d.status == "warn" for d in usable):
        status = "warn"
    else:
        status = "ok"
    return overall, status, gates_failed


def _default_roles_path() -> Path | None:
    cand = Path(__file__).resolve().parents[1] / "configs" / "default_roles.yaml"
    return cand if cand.exists() else None


def review_dataset(
    dataset_path: str | Path,
    *,
    max_episodes: int = 20,
    episode_ids: list[int] | None = None,
    roles_config: str | Path | None = None,
    seed: int = 0,
    video_probe_per_ep: int = 2,
    profile: DatasetProfile | None = None,
) -> ReviewReport:
    root = Path(dataset_path)
    roles_path = Path(roles_config) if roles_config else _default_roles_path()

    adapter = open_dataset(root, role_config=roles_path)
    available_ids = adapter.available_episode_ids()
    episodes = list(
        adapter.iter_episodes(max_episodes=max_episodes, episode_ids=episode_ids, seed=seed)
    )

    if profile is None:
        profile = infer_profile_from_text(
            name=root.name,
            description=" ".join(adapter.role_map().keys()),
            contents=",".join(
                [
                    "灵巧手动作" if adapter.coverage().get("has_action") else "",
                    "触觉" if adapter.coverage().get("has_tactile") else "",
                ]
            ),
        )

    applicable = set(profile.applicable_dimensions)
    ctx: dict[str, Any] = {
        "role_map": adapter.role_map(),
        "coverage": adapter.coverage(),
        "fps": adapter.fps,
        "dataset_meta": adapter.info,
        "video_probe_per_ep": video_probe_per_ep,
        "profile": profile,
    }

    dimensions: list[DimensionScore] = []
    for scorer in SCORERS:
        dim = scorer(episodes, **ctx)
        # If catalog says dimension not applicable, force N/A (unless already na)
        if dim.id not in applicable and dim.score is not None:
            dim = DimensionScore(
                id=dim.id,
                name=dim.name,
                score=None,
                status="na",
                confidence=dim.confidence,
                evidence={**dim.evidence, "skipped_by_profile": True},
                notes=[f"not applicable for family={profile.family}"] + dim.notes,
            )
        dimensions.append(dim)

    overall, status, gates_failed = aggregate_scores(dimensions)
    return ReviewReport(
        dataset_path=str(adapter.root),
        format=adapter.format_id,
        n_episodes_total=adapter.n_episodes,
        n_episodes_scored=len(episodes),
        role_map=adapter.role_map(),
        dimensions=dimensions,
        overall_score=overall,
        status=status,
        gates_failed=gates_failed,
        meta={
            "fps": adapter.fps,
            "coverage": adapter.coverage(),
            "adapter_info": adapter.info,
            "n_episodes_on_disk": len(available_ids),
            "seed": seed,
            "max_episodes": max_episodes,
            "profile": profile.to_dict(),
            "applicable_dimensions": sorted(applicable),
        },
    )


def review_catalog(
    catalog_csv: str | Path,
    *,
    search_roots: list[str | Path] | None = None,
    max_episodes: int = 10,
    seed: int = 0,
    roles_config: str | Path | None = None,
) -> dict[str, Any]:
    """Review all datasets listed in Notion CSV.

    - Always emit capability profile.
    - If a local root is found under search_roots, run full scoring.
    """
    profiles = load_catalog(Path(catalog_csv))
    roots = [Path(r) for r in (search_roots or [])]
    results = []
    for p in profiles:
        entry: dict[str, Any] = {
            "name": p.name,
            "profile": p.to_dict(),
            "applicable_dimensions": p.applicable_dimensions,
            "local_hits": [],
            "review": None,
            "status": "profile_only",
        }
        if roots:
            hits = suggest_local_roots(p, roots)
            entry["local_hits"] = [str(h) for h in hits]
            for hit in hits[:1]:
                try:
                    report = review_dataset(
                        hit,
                        max_episodes=max_episodes,
                        seed=seed,
                        roles_config=roles_config,
                        profile=p,
                    )
                    entry["review"] = report.to_dict()
                    entry["status"] = report.status
                    break
                except Exception as e:  # noqa: BLE001
                    entry.setdefault("errors", []).append(f"{hit}: {e}")
        results.append(entry)

    n_local = sum(1 for r in results if r.get("review"))
    return {
        "catalog": str(Path(catalog_csv).resolve()),
        "n_datasets": len(results),
        "n_with_local_review": n_local,
        "results": results,
    }


def save_report(report: ReviewReport | dict[str, Any], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, ReviewReport) else report
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
