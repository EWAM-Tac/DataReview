from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from ..loader_lerobot import LeRobotDatasetView
from ..types import EpisodeArrays
from .base import DatasetAdapter


class LeRobotAdapter(DatasetAdapter):
    format_id = "lerobot_v2.1"

    def __init__(self, root: Path, role_config: Path | None = None):
        super().__init__(root)
        self._view = LeRobotDatasetView(root, role_config=role_config)

    @classmethod
    def can_open(cls, root: Path) -> bool:
        root = Path(root)
        if (root / "meta" / "info.json").exists():
            return True
        # nested
        return (root / "lerobot_v2.1" / "meta" / "info.json").exists()

    @classmethod
    def resolve_root(cls, root: Path) -> Path:
        root = Path(root)
        if (root / "meta" / "info.json").exists():
            return root
        nested = root / "lerobot_v2.1"
        if (nested / "meta" / "info.json").exists():
            return nested
        # deepest meta/info.json
        hits = list(root.rglob("meta/info.json"))
        if hits:
            return hits[0].parent.parent
        raise FileNotFoundError(root)

    def role_map(self) -> dict[str, list[str]]:
        return self._view.role_map

    def coverage(self) -> dict[str, Any]:
        return self._view.coverage

    @property
    def fps(self) -> float:
        return self._view.fps

    @property
    def info(self) -> dict[str, Any]:
        return self._view.info

    def available_episode_ids(self) -> list[int]:
        return self._view.available_episode_ids()

    def iter_episodes(self, **kwargs) -> Iterator[EpisodeArrays]:
        return self._view.iter_episodes(**kwargs)
