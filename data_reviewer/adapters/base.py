from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from ..types import EpisodeArrays


class DatasetAdapter(ABC):
    """Unified read interface for heterogeneous dataset layouts."""

    format_id: str = "unknown"

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    @abstractmethod
    def can_open(cls, root: Path) -> bool: ...

    @abstractmethod
    def role_map(self) -> dict[str, list[str]]: ...

    @abstractmethod
    def coverage(self) -> dict[str, Any]: ...

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def info(self) -> dict[str, Any]:
        return {}

    @property
    def n_episodes(self) -> int:
        return len(self.available_episode_ids())

    @abstractmethod
    def available_episode_ids(self) -> list[int]: ...

    @abstractmethod
    def iter_episodes(
        self,
        *,
        max_episodes: int | None = None,
        episode_ids: list[int] | None = None,
        seed: int = 0,
    ) -> Iterator[EpisodeArrays]: ...
