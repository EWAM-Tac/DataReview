from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DimensionScore:
    id: str
    name: str
    score: float | None  # None = N/A (modality missing)
    confidence: float = 1.0
    status: str = "ok"  # ok | warn | fail | na
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewReport:
    dataset_path: str
    format: str
    n_episodes_total: int
    n_episodes_scored: int
    role_map: dict[str, list[str]]
    dimensions: list[DimensionScore]
    overall_score: float | None
    status: str  # ok | warn | fail
    gates_failed: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dimensions"] = [x.to_dict() if hasattr(x, "to_dict") else x for x in self.dimensions]
        return d


@dataclass
class EpisodeArrays:
    """Numeric arrays for one episode after role resolution."""

    episode_index: int
    n_frames: int
    arrays: dict[str, Any]  # role -> np.ndarray (T, ...) or list of video paths
    video_paths: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
