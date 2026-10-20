from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter
from .folder_adapter import FolderAdapter
from .hdf5_adapter import Hdf5Adapter
from .lerobot_adapter import LeRobotAdapter


ADAPTERS: list[type[DatasetAdapter]] = [
    LeRobotAdapter,
    Hdf5Adapter,
    FolderAdapter,
]


def open_dataset(root: Path, role_config: Path | None = None) -> DatasetAdapter:
    """Auto-detect dataset format and return an adapter.

    Accepts a leaf LeRobot root, a season folder with nested ``lerobot_v2.1``,
    or a sample tree (task/season/...) — first openable match wins.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    errors: list[str] = []

    def _try(path: Path) -> DatasetAdapter | None:
        try:
            if LeRobotAdapter.can_open(path):
                return LeRobotAdapter(LeRobotAdapter.resolve_root(path), role_config=role_config)
            if Hdf5Adapter.can_open(path):
                return Hdf5Adapter(path, role_config=role_config)
            if FolderAdapter.can_open(path):
                return FolderAdapter(path, role_config=role_config)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path}: {e}")
        return None

    hit = _try(root)
    if hit is not None:
        return hit

    # Nested sample trees: …/task/season/lerobot_v2.1/meta/info.json
    info_hits = sorted(root.rglob("meta/info.json"))
    for info in info_hits[:20]:
        cand = info.parent.parent
        hit = _try(cand)
        if hit is not None:
            return hit

    for h5 in sorted(root.rglob("*.hdf5"))[:10] + sorted(root.rglob("*.h5"))[:10]:
        hit = _try(h5.parent)
        if hit is not None:
            return hit

    msg = " ; ".join(errors) if errors else "no adapter matched"
    raise RuntimeError(f"Cannot open dataset at {root}: {msg}")


__all__ = ["DatasetAdapter", "open_dataset", "LeRobotAdapter", "Hdf5Adapter", "FolderAdapter"]
