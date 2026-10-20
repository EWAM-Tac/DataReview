from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..roles import summarize_role_coverage
from ..types import EpisodeArrays
from .base import DatasetAdapter


def _episode_dirs(root: Path) -> list[Path]:
    # episode_000 / demo_0 / traj_0 / ep0
    cands = []
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_dir() and (
            p.name.startswith(("episode", "demo", "traj", "ep", "seq"))
            or p.name.isdigit()
        ):
            cands.append(p)
    if cands:
        return cands
    # one level deeper
    for sub in sorted(root.iterdir()) if root.is_dir() else []:
        if not sub.is_dir():
            continue
        for p in sorted(sub.iterdir()):
            if p.is_dir() and p.name.startswith(("episode", "demo", "traj", "ep")):
                cands.append(p)
    return cands


class FolderAdapter(DatasetAdapter):
    """Generic folder of episodes with .npz / .npy / images.

    Useful for contact-pair / custom dumps that are not LeRobot or HDF5.
    """

    format_id = "generic_folder"

    def __init__(self, root: Path, role_config: Path | None = None):
        super().__init__(root)
        self._eps = _episode_dirs(root)
        if not self._eps:
            # treat root itself as single episode if it has arrays
            if any(root.glob("*.npz")) or any(root.glob("*.npy")):
                self._eps = [root]
            else:
                raise FileNotFoundError(f"no episode folders under {root}")
        self._role_map = self._infer_roles(self._eps[0])
        self._coverage = summarize_role_coverage(self._role_map)

    @classmethod
    def can_open(cls, root: Path) -> bool:
        root = Path(root)
        try:
            return bool(_episode_dirs(root)) or bool(list(root.glob("*.npz"))) or bool(
                list(root.glob("*.npy"))
            )
        except Exception:
            return False

    def _infer_roles(self, ep_dir: Path) -> dict[str, list[str]]:
        role_map: dict[str, list[str]] = {}
        name_map = {
            "action": ["action", "actions", "qpos_action"],
            "state": ["state", "states", "qpos", "proprio", "obs"],
            "tactile": [
                "tactile",
                "force",
                "wrench",
                "pressure",
                "gelsight",
                "digit",
                "taxel",
            ],
            "hand_pose": ["mano", "hand_pose", "joints", "keypoints"],
        }
        files = {p.stem.lower(): p.name for p in ep_dir.glob("*") if p.suffix in {".npz", ".npy"}}
        for role, aliases in name_map.items():
            for a in aliases:
                if a in files:
                    role_map.setdefault(role, []).append(files[a])
        # images
        img_dirs = []
        for p in ep_dir.iterdir():
            if p.is_dir() and (
                any(p.glob("*.png")) or any(p.glob("*.jpg")) or any(p.glob("*.jpeg"))
            ):
                img_dirs.append(p.name)
        if img_dirs:
            # classify tactile vs rgb by name
            for d in img_dirs:
                dl = d.lower()
                if any(k in dl for k in ("tactile", "gelsight", "digit", "deform")):
                    role_map.setdefault("tactile_image", []).append(d)
                else:
                    role_map.setdefault("image", []).append(d)
        return role_map

    def role_map(self) -> dict[str, list[str]]:
        return self._role_map

    def coverage(self) -> dict[str, Any]:
        return self._coverage

    def available_episode_ids(self) -> list[int]:
        return list(range(len(self._eps)))

    def iter_episodes(
        self,
        *,
        max_episodes: int | None = None,
        episode_ids: list[int] | None = None,
        seed: int = 0,
    ) -> Iterator[EpisodeArrays]:
        ids = episode_ids if episode_ids is not None else self.available_episode_ids()
        if max_episodes is not None and max_episodes < len(ids):
            rng = np.random.default_rng(seed)
            ids = sorted(rng.choice(ids, size=max_episodes, replace=False).tolist())

        for ei in ids:
            ep_dir = self._eps[ei]
            arrays: dict[str, Any] = {}
            for role, files in self._role_map.items():
                if role in ("image", "tactile_image"):
                    continue
                for fn in files:
                    path = ep_dir / fn
                    if not path.exists():
                        # stem match
                        hits = list(ep_dir.glob(fn + ".*")) + list(ep_dir.glob(fn))
                        if not hits:
                            continue
                        path = hits[0]
                    if path.suffix == ".npz":
                        z = np.load(path)
                        # prefer array matching role name else first
                        key = role if role in z.files else z.files[0]
                        arr = z[key]
                    else:
                        arr = np.load(path)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    if arr.ndim >= 3:
                        continue
                    arrays[role] = arr.astype(np.float32, copy=False)
                    break
            n = max((a.shape[0] for a in arrays.values()), default=0)
            # image presence as video_paths markers (directory)
            videos = {}
            for role in ("image", "tactile_image"):
                for d in self._role_map.get(role, []):
                    p = ep_dir / d
                    if p.is_dir():
                        videos[d] = str(p)
            if n == 0 and not videos:
                continue
            yield EpisodeArrays(
                episode_index=ei,
                n_frames=max(n, 1),
                arrays=arrays,
                video_paths=videos,
                meta={"dir": str(ep_dir)},
            )
