from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..roles import load_role_config, map_features_to_roles, summarize_role_coverage
from ..types import EpisodeArrays
from .base import DatasetAdapter


def _list_h5_files(root: Path) -> list[Path]:
    files = []
    for pat in ("*.hdf5", "*.h5", "**/*.hdf5", "**/*.h5"):
        files.extend(root.glob(pat))
    # unique
    seen = set()
    out = []
    for f in sorted(files):
        r = f.resolve()
        if r not in seen and f.is_file():
            seen.add(r)
            out.append(f)
    return out


def _discover_keys(h5) -> list[str]:
    keys: list[str] = []

    def visitor(name, obj):
        import h5py

        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    h5.visititems(visitor)
    return keys


def _pick_array(h5, candidates: list[str]) -> np.ndarray | None:
    for c in candidates:
        if c in h5:
            arr = np.asarray(h5[c])
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr.astype(np.float32, copy=False)
        # try nested
        for k in h5.keys():
            if k.endswith(c) or c.endswith(k):
                try:
                    arr = np.asarray(h5[k])
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    return arr.astype(np.float32, copy=False)
                except Exception:
                    pass
    return None


class Hdf5Adapter(DatasetAdapter):
    """Best-effort HDF5 reader for robot / tactile trajectories.

    Supports:
      - one episode per file
      - one file with groups episode_000 / demo_0 / traj_0
    """

    format_id = "hdf5"

    def __init__(self, root: Path, role_config: Path | None = None):
        super().__init__(root)
        self._files = _list_h5_files(root)
        if not self._files:
            raise FileNotFoundError(f"no hdf5 under {root}")
        self._role_config = role_config
        self._patterns = load_role_config(role_config)
        # probe first file for feature keys
        self._feature_keys = self._probe_features()
        self._role_map = map_features_to_roles(self._feature_keys, self._patterns)
        self._coverage = summarize_role_coverage(self._role_map)
        self._episode_index = self._build_index()

    @classmethod
    def can_open(cls, root: Path) -> bool:
        return len(_list_h5_files(Path(root))) > 0

    def _probe_features(self) -> list[str]:
        import h5py

        keys = set()
        with h5py.File(self._files[0], "r") as f:
            # if top-level looks like episodes, open first child
            child_eps = [k for k in f.keys() if re.search(r"(episode|demo|traj|ep)_?\d+", k, re.I)]
            if child_eps:
                g = f[sorted(child_eps)[0]]
                keys.update(_discover_keys(g) if hasattr(g, "visititems") else list(g.keys()))
            else:
                keys.update(_discover_keys(f))
        # also add common aliases as synthetic keys for role mapping from leaf names
        leaves = set()
        for k in keys:
            leaves.add(k)
            leaves.add(k.split("/")[-1])
        return sorted(leaves)

    def _build_index(self) -> list[tuple[Path, str | None]]:
        """List of (file, group_or_None)."""
        import h5py

        index: list[tuple[Path, str | None]] = []
        for fp in self._files:
            with h5py.File(fp, "r") as f:
                child_eps = sorted(
                    [k for k in f.keys() if re.search(r"(episode|demo|traj|ep)_?\d+", k, re.I)]
                )
                if child_eps:
                    for g in child_eps:
                        index.append((fp, g))
                else:
                    index.append((fp, None))
        return index

    def role_map(self) -> dict[str, list[str]]:
        return self._role_map

    def coverage(self) -> dict[str, Any]:
        return self._coverage

    @property
    def info(self) -> dict[str, Any]:
        return {"n_h5_files": len(self._files), "feature_keys_sample": self._feature_keys[:40]}

    def available_episode_ids(self) -> list[int]:
        return list(range(len(self._episode_index)))

    def iter_episodes(
        self,
        *,
        max_episodes: int | None = None,
        episode_ids: list[int] | None = None,
        seed: int = 0,
    ) -> Iterator[EpisodeArrays]:
        import h5py

        ids = episode_ids if episode_ids is not None else self.available_episode_ids()
        if max_episodes is not None and max_episodes < len(ids):
            rng = np.random.default_rng(seed)
            ids = sorted(rng.choice(ids, size=max_episodes, replace=False).tolist())

        # map roles to candidate dataset names
        cand = {
            "action": ["action", "actions", "action/joint_position", "abs_actions"],
            "state": ["state", "states", "observation/state", "obs", "proprio"],
            "tactile": [
                "tactile",
                "tactile_force",
                "force",
                "wrench",
                "gelsight",
                "digit",
                "pressure",
                "observation/tactile",
            ],
            "timestamp": ["timestamp", "time", "t"],
        }
        # also include role_map keys
        for role, keys in self._role_map.items():
            cand.setdefault(role, [])
            cand[role] = list(dict.fromkeys(list(keys) + cand.get(role, [])))

        for ei in ids:
            fp, group = self._episode_index[ei]
            with h5py.File(fp, "r") as f:
                node = f[group] if group else f
                arrays: dict[str, Any] = {}
                for role, names in cand.items():
                    # try exact then endswith
                    found = None
                    for n in names:
                        if n in node:
                            found = np.asarray(node[n])
                            break
                        # search datasets
                        matches = []

                        def vis(name, obj, _n=n, _m=matches):
                            import h5py as _h5

                            if isinstance(obj, _h5.Dataset) and (
                                name == _n or name.endswith("/" + _n) or name.split("/")[-1] == _n
                            ):
                                _m.append(name)

                        if hasattr(node, "visititems"):
                            node.visititems(vis)
                        if matches:
                            found = np.asarray(node[matches[0]])
                            break
                    if found is None:
                        continue
                    if found.ndim == 1:
                        found = found[:, None]
                    # skip huge image tensors in hdf5 for scoring speed unless tactile small
                    if found.ndim >= 3 and found.shape[-1] in (1, 3) and found.shape[1] > 32:
                        # treat as video-like — skip numeric scorer load
                        continue
                    arrays[role] = found.astype(np.float32, copy=False)

                # length
                n = 0
                for v in arrays.values():
                    n = max(n, int(v.shape[0]))
                if n == 0:
                    continue
                yield EpisodeArrays(
                    episode_index=ei,
                    n_frames=n,
                    arrays=arrays,
                    video_paths={},
                    meta={"hdf5": str(fp), "group": group},
                )
