from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .roles import load_role_config, map_features_to_roles, summarize_role_coverage
from .types import EpisodeArrays


def _as_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _col_to_numpy(table, name: str) -> np.ndarray:
    """Convert a pyarrow column (possibly fixed_size_list) to (T, D) float/int array."""
    import pyarrow as pa

    col = table[name]
    # Try to convert list columns
    try:
        # fixed_size_list / list
        if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
            flat = col.values.to_numpy(zero_copy_only=False)
            n = len(col)
            # infer width
            if pa.types.is_fixed_size_list(col.type):
                w = col.type.list_size
            else:
                # variable — fall back to row iteration
                rows = []
                for i in range(n):
                    rows.append(np.asarray(col[i].as_py(), dtype=np.float32).reshape(-1))
                return np.stack(rows, axis=0)
            return np.asarray(flat, dtype=np.float32).reshape(n, w)
        return _as_2d(col.to_numpy(zero_copy_only=False))
    except Exception:
        rows = []
        for i in range(len(col)):
            v = col[i].as_py()
            rows.append(np.asarray(v, dtype=np.float32).reshape(-1))
        return np.stack(rows, axis=0)


class LeRobotDatasetView:
    """Minimal LeRobot v2.1 reader for quality scoring."""

    def __init__(self, root: Path, role_config: Path | None = None):
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        if not self.meta_dir.exists():
            raise FileNotFoundError(f"no meta/ under {root}")
        self.info = json.loads((self.meta_dir / "info.json").read_text())
        self.features: dict[str, Any] = self.info.get("features", {})
        self.fps = float(self.info.get("fps") or 30)
        self.video_root = self._read_video_root()
        patterns = load_role_config(role_config)
        self.role_map = map_features_to_roles(list(self.features.keys()), patterns)
        self.coverage = summarize_role_coverage(self.role_map)
        self._episodes_meta = self._load_episodes_meta()

    def _load_episodes_meta(self) -> list[dict[str, Any]]:
        path = self.meta_dir / "episodes.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    @property
    def n_episodes(self) -> int:
        return int(self.info.get("total_episodes") or len(self._episodes_meta) or 0)

    def available_episode_ids(self) -> list[int]:
        """Episode indices that actually have parquet on disk (partial samples OK)."""
        ids: list[int] = []
        for p in sorted((self.root / "data").rglob("episode_*.parquet")):
            try:
                ids.append(int(p.stem.split("_")[1]))
            except Exception:
                continue
        return ids

    def episode_parquet_path(self, ep: int) -> Path | None:
        # data/chunk-XXX/episode_YYYYYY.parquet
        pattern = self.info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        chunk_size = int(self.info.get("chunks_size") or 1000)
        chunk = ep // chunk_size
        try:
            path = self.root / pattern.format(episode_chunk=chunk, episode_index=ep)
        except Exception:
            path = self.root / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        return path if path.exists() else None

    def _read_video_root(self) -> Path | None:
        path = self.meta_dir / "adjust.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = payload.get("video_root") if isinstance(payload, dict) else None
        return Path(raw) if raw else None

    def _video_candidate(self, root: Path, key: str, ep: int) -> Path:
        pattern = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        chunk_size = int(self.info.get("chunks_size") or 1000)
        chunk = ep // chunk_size
        try:
            return root / pattern.format(episode_chunk=chunk, episode_index=ep, video_key=key)
        except Exception:
            return root / f"videos/chunk-{chunk:03d}/{key}/episode_{ep:06d}.mp4"

    def video_path(self, key: str, ep: int) -> Path | None:
        path = self._video_candidate(self.root, key, ep)
        if path.is_file():
            return path
        if self.video_root is not None:
            shared = self._video_candidate(self.video_root, key, ep)
            if shared.is_file():
                return shared
        return None

    def iter_episodes(
        self,
        *,
        max_episodes: int | None = None,
        episode_ids: list[int] | None = None,
        seed: int = 0,
    ) -> Iterator[EpisodeArrays]:
        import pyarrow.parquet as pq

        if episode_ids is None:
            ids = self.available_episode_ids()
            if not ids:
                ids = list(range(self.n_episodes))
            if max_episodes is not None and max_episodes < len(ids):
                rng = np.random.default_rng(seed)
                ids = sorted(rng.choice(ids, size=max_episodes, replace=False).tolist())
        else:
            ids = list(episode_ids)

        # roles that live in parquet (not video dtype)
        video_keys = {
            k for k, v in self.features.items() if str(v.get("dtype", "")).lower() == "video"
        }
        numeric_roles = [
            r
            for r, keys in self.role_map.items()
            if any(k not in video_keys for k in keys)
        ]

        for ep in ids:
            pq_path = self.episode_parquet_path(ep)
            if pq_path is None:
                continue
            table = pq.read_table(pq_path)
            n = table.num_rows
            arrays: dict[str, Any] = {}
            for role in numeric_roles:
                keys = [k for k in self.role_map[role] if k in table.column_names]
                if not keys:
                    continue
                # primary key = first
                arrays[role] = _col_to_numpy(table, keys[0])
                if len(keys) > 1 and role in ("image", "tactile_image"):
                    pass  # videos handled below
                elif len(keys) > 1 and role.startswith("state"):
                    # keep primary only; extras available under role__key
                    for k in keys[1:]:
                        arrays[f"{role}__{k}"] = _col_to_numpy(table, k)

            videos: dict[str, str] = {}
            for role in ("image", "tactile_image"):
                for k in self.role_map.get(role, []):
                    vp = self.video_path(k, ep)
                    if vp is not None:
                        videos[k] = str(vp)

            meta = {}
            if ep < len(self._episodes_meta):
                meta = dict(self._episodes_meta[ep])
            yield EpisodeArrays(
                episode_index=ep,
                n_frames=int(n),
                arrays=arrays,
                video_paths=videos,
                meta=meta,
            )


def detect_and_open(root: Path, role_config: Path | None = None) -> LeRobotDatasetView:
    root = Path(root)
    if (root / "meta" / "info.json").exists():
        return LeRobotDatasetView(root, role_config=role_config)
    # maybe nested lerobot_v2.1
    cand = list(root.rglob("meta/info.json"))
    if cand:
        return LeRobotDatasetView(cand[0].parent.parent, role_config=role_config)
    raise FileNotFoundError(f"Cannot find LeRobot meta/info.json under {root}")
