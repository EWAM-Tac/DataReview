from __future__ import annotations

import csv
import re
from pathlib import Path

from .profile import DatasetProfile, infer_profile_from_text


def _norm_key(k: str) -> str:
    return re.sub(r'[\s"\ufeff]', "", k)


# Known logical fields. Extra CSV columns are preserved in profile.extra /
# overrides — the catalog is a living reference and will grow.
COLUMN_ALIASES = {
    "name": ["数据集名称", "Name", "Dataset", "dataset_name", "名称"],
    "train_type": ["训练类型", "Type", "train_type", "类型"],
    "description": ["描述字段", "Description", "描述", "备注"],
    "volume": ["数据体量", "Volume", "Scale", "体量", "规模"],
    "url": ["下载链接", "URL", "Link", "链接", "HF"],
    "contents": ["包含内容", "Contents", "内容", "模态"],
    "subject": ["主体", "Subject", "Embodiment"],
}

# Optional override columns when expanding the catalog with explicit flags
OVERRIDE_ALIASES = {
    "family": ["family", "族", "数据族"],
    "expects_action": ["expects_action", "有动作"],
    "expects_state": ["expects_state", "有状态"],
    "expects_image": ["expects_image", "有图像"],
    "expects_tactile": ["expects_tactile", "有触觉"],
    "expects_force": ["expects_force", "有力矩"],
    "expects_hand_pose": ["expects_hand_pose", "有手姿态"],
    "expects_language": ["expects_language", "有语言"],
    "expects_sequence": ["expects_sequence", "有序列"],
    "expects_contact_pairs": ["expects_contact_pairs", "接触配对"],
}


def _pick(row: dict[str, str], field: str, aliases: dict[str, list[str]] | None = None) -> str:
    aliases = aliases or COLUMN_ALIASES
    names = aliases[field]
    for a in names:
        for k, v in row.items():
            if _norm_key(k) == _norm_key(a):
                return (v or "").strip()
    for a in names:
        for k, v in row.items():
            if _norm_key(a) in _norm_key(k):
                return (v or "").strip()
    return ""


def _claimed_keys(row: dict[str, str]) -> set[str]:
    claimed: set[str] = set()
    for aliases in (COLUMN_ALIASES, OVERRIDE_ALIASES):
        for names in aliases.values():
            for a in names:
                for k in row:
                    if _norm_key(k) == _norm_key(a) or _norm_key(a) in _norm_key(k):
                        claimed.add(k)
    return claimed


def load_catalog(csv_path: Path | str, *, rules_path: Path | None = None) -> list[DatasetProfile]:
    """Load a living dataset catalog CSV (Notion export or compatible).

    - Required-ish: a name column (数据集名称 / Name / …)
    - Other columns are optional; unknown columns are kept in ``profile.extra``
    - Optional override columns (family / expects_*) pin inference for new rows
    - New datasets = append rows; no scorer code changes
    """
    csv_path = Path(csv_path)
    profiles: list[DatasetProfile] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip fully empty rows
            if not any((v or "").strip() for v in row.values()):
                continue
            name = _pick(row, "name")
            if not name:
                continue
            claimed = _claimed_keys(row)
            extra = {
                k: (v or "").strip()
                for k, v in row.items()
                if k not in claimed and (v or "").strip()
            }
            overrides = {
                field: _pick(row, field, OVERRIDE_ALIASES)
                for field in OVERRIDE_ALIASES
                if _pick(row, field, OVERRIDE_ALIASES)
            }
            profiles.append(
                infer_profile_from_text(
                    name=name,
                    train_type=_pick(row, "train_type"),
                    description=_pick(row, "description"),
                    contents=_pick(row, "contents"),
                    subject=_pick(row, "subject"),
                    urls=_pick(row, "url"),
                    volume=_pick(row, "volume"),
                    extra=extra,
                    overrides=overrides or None,
                    rules_path=rules_path,
                )
            )
    return profiles


# backwards-compatible alias
load_notion_catalog = load_catalog


def _looks_like_dataset_root(p: Path) -> bool:
    if (p / "meta" / "info.json").exists():
        return True
    if (p / "lerobot_v2.1" / "meta" / "info.json").exists():
        return True
    if any(p.glob("*.hdf5")) or any(p.glob("*.h5")):
        return True
    return False


def _normalize_hit(p: Path) -> Path:
    if (p / "meta" / "info.json").exists():
        return p
    nested = p / "lerobot_v2.1"
    if (nested / "meta" / "info.json").exists():
        return nested
    return p


_STOP_TOKENS = {
    "sample",
    "samples",
    "data",
    "dataset",
    "datasets",
    "train",
    "test",
    "val",
    "real",
    "world",
    "open",
    "public",
    "v1",
    "v2",
    "v3",
    "the",
    "and",
    "for",
}


def _name_tokens(name: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9]+", name)
    toks = [t for t in raw if len(t) >= 4 and t.lower() not in _STOP_TOKENS]
    toks.sort(key=lambda t: (-len(t), t.lower()))
    return toks


def suggest_local_roots(profile: DatasetProfile, search_roots: list[Path]) -> list[Path]:
    """Heuristic: find local folders whose name fuzzy-matches dataset name."""
    tokens = _name_tokens(profile.name)
    if not tokens:
        return []
    primary = tokens[0].lower()
    hits: list[Path] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates = [root, *root.rglob("*")]
        for p in candidates:
            if not p.is_dir():
                continue
            blob = f"{p.name} {root.name}".lower()
            if primary not in blob:
                continue
            if _looks_like_dataset_root(p):
                hits.append(_normalize_hit(p))
    seen = set()
    out: list[Path] = []
    for h in sorted(
        hits,
        key=lambda x: (0 if x.name == "lerobot_v2.1" or (x / "meta").exists() else 1, str(x)),
    ):
        rp = h.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(h)
    return out[:10]
