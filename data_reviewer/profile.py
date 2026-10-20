from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DatasetProfile:
    """Capability profile — modalities / family claimed by metadata or local coverage.

    Catalog CSV rows are a living reference: new datasets only need text fields
    (or optional override columns). Unknown CSV columns are kept in ``extra``.
    """

    name: str
    train_type: str = ""
    description: str = ""
    contents: list[str] = field(default_factory=list)
    subject: str = ""
    urls: list[str] = field(default_factory=list)
    volume_text: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    expects_action: bool = False
    expects_state: bool = False
    expects_image: bool = False
    expects_tactile: bool = False
    expects_force: bool = False
    expects_hand_pose: bool = False
    expects_language: bool = False
    expects_sequence: bool = True
    expects_contact_pairs: bool = False
    family: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def applicable_dimensions(self) -> list[str]:
        dims = ["completeness", "integrity"]
        if self.expects_sequence and (
            self.expects_action or self.expects_state or self.expects_hand_pose
        ):
            dims.append("dynamics")
        if self.expects_sequence and (
            (self.expects_action and self.expects_state)
            or self.expects_tactile
            or self.expects_force
        ):
            dims.append("consistency")
        if self.expects_language or "语言" in self.description:
            dims.append("annotation")
        if self.expects_tactile or self.expects_force:
            dims.append("tactile")
        dims.append("trainability")
        return dims


_RULES_CACHE: dict[str, Any] | None = None


def _default_rules_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "profile_rules.yaml"


def load_profile_rules(path: Path | None = None) -> dict[str, Any]:
    global _RULES_CACHE
    path = Path(path) if path else _default_rules_path()
    key = str(path.resolve()) if path.exists() else "__builtin__"
    if _RULES_CACHE is not None and _RULES_CACHE.get("_key") == key:
        return _RULES_CACHE
    if path.exists() and yaml is not None:
        raw = yaml.safe_load(path.read_text()) or {}
    else:
        raw = {}
    raw["_key"] = key
    _RULES_CACHE = raw
    return raw


def _kw_hit(blob: str, keys: list[str]) -> bool:
    return any(k.lower() in blob for k in keys)


def _apply_overrides(p: DatasetProfile, overrides: dict[str, str]) -> None:
    """Optional CSV columns for explicit control when the catalog grows.

    Recognized overrides (case-insensitive keys):
      family, expects_action, expects_state, expects_image, expects_tactile,
      expects_force, expects_hand_pose, expects_language, expects_sequence
    Boolean values: 1/0, true/false, yes/no, 是/否
    """
    bool_map = {
        "1": True,
        "true": True,
        "yes": True,
        "y": True,
        "是": True,
        "0": False,
        "false": False,
        "no": False,
        "n": False,
        "否": False,
    }
    flag_fields = (
        "expects_action",
        "expects_state",
        "expects_image",
        "expects_tactile",
        "expects_force",
        "expects_hand_pose",
        "expects_language",
        "expects_sequence",
        "expects_contact_pairs",
    )
    for k, v in overrides.items():
        key = k.strip().lower()
        val = (v or "").strip()
        if not val:
            continue
        if key == "family":
            p.family = val
            continue
        if key in flag_fields:
            b = bool_map.get(val.lower())
            if b is not None:
                setattr(p, key, b)


def infer_profile_from_text(
    *,
    name: str,
    train_type: str = "",
    description: str = "",
    contents: str = "",
    subject: str = "",
    urls: str = "",
    volume: str = "",
    extra: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
    rules_path: Path | None = None,
) -> DatasetProfile:
    rules = load_profile_rules(rules_path)
    mod = rules.get("modality_keywords") or {}
    tags = rules.get("content_tags") or {"action": ["动作"], "tactile": ["触觉"]}
    families = rules.get("families") or []

    blob = " ".join([name, train_type, description, contents, subject]).lower()
    contents_list = [c.strip() for c in contents.replace("，", ",").split(",") if c.strip()]

    p = DatasetProfile(
        name=name,
        train_type=train_type,
        description=description,
        contents=contents_list,
        subject=subject,
        urls=[u.strip() for u in urls.replace("|", "\n").split() if u.strip().startswith("http")],
        volume_text=volume,
        extra=dict(extra or {}),
    )

    action_tags = tags.get("action") or ["动作"]
    tactile_tags = tags.get("tactile") or ["触觉"]
    has_action_tag = any(any(t in c for t in action_tags) for c in contents_list)
    has_tactile_tag = any(any(t in c for t in tactile_tags) for c in contents_list)

    p.expects_tactile = has_tactile_tag or _kw_hit(blob, mod.get("tactile") or [])
    p.expects_force = _kw_hit(blob, mod.get("force") or [])
    p.expects_image = _kw_hit(blob, mod.get("image") or [])
    p.expects_hand_pose = _kw_hit(blob, mod.get("hand_pose") or [])
    p.expects_language = _kw_hit(blob, mod.get("language") or [])
    p.expects_action = has_action_tag or _kw_hit(blob, mod.get("action") or [])

    # family: first matching rule
    family_set = False
    for rule in families:
        rid = rule.get("id") or "unknown"
        any_keys = [str(x).lower() for x in (rule.get("any") or [])]
        if any_keys and any(k in blob for k in any_keys):
            p.family = rid
            for sk, sv in (rule.get("set") or {}).items():
                if hasattr(p, sk):
                    setattr(p, sk, sv)
            family_set = True
            break
        if rule.get("subject_only_human") and "人类" in subject and "机器人" not in subject:
            p.family = rid
            family_set = True
            break
        if rule.get("subject_only_robot") and "机器人" in subject and "人类" not in subject:
            p.family = rid
            family_set = True
            break
        if rule.get("fallback") and not family_set:
            p.family = rid
            family_set = True
            break
    if not family_set:
        p.family = "mixed"

    p.expects_state = p.expects_action or _kw_hit(blob, mod.get("state") or [])

    if p.family == "human_hoi" and not has_action_tag:
        p.expects_hand_pose = True
    if "不含" in description and "手关节" in description:
        p.expects_action = False
        p.expects_hand_pose = False

    # explicit overrides win (for catalog expansion without fighting heuristics)
    if overrides:
        _apply_overrides(p, overrides)

    return p
