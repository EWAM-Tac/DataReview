from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_ROLE_PATTERNS: dict[str, list[str]] = {
    "state": [
        r"^observation\.state$",
        r"^observation\.states?$",
        r"^state$",
        r"^states$",
        r"^obs\.state$",
        r"^proprio",
        r"^qpos$",
        r"observation/state",
    ],
    "state_tcp": [
        r"^observation\.state\.tcp",
        r"^observation\.tcp",
        r"^observation\..*ee_pose",
        r"^observation\..*tcp_pose",
    ],
    "state_torque": [r"joint_torque", r"(^|[._/])torque([._/]|$)", r"(^|[._/])effort([._/]|$)"],
    "action": [r"^action$", r"^actions$", r"^action\.joints?$", r"abs_actions", r"action/joint"],
    "action_tcp": [r"^action\.tcp", r"^action\..*ee", r"action/ee", r"^action\..*tcp"],
    "tactile": [
        r"^observation\.tactile$",
        r"^observation\.tactiles?",
        r"^tactile$",
        r"tactile_force",
        r"tactile_taxel",
        r"taxel",
        r"pressure",
        r"uskin",
        r"observation/tactile",
    ],
    "force": [
        r"^force$",
        r"^forces$",
        r"wrench",
        r"ft_sensor",
        r"force_torque",
        r"force-torque",
        r"(^|[._/])ati([._/]|$)",
        r"(^|[._/])axia([._/]|$)",
        r"observation\.force",
        r"observation\.wrench",
    ],
    "tactile_image": [
        r"tactile_deform",
        r"tactile_raw",
        r"gelsight",
        r"gelslim",
        r"digit",
        r"freetacman",
        r"exumi",
        r"vitam",
        r"mctac",
    ],
    "image": [
        r"^observation\.images\.",
        r"^observation\.image\.",
        r"^images?\.",
        r"^camera",
        r"wrist",
        r"head_",
        r"^ego",
        r"front",
    ],
    "hand_pose": [r"mano", r"hand_pose", r"hand_joints", r"keypoints", r"skeleton"],
    "timestamp": [r"^timestamp$", r"^time$"],
    "frame_index": [r"^frame_index$"],
    "episode_index": [r"^episode_index$"],
    "task_index": [r"^task_index$"],
    "subtask_index": [r"^subtask_index$", r"^subtask$"],
    "language": [r"instruction", r"caption", r"language", r"task_description"],
}


def load_role_config(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {k: list(v) for k, v in DEFAULT_ROLE_PATTERNS.items()}
    if yaml is None:
        raise RuntimeError("PyYAML required to load role config; pip install pyyaml")
    raw = yaml.safe_load(path.read_text())
    roles = raw.get("roles", raw)
    return {str(role): [str(p) for p in pats] for role, pats in roles.items()}


def map_features_to_roles(
    feature_keys: list[str],
    patterns: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    patterns = patterns or DEFAULT_ROLE_PATTERNS
    compiled = {
        role: [re.compile(p, re.IGNORECASE) for p in pats] for role, pats in patterns.items()
    }
    # Specific roles first; loose proxies (force) after state/action/image
    # so short substrings cannot steal RGB / proprio keys.
    priority = [
        "tactile_image",
        "tactile",
        "action_tcp",
        "state_tcp",
        "state_torque",
        "hand_pose",
        "language",
        "state",
        "action",
        "image",
        "force",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
        "subtask_index",
    ]
    for r in patterns:
        if r not in priority:
            priority.append(r)

    role_map: dict[str, list[str]] = {r: [] for r in patterns}
    claimed: set[str] = set()

    for role in priority:
        if role not in compiled:
            continue
        for key in feature_keys:
            if key in claimed:
                continue
            if any(cre.search(key) for cre in compiled[role]):
                role_map[role].append(key)
                claimed.add(key)

    return {r: ks for r, ks in role_map.items() if ks}


def summarize_role_coverage(role_map: dict[str, list[str]]) -> dict[str, Any]:
    has = set(role_map)
    return {
        "has_state": "state" in has or "state_tcp" in has,
        "has_action": "action" in has or "action_tcp" in has,
        "has_image": "image" in has,
        "has_tactile": "tactile" in has or "tactile_image" in has or "force" in has,
        "has_force": "force" in has,
        "has_hand_pose": "hand_pose" in has,
        "has_timestamp": "timestamp" in has,
        "has_subtask": "subtask_index" in has,
        "has_task": "task_index" in has,
        "n_image_streams": len(role_map.get("image", [])),
        "n_tactile_image_streams": len(role_map.get("tactile_image", [])),
    }
