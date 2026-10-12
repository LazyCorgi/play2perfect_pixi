"""Shared storage helpers for the calibrated UR5e-O30 scene state."""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = REPO_ROOT / "isaacsimenvs/cfg/robot/ur5e_o30_scene.json"
POSE_KEYS = ("table", "hole", "peg")


def _validate_pose(name: str, pose: object) -> None:
    if not isinstance(pose, dict):
        raise ValueError(f"{name} must be an object")
    position = pose.get("position")
    rotation = pose.get("rotation_wxyz")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"{name}.position must contain 3 numbers")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ValueError(f"{name}.rotation_wxyz must contain 4 numbers")
    if not all(math.isfinite(float(value)) for value in position + rotation):
        raise ValueError(f"{name} contains a non-finite value")


def load_scene_state(path: Path = DEFAULT_STATE_PATH) -> dict:
    state = json.loads(path.read_text())
    if state.get("schema_version") != 1:
        raise ValueError(f"Unsupported scene-state schema in {path}")
    robot = state.get("robot")
    if not isinstance(robot, dict) or not isinstance(robot.get("joint_positions"), dict):
        raise ValueError("robot.joint_positions must be an object")
    _validate_pose("robot.base_pose", robot.get("base_pose"))
    for key in POSE_KEYS:
        _validate_pose(key, state.get(key))
    return state


def save_scene_state(state: dict, path: Path = DEFAULT_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)
