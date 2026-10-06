"""Manipulation episode schema: what a demonstration log must carry to be retargeted.

One JSON file per episode. Arrays are per-frame; a missing pose is a row of NaN
(never a plausible coordinate). Quaternions are (w, x, y, z), lengths in metres,
time in seconds, gripper command normalised to 0 (open) .. 1 (closed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import transforms as tf

SCHEMA_VERSION = "physics_replay.episode.v1"
POSE_FIELDS = ("tcp", "object")
REQUIRED = ("schema_version", "robot", "frame_convention", "tcp_definition", "scene",
            "timestamps_s", "tcp_position_m", "tcp_quat_wxyz", "object_position_m",
            "object_quat_wxyz", "gripper_command", "joint_positions_rad")


def poses_to_arrays(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=float)
    positions = np.full((len(poses), 3), np.nan)
    quats = np.full((len(poses), 4), np.nan)
    for i, pose in enumerate(poses):
        if np.isfinite(pose).all():
            positions[i] = pose[:3, 3]
            quats[i] = tf.matrix_to_quat_wxyz(pose[:3, :3])
    return positions, quats


def arrays_to_poses(positions: np.ndarray, quats: np.ndarray) -> np.ndarray:
    poses = np.full((len(positions), 4, 4), np.nan)
    for i, (position, quat) in enumerate(zip(positions, quats)):
        if np.isfinite(position).all() and np.isfinite(quat).all():
            poses[i] = tf.make(tf.quat_wxyz_to_matrix(quat), position)
    return poses


def validate(episode: dict) -> dict:
    """Raise ValueError on any contract violation; return the episode unchanged."""
    missing = [key for key in REQUIRED if key not in episode]
    if missing:
        raise ValueError(f"episode is missing fields: {missing}")
    if episode["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {episode['schema_version']!r}")
    timestamps = np.asarray(episode["timestamps_s"], dtype=float)
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all() \
            or (np.diff(timestamps) <= 0).any():
        raise ValueError("timestamps_s must be a strictly increasing finite vector with >= 2 frames")
    n = len(timestamps)
    for name in POSE_FIELDS:
        positions = np.asarray(episode[f"{name}_position_m"], dtype=float)
        quats = np.asarray(episode[f"{name}_quat_wxyz"], dtype=float)
        if positions.shape != (n, 3) or quats.shape != (n, 4):
            raise ValueError(f"{name} pose arrays must have shape ({n}, 3) and ({n}, 4)")
        row_valid = np.isfinite(positions).all(1) & np.isfinite(quats).all(1)
        row_missing = np.isnan(positions).all(1) & np.isnan(quats).all(1)
        if not (row_valid | row_missing).all():
            raise ValueError(f"{name} pose rows must be fully finite or fully NaN (missing)")
        norms = np.linalg.norm(quats[row_valid], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError(f"{name}_quat_wxyz must hold unit quaternions")
    if not np.isfinite(np.asarray(episode["tcp_position_m"], dtype=float)).all():
        raise ValueError("a robot's own TCP log may not have missing frames")
    gripper = np.asarray(episode["gripper_command"], dtype=float)
    if gripper.shape != (n,) or not np.isfinite(gripper).all() or (gripper < 0).any() or (gripper > 1).any():
        raise ValueError("gripper_command must be n values in [0, 1]")
    joints = np.asarray(episode["joint_positions_rad"], dtype=float)
    if joints.ndim != 2 or len(joints) != n or not np.isfinite(joints).all():
        raise ValueError("joint_positions_rad must be a finite (n, dof) array")
    for key in ("T_world_object", "T_world_goal"):
        if not tf.is_rigid(np.asarray(episode["scene"][key], dtype=float)):
            raise ValueError(f"scene.{key} must be a rigid 4x4 transform")
    return episode


def build(robot: str, tcp_definition: str, scene: dict, timestamps: np.ndarray, tcp_poses: np.ndarray,
          object_poses: np.ndarray, gripper: np.ndarray, joints: np.ndarray, extra: dict | None = None) -> dict:
    tcp_position, tcp_quat = poses_to_arrays(tcp_poses)
    object_position, object_quat = poses_to_arrays(object_poses)
    episode = {
        "schema_version": SCHEMA_VERSION,
        "robot": robot,
        "frame_convention": "T_A_B maps coordinates in frame B into frame A; world z up, table surface z = 0",
        "units": {"length": "m", "angle": "rad", "time": "s", "quaternion": "wxyz"},
        "tcp_definition": tcp_definition,
        "scene": scene,
        "timestamps_s": np.asarray(timestamps, dtype=float).tolist(),
        "tcp_position_m": tcp_position.tolist(),
        "tcp_quat_wxyz": tcp_quat.tolist(),
        "object_position_m": object_position.tolist(),
        "object_quat_wxyz": object_quat.tolist(),
        "gripper_command": np.asarray(gripper, dtype=float).tolist(),
        "joint_positions_rad": np.asarray(joints, dtype=float).tolist(),
    }
    episode.update(extra or {})
    return validate(episode)


def tcp_poses(episode: dict) -> np.ndarray:
    return arrays_to_poses(np.asarray(episode["tcp_position_m"]), np.asarray(episode["tcp_quat_wxyz"]))


def object_poses(episode: dict) -> np.ndarray:
    return arrays_to_poses(np.asarray(episode["object_position_m"], dtype=float),
                           np.asarray(episode["object_quat_wxyz"], dtype=float))


def to_json_value(value):
    """Recursively convert numpy values to JSON types, writing NaN as null."""
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_json_value(item) for item in (value.tolist() if isinstance(value, np.ndarray) else value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def save(episode: dict, path: Path) -> None:
    validate(episode)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_value(episode), allow_nan=False))


def load(path: Path) -> dict:
    episode = json.loads(Path(path).read_text())
    for name in POSE_FIELDS:
        for key in (f"{name}_position_m", f"{name}_quat_wxyz"):
            episode[key] = [[np.nan if v is None else v for v in row] for row in episode[key]]
    return validate(episode)
