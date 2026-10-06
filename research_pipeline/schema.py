"""Frame conventions: T_A_B maps coordinates in B into coordinates in A."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def rigid_transform(value: object, name: str = "transform") -> np.ndarray:
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has invalid homogeneous row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal and right-handed")
    return t


def timestamps(values: object) -> np.ndarray:
    ts = np.asarray(values, dtype=float)
    if ts.ndim != 1 or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("timestamps must be a strictly increasing finite vector")
    return ts


def calibration(value: dict) -> dict:
    camera = value["camera"]
    k = np.asarray(camera["K"], dtype=float)
    d = np.asarray(camera["distortion"], dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("camera K must have positive focal lengths")
    if d.ndim != 1 or len(d) not in (4, 5, 8) or not np.isfinite(d).all():
        raise ValueError("camera distortion must have 4, 5, or 8 finite coefficients")
    rigid_transform(camera["T_world_camera"], "T_world_camera")
    if not all(isinstance(camera[key], int) and not isinstance(camera[key], bool) and camera[key] > 0
               for key in ("width", "height")):
        raise ValueError("image dimensions must be positive integers")
    marker_size = value["marker_size_m"]
    if not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("image size and marker size must be positive")
    if not np.isfinite(value["fps"]) or value["fps"] <= 0:
        raise ValueError("fps must be positive and finite")
    for key in ("frames", "segment_length"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("budget_s", "oracle_edit_cost_s"):
        number = value[key]
        if not np.isfinite(number) or (number < 0 if key == "budget_s" else number <= 0):
            raise ValueError(f"{key} must be finite and nonnegative; edit cost positive")
    if not isinstance(value["marker_id"], int) or not 0 <= value["marker_id"] < 50:
        raise ValueError("marker_id must be an integer in DICT_4X4_50")
    return value


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != "il-research-prototype.v1":
        raise ValueError("unsupported configuration schema")
    return calibration(value)


def validate_poses(ts: object, transforms: object, valid: object) -> None:
    t = timestamps(ts)
    poses = np.asarray(transforms)
    mask = np.asarray(valid)
    if poses.shape != (len(t), 4, 4) or mask.shape != (len(t),) or mask.dtype != bool:
        raise ValueError("pose arrays must agree on frame count and validity")
    for pose in poses[mask]:
        rigid_transform(pose, "valid pose")
    if not np.isnan(poses[~mask]).all():
        raise ValueError("missing poses must be NaN, never plausible coordinates")
