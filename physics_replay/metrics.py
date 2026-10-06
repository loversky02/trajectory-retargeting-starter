"""Pick-and-place success predicate and failure taxonomy.

Success = the cube was lifted, ends at rest inside the goal tolerance, is not
tipped, the gripper is open at the end, every commanded pose was reachable, and
no robot geom touched the table. Failure reasons are checked in the order of
FAILURE_ORDER and the first match is reported; every flag is kept as well.
"""
from __future__ import annotations

import numpy as np

from . import episode as ep

FAILURE_ORDER = ("ik_unreachable", "floor_collision", "grasp_failed", "dropped_in_transit",
                 "tipped", "place_miss", "not_released")


def _tilt_rad(rotation: np.ndarray) -> float:
    """Angle between world z and the closest cube face normal (any face up counts)."""
    return float(np.arccos(np.clip(np.max(np.abs(rotation[2, :])), -1.0, 1.0)))


def evaluate(episode: dict, diagnostics: dict, success: dict, ik: dict) -> dict:
    positions = np.asarray(episode["object_position_m"], dtype=float)
    objects = ep.object_poses(episode)
    gripper = np.asarray(episode["gripper_command"], dtype=float)
    goal = np.asarray(episode["scene"]["T_world_goal"], dtype=float)
    half = float(episode["scene"]["cube"]["half_size_m"])
    closed = gripper >= 0.5
    rest_z = positions[0, 2]

    lift_m = float(positions[closed, 2].max() - rest_z) if closed.any() else 0.0
    lifted = lift_m >= success["lift_min_m"]
    xy_to_goal = np.linalg.norm(positions[:, :2] - goal[:2, 3], axis=1)
    on_table_away = (positions[:, 2] < half + success["rest_z_tolerance_m"]) & (xy_to_goal > success["place_xy_tolerance_m"])
    first_lift = int(np.argmax(positions[:, 2] - rest_z >= success["lift_min_m"])) if lifted else None
    dropped = bool(lifted and (closed[first_lift:] & on_table_away[first_lift:]).any())

    final_xy = float(xy_to_goal[-1])
    final_z = float(abs(positions[-1, 2] - half))
    tilt = _tilt_rad(objects[-1][:3, :3])
    ik_position = float(np.max(diagnostics["ik_position_residual_m"]))
    ik_rotation = float(np.max(diagnostics["ik_rotation_residual_rad"]))
    contact_ticks = int(np.count_nonzero(diagnostics["robot_floor_contacts"]))
    flags = {
        "ik_unreachable": ik_position > ik["max_position_residual_m"] or ik_rotation > ik["max_rotation_residual_rad"],
        "floor_collision": contact_ticks > 0,
        "grasp_failed": not lifted,
        "dropped_in_transit": dropped,
        "tipped": tilt > success["max_tilt_rad"],
        "place_miss": final_xy > success["place_xy_tolerance_m"] or final_z > success["rest_z_tolerance_m"],
        "not_released": bool(closed[-1]),
    }
    reason = next((name for name in FAILURE_ORDER if flags[name]), None)
    tracking = np.asarray(diagnostics["tracking_position_error_m"], dtype=float)
    return {
        "success": reason is None,
        "failure_reason": reason or "none",
        "flags": flags,
        "lift_m": lift_m,
        "final_xy_error_m": final_xy,
        "final_z_error_m": final_z,
        "final_tilt_rad": tilt,
        "max_ik_position_residual_m": ik_position,
        "max_ik_rotation_residual_rad": ik_rotation,
        "floor_contact_ticks": contact_ticks,
        "tracking_position_error_mean_m": float(tracking.mean()),
        "tracking_position_error_max_m": float(tracking.max()),
        "tracking_rotation_error_max_rad": float(np.max(diagnostics["tracking_rotation_error_rad"])),
        "duration_s": float(episode["timestamps_s"][-1]),
        "frames": len(episode["timestamps_s"]),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    # The bounds are exactly 0 and 1 at the extremes; rounding would otherwise put 1.0 just outside n/n.
    low = 0.0 if successes == 0 else max(0.0, float(centre - half))
    high = 1.0 if successes == trials else min(1.0, float(centre + half))
    return (low, high)
