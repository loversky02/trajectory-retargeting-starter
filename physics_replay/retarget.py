"""Source episode -> target-robot TCP plan.

``world_frame`` replays the source TCP poses at the same world coordinates.
``object_relative`` splits the demo once the grasped object has been lifted:
frames before the split are expressed relative to the object's initial pose,
frames after it relative to the goal pose, and each part is re-anchored on the
target scene's object / goal. A free-space transition joins the two parts.
``object_relative_trimmed`` does the same but drops the source's free-space
approach: object-anchored frames start where the source TCP last came within
``approach_radius_m`` of the object, and the planned transition from the target
robot's own start pose covers the rest.

Segmentation reads only the recorded gripper command and TCP height, never the
script's phase labels, so any episode that satisfies the schema can be retargeted.
"""
from __future__ import annotations

import numpy as np

from . import episode as ep
from . import transforms as tf
from .execute import Plan, min_jerk
from .scene import SceneSpec

METHODS = ("world_frame", "object_relative", "object_relative_trimmed")


def segment(episode: dict, lift_threshold_m: float) -> dict:
    gripper = np.asarray(episode["gripper_command"], dtype=float)
    closed = gripper >= 0.5
    if not closed.any():
        raise ValueError("source episode never closes the gripper")
    grasp = int(np.argmax(closed))
    z = np.asarray(episode["tcp_position_m"], dtype=float)[:, 2]
    risen = np.flatnonzero(closed[grasp:] & (z[grasp:] >= z[grasp] + lift_threshold_m))
    if not len(risen):
        raise ValueError("source episode never lifts after grasping")
    reopened = np.flatnonzero(~closed[grasp:])
    return {"grasp_frame": grasp, "split_frame": grasp + int(risen[0]),
            "release_frame": grasp + int(reopened[0]) if len(reopened) else None}


def source_object_reference(episode: dict, grasp_frame: int) -> np.ndarray:
    """Object pose at the first frame where it was observed before the grasp."""
    poses = ep.object_poses(episode)
    observed = [i for i in range(grasp_frame + 1) if np.isfinite(poses[i]).all()]
    if not observed:
        raise ValueError("object pose is never observed before the grasp")
    return poses[observed[0]]


def approach_start(episode: dict, grasp_frame: int, T_world_object: np.ndarray, radius_m: float) -> int:
    """First frame of the last run before the grasp in which the TCP stays within ``radius_m`` of the object."""
    distance = np.linalg.norm(np.asarray(episode["tcp_position_m"], dtype=float)[:grasp_frame + 1]
                              - T_world_object[:3, 3], axis=1)
    outside = np.flatnonzero(distance > radius_m)
    return int(outside[-1]) + 1 if len(outside) else 0


def _transition(start: np.ndarray, end: np.ndarray, gripper: float, control_hz: float, config: dict):
    position_m, rotation_rad = tf.pose_error(start, end)
    duration = max(config["transition_min_s"], position_m / config["transition_speed_mps"],
                   rotation_rad / config["transition_rotation_speed_radps"])
    steps = max(1, int(round(duration * control_hz)))
    poses = [tf.interpolate(start, end, min_jerk(i / steps)) for i in range(1, steps + 1)]
    return poses, [gripper] * steps


def _resample(times: np.ndarray, poses: np.ndarray, gripper: np.ndarray, control_hz: float):
    """Resample a pose stream onto the control clock; gripper is zero-order hold."""
    grid = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * control_hz)) + 1) / control_hz
    out_poses, out_gripper, out_index = [], [], []
    for t in grid:
        j = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
        k = min(j + 1, len(times) - 1)
        alpha = 0.0 if k == j else (t - times[j]) / (times[k] - times[j])
        out_poses.append(tf.interpolate(poses[j], poses[k], alpha))
        out_gripper.append(gripper[j])
        out_index.append(j if alpha < 0.5 else k)
    return out_poses, out_gripper, out_index


def retarget(episode: dict, target: SceneSpec, target_start_tcp: np.ndarray, method: str,
             control_hz: float, config: dict,
             T_source_tcp_target_tcp: np.ndarray | None = None) -> Plan:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    ep.validate(episode)
    offset = np.eye(4) if T_source_tcp_target_tcp is None else T_source_tcp_target_tcp
    times = np.asarray(episode["timestamps_s"], dtype=float)
    source_tcp = ep.tcp_poses(episode)
    gripper = np.asarray(episode["gripper_command"], dtype=float)
    cuts = segment(episode, config["segment_lift_threshold_m"])
    split = cuts["split_frame"]

    if method == "world_frame":
        anchors = [("world", np.eye(4), 0, len(times))]
        notes = {"anchor": "source world coordinates replayed unchanged"}
    else:
        T_obj_src = source_object_reference(episode, cuts["grasp_frame"])
        T_goal_src = np.asarray(episode["scene"]["T_world_goal"], dtype=float)
        to_object = np.asarray(target.T_world_object) @ tf.inverse(T_obj_src)
        to_goal = np.asarray(target.T_world_goal) @ tf.inverse(T_goal_src)
        first = 0
        if method == "object_relative_trimmed":
            first = approach_start(episode, cuts["grasp_frame"], T_obj_src, config["approach_radius_m"])
            cuts["approach_frame"] = first
        anchors = [("object", to_object, first, split), ("goal", to_goal, split, len(times))]
        notes = {"anchor": "pre-split frames follow the object, post-split frames follow the goal",
                 "T_target_source_object": to_object.tolist(), "T_target_source_goal": to_goal.tolist()}

    poses, grip, frames, labels = [], [], [], []
    current = np.asarray(target_start_tcp, dtype=float)
    for name, transform, start, stop in anchors:
        mapped = np.array([transform @ pose @ offset for pose in source_tcp[start:stop]])
        seg_poses, seg_grip, seg_index = _resample(times[start:stop], mapped, gripper[start:stop], control_hz)
        bridge, bridge_grip = _transition(current, seg_poses[0], seg_grip[0], control_hz, config)
        poses += bridge + seg_poses
        grip += bridge_grip + seg_grip
        frames += [-1] * len(bridge) + [start + i for i in seg_index]
        labels += [f"transition_to_{name}"] * len(bridge) + [name] * len(seg_poses)
        current = seg_poses[-1]
    notes.update({"method": method, "segmentation": cuts,
                  "T_source_tcp_target_tcp": offset.tolist(),
                  "source_episode_robot": episode["robot"]})
    n = len(poses)
    return Plan(np.arange(n) / control_hz, np.array(poses), np.array(grip, dtype=float),
                np.array(frames), labels, notes)
