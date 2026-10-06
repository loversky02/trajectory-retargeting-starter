"""Closed-loop physics execution of a dense TCP plan on one robot scene."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from . import episode as ep
from . import transforms as tf
from .kinematics import TcpIK
from .scene import RobotScene

TCP_DEFINITIONS = {
    "panda": "Panda hand body + 0.1034 m along hand z (finger pad centre); z approach, y finger closing axis",
    "ur5e_2f85": "Robotiq 2F-85 base + 0.140 m along base z (pad centre, 5 mm inside pinch site); "
                 "z approach, y finger closing axis",
}


@dataclass
class Plan:
    """Per-control-tick TCP targets. ``source_frame`` is -1 for inserted transitions."""
    timestamps: np.ndarray
    tcp_targets: np.ndarray
    gripper: np.ndarray
    source_frame: np.ndarray
    segment: list[str]
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.timestamps)
        if not (len(self.tcp_targets) == len(self.gripper) == len(self.source_frame) == len(self.segment) == n):
            raise ValueError("plan arrays must have equal length")
        if n < 2 or (np.diff(self.timestamps) <= 0).any():
            raise ValueError("plan timestamps must be strictly increasing")
        if not all(tf.is_rigid(t) for t in self.tcp_targets):
            raise ValueError("every plan TCP target must be a finite rigid transform")

    def to_json(self) -> dict:
        positions, quats = ep.poses_to_arrays(self.tcp_targets)
        return {"schema_version": "physics_replay.plan.v1", "timestamps_s": self.timestamps,
                "tcp_target_position_m": positions, "tcp_target_quat_wxyz": quats,
                "gripper_command": self.gripper, "source_frame": self.source_frame,
                "segment": self.segment, "notes": self.notes}


def execute(scene: RobotScene, plan: Plan, control_hz: float,
            on_frame: Callable[[RobotScene, int], None] | None = None) -> tuple[dict, dict]:
    """Track the plan with IK + joint position servos; return (episode, diagnostics).

    Frame k of the episode is the state at the end of control period k.
    """
    scene.reset()
    ik = TcpIK(scene)
    dt = 1.0 / control_hz
    substeps = int(round(dt / scene.model.opt.timestep))
    q_command = scene.home_q.copy()
    n = len(plan.timestamps)
    tcp, objects = np.zeros((n, 4, 4)), np.zeros((n, 4, 4))
    joints = np.zeros((n, len(scene.arm_qpos_ids)))
    ik_position, ik_rotation = np.zeros(n), np.zeros(n)
    track_position, track_rotation = np.zeros(n), np.zeros(n)
    floor_contacts = np.zeros(n, dtype=int)
    joint_limit = np.zeros(n, dtype=bool)
    for k in range(n):
        result = ik.solve(plan.tcp_targets[k], q_command)
        q_command = result.q
        ik_position[k], ik_rotation[k], joint_limit[k] = (result.position_error_m, result.rotation_error_rad,
                                                           result.at_joint_limit)
        scene.data.ctrl[scene.arm_actuator_ids] = q_command
        scene.data.ctrl[scene.gripper_actuator_id] = scene.gripper_ctrl(plan.gripper[k])
        contacts = 0
        for _ in range(substeps):
            mujoco.mj_step(scene.model, scene.data)
            contacts = max(contacts, scene.robot_floor_contacts())
        floor_contacts[k] = contacts
        tcp[k], objects[k] = scene.tcp_pose(), scene.object_pose()
        joints[k] = scene.data.qpos[scene.arm_qpos_ids]
        track_position[k], track_rotation[k] = tf.pose_error(tcp[k], plan.tcp_targets[k])
        if on_frame is not None:
            on_frame(scene, k)
    episode = ep.build(scene.robot, TCP_DEFINITIONS[scene.robot], scene.spec.to_json(),
                       np.asarray(plan.timestamps) + dt, tcp, objects, plan.gripper, joints,
                       extra={"control_hz": control_hz, "segment": list(plan.segment),
                              "source_frame": np.asarray(plan.source_frame).tolist()})
    diagnostics = {
        "ik_position_residual_m": ik_position, "ik_rotation_residual_rad": ik_rotation,
        "tracking_position_error_m": track_position, "tracking_rotation_error_rad": track_rotation,
        "robot_floor_contacts": floor_contacts, "commanded_joint_at_limit": joint_limit,
    }
    return episode, diagnostics


def min_jerk(alpha: np.ndarray | float):
    return 10 * alpha ** 3 - 15 * alpha ** 4 + 6 * alpha ** 5


def waypoint_plan(start: np.ndarray, waypoints: list[tuple[str, np.ndarray, float, float]],
                  control_hz: float, t0: float = 0.0) -> Plan:
    """Min-jerk interpolate (label, T_world_tcp, gripper, duration_s) waypoints from ``start``."""
    targets, gripper, labels = [], [], []
    previous = start
    for label, target, closed, duration in waypoints:
        steps = max(1, int(round(duration * control_hz)))
        for i in range(1, steps + 1):
            targets.append(tf.interpolate(previous, target, min_jerk(i / steps)))
            gripper.append(closed)
            labels.append(label)
        previous = target
    n = len(targets)
    return Plan(t0 + np.arange(n) / control_hz, np.array(targets), np.array(gripper, dtype=float),
                np.full(n, -1), labels)
