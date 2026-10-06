"""Damped least-squares 6-D TCP inverse kinematics on a scratch MjData."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import transforms as tf


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    iterations: int
    at_joint_limit: bool


class TcpIK:
    """Solve q such that the scene's TCP site matches a world pose.

    Uses a separate MjData so the physics state is never touched. A small
    null-space pull toward the home posture keeps the 7-DoF Panda from drifting.
    """

    def __init__(self, scene, damping: float = 0.03, rotation_weight: float = 0.3,
                 max_step_rad: float = 0.25, posture_gain: float = 0.05):
        self.scene = scene
        self.model = scene.model
        self.data = mujoco.MjData(scene.model)
        self.damping = damping
        self.rotation_weight = rotation_weight
        self.max_step = max_step_rad
        self.posture_gain = posture_gain
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def forward(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[self.scene.arm_qpos_ids] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.scene.tcp_pose(self.data)

    def solve(self, T_world_target: np.ndarray, q_init: np.ndarray, iterations: int = 60,
              position_tolerance_m: float = 2e-4, rotation_tolerance_rad: float = 2e-3) -> IKResult:
        lower, upper = self.scene.arm_lower, self.scene.arm_upper
        q = np.clip(np.asarray(q_init, dtype=float), lower, upper)
        n = len(q)
        weights = np.r_[np.ones(3), np.full(3, self.rotation_weight)]
        position_error = rotation_error = np.inf
        iteration = 0
        for iteration in range(1, iterations + 1):
            current = self.forward(q)
            dp = T_world_target[:3, 3] - current[:3, 3]
            dr = current[:3, :3] @ tf.rotation_log(current[:3, :3].T @ T_world_target[:3, :3])
            position_error, rotation_error = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            if position_error < position_tolerance_m and rotation_error < rotation_tolerance_rad:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.scene.tcp_site_id)
            jacobian = np.vstack([self._jacp, self._jacr])[:, self.scene.arm_dof_ids] * weights[:, None]
            error = np.r_[dp, dr] * weights
            gram = jacobian @ jacobian.T + (self.damping ** 2) * np.eye(6)
            pseudo_inverse = jacobian.T @ np.linalg.solve(gram, np.eye(6))
            step = pseudo_inverse @ error
            if n > 6:
                step += (np.eye(n) - pseudo_inverse @ jacobian) @ (self.posture_gain * (self.scene.home_q - q))
            norm = np.linalg.norm(step)
            if norm > self.max_step:
                step *= self.max_step / norm
            q = np.clip(q + step, lower, upper)
        else:
            current = self.forward(q)
            position_error, rotation_error = tf.pose_error(current, T_world_target)
        at_limit = bool(np.any((q - lower < 1e-3) | (upper - q < 1e-3)))
        return IKResult(q, position_error, rotation_error, iteration, at_limit)
