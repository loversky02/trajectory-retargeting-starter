"""Tabletop pick-and-place scenes around pinned Menagerie robots.

Both robots stand at the world origin on the floor (z = 0 is the table surface).
The TCP site of each gripper sits at the centre of its finger pads with +z along
the approach direction and +y along the finger closing axis, so a TCP pose
transfers between the two grippers without an extra offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import transforms as tf
from .models import XML, verify_models

ROBOTS = ("panda", "ur5e_2f85")
PANDA_ARM = [f"joint{i}" for i in range(1, 8)]
UR5E_ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# Pad centre along the approach axis, measured from the official finger geoms.
PANDA_TCP_OFFSET_M = 0.1034          # from the hand body origin
ROBOTIQ_TCP_OFFSET_M = 0.140         # from the 2F-85 base body (pinch site is at 0.145)


@dataclass(frozen=True)
class CubeSpec:
    half_size_m: float = 0.025
    mass_kg: float = 0.1
    friction: tuple[float, float, float] = (1.5, 0.02, 0.0002)


@dataclass(frozen=True)
class SceneSpec:
    """World poses of the cube centre at rest and of the goal (placed cube centre)."""
    T_world_object: np.ndarray
    T_world_goal: np.ndarray
    cube: CubeSpec = field(default_factory=CubeSpec)

    def to_json(self) -> dict:
        return {"T_world_object": np.asarray(self.T_world_object).tolist(),
                "T_world_goal": np.asarray(self.T_world_goal).tolist(),
                "cube": {"half_size_m": self.cube.half_size_m, "mass_kg": self.cube.mass_kg,
                         "friction": list(self.cube.friction)}}


def tabletop_pose(x: float, y: float, yaw: float, half_size: float) -> np.ndarray:
    return tf.make(tf.rot_z(yaw), np.array([x, y, half_size]))


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # MuJoCo cameras look along -z with +y up.
    return tf.matrix_to_quat_wxyz(np.column_stack([right, up, -forward]))


class RobotScene:
    """A compiled MuJoCo model plus the indices a controller needs."""

    def __init__(self, robot: str, spec: SceneSpec, cache: Path, verify: bool = True):
        if robot not in ROBOTS:
            raise ValueError(f"robot must be one of {ROBOTS}")
        cache = Path(cache)
        if verify:
            verify_models(cache)
        self.robot = robot
        self.spec = spec
        model_spec, arm_joints, home = self._robot_spec(robot, cache)
        self._add_world(model_spec, spec)
        self.model = model_spec.compile()
        self.data = mujoco.MjData(self.model)

        self.arm_joint_ids = [self.model.joint(name).id for name in arm_joints]
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_lower = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_upper = self.model.jnt_range[self.arm_joint_ids, 1].copy()
        actuated = {int(self.model.actuator_trnid[a, 0]): a for a in range(self.model.nu)
                    if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self.arm_actuator_ids = np.array([actuated[j] for j in self.arm_joint_ids])
        self.gripper_actuator_id = next(a for a in range(self.model.nu) if a not in set(self.arm_actuator_ids))
        self.tcp_site_id = self.model.site("tcp").id
        self.object_body_id = self.model.body("object").id
        self.object_qpos_adr = self.model.jnt_qposadr[self.model.joint("object_free").id]
        self.floor_geom_id = self.model.geom("floor").id
        self.home_q = home
        robot_bodies = {b for b in range(1, self.model.nbody)
                        if b != self.object_body_id and self.model.body_rootid[b] != self.object_body_id}
        self.robot_geom = np.isin(self.model.geom_bodyid, list(robot_bodies))

    # --- construction -------------------------------------------------
    @staticmethod
    def _robot_spec(robot: str, cache: Path):
        if robot == "panda":
            spec = mujoco.MjSpec.from_file(str(cache / XML["panda"]))
            spec.body("hand").add_site(name="tcp", pos=[0, 0, PANDA_TCP_OFFSET_M], size=[0.004], group=5)
            arm = PANDA_ARM
        else:
            spec = mujoco.MjSpec.from_file(str(cache / XML["ur5e"]))
            gripper = mujoco.MjSpec.from_file(str(cache / XML["2f85"]))
            # Match the gripper's solver options so attaching it reports no conflict.
            spec.option.impratio = gripper.option.impratio
            spec.option.cone = gripper.option.cone
            spec.site("attachment_site").attach_body(gripper.worldbody.first_body(), "gripper_", "")
            spec.body("gripper_base").add_site(name="tcp", pos=[0, 0, ROBOTIQ_TCP_OFFSET_M], size=[0.004], group=5)
            arm = UR5E_ARM
        # Real arm controllers compensate gravity; the Menagerie servos alone sag under load.
        for body in spec.bodies:
            if body.name not in ("", "world"):
                body.gravcomp = 1.0
        home_key = spec.key("home")
        source = mujoco.MjSpec.from_file(str(cache / (XML["panda"] if robot == "panda" else XML["ur5e"])))
        source_model = source.compile()
        key_qpos = np.array(source_model.key_qpos[source_model.key("home").id])
        home = np.array([key_qpos[source_model.jnt_qposadr[source_model.joint(n).id]] for n in arm])
        spec.delete(home_key)  # qpos layout changes once the cube is added; we set state explicitly
        return spec, arm, home

    @staticmethod
    def _add_world(model_spec: mujoco.MjSpec, spec: SceneSpec) -> None:
        model_spec.option.impratio = 10
        model_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model_spec.visual.global_.offwidth = 1280
        model_spec.visual.global_.offheight = 960
        world = model_spec.worldbody
        world.add_light(pos=[0.5, -0.5, 2.5], dir=[0, 0.2, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)
        world.add_light(pos=[-1, 1, 2], dir=[0.5, -0.5, -1], diffuse=[0.3, 0.3, 0.3], castshadow=False)
        world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05],
                       rgba=[0.82, 0.8, 0.76, 1], friction=[1.0, 0.005, 0.0001])
        cube = spec.cube
        goal = np.asarray(spec.T_world_goal)
        world.add_geom(name="goal_marker", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[cube.half_size_m * 1.4, cube.half_size_m * 1.4, 0.0005],
                       pos=[goal[0, 3], goal[1, 3], 0.0005], quat=tf.matrix_to_quat_wxyz(goal[:3, :3]),
                       rgba=[0.15, 0.7, 0.3, 0.6], contype=0, conaffinity=0)
        pose = np.asarray(spec.T_world_object)
        body = world.add_body(name="object", pos=pose[:3, 3], quat=tf.matrix_to_quat_wxyz(pose[:3, :3]))
        body.add_freejoint(name="object_free")
        body.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube.half_size_m] * 3,
                      mass=cube.mass_kg, friction=list(cube.friction), condim=4, rgba=[0.85, 0.2, 0.15, 1])
        focus = np.array([0.45, 0.0, 0.1])
        for name, eye in (("overview", np.array([1.35, -1.0, 0.85])), ("side", np.array([0.5, -1.25, 0.45]))):
            world.add_camera(name=name, pos=eye, quat=_look_at_quat(eye, focus), fovy=45)

    # --- state --------------------------------------------------------
    def reset(self, settle_s: float = 0.3) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_ids] = self.home_q
        pose = np.asarray(self.spec.T_world_object)
        adr = self.object_qpos_adr
        self.data.qpos[adr:adr + 3] = pose[:3, 3]
        self.data.qpos[adr + 3:adr + 7] = tf.matrix_to_quat_wxyz(pose[:3, :3])
        self.data.ctrl[self.arm_actuator_ids] = self.home_q
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_ctrl(0.0)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(round(settle_s / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def gripper_ctrl(self, closed: float) -> float:
        """Map a normalised command (0 open, 1 closed) to the actuator's native range."""
        closed = float(np.clip(closed, 0.0, 1.0))
        # Panda: ctrl 255 = fully open fingers. Robotiq 2F-85: ctrl 255 = fully closed.
        return 255.0 * (1.0 - closed) if self.robot == "panda" else 255.0 * closed

    def tcp_pose(self, data: mujoco.MjData | None = None) -> np.ndarray:
        data = self.data if data is None else data
        return tf.make(data.site_xmat[self.tcp_site_id].reshape(3, 3).copy(), data.site_xpos[self.tcp_site_id].copy())

    def home_tcp_pose(self) -> np.ndarray:
        """TCP pose at the home posture, where every executed plan starts."""
        data = mujoco.MjData(self.model)
        data.qpos[self.arm_qpos_ids] = self.home_q
        mujoco.mj_kinematics(self.model, data)
        return self.tcp_pose(data)

    def object_pose(self) -> np.ndarray:
        return tf.make(self.data.xmat[self.object_body_id].reshape(3, 3).copy(), self.data.xpos[self.object_body_id].copy())

    def robot_floor_contacts(self) -> int:
        count = 0
        for contact in self.data.contact[:self.data.ncon]:
            pair = (contact.geom1, contact.geom2)
            if self.floor_geom_id in pair and any(self.robot_geom[g] for g in pair if g != self.floor_geom_id):
                count += 1
        return count
