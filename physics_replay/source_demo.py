"""Scripted Panda pick-and-place: the clean-state source demonstration."""
from __future__ import annotations

import numpy as np

from . import transforms as tf
from .execute import Plan, waypoint_plan

# TCP pointing down (z = -world z) with the finger closing axis along world x.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def wrap_quarter_turn(angle: float) -> float:
    """A cube looks the same every 90 degrees; return the equivalent yaw in [-pi/4, pi/4)."""
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def yaw_of(transform: np.ndarray) -> float:
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def scripted_waypoints(T_world_object: np.ndarray, T_world_goal: np.ndarray, demo: dict) -> list:
    """Approach, grasp, lift, carry, place, release, retreat. Durations come from the config."""
    grasp_yaw = wrap_quarter_turn(yaw_of(T_world_object))
    place_yaw = grasp_yaw + wrap_quarter_turn(yaw_of(T_world_goal) - yaw_of(T_world_object))
    grasp_rotation = tf.rot_z(grasp_yaw) @ R_DOWN
    place_rotation = tf.rot_z(place_yaw) @ R_DOWN
    up = np.array([0.0, 0.0, 1.0])
    grasp = T_world_object[:3, 3] + up * demo["grasp_above_center_m"]
    place = T_world_goal[:3, 3] + up * (demo["grasp_above_center_m"] + demo["place_clearance_m"])
    d = demo["durations_s"]
    return [
        ("approach", tf.make(grasp_rotation, grasp + up * demo["pregrasp_height_m"]), 0.0, d["approach"]),
        ("descend", tf.make(grasp_rotation, grasp), 0.0, d["descend"]),
        ("close", tf.make(grasp_rotation, grasp), 1.0, d["close"]),
        ("lift", tf.make(grasp_rotation, grasp + up * demo["lift_height_m"]), 1.0, d["lift"]),
        ("carry", tf.make(place_rotation, place + up * demo["lift_height_m"]), 1.0, d["carry"]),
        ("lower", tf.make(place_rotation, place), 1.0, d["lower"]),
        ("release", tf.make(place_rotation, place), 0.0, d["release"]),
        ("retreat", tf.make(place_rotation, place + up * demo["pregrasp_height_m"]), 0.0, d["retreat"]),
    ]


def source_plan(scene, demo: dict, control_hz: float) -> Plan:
    """Dense plan from the robot's home TCP through the scripted waypoints."""
    return waypoint_plan(scene.home_tcp_pose(), scripted_waypoints(np.asarray(scene.spec.T_world_object),
                                                   np.asarray(scene.spec.T_world_goal), demo), control_hz)
