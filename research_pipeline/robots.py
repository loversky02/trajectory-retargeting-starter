"""Actual Menagerie FK and bounded position IK for Panda and UR5e."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .models import MODELS, verify_models


SPECS = {
    "panda": ("hand", [f"joint{i}" for i in range(1, 8)]),
    "ur5e": ("attachment_site", ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]),
}


class Robot:
    def __init__(self, name: str, cache: Path):
        verify_models(cache)
        directory, xml = MODELS[name]
        self.model = mujoco.MjModel.from_xml_path(str(cache / directory / xml))
        self.data = mujoco.MjData(self.model)
        endpoint, joints = SPECS[name]
        self.site = name == "ur5e"
        self.endpoint_id = (self.model.site(endpoint).id if self.site else self.model.body(endpoint).id)
        self.qpos_ids = [self.model.jnt_qposadr[self.model.joint(joint).id] for joint in joints]
        joint_ids = [self.model.joint(joint).id for joint in joints]
        self.lower = self.model.jnt_range[joint_ids, 0].copy()
        self.upper = self.model.jnt_range[joint_ids, 1].copy()
        self.name = name
        home_id = self.model.key("home").id
        self.home = self.model.key_qpos[home_id][self.qpos_ids].copy()
        self.home_position = self.fk(self.home)

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[self.qpos_ids] = q
        mujoco.mj_forward(self.model, self.data)
        return (self.data.site_xpos[self.endpoint_id] if self.site else self.data.xpos[self.endpoint_id]).copy()

    def solve(self, target: np.ndarray, initial: np.ndarray) -> dict:
        # Translation-only research transfer: endpoint orientation is unconstrained.
        result = least_squares(lambda q: self.fk(q) - target, np.clip(initial, self.lower, self.upper),
                               bounds=(self.lower, self.upper), max_nfev=70, ftol=1e-8, xtol=1e-8)
        executed = self.fk(result.x)
        error = float(np.linalg.norm(executed - target))
        at_limit = bool(np.any((result.x - self.lower < 1e-4) | (self.upper - result.x < 1e-4)))
        status = "limit_or_unreachable" if (at_limit and error > 0.005) else ("ik_residual" if error > 0.005 else "ok")
        return {"requested_position_m": target.tolist(), "executed_position_m": executed.tolist(),
                "joint_positions_rad": result.x.tolist(), "position_error_m": error,
                "at_joint_limit": at_limit, "status": status}


def retarget(observations: dict, cache: Path, reference_marker_position: np.ndarray | None = None) -> dict:
    """Marker displacement -> Panda FK execution -> UR5e relative displacement."""
    panda, ur5e = Robot("panda", cache), Robot("ur5e", cache)
    valid = observations["valid"]
    if reference_marker_position is not None:
        origin = np.asarray(reference_marker_position, dtype=float)
    elif valid.any():
        origin = observations["T_world_marker"][np.flatnonzero(valid)[0], :3, 3]
    else:
        origin = None
    if origin is not None and (origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError("reference marker position must be a finite 3-vector")
    previous_panda, previous_ur = panda.home.copy(), ur5e.home.copy()
    rows = []
    for index, (timestamp, is_valid) in enumerate(zip(observations["timestamps"], valid)):
        if not is_valid:
            rows.append({"frame": index, "timestamp_s": float(timestamp), "status": "missing_marker",
                         "panda": None, "ur5e": None})
            continue
        if origin is None:
            rows.append({"frame": index, "timestamp_s": float(timestamp), "status": "no_alignment",
                         "panda": None, "ur5e": None})
            continue
        displacement = observations["T_world_marker"][index, :3, 3] - origin
        panda_result = panda.solve(panda.home_position + displacement, previous_panda)
        previous_panda = np.array(panda_result["joint_positions_rad"])
        if panda_result["status"] != "ok":
            rows.append({"frame": index, "timestamp_s": float(timestamp), "status": "source_panda_ik_failure",
                         "panda": panda_result, "ur5e": None})
            continue
        # Transfer only the *executed* Panda displacement, not the requested one.
        panda_executed_delta = np.array(panda_result["executed_position_m"]) - panda.home_position
        ur_result = ur5e.solve(ur5e.home_position + panda_executed_delta, previous_ur)
        previous_ur = np.array(ur_result["joint_positions_rad"])
        rows.append({"frame": index, "timestamp_s": float(timestamp),
                     "status": "ok" if panda_result["status"] == ur_result["status"] == "ok" else "kinematic_failure",
                     "panda": panda_result, "ur5e": ur_result})
    return {"convention": "T_A_B maps B coordinates into A; marker position in world is displacement source",
            "alignment_status": "aligned" if origin is not None else "no_valid_marker",
            "transfer": "position displacement relative to first detected marker; Panda home endpoint to UR5e home endpoint",
            "panda_endpoint": "hand body origin", "ur5e_endpoint": "attachment_site",
            "orientation": "not constrained; no grasp transform or physical task success inferred",
            "reference_marker_position_m": origin.tolist() if origin is not None else None,
            "panda_home_position_m": panda.home_position.tolist(), "ur5e_home_position_m": ur5e.home_position.tolist(),
            "frames": rows}
