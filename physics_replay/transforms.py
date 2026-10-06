"""Minimal SE(3) helpers. ``T_A_B`` maps coordinates in frame B into frame A."""
from __future__ import annotations

import numpy as np


def make(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def inverse(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    return make(rotation.T, -rotation.T @ transform[:3, 3])


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns a unit quaternion with w >= 0."""
    m = rotation
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    quat /= np.linalg.norm(quat)
    return quat if quat[0] >= 0 else -quat


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_angle(rotation: np.ndarray) -> float:
    """Geodesic angle (rad) of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (robust near 0 and pi)."""
    angle = rotation_angle(rotation)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near pi the skew part vanishes; recover the axis from the symmetric part.
        axis = np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None))
        index = int(np.argmax(axis))
        axis[np.arange(3) != index] *= np.sign(rotation[index, np.arange(3) != index] + 1e-12)
        return angle * axis / np.linalg.norm(axis)
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return angle * skew / (2.0 * np.sin(angle))


def slerp(rotation_a: np.ndarray, rotation_b: np.ndarray, alpha: float) -> np.ndarray:
    delta = rotation_a.T @ rotation_b
    omega = rotation_log(delta) * alpha
    angle = np.linalg.norm(omega)
    if angle < 1e-12:
        return rotation_a.copy()
    k = omega / angle
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return rotation_a @ (np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew)


def interpolate(transform_a: np.ndarray, transform_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear position, geodesic rotation interpolation between two poses."""
    position = (1 - alpha) * transform_a[:3, 3] + alpha * transform_b[:3, 3]
    return make(slerp(transform_a[:3, :3], transform_b[:3, :3], alpha), position)


def pose_error(transform_a: np.ndarray, transform_b: np.ndarray) -> tuple[float, float]:
    """Return (position error m, rotation error rad) between two poses."""
    position = float(np.linalg.norm(transform_a[:3, 3] - transform_b[:3, 3]))
    return position, rotation_angle(transform_a[:3, :3].T @ transform_b[:3, :3])


def is_rigid(transform: np.ndarray, atol: float = 1e-6) -> bool:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    rotation = transform[:3, :3]
    return bool(np.allclose(transform[3], [0, 0, 0, 1], atol=atol)
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=atol * 10)
                and np.isclose(np.linalg.det(rotation), 1.0, atol=atol * 10))
