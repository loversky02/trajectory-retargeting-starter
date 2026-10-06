"""Synthetic calibrated marker fixture and image-only pose extraction."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .schema import rigid_transform, validate_poses


def marker_points(size: float) -> np.ndarray:
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def generate_fixture(video: Path, ground_truth: Path, config: dict) -> None:
    if video.exists() or ground_truth.exists():
        raise FileExistsError("fixture output already exists")
    fps, frames = config["fps"], config["frames"]
    camera = config["camera"]
    width, height = camera["width"], camera["height"]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create video")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, config["marker_id"], 240)
    # The camera sees a slightly tilted square. T_camera_marker is evaluator-only.
    # Marker y axis points up; camera pixel y points down. Face the marker toward camera.
    rvec = np.array([3.0, -0.1, 0.03], dtype=float)
    rotation, _ = cv2.Rodrigues(rvec)
    t_world_camera = rigid_transform(camera["T_world_camera"])
    poses = np.empty((frames, 4, 4))
    k = np.asarray(camera["K"])
    distortion = np.asarray(camera["distortion"])
    source = np.float32([[0, 0], [239, 0], [239, 239], [0, 239]])
    try:
        for index in range(frames):
            phase = index / max(frames - 1, 1)
            translation = np.array([0.035 * np.sin(2 * np.pi * phase),
                                    0.025 * np.cos(2 * np.pi * phase), 0.55 + 0.012 * phase])
            t_camera_marker = np.eye(4)
            t_camera_marker[:3, :3] = rotation
            t_camera_marker[:3, 3] = translation
            poses[index] = t_world_camera @ t_camera_marker
            projected, _ = cv2.projectPoints(marker_points(config["marker_size_m"]), rvec,
                                              translation, k, distortion)
            quad = projected.reshape(4, 2).astype(np.float32)
            warp = cv2.getPerspectiveTransform(source, quad)
            image = np.full((height, width), 225, np.uint8)
            warped = cv2.warpPerspective(marker, warp, (width, height), borderValue=225)
            mask = cv2.warpPerspective(np.full_like(marker, 255), warp, (width, height)) > 127
            image[mask] = warped[mask]
            # Controlled dropout is generated from fixture design, not fed to the detector.
            if index in (9, 10, 21):
                image[:] = 225
            writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    np.savez_compressed(ground_truth, timestamps=np.arange(frames) / fps, T_world_marker=poses)


def extract_video(video: Path, config: dict) -> dict:
    """Return observations only. This function has no GT argument or GT file access."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"video unavailable: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("video has no usable frame rate")
    camera = config["camera"]
    k = np.asarray(camera["K"], dtype=float)
    distortion = np.asarray(camera["distortion"], dtype=float)
    t_world_camera = rigid_transform(camera["T_world_camera"])
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    object_points = marker_points(config["marker_size_m"])
    poses, valid, diagnostics = [], [], []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != camera["width"] or frame.shape[0] != camera["height"]:
                raise ValueError("video dimensions do not match calibration")
            corners, ids, _ = detector.detectMarkers(frame)
            matches = [] if ids is None else [i for i, marker_id in enumerate(ids.flat) if marker_id == config["marker_id"]]
            pose = np.full((4, 4), np.nan)
            diagnostic = {"frame": frame_index, "status": "missing_marker", "reprojection_rmse_px": None}
            if len(matches) == 1:
                image_points = corners[matches[0]].reshape(4, 2)
                solved, rvec, tvec = cv2.solvePnP(object_points, image_points, k, distortion,
                                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if solved and tvec[2, 0] > 0:
                    rotation, _ = cv2.Rodrigues(rvec)
                    t_camera_marker = np.eye(4)
                    t_camera_marker[:3, :3] = rotation
                    t_camera_marker[:3, 3] = tvec.flat
                    pose = t_world_camera @ t_camera_marker
                    projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, distortion)
                    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1))))
                    diagnostic = {"frame": frame_index, "status": "detected", "reprojection_rmse_px": rmse}
            poses.append(pose)
            valid.append(diagnostic["status"] == "detected")
            diagnostics.append(diagnostic)
            frame_index += 1
    finally:
        capture.release()
    if not poses:
        raise ValueError("video contains no frames")
    result = {"timestamps": np.arange(len(poses), dtype=float) / fps,
              "T_world_marker": np.asarray(poses), "valid": np.asarray(valid, dtype=bool),
              "diagnostics": diagnostics, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "fps": fps}
    validate_poses(result["timestamps"], result["T_world_marker"], result["valid"])
    return result
