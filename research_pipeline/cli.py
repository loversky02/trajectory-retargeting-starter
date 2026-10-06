"""Offline research command-line entrypoints and run artifact receipts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import platform
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy
from PIL import __version__ as pillow_version

from .benchmark import condition_observations, evaluate, manual_edit
from .models import fetch_models, sha256, verify_models
from .robots import retarget
from .schema import load_config, validate_poses
from .video import extract_video, generate_fixture


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/research-prototype/default.json"
DEFAULT_CACHE = ROOT / "data/research-prototype/models"


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _save_observations(output: Path, observations: dict) -> None:
    np.savez_compressed(output / "observations.npz", timestamps=observations["timestamps"],
                        T_world_marker=observations["T_world_marker"], valid=observations["valid"])
    _json(output / "observations.json", {"schema": "il-marker-poses.v1",
        "frame_convention": "T_A_B maps coordinates in B into coordinates in A",
        "pose_field": "T_world_marker", "missing": "NaN transform and valid=false",
        "video_sha256": observations["video_sha256"], "fps": observations["fps"],
        "timestamps_s": observations["timestamps"].tolist(), "valid": observations["valid"].tolist(),
        "diagnostics": observations["diagnostics"]})


def _mean_robot_error(trajectory: dict, robot: str) -> float | None:
    errors = [frame[robot]["position_error_m"] for frame in trajectory["frames"]
              if frame[robot] is not None]
    return float(np.mean(errors)) if errors else None


def _load_observations(path: Path) -> dict:
    archive = np.load(path, allow_pickle=False)
    observations = {key: archive[key] for key in ("timestamps", "T_world_marker", "valid")}
    validate_poses(**{"ts": observations["timestamps"], "transforms": observations["T_world_marker"],
                      "valid": observations["valid"]})
    return observations


def _plot(output: Path, observations: dict, gt: np.ndarray, results: dict) -> None:
    positions = observations["T_world_marker"][:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(gt[:, 0, 3], gt[:, 1, 3], label="evaluator GT")
    axes[0].scatter(positions[:, 0], positions[:, 1], s=12, label="PnP observations")
    axes[0].set(xlabel="world x (m)", ylabel="world y (m)", title="Marker trajectory")
    axes[0].legend()
    names = list(results["methods"])
    axes[1].bar(names, [results["methods"][name]["mean_penalized_error_m"] for name in names])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set(ylabel="penalized position error (m)", title="Synthetic oracle comparison")
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=140)
    plt.close(fig)


def _report(output: Path, observations: dict, retargeted: dict, results: dict, provenance: dict) -> None:
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{value['mean_penalized_error_m']:.5f}</td>"
                   f"<td>{value['frames_missing']}</td><td>{value['simulated_elapsed_s']}</td></tr>"
                   for name, value in results["methods"].items())
    failures = sum(frame["status"] == "kinematic_failure" for frame in retargeted["frames"])
    content = f"""<!doctype html><html><meta charset='utf-8'><title>IL prototype run</title>
<body><h1>IL (Invested Lamda) offline research prototype</h1>
<p>Generated marker-assisted video, OpenCV ArUco/PnP extraction, actual Menagerie Panda and UR5e kinematics.</p>
<p>Detected {int(observations['valid'].sum())}/{len(observations['valid'])} frames; kinematic failures {failures}.
Missing samples remain in every denominator. Position-only retargeting does not establish grasp, collision clearance, dynamics, or manipulation success.</p>
<p>Oracle pose offsets and segment costs are simulated. These results do not measure human correction time or benefit.
Manual terminal edits have a separate monotonic timing receipt.</p>
<img src='summary.png' width='900' alt='Trajectory and benchmark plots'>
<table border='1'><tr><th>Method</th><th>Penalized position error (m)</th><th>Missing frames</th><th>Simulated seconds</th></tr>{rows}</table>
<p>Source revision: {html.escape(provenance['models']['commit'])}; seed: {results['seed']}.</p>
<p>Camera/marker and robot conventions are in observations.json, retarget.json and provenance.json.</p>
</body></html>"""
    (output / "report.html").write_text(content)


def demo(output: Path, config_path: Path, cache: Path) -> None:
    config = load_config(config_path)
    model_receipt = verify_models(cache)
    if output.exists():
        raise FileExistsError(f"run output already exists: {output}")
    output.mkdir(parents=True)
    video = output / "video.mp4"
    gt_path = output / "evaluator-ground-truth.npz"
    generate_fixture(video, gt_path, config)
    observations = extract_video(video, config)
    _save_observations(output, observations)
    gt_archive = np.load(gt_path, allow_pickle=False)
    gt = gt_archive["T_world_marker"]
    errors = np.linalg.norm(observations["T_world_marker"][observations["valid"], :3, 3]
                            - gt[observations["valid"], :3, 3], axis=1)
    rotations_est = observations["T_world_marker"][observations["valid"], :3, :3]
    rotations_gt = gt[observations["valid"], :3, :3]
    rotation_angles = [float(np.arccos(np.clip((np.trace(expected.T @ observed) - 1) / 2, -1, 1)))
                       for expected, observed in zip(rotations_gt, rotations_est)]
    results = evaluate(observations, gt, config["segment_length"], config["budget_s"],
                       config["oracle_edit_cost_s"], config["seed"])
    reference = (observations["T_world_marker"][np.flatnonzero(observations["valid"])[0], :3, 3].copy()
                 if observations["valid"].any() else None)
    trajectories = {}
    for name, method in results["methods"].items():
        conditioned = condition_observations(observations, name, method["edits"])
        trajectory = retarget(conditioned, cache, reference_marker_position=reference)
        trajectories[name] = trajectory
        statuses = [frame["status"] for frame in trajectory["frames"]]
        method["kinematic_outcomes"] = {status: statuses.count(status) for status in sorted(set(statuses))}
        method["kinematic_coverage"] = {"total_frames": len(statuses), "ok_frames": statuses.count("ok"),
            "failed_or_skipped_frames": len(statuses) - statuses.count("ok"),
            "missing_marker_frames": statuses.count("missing_marker"),
            "source_panda_ik_failure_frames": statuses.count("source_panda_ik_failure"),
            "other_kinematic_failure_frames": statuses.count("kinematic_failure")}
        method["panda_mean_position_error_m"] = _mean_robot_error(trajectory, "panda")
        method["ur5e_mean_position_error_m"] = _mean_robot_error(trajectory, "ur5e")
    retargeted = trajectories["no_edit"]
    _json(output / "retarget.json", retargeted)
    _json(output / "condition-retarget.json", trajectories)
    results["extraction"] = {"detected": int(observations["valid"].sum()),
                             "total": len(observations["valid"]),
                             "mean_valid_position_error_m": float(np.mean(errors)) if len(errors) else None,
                             "mean_valid_rotation_angle_error_rad": float(np.mean(rotation_angles)) if rotation_angles else None}
    _json(output / "results.json", results)
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "frames_total", "frames_valid", "frames_missing",
                                                       "mean_valid_position_error_m", "mean_penalized_error_m", "simulated_elapsed_s"])
        writer.writeheader()
        for name, value in results["methods"].items():
            writer.writerow({"method": name, **{key: value[key] for key in writer.fieldnames if key != "method"}})
    provenance = {"models": {"repository": model_receipt["repository"], "commit": model_receipt["commit"],
                             "receipt_sha256": sha256(cache / "receipt.json"), "licenses": model_receipt["license_paths"]},
                  "inputs": {"config_sha256": sha256(config_path), "video_sha256": sha256(video),
                             "fixture_gt_sha256": sha256(gt_path)},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                              "opencv": cv2.__version__, "mujoco": mujoco.__version__,
                              "matplotlib": matplotlib.__version__, "Pillow": pillow_version},
                  "claims": ["synthetic marker-assisted extraction", "position-only FK/IK retargeting; orientation unconstrained; no SE(3) tracking",
                             "simulated oracle correction cost; no human or physical task success"]}
    _json(output / "provenance.json", provenance)
    _plot(output, observations, gt, results)
    _report(output, observations, retargeted, results, provenance)
    print(json.dumps({"output": str(output), "detected": results["extraction"]["detected"],
                      "total": results["extraction"]["total"],
                      "mean_valid_position_error_m": results["extraction"]["mean_valid_position_error_m"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description="IL offline marker/kinematics research prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-models", help="one-time public immutable Menagerie cache fetch")
    fetch.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    run = commands.add_parser("demo", help="generated video to benchmark")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--models", type=Path, default=DEFAULT_CACHE)
    imported = commands.add_parser("import-video", help="calibrated marker video to observations and robot trajectories")
    imported.add_argument("--video", type=Path, required=True)
    imported.add_argument("--config", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--models", type=Path, default=DEFAULT_CACHE)
    manual = commands.add_parser("manual-edit", help="interactive timed terminal pose-offset receipt")
    manual.add_argument("--poses", type=Path, required=True)
    manual.add_argument("--segment", type=int, required=True)
    manual.add_argument("--segment-length", type=int, default=5)
    manual.add_argument("--budget-s", type=float, required=True)
    manual.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch-models":
        receipt = fetch_models(args.cache)
        print(json.dumps({"cache": str(args.cache), "files": len(receipt["files_sha256"]), "commit": receipt["commit"]}))
    elif args.command == "demo":
        demo(args.output, args.config, args.models)
    elif args.command == "import-video":
        config = load_config(args.config)
        verify_models(args.models)
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        args.output.mkdir(parents=True)
        observations = extract_video(args.video, config)
        _save_observations(args.output, observations)
        _json(args.output / "retarget.json", retarget(observations, args.models))
        _json(args.output / "provenance.json", {"source_video_sha256": sha256(args.video),
                                                 "calibration_sha256": sha256(args.config),
                                                 "model_receipt_sha256": sha256(args.models / "receipt.json"),
                                                 "ground_truth": "unavailable; no benchmark claim"})
        print(json.dumps({"output": str(args.output), "frames": len(observations["valid"])}))
    else:
        if args.output.exists():
            raise FileExistsError(f"manual receipt already exists: {args.output}")
        observation = _load_observations(args.poses)
        corrected, receipt = manual_edit(observation, args.segment, args.segment_length, args.budget_s)
        args.output.mkdir(parents=True)
        inferred_fps = (float(1 / np.median(np.diff(corrected["timestamps"])))
                        if len(corrected["timestamps"]) > 1 else None)
        _save_observations(args.output, {**corrected, "diagnostics": [], "video_sha256": "from-input-npz",
                                          "fps": inferred_fps})
        receipt["source_poses_sha256"] = sha256(args.poses)
        receipt["edited_poses"] = "observations.npz"
        _json(args.output / "manual-edit-receipt.json", receipt)


if __name__ == "__main__":
    main()
