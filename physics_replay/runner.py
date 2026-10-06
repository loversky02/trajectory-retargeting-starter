"""Reproducible runs: a demo with videos, external-episode replay, and a seeded paired benchmark.

Every run directory gets a ``manifest.json`` with the config hash, the model
receipt hash, the git commit and package versions, so a table on a slide can be
traced back to the exact inputs that produced it.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from . import episode as ep
from . import report
from .execute import execute
from .metrics import FAILURE_ORDER, evaluate, wilson_interval
from .models import verify_models
from .retarget import retarget
from .scene import CubeSpec, RobotScene, SceneSpec, tabletop_pose
from .source_demo import source_plan

CONFIG_VERSION = "physics_replay.pick_place.v1"
METHODS = ("world_frame", "object_relative", "object_relative_trimmed", "target_scripted")
METHOD_NOTES = {
    "world_frame": "source TCP poses replayed at the same world coordinates (raw end-effector baseline)",
    "object_relative": "source TCP re-anchored on the target object, then on the target goal (clean-state poses)",
    "object_relative_trimmed": "object_relative without the source's free-space approach; a planned transition "
                               "from the target robot's start pose reaches the object-anchored segment",
    "target_scripted": "target robot runs the scripted demo on the target scene itself; an upper bound for the "
                       "controller and gripper, not a retargeting method",
}
# (baseline, method) pairs compared on the same layouts with an exact McNemar test.
PAIRED_COMPARISONS = (("world_frame", "object_relative"), ("object_relative", "object_relative_trimmed"))
REPO_ROOT = Path(__file__).resolve().parents[1]


# --- configuration --------------------------------------------------------
def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(Path(path).read_text())
    if config.get("config_version") != CONFIG_VERSION:
        raise ValueError(f"config_version must be {CONFIG_VERSION!r}")
    return config, hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def scene_spec(config: dict, layout: dict) -> SceneSpec:
    cube = CubeSpec(config["cube"]["half_size_m"], config["cube"]["mass_kg"], tuple(config["cube"]["friction"]))

    def pose(part: dict) -> np.ndarray:
        return tabletop_pose(part["x_m"], part["y_m"], part["yaw_rad"], cube.half_size_m)

    return SceneSpec(pose(layout["object"]), pose(layout["goal"]), cube)


def sample_layouts(config: dict, count: int, seed: int) -> list[dict]:
    """Uniform object/goal placements, rejecting pairs closer than the configured distance."""
    ranges = config["target_sampling"]
    rng = np.random.default_rng(seed)
    layouts = []
    while len(layouts) < count:
        layout = {part: {key: float(rng.uniform(*ranges[part][key])) for key in ("x_m", "y_m", "yaw_rad")}
                  for part in ("object", "goal")}
        distance = math.hypot(layout["object"]["x_m"] - layout["goal"]["x_m"],
                              layout["object"]["y_m"] - layout["goal"]["y_m"])
        if distance >= ranges["min_object_goal_distance_m"]:
            layouts.append(layout)
    return layouts


def code_hash() -> str:
    """SHA-256 over the package sources, so uncommitted runs still name the exact code."""
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def provenance(config_path: Path, config_hash: str, cache: Path, command: list[str]) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": command,
            "config_path": str(config_path), "config_sha256": config_hash,
            "model_receipt_sha256": verify_models(cache), "code_sha256": code_hash(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "physics_replay", "config/physics-replay")),
            "python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
            "platform": platform.platform()}


# --- single runs ----------------------------------------------------------
def _recorder(scene: RobotScene, every: int, frames: list, camera: str = "overview"):
    renderer = mujoco.Renderer(scene.model, report.VIDEO_HEIGHT, report.VIDEO_WIDTH)

    def on_frame(current: RobotScene, k: int) -> None:
        if k % every == 0:
            renderer.update_scene(current.data, camera)
            frames.append(renderer.render().copy())

    return on_frame, renderer


def run(robot: str, spec: SceneSpec, config: dict, cache: Path, method: str,
        source_episode: dict | None = None, frames: list | None = None) -> dict:
    """Plan and execute one episode; ``frames`` collects video frames when given."""
    hz = config["control_hz"]
    scene = RobotScene(robot, spec, cache, verify=False)
    if method in ("source", "target_scripted"):
        plan = source_plan(scene, config["demo"], hz)
    else:
        plan = retarget(source_episode, spec, scene.home_tcp_pose(), method, hz, config["retarget"])
    renderer = None
    on_frame = None
    if frames is not None:
        on_frame, renderer = _recorder(scene, max(1, round(hz / report.VIDEO_FPS)), frames)
    try:
        episode, diagnostics = execute(scene, plan, hz, on_frame)
    finally:
        if renderer is not None:
            renderer.close()
    metrics = evaluate(episode, diagnostics, config["success"], config["ik"])
    return {"plan": plan, "episode": episode, "diagnostics": diagnostics, "metrics": metrics}


def record_source(config: dict, cache: Path, frames: list | None = None) -> dict:
    result = run(config["source_robot"], scene_spec(config, config["source_scene"]), config, cache, "source",
                 frames=frames)
    if not result["metrics"]["success"]:
        raise RuntimeError(f"source demonstration failed: {result['metrics']['failure_reason']}")
    return result


def _save_run(out: Path, name: str, result: dict) -> None:
    ep.save(result["episode"], out / f"{name}_episode.json")
    (out / f"{name}_plan.json").write_text(json.dumps(ep.to_json_value(result["plan"].to_json()), allow_nan=False))
    report.plot_tracking(result["plan"], result["episode"], result["diagnostics"], out / f"{name}_tracking.png", name)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(ep.to_json_value(value), indent=1, allow_nan=False))


# --- commands -------------------------------------------------------------
def demo(config_path: Path, cache: Path, out: Path, video: bool, command: list[str]) -> dict:
    config, config_hash = load_config(config_path)
    out.mkdir(parents=True, exist_ok=True)
    manifest = provenance(config_path, config_hash, cache, command)
    clips = {}
    source_frames = [] if video else None
    source = record_source(config, cache, source_frames)
    _save_run(out, "source", source)
    clips["Panda source demo (source scene)"] = source_frames
    target = scene_spec(config, config["demo_target_scene"])
    metrics = {"source": source["metrics"]}
    for method in METHODS:
        frames = [] if video else None
        result = run(config["target_robot"], target, config, cache, method, source["episode"], frames)
        _save_run(out, f"target_{method}", result)
        metrics[method] = result["metrics"]
        clips[f"UR5e {method.replace('_', ' ')} (new scene)"] = frames
    if video:
        for label, frames in clips.items():
            report.write_video(frames, out / f"{report.slug(label)}.mp4", label)
        report.write_comparison(clips, out / "comparison.mp4")
    manifest.update({"kind": "demo", "methods": METHOD_NOTES, "target_layout": config["demo_target_scene"],
                     "metrics": metrics})
    _write_json(out / "manifest.json", manifest)
    return manifest


def replay(episode_path: Path, config_path: Path, cache: Path, out: Path, method: str, video: bool,
           command: list[str]) -> dict:
    """Retarget any schema-valid episode onto the target robot in the demo target scene."""
    config, config_hash = load_config(config_path)
    source_episode = ep.load(episode_path)
    out.mkdir(parents=True, exist_ok=True)
    manifest = provenance(config_path, config_hash, cache, command)
    frames = [] if video else None
    result = run(config["target_robot"], scene_spec(config, config["demo_target_scene"]), config, cache, method,
                 source_episode, frames)
    _save_run(out, f"target_{method}", result)
    if video:
        report.write_video(frames, out / f"target_{method}.mp4", f"UR5e {method} replay of {episode_path.name}")
    manifest.update({"kind": "replay", "episode": str(episode_path),
                     "episode_sha256": hashlib.sha256(Path(episode_path).read_bytes()).hexdigest(),
                     "method": method, "metrics": result["metrics"]})
    _write_json(out / "manifest.json", manifest)
    return manifest


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarise(trials: list[dict], methods: list[str]) -> dict:
    per_method = {}
    for method in methods:
        rows = [t for t in trials if t["method"] == method]
        wins = sum(t["metrics"]["success"] for t in rows)
        low, high = wilson_interval(wins, len(rows))
        failures = {reason: sum(t["metrics"]["failure_reason"] == reason for t in rows) for reason in FAILURE_ORDER}
        per_method[method] = {"successes": wins, "trials": len(rows), "success_rate": wins / len(rows),
                              "wilson95": [low, high], "failure_reasons": failures,
                              "final_xy_error_m_median_success": float(np.median(
                                  [t["metrics"]["final_xy_error_m"] for t in rows if t["metrics"]["success"]]
                              )) if wins else None}
    paired = {}
    outcome = {(t["layout_index"], t["method"]): t["metrics"]["success"] for t in trials}
    indices = sorted({t["layout_index"] for t in trials})
    for baseline, method in PAIRED_COMPARISONS:
        if {baseline, method} <= set(methods):
            only_baseline = sum(outcome[i, baseline] and not outcome[i, method] for i in indices)
            only_method = sum(outcome[i, method] and not outcome[i, baseline] for i in indices)
            paired[f"{method}_vs_{baseline}"] = {
                "baseline": baseline, "method": method, "pairs": len(indices),
                "only_baseline_succeeds": only_baseline, "only_method_succeeds": only_method,
                "mcnemar_exact_p": mcnemar_exact(only_baseline, only_method)}
    return {"methods": per_method, "paired": paired}


def benchmark(config_path: Path, cache: Path, out: Path, configs: int | None, seed: int | None,
              methods: list[str], command: list[str]) -> dict:
    config, config_hash = load_config(config_path)
    count = configs or config["evaluation"]["configs"]
    seed = config["evaluation"]["seed"] if seed is None else seed
    out.mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(exist_ok=True)
    manifest = provenance(config_path, config_hash, cache, command)
    source = record_source(config, cache)
    ep.save(source["episode"], out / "source_episode.json")
    layouts = sample_layouts(config, count, seed)
    _write_json(out / "layouts.json", {"seed": seed, "layouts": layouts})
    trials = []
    started = time.time()
    with (out / "trials.jsonl").open("w") as log, (out / "failures.jsonl").open("w") as failure_log:
        for index, layout in enumerate(layouts):
            spec = scene_spec(config, layout)
            for method in methods:
                result = run(config["target_robot"], spec, config, cache, method, source["episode"])
                row = {"layout_index": index, "seed": seed, "method": method, "layout": layout,
                       "metrics": result["metrics"]}
                trials.append(row)
                log.write(json.dumps(ep.to_json_value(row), allow_nan=False) + "\n")
                if not result["metrics"]["success"]:
                    failure_log.write(json.dumps(ep.to_json_value(row), allow_nan=False) + "\n")
                    _save_run(out / "failures", f"{index:03d}_{method}", result)
            print(f"[{index + 1}/{count}] " + "  ".join(
                f"{t['method']}={'ok' if t['metrics']['success'] else t['metrics']['failure_reason']}"
                for t in trials[-len(methods):]), flush=True)
    summary = summarise(trials, methods)
    (out / "summary.md").write_text(report.summary_markdown(summary, count, seed, config_hash))
    manifest.update({"kind": "benchmark", "seed": seed, "configs": count, "methods": {m: METHOD_NOTES[m] for m in methods},
                     "elapsed_s": round(time.time() - started, 1), "summary": summary})
    _write_json(out / "manifest.json", manifest)
    report.plot_benchmark(summary, out / "success_rates.png")
    return manifest


def default_out(kind: str) -> Path:
    return REPO_ROOT / "data" / "physics-replay" / "runs" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"


def argv() -> list[str]:
    return [Path(sys.executable).name, "-m", "physics_replay", *sys.argv[1:]]
