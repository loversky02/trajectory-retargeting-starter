"""Videos, plots and the markdown summary written next to every run."""
from __future__ import annotations

import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .metrics import FAILURE_ORDER  # noqa: E402

VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = 640, 480, 25


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    box = draw.textbbox((10, 8), text, font=font)
    draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def write_video(frames: list, path: Path, label: str) -> None:
    imageio.mimwrite(path, [_label(f, label) for f in frames], fps=VIDEO_FPS, codec="libx264",
                     quality=7, macro_block_size=16)


def write_comparison(clips: dict[str, list], path: Path) -> None:
    """Side-by-side clips; shorter ones hold their last frame."""
    length = max(len(frames) for frames in clips.values())
    labelled = [[_label(f, label) for f in frames] for label, frames in clips.items()]
    rows = [np.hstack([frames[min(i, len(frames) - 1)] for frames in labelled]) for i in range(length)]
    imageio.mimwrite(path, rows, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=16)


def plot_tracking(plan, episode: dict, diagnostics: dict, path: Path, title: str) -> None:
    """Requested vs executed TCP, tracking error and gripper, shaded by plan segment."""
    t = np.asarray(episode["timestamps_s"])
    requested = np.asarray(plan.tcp_targets)[:, :3, 3]
    executed = np.asarray(episode["tcp_position_m"])
    obj = np.asarray(episode["object_position_m"])
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    for axis, name, i in zip(axes[:3], "xyz", range(3)):
        axis.plot(t, requested[:, i], "--", color="0.4", label="requested TCP")
        axis.plot(t, executed[:, i], color="tab:blue", label="executed TCP")
        axis.plot(t, obj[:, i], color="tab:red", alpha=0.7, label="object")
        axis.set_ylabel(f"{name} (m)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[3].plot(t, 1000 * np.asarray(diagnostics["tracking_position_error_m"]), color="tab:orange")
    axes[3].set_ylabel("tracking err (mm)")
    axes[4].step(t, episode["gripper_command"], where="post", color="k")
    axes[4].set_ylabel("gripper cmd")
    axes[4].set_xlabel("time (s)")
    segments = list(plan.segment)
    start = 0
    colours = plt.cm.tab20(np.linspace(0, 1, 20))
    names = list(dict.fromkeys(segments))
    for i in range(1, len(segments) + 1):
        if i == len(segments) or segments[i] != segments[start]:
            colour = colours[names.index(segments[start]) % 20]
            for axis in axes:
                axis.axvspan(t[start], t[i - 1], color=colour, alpha=0.12, lw=0)
            axes[0].text((t[start] + t[i - 1]) / 2, axes[0].get_ylim()[1], segments[start], rotation=90,
                         fontsize=6, va="top", ha="center")
            start = i
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_benchmark(summary: dict, path: Path) -> None:
    methods = list(summary["methods"])
    stats = summary["methods"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    rates = [stats[m]["success_rate"] for m in methods]
    errors = np.array([[r - stats[m]["wilson95"][0], stats[m]["wilson95"][1] - r] for m, r in zip(methods, rates)]).T
    left.bar(methods, rates, yerr=errors, capsize=6, color=["tab:gray", "tab:blue", "tab:purple", "tab:green"][:len(methods)])
    for i, m in enumerate(methods):
        left.text(i, 0.02, f"{stats[m]['successes']}/{stats[m]['trials']}", ha="center", color="white")
    left.set_ylim(0, 1.05)
    left.set_ylabel("success rate (Wilson 95% CI)")
    bottom = np.zeros(len(methods))
    for reason in FAILURE_ORDER:
        counts = np.array([stats[m]["failure_reasons"][reason] for m in methods])
        if counts.any():
            right.bar(methods, counts, bottom=bottom, label=reason)
            bottom += counts
    right.set_ylabel("failed trials")
    right.set_title("first failure reason")
    if bottom.any():
        right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summary_markdown(summary: dict, configs: int, seed: int, config_hash: str) -> str:
    lines = [f"# Panda -> UR5e pick-and-place replay benchmark", "",
             f"{configs} sampled target layouts, seed {seed}, config sha256 `{config_hash[:12]}`.", "",
             "| Method | Success | Rate | Wilson 95% CI | Median place error (success) | Failure reasons |",
             "|---|---|---|---|---|---|"]
    for method, stats in summary["methods"].items():
        reasons = ", ".join(f"{k} {v}" for k, v in stats["failure_reasons"].items() if v) or "none"
        error = stats["final_xy_error_m_median_success"]
        lines.append(f"| {method} | {stats['successes']}/{stats['trials']} | {stats['success_rate']:.2f} | "
                     f"{stats['wilson95'][0]:.2f}-{stats['wilson95'][1]:.2f} | "
                     f"{'-' if error is None else f'{1000 * error:.1f} mm'} | {reasons} |")
    for paired in summary["paired"].values():
        lines += ["", f"Paired {paired['method']} vs {paired['baseline']}: {paired['pairs']} layouts, only "
                      f"{paired['baseline']} succeeds {paired['only_baseline_succeeds']}, only {paired['method']} "
                      f"succeeds {paired['only_method_succeeds']}, exact McNemar p = {paired['mcnemar_exact_p']:.3g}."]
    return "\n".join(lines) + "\n"
