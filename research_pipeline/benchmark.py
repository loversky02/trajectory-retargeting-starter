"""Equal-pool correction comparison; oracle edits are simulated, never human data."""
from __future__ import annotations

import time

import numpy as np


def segments(count: int, length: int) -> list[list[int]]:
    if length <= 0:
        raise ValueError("segment length must be positive")
    return [list(range(i, min(i + length, count))) for i in range(0, count, length)]


def rank_observations(observations: dict, length: int) -> list[int]:
    """Uses detector status, reprojection error and observed temporal jumps only."""
    position = observations["T_world_marker"][:, :3, 3]
    valid = observations["valid"]
    diagnostics = observations["diagnostics"]
    scores = []
    for segment_id, frames in enumerate(segments(len(valid), length)):
        score = 0.0
        for i in frames:
            if not valid[i]:
                score += 10.0
            else:
                score += float(diagnostics[i]["reprojection_rmse_px"] or 0) / 10
                if i and valid[i - 1]:
                    score += min(10.0, float(np.linalg.norm(position[i] - position[i - 1]) * 50))
        scores.append((score, segment_id))
    return [item[1] for item in sorted(scores, key=lambda x: (-x[0], x[1]))]


def select(pool: list[int], budget_s: float, cost_s: float) -> list[int]:
    if cost_s <= 0 or not np.isfinite(cost_s) or budget_s < 0 or not np.isfinite(budget_s):
        raise ValueError("seconds budget and edit cost must be finite and nonnegative; cost must be positive")
    count = int(np.floor(budget_s / cost_s))
    if count > len(pool):
        count = len(pool)
    chosen = pool[:count]
    if len(chosen) * cost_s > budget_s + 1e-9:
        raise ValueError("budget overrun")
    return chosen


def _score(estimated: np.ndarray, gt: np.ndarray, valid: np.ndarray, failure_penalty_m: float) -> dict:
    errors = np.linalg.norm(estimated[valid] - gt[valid], axis=1)
    missing = int((~valid).sum())
    return {"frames_total": int(len(valid)), "frames_valid": int(valid.sum()), "frames_missing": missing,
            "mean_valid_position_error_m": float(np.mean(errors)) if len(errors) else None,
            "failure_penalty_m_per_missing": failure_penalty_m,
            "mean_penalized_error_m": float((np.sum(errors) + missing * failure_penalty_m) / len(valid))}


def condition_observations(observations: dict, method: str, edits: list[dict]) -> dict:
    """Apply the same constant offset primitive used by the terminal editor."""
    poses = observations["T_world_marker"].copy()
    valid = observations["valid"]
    if method == "automatic_local_mean":
        original = poses[:, :3, 3].copy()
        for i in np.flatnonzero(valid):
            neighborhood = np.arange(max(0, i - 1), min(len(valid), i + 2))
            neighborhood = neighborhood[valid[neighborhood]]
            poses[i, :3, 3] = np.mean(original[neighborhood], axis=0)
    else:
        for edit in edits:
            if edit["offset_m"] is not None:
                frames = np.asarray(edit["frames"], dtype=int)
                present = frames[valid[frames]]
                poses[present, :3, 3] += np.asarray(edit["offset_m"])
    return {**observations, "T_world_marker": poses}


def evaluate(observations: dict, ground_truth: np.ndarray, length: int, budget_s: float, cost_s: float, seed: int,
             failure_penalty_m: float = 0.1) -> dict:
    valid = observations["valid"]
    estimated = observations["T_world_marker"][:, :3, 3]
    if ground_truth.shape != (len(valid), 4, 4):
        raise ValueError("GT frame count mismatch")
    gt = ground_truth[:, :3, 3]
    pool = list(range(len(segments(len(valid), length))))
    ranked = select(rank_observations(observations, length), budget_s, cost_s)
    random = select(np.random.default_rng(seed).permutation(pool).tolist(), budget_s, cost_s)
    methods = {}
    for name, chosen in (("no_edit", []), ("ranked_oracle", ranked), ("random_oracle", random)):
        edits = []
        for segment_id in chosen:
            frames = np.asarray(segments(len(valid), length)[segment_id])
            present = frames[valid[frames]]
            if len(present):
                offset = np.median(gt[present] - estimated[present], axis=0)
                edits.append({"segment": segment_id, "frames": frames.tolist(), "offset_m": offset.tolist(), "frames_edited": int(len(present)),
                              "simulated_cost_s": cost_s})
            else:
                edits.append({"segment": segment_id, "frames": frames.tolist(), "offset_m": None, "frames_edited": 0, "simulated_cost_s": cost_s})
        corrected = condition_observations(observations, name, edits)["T_world_marker"][:, :3, 3]
        methods[name] = {"selected_segments": chosen, "edits": edits, "simulated_elapsed_s": len(chosen) * cost_s,
                         **_score(corrected, gt, valid, failure_penalty_m)}
    smoothed = condition_observations(observations, "automatic_local_mean", [])["T_world_marker"][:, :3, 3]
    methods["automatic_local_mean"] = {"selected_segments": [], "edits": [], "simulated_elapsed_s": 0,
                                       **_score(smoothed, gt, valid, failure_penalty_m)}
    return {"comparison": "synthetic marker fixture; oracle GT pose-offset edits; no measured human benefit",
            "ranking_inputs": "estimated validity, reprojection RMSE, observed temporal jump only",
            "ground_truth_use": "evaluator and oracle edit offsets only",
            "segment_pool": segments(len(valid), length), "budget_s": budget_s,
            "oracle_edit_cost_s": cost_s, "seed": seed,
            "edit_primitive": "median position offset for observed frames in selected segment; no missing-frame repair",
            "methods": methods}


def manual_edit(observations: dict, segment: int, length: int, budget_s: float) -> tuple[dict, dict]:
    """One interactive edit; measured session duration cannot establish active human work."""
    pool = segments(len(observations["valid"]), length)
    if segment < 0 or segment >= len(pool):
        raise ValueError("segment outside pool")
    if budget_s <= 0 or not np.isfinite(budget_s):
        raise ValueError("manual seconds budget must be finite and positive")
    frames = pool[segment]
    start_ns = time.monotonic_ns()
    print(f"Segment {segment}, frames {frames}; enter three position offsets in metres, or blank to skip.")
    for i in frames:
        print(i, "valid" if observations["valid"][i] else "missing",
              observations["T_world_marker"][i, :3, 3].tolist())
    review_end_ns = time.monotonic_ns()
    answer = input("dx dy dz: ").strip()
    end_ns = time.monotonic_ns()
    offset = None if not answer else [float(x) for x in answer.split()]
    if offset is not None and (len(offset) != 3 or not np.isfinite(offset).all()):
        raise ValueError("enter exactly three finite offsets")
    total_s = (end_ns - start_ns) / 1e9
    if total_s > budget_s:
        raise ValueError(f"manual session exceeded {budget_s} s budget: {total_s:.3f} s; no edited artifact written")
    edit = {"segment": segment, "frames": frames, "offset_m": offset}
    corrected = condition_observations(observations, "manual_pose_offset", [edit])
    receipt = {"method": "manual_pose_offset", "segment": segment, "frames": frames,
               "offset_m": offset, "frames_edited": int(sum(observations["valid"][frames])) if offset is not None else 0,
               "budget_s": budget_s, "display_elapsed_s": (review_end_ns - start_ns) / 1e9,
               "prompt_response_elapsed_s": (end_ns - review_end_ns) / 1e9,
               "total_session_elapsed_s": total_s, "timing_source": "time.monotonic_ns",
               "timing_limit": "includes display, reading, thinking, typing and idle time; active human time cannot be inferred",
               "primitive": "constant position offset on valid frames only; missing frames preserved"}
    return corrected, receipt
