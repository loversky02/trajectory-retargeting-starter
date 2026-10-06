"""Meaningful contracts for the isolated offline research slice."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research_pipeline.benchmark import evaluate, manual_edit, rank_observations, select
from research_pipeline.models import verify_models
from research_pipeline.robots import retarget
from research_pipeline.schema import load_config, rigid_transform, timestamps, validate_poses
from research_pipeline.video import extract_video, generate_fixture


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/research-prototype/default.json"
CACHE = ROOT / "data/research-prototype/models"


class SchemaTests(unittest.TestCase):
    def test_frame_transform_and_time_contract(self):
        self.assertEqual(rigid_transform(np.eye(4)).shape, (4, 4))
        invalid = np.eye(4)
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            rigid_transform(invalid)
        with self.assertRaises(ValueError):
            timestamps([0, 0.1, 0.1])
        with self.assertRaises(ValueError):
            validate_poses([0, 1], [np.eye(4), np.eye(4)], [True, False])
        infinity_missing = np.stack([np.eye(4), np.full((4, 4), np.inf)])
        with self.assertRaises(ValueError):
            validate_poses([0, 1], infinity_missing, np.array([True, False]))

    def test_configuration_numeric_contract(self):
        base = load_config(CONFIG)
        from research_pipeline.schema import calibration
        for key, bad in (("fps", float("inf")), ("marker_size_m", float("nan")),
                         ("frames", 0), ("segment_length", 1.2), ("budget_s", -1),
                         ("oracle_edit_cost_s", 0)):
            value = copy.deepcopy(base)
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                calibration(value)

    def test_budget_rejects_overrun_and_invalid_seconds(self):
        self.assertEqual(select([2, 1, 0], 20.0, 10.0), [2, 1])
        self.assertEqual(select([2, 1, 0], 19.9, 10.0), [2])
        for budget, cost in ((-1, 10), (20, 0), (float("nan"), 10)):
            with self.assertRaises(ValueError):
                select([0], budget, cost)

    def test_manual_offset_preserves_missing_and_times_session(self):
        poses = np.stack([np.eye(4), np.full((4, 4), np.nan)])
        observations = {"timestamps": np.array([0., 1.]), "T_world_marker": poses,
                        "valid": np.array([True, False])}
        with patch("builtins.input", return_value="0.1 0 0"), patch(
                "research_pipeline.benchmark.time.monotonic_ns", side_effect=[0, 100_000_000, 200_000_000]):
            corrected, receipt = manual_edit(observations, 0, 2, 1.0)
        self.assertAlmostEqual(corrected["T_world_marker"][0, 0, 3], 0.1)
        self.assertTrue(np.isnan(corrected["T_world_marker"][1]).all())
        self.assertEqual(receipt["frames_edited"], 1)
        self.assertAlmostEqual(receipt["total_session_elapsed_s"], 0.2)
        with patch("builtins.input", return_value="0 0 0"), patch(
                "research_pipeline.benchmark.time.monotonic_ns", side_effect=[0, 100_000_000, 2_000_000_000]):
            with self.assertRaises(ValueError):
                manual_edit(observations, 0, 2, 1.0)
        with patch("builtins.input", return_value="0 0 0"), patch(
                "research_pipeline.benchmark.time.monotonic_ns", side_effect=[0, 100_000_000, 200_000_000]):
            _, empty_receipt = manual_edit(observations, 1, 1, 1.0)
        self.assertEqual(empty_receipt["frames_edited"], 0)


class VideoAndBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = copy.deepcopy(load_config(CONFIG))
        self.config["frames"] = 15
        self.video = self.root / "video.mp4"
        self.gt_path = self.root / "evaluator-only.npz"
        generate_fixture(self.video, self.gt_path, self.config)

    def test_real_video_extraction_preserves_dropout_and_pose_accuracy(self):
        observations = extract_video(self.video, self.config)
        gt = np.load(self.gt_path, allow_pickle=False)["T_world_marker"]
        self.assertEqual(len(observations["valid"]), 15)
        self.assertFalse(observations["valid"][9])
        self.assertFalse(observations["valid"][10])
        self.assertGreaterEqual(int(observations["valid"].sum()), 12)
        self.assertTrue(np.isnan(observations["T_world_marker"][~observations["valid"]]).all())
        error = np.linalg.norm(observations["T_world_marker"][observations["valid"], :3, 3]
                               - gt[observations["valid"], :3, 3], axis=1)
        self.assertLess(float(np.mean(error)), 0.025)
        with self.assertRaises(FileExistsError):
            generate_fixture(self.video, self.gt_path, self.config)

    def test_ranking_is_independent_of_evaluator_ground_truth(self):
        observations = extract_video(self.video, self.config)
        ranking = rank_observations(observations, 5)
        original_gt = np.load(self.gt_path, allow_pickle=False)["T_world_marker"]
        altered_gt = original_gt.copy()
        altered_gt[:, :3, 3] += np.array([1.0, -2.0, 3.0])
        first = evaluate(observations, original_gt, 5, 10.0, 10.0, 42)
        second = evaluate(observations, altered_gt, 5, 10.0, 10.0, 42)
        self.assertEqual(ranking, rank_observations(observations, 5))
        self.assertEqual(first["methods"]["ranked_oracle"]["selected_segments"],
                         second["methods"]["ranked_oracle"]["selected_segments"])
        self.assertEqual(first["methods"]["random_oracle"]["selected_segments"],
                         second["methods"]["random_oracle"]["selected_segments"])
        self.assertEqual(first["methods"]["ranked_oracle"]["frames_missing"], 2)
        self.assertNotEqual(first["methods"]["no_edit"]["mean_penalized_error_m"],
                            second["methods"]["no_edit"]["mean_penalized_error_m"])


class IntegrationTests(unittest.TestCase):
    def test_generated_video_to_real_robots_and_benchmark(self):
        try:
            verify_models(CACHE)
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = copy.deepcopy(load_config(CONFIG))
            config["frames"] = 15
            video, gt_path = root / "fixture.mp4", root / "gt.npz"
            generate_fixture(video, gt_path, config)
            observations = extract_video(video, config)
            trajectory = retarget(observations, CACHE)
            self.assertEqual(len(trajectory["frames"]), 15)
            valid_rows = [row for row in trajectory["frames"] if row["status"] == "ok"]
            self.assertGreaterEqual(len(valid_rows), 10)
            for row in valid_rows:
                self.assertEqual(len(row["panda"]["joint_positions_rad"]), 7)
                self.assertEqual(len(row["ur5e"]["joint_positions_rad"]), 6)
                self.assertLess(row["panda"]["position_error_m"], 0.005)
                self.assertLess(row["ur5e"]["position_error_m"], 0.005)
            gt = np.load(gt_path, allow_pickle=False)["T_world_marker"]
            results = evaluate(observations, gt, 5, 20.0, 10.0, 42)
            self.assertLessEqual(results["methods"]["ranked_oracle"]["simulated_elapsed_s"], 20)
            self.assertEqual(results["methods"]["ranked_oracle"]["frames_total"], 15)


if __name__ == "__main__":
    unittest.main()
