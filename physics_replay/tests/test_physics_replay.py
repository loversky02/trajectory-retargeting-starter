"""End-to-end physics replay on the pinned Menagerie models.

Skipped when the model cache is missing; run ``python -m physics_replay fetch-models`` once.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physics_replay import episode as ep
from physics_replay import runner
from physics_replay import transforms as tf
from physics_replay.__main__ import DEFAULT_CACHE, DEFAULT_CONFIG
from physics_replay.models import verify_models

EXAMPLE = Path(runner.__file__).parent / "examples" / "panda_pick_place_10hz.json"

try:
    verify_models(DEFAULT_CACHE)
    MODELS_READY = True
except (FileNotFoundError, ValueError):
    MODELS_READY = False


@unittest.skipUnless(MODELS_READY, "model cache missing: python -m physics_replay fetch-models")
class PhysicsReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = runner.load_config(DEFAULT_CONFIG)
        cls.source = runner.record_source(cls.config, DEFAULT_CACHE)
        cls.target = runner.scene_spec(cls.config, cls.config["demo_target_scene"])

    def run_target(self, method, source_episode=None):
        return runner.run(self.config["target_robot"], self.target, self.config, DEFAULT_CACHE, method,
                          source_episode if source_episode is not None else self.source["episode"])

    def test_source_episode_is_schema_valid_and_tracks_plan(self):
        episode = self.source["episode"]
        ep.validate(episode)
        self.assertEqual(episode["robot"], "panda")
        executed = ep.tcp_poses(episode)
        requested = np.asarray(self.source["plan"].tcp_targets)
        errors = np.array([tf.pose_error(a, b)[0] for a, b in zip(executed, requested)])
        # The position servos lag a moving target by centimetres; once the target holds still they settle.
        still = np.r_[False, np.linalg.norm(np.diff(requested[:, :3, 3], axis=0), axis=1) < 1e-6]
        self.assertGreater(still.sum(), 20)
        self.assertLess(float(errors[still].max()), 0.003)
        self.assertLess(self.source["metrics"]["final_xy_error_m"], self.config["success"]["place_xy_tolerance_m"])

    def test_object_relative_succeeds_where_world_frame_misses(self):
        world = self.run_target("world_frame")["metrics"]
        relative = self.run_target("object_relative")
        self.assertFalse(world["success"])
        self.assertEqual(world["failure_reason"], "grasp_failed")
        self.assertTrue(relative["metrics"]["success"], relative["metrics"]["failure_reason"])
        ep.validate(relative["episode"])
        self.assertEqual(relative["episode"]["robot"], "ur5e_2f85")

    def test_replay_is_deterministic(self):
        first = self.run_target("object_relative_trimmed")
        second = self.run_target("object_relative_trimmed")
        np.testing.assert_array_equal(first["episode"]["object_position_m"], second["episode"]["object_position_m"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_committed_example_replays_through_occlusion(self):
        example = ep.load(EXAMPLE)
        result = self.run_target("object_relative_trimmed", example)
        self.assertTrue(result["metrics"]["success"], result["metrics"]["failure_reason"])

    def test_replay_command_writes_manifest_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = runner.replay(EXAMPLE, DEFAULT_CONFIG, DEFAULT_CACHE, out, "object_relative_trimmed",
                                     False, ["test"])
            for name in ("manifest.json", "target_object_relative_trimmed_episode.json",
                         "target_object_relative_trimmed_plan.json", "target_object_relative_trimmed_tracking.png"):
                self.assertTrue((out / name).exists(), name)
            ep.load(out / "target_object_relative_trimmed_episode.json")
        for key in ("config_sha256", "model_receipt_sha256", "code_sha256", "episode_sha256"):
            self.assertEqual(len(manifest[key]), 64, key)
        self.assertEqual(manifest["metrics"]["success"], True)


if __name__ == "__main__":
    unittest.main()
