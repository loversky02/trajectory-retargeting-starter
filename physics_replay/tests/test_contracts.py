"""Frame, schema, segmentation and statistics contracts. No robot models needed."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physics_replay import episode as ep
from physics_replay import retarget as rt
from physics_replay import runner
from physics_replay import transforms as tf
from physics_replay.metrics import wilson_interval
from physics_replay.source_demo import R_DOWN

CONFIG_PATH = runner.REPO_ROOT / "config" / "physics-replay" / "pick-place-v1.json"
EXAMPLE = Path(rt.__file__).parent / "examples" / "panda_pick_place_10hz.json"


def random_pose(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    return tf.make(tf.quat_wxyz_to_matrix(quat / np.linalg.norm(quat)), rng.uniform(-1, 1, 3))


def synthetic_episode(T_world_object: np.ndarray, T_world_goal: np.ndarray, hz: float = 50.0) -> dict:
    """Straight-line pick and place: approach from far away, grasp, lift, carry, lower, release."""
    grasp = tf.make(R_DOWN, T_world_object[:3, 3] + [0, 0, 0.01])
    place = tf.make(R_DOWN, T_world_goal[:3, 3] + [0, 0, 0.01])
    up = tf.make(np.eye(3), [0, 0, 0.15])
    waypoints = [(tf.make(R_DOWN, [0.1, 0.4, 0.5]), 0.0, 10), (up @ grasp, 0.0, 30), (grasp, 0.0, 20),
                 (grasp, 1.0, 10), (up @ grasp, 1.0, 20), (up @ place, 1.0, 30), (place, 1.0, 20),
                 (place, 0.0, 10), (up @ place, 0.0, 10)]
    poses, gripper, objects = [], [], []
    previous = waypoints[0][0]
    for target, closed, steps in waypoints:
        for i in range(1, steps + 1):
            pose = tf.interpolate(previous, target, i / steps)
            poses.append(pose)
            gripper.append(closed)
            holding = closed and not np.allclose(target, grasp)
            objects.append(pose @ tf.inverse(grasp) @ T_world_object if holding or (
                closed and np.allclose(target, place)) else (objects[-1] if objects else T_world_object))
        previous = target
    n = len(poses)
    scene = {"T_world_object": T_world_object.tolist(), "T_world_goal": T_world_goal.tolist()}
    return ep.build("panda", "synthetic", scene, np.arange(n) / hz, np.array(poses), np.array(objects),
                    np.array(gripper), np.zeros((n, 7)))


class TransformTest(unittest.TestCase):
    def test_inverse_and_quaternion_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            pose = random_pose(rng)
            np.testing.assert_allclose(pose @ tf.inverse(pose), np.eye(4), atol=1e-12)
            quat = tf.matrix_to_quat_wxyz(pose[:3, :3])
            self.assertGreaterEqual(quat[0], 0.0)
            np.testing.assert_allclose(tf.quat_wxyz_to_matrix(quat), pose[:3, :3], atol=1e-12)

    def test_rotation_log_near_pi(self):
        for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 1.0]) / math.sqrt(2), np.array([-0.3, 0.5, 0.81])):
            axis = axis / np.linalg.norm(axis)
            angle = math.pi - 1e-8
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
            omega = tf.rotation_log(rotation)
            self.assertAlmostEqual(np.linalg.norm(omega), angle, places=6)
            self.assertAlmostEqual(abs(float(omega @ axis)), angle, places=5)

    def test_interpolation_endpoints_and_midpoint(self):
        a = tf.make(np.eye(3), [0, 0, 0])
        b = tf.make(tf.rot_z(1.0), [1, 2, 3])
        np.testing.assert_allclose(tf.interpolate(a, b, 0.0), a, atol=1e-12)
        np.testing.assert_allclose(tf.interpolate(a, b, 1.0), b, atol=1e-12)
        middle = tf.interpolate(a, b, 0.5)
        np.testing.assert_allclose(middle[:3, :3], tf.rot_z(0.5), atol=1e-12)
        np.testing.assert_allclose(middle[:3, 3], [0.5, 1, 1.5])


class EpisodeSchemaTest(unittest.TestCase):
    def setUp(self):
        self.episode = synthetic_episode(tf.make(tf.rot_z(0.3), [0.5, -0.1, 0.025]),
                                         tf.make(tf.rot_z(-0.2), [0.5, 0.15, 0.025]))

    def test_round_trip_keeps_missing_rows_as_nan(self):
        episode = dict(self.episode)
        positions = np.asarray(episode["object_position_m"])
        quats = np.asarray(episode["object_quat_wxyz"])
        positions[40:60] = np.nan
        quats[40:60] = np.nan
        episode["object_position_m"], episode["object_quat_wxyz"] = positions.tolist(), quats.tolist()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.json"
            ep.save(episode, path)
            self.assertIn("null", path.read_text())
            back = ep.load(path)
        np.testing.assert_array_equal(np.isnan(back["object_position_m"]), np.isnan(positions))
        np.testing.assert_allclose(np.nan_to_num(back["object_position_m"]), np.nan_to_num(positions))
        self.assertTrue(np.isnan(ep.object_poses(back)[50]).all())

    def assertRejected(self, **changes):
        episode = json.loads(json.dumps(ep.to_json_value(self.episode)))
        for key, value in changes.items():
            episode[key] = value
        with self.assertRaises(ValueError):
            ep.validate(episode)

    def test_rejects_contract_violations(self):
        n = len(self.episode["timestamps_s"])
        timestamps = list(self.episode["timestamps_s"])
        timestamps[5] = timestamps[4]
        self.assertRejected(timestamps_s=timestamps)
        self.assertRejected(schema_version="physics_replay.episode.v0")
        partial = [list(row) for row in self.episode["object_position_m"]]
        partial[3][1] = float("nan")
        self.assertRejected(object_position_m=partial)
        quats = [list(row) for row in self.episode["tcp_quat_wxyz"]]
        quats[2] = [0.9, 0.0, 0.0, 0.0]
        self.assertRejected(tcp_quat_wxyz=quats)
        tcp = [list(row) for row in self.episode["tcp_position_m"]]
        tcp[7] = [float("nan")] * 3
        self.assertRejected(tcp_position_m=tcp, tcp_quat_wxyz=[[float("nan")] * 4 if i == 7 else q
                                                                for i, q in enumerate(self.episode["tcp_quat_wxyz"])])
        self.assertRejected(gripper_command=[1.5] * n)
        missing = dict(self.episode)
        del missing["scene"]
        with self.assertRaises(ValueError):
            ep.validate(missing)

    def test_committed_example_is_valid_and_occluded_in_transit(self):
        example = ep.load(EXAMPLE)
        missing = np.isnan(np.asarray(example["object_position_m"])[:, 0])
        closed = np.asarray(example["gripper_command"]) >= 0.5
        self.assertTrue(missing.any())
        self.assertTrue(closed[missing].all(), "object is only hidden while it is held")
        self.assertFalse(missing[0])


class RetargetTest(unittest.TestCase):
    def setUp(self):
        self.config, _ = runner.load_config(CONFIG_PATH)
        self.source = synthetic_episode(tf.make(tf.rot_z(0.3), [0.5, -0.1, 0.025]),
                                        tf.make(tf.rot_z(-0.2), [0.5, 0.15, 0.025]))
        self.target = runner.scene_spec(self.config, self.config["demo_target_scene"])
        self.start = tf.make(R_DOWN, [0.2, 0.3, 0.4])

    def plan(self, method):
        return rt.retarget(self.source, self.target, self.start, method, 50.0, self.config["retarget"])

    def test_segmentation_reads_gripper_and_height(self):
        threshold = self.config["retarget"]["segment_lift_threshold_m"]
        cuts = rt.segment(self.source, threshold)
        gripper = np.asarray(self.source["gripper_command"])
        self.assertEqual(cuts["grasp_frame"], int(np.argmax(gripper >= 0.5)))
        z = np.asarray(self.source["tcp_position_m"])[:, 2]
        self.assertGreaterEqual(z[cuts["split_frame"]], z[cuts["grasp_frame"]] + threshold)
        self.assertLess(z[cuts["split_frame"] - 1], z[cuts["grasp_frame"]] + threshold)
        self.assertEqual(gripper[cuts["release_frame"]], 0.0)
        self.assertEqual(gripper[cuts["release_frame"] - 1], 1.0)

    def test_world_frame_replays_source_poses(self):
        plan = self.plan("world_frame")
        source = ep.tcp_poses(self.source)
        for k in np.flatnonzero(plan.source_frame >= 0)[::25]:
            np.testing.assert_allclose(plan.tcp_targets[k], source[plan.source_frame[k]], atol=1e-9)

    def test_object_relative_keeps_grasp_relative_to_object(self):
        plan = self.plan("object_relative")
        cuts = plan.notes["segmentation"]
        source = ep.tcp_poses(self.source)
        T_src_object = rt.source_object_reference(self.source, cuts["grasp_frame"])
        T_src_goal = np.asarray(self.source["scene"]["T_world_goal"])
        grasp = int(np.flatnonzero(plan.source_frame == cuts["grasp_frame"])[0])
        np.testing.assert_allclose(tf.inverse(self.target.T_world_object) @ plan.tcp_targets[grasp],
                                   tf.inverse(T_src_object) @ source[cuts["grasp_frame"]], atol=1e-9)
        release = int(np.flatnonzero(plan.source_frame == cuts["release_frame"])[0])
        np.testing.assert_allclose(tf.inverse(self.target.T_world_goal) @ plan.tcp_targets[release],
                                   tf.inverse(T_src_goal) @ source[cuts["release_frame"]], atol=1e-9)
        self.assertEqual(plan.segment[0], "transition_to_object")
        np.testing.assert_allclose(plan.tcp_targets[-1],
                                   self.target.T_world_goal @ tf.inverse(T_src_goal) @ source[-1], atol=1e-9)

    def test_trimmed_variant_drops_free_space_approach(self):
        full, trimmed = self.plan("object_relative"), self.plan("object_relative_trimmed")
        first = trimmed.notes["segmentation"]["approach_frame"]
        radius = self.config["retarget"]["approach_radius_m"]
        T_src_object = rt.source_object_reference(self.source, 0)
        distance = np.linalg.norm(np.asarray(self.source["tcp_position_m"])[first:trimmed.notes["segmentation"]
                                  ["grasp_frame"] + 1] - T_src_object[:3, 3], axis=1)
        self.assertTrue((distance <= radius).all())
        self.assertGreater(np.linalg.norm(np.asarray(self.source["tcp_position_m"])[first - 1]
                                          - T_src_object[:3, 3]), radius)
        self.assertEqual(trimmed.source_frame[trimmed.source_frame >= 0].min(), first)
        cut = int(np.flatnonzero(full.source_frame == first)[0])
        k = int(np.flatnonzero(trimmed.source_frame == first)[0])
        np.testing.assert_allclose(trimmed.tcp_targets[k], full.tcp_targets[cut], atol=1e-9)

    def test_object_missing_before_grasp_uses_first_observation(self):
        episode = json.loads(json.dumps(ep.to_json_value(self.source)))
        for key in ("object_position_m", "object_quat_wxyz"):
            for i in range(5):
                episode[key][i] = [float("nan")] * len(episode[key][i])
        reference = rt.source_object_reference(ep.validate(episode), 100)
        np.testing.assert_allclose(reference, ep.object_poses(self.source)[5], atol=1e-12)


class StatisticsTest(unittest.TestCase):
    def test_wilson_interval(self):
        low, high = wilson_interval(41, 50)
        self.assertAlmostEqual(low, 0.6920, places=3)
        self.assertAlmostEqual(high, 0.9022, places=3)
        self.assertEqual(wilson_interval(0, 50)[0], 0.0)
        self.assertAlmostEqual(wilson_interval(0, 50)[1], 0.0713, places=3)
        for trials in range(1, 301):
            self.assertEqual(wilson_interval(0, trials)[0], 0.0)
            self.assertEqual(wilson_interval(trials, trials)[1], 1.0)
            low, high = wilson_interval(trials // 3, trials)
            self.assertLessEqual(low, (trials // 3) / trials)
            self.assertGreaterEqual(high, (trials // 3) / trials)

    def test_exact_mcnemar(self):
        self.assertEqual(runner.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(runner.mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(runner.mcnemar_exact(0, 3), 0.25)
        self.assertAlmostEqual(runner.mcnemar_exact(1, 9), 2 * 11 / 1024)

    def test_layout_sampling_is_seeded_and_spaced(self):
        config, _ = runner.load_config(CONFIG_PATH)
        first = runner.sample_layouts(config, 20, 3)
        self.assertEqual(first, runner.sample_layouts(config, 20, 3))
        self.assertNotEqual(first, runner.sample_layouts(config, 20, 4))
        for layout in first:
            distance = math.hypot(layout["object"]["x_m"] - layout["goal"]["x_m"],
                                  layout["object"]["y_m"] - layout["goal"]["y_m"])
            self.assertGreaterEqual(distance, config["target_sampling"]["min_object_goal_distance_m"])

    def test_config_version_is_enforced(self):
        config = json.loads(CONFIG_PATH.read_text())
        config["config_version"] = "physics_replay.pick_place.v0"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                runner.load_config(path)


if __name__ == "__main__":
    unittest.main()
