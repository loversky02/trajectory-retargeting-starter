# physics_replay: Panda → UR5e pick-and-place retargeting in MuJoCo

A reference demo for the first two steps of the trajectory-retargeting capstone:

1. a trajectory schema that a demonstration log must satisfy;
2. retargeting one Panda pick-and-place demonstration onto a UR5e with a Robotiq
   2F-85, executed in physics in a scene the demo never saw.

The physics demo starts from clean simulator state (ground-truth object pose), so a failure
points at the recorded trajectory, grasp, controller or reachability rather than at vision.
A separate [marker-assisted prototype guide](../docs/marker-prototype.md) covers `research_pipeline/`;
it uses synthetic calibrated pixels and transfers translation only. The packages are not
connected as a video-to-physics pipeline. See [evidence and limitations](../docs/evidence-and-limitations.md).

Everything runs on a CPU. One physics run takes about 0.5 s.

## Run it

Python 3.12. From the repository root:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r physics_replay/requirements.txt
python -m physics_replay fetch-models      # once: pinned Menagerie Panda, UR5e, 2F-85 (117 files)
python -m physics_replay demo              # source demo + 4 target runs, videos and plots
python -m physics_replay benchmark         # 50 seeded target layouts x 4 methods
python -m pytest -q physics_replay/tests   # 20 tests
```

Outputs go to `data/physics-replay/runs/<kind>-<time>/` unless `--out` is given.
That folder is git-ignored. Other commands:

```bash
# retarget any schema-valid episode onto the UR5e in the demo target scene
python -m physics_replay replay --episode physics_replay/examples/panda_pick_place_10hz.json --method object_relative_trimmed
# held-out layouts
python -m physics_replay benchmark --seed 7 --configs 100
```

On Google Colab, open [`colab_physics_replay.ipynb`](colab_physics_replay.ipynb), which clones this
public repository's `main` branch directly. No ZIP upload or authentication is required.
A CPU runtime is enough; without a GPU the videos render through OSMesa.

The notebook runs the tests, the demo, the example replay and both benchmarks. It then
reports how many benchmark trials agree with `results/`. Runs are deterministic on one
machine but not guaranteed across machines, so the notebook reports agreement rather than
asserting it.

The source reference pack reports a Colab CPU run on 2026-10-06 (Linux x86-64, Python 3.13,
OSMesa): 20 tests passed and 200/200 dev plus 400/400 held-out trials agreed with these
artifacts in success and failure reason. This public packaging pass did not repeat those 600
benchmark trials or independently rerun the notebook in Colab. The notebook's acquisition
cell now clones this public starter directly. The source pack reported benchmark times of 7.6
and 14.8 min in Colab and 1.8 and 3.3 min on an Apple-silicon laptop.

## What the demo does

- The **source**: a scripted Panda pick-and-place in the source scene. The cube is
  at (0.50, −0.12) m, yaw 0.30 rad. The goal is at (0.52, 0.15) m, yaw −0.20 rad.
  The run is recorded as an episode.
- The **target**: the episode is retargeted onto the UR5e + 2F-85 in a different
  scene and executed in physics. The cube moves to (0.45, −0.20) m, yaw −0.50 rad.
  The goal moves to (0.58, 0.22) m, yaw 0.60 rad.
- Both robots stand at the world origin and the table surface is z = 0. Each
  gripper's TCP is its finger-pad centre, with +z along the approach direction and
  +y along the closing axis. A TCP pose therefore transfers between grippers
  without an extra offset. The offsets are in [`scene.py`](scene.py).
- Execution uses damped least-squares 6-D IK on a scratch `MjData`, then joint
  position servos at 50 Hz with gravity compensation ([`execute.py`](execute.py)).

The four target methods (`METHOD_NOTES` in [`runner.py`](runner.py)):

| Method | What it does |
|---|---|
| `world_frame` | Replays the source TCP poses at the same world coordinates. This is the raw end-effector baseline. |
| `object_relative` | Splits the demo once the grasped cube is lifted. The part before the split is re-anchored on the target cube, the part after on the target goal, and a planned free-space transition joins them. |
| `object_relative_trimmed` | Same as `object_relative`, but drops the source's free-space approach. Object-anchored frames start where the source TCP came within 0.15 m of the cube, and a transition from the UR5e's own start pose covers the rest. |
| `target_scripted` | The UR5e runs the scripted demo directly on the target scene. This task-informed reference uses target-specific information; it is not a retargeting method or a mathematical success upper bound. |

Segmentation reads only the gripper command and TCP height, never the script's phase
labels. Any schema-valid episode can therefore be retargeted.

## Episode schema

Defined in [`episode.py`](episode.py) as `physics_replay.episode.v1`, one JSON file
per episode:

- **Per-frame arrays:** timestamps, TCP pose, object pose, gripper command (0 open …
  1 closed) and joint positions.
- **Units:** metres, radians, seconds; quaternions are (w, x, y, z).
- **Missing poses:** a row of NaN (`null` in JSON), never a plausible coordinate.
- **Scene and TCP:** each episode carries the scene (`T_world_object`, `T_world_goal`,
  cube spec), the TCP definition and the frame convention. `T_A_B` maps coordinates
  in frame B into frame A.

[`examples/panda_pick_place_10hz.json`](examples/panda_pick_place_10hz.json) is the
source demo downsampled to 10 Hz. Its object pose is `null` in 36 of 102 frames,
the frames where the gripper would hide the cube in transit. It still replays
successfully, because retargeting needs the object pose only before the lift.

## Success and failure taxonomy

A run succeeds only when every flag below is false ([`metrics.py`](metrics.py), thresholds
in [`pick-place-v1.json`](../config/physics-replay/pick-place-v1.json)). The first true
flag, in this order, is the reported failure reason. All flags are kept in the output.

| Flag | Condition |
|---|---|
| `ik_unreachable` | IK residual > 5 mm or > 0.05 rad on any tick |
| `floor_collision` | any robot–floor contact |
| `grasp_failed` | cube never lifted ≥ 5 cm while the gripper is closed |
| `dropped_in_transit` | cube back on the table away from the goal while the gripper is still closed |
| `tipped` | final tilt > 0.17 rad |
| `place_miss` | final xy error > 2 cm, or the cube is not resting on the table (±5 mm) |
| `not_released` | gripper still closed at the end |

## Results

These were produced with the code and config in this folder. Summaries, manifests,
per-trial logs and plots are in [`results/`](results/). Every trial uses one sampled
target layout, and every method runs on the same layouts, so comparisons are paired.

| Method | Dev (seed 20261006, 50 layouts) | Held-out (seed 7, 100 layouts) | Failure reasons (held-out) |
|---|---|---|---|
| `world_frame` | 0/50 (Wilson 95% 0.00–0.07) | 0/100 (0.00–0.04) | grasp_failed 94, place_miss 4, dropped_in_transit 2 |
| `object_relative` | 41/50 (0.69–0.90) | 89/100 (0.81–0.94) | ik_unreachable 11 |
| `object_relative_trimmed` | 50/50 (0.93–1.00) | 100/100 (0.96–1.00) | none |
| `target_scripted` | 50/50 (0.93–1.00) | 100/100 (0.96–1.00) | none |

The target-scripted reference uses task-specific target information. Its matching 100/100 is not an upper bound on success across methods or tasks. The trimmed method was selected after the development set exposed reachability failures; seed 7 must not be reused for tuning and then described as held out. Any tuning against these results requires a new held-out set.

Median final placement error of successful held-out runs:

| Method | Error |
|---|---|
| `object_relative` | 1.3 mm |
| `object_relative_trimmed` | 1.2 mm |
| `target_scripted` | 0.8 mm |

Held-out paired comparisons (exact McNemar):

- `object_relative` vs `world_frame`: 89 layouts where only `object_relative` succeeds, p = 3e-27.
- `object_relative_trimmed` vs `object_relative`: 11 layouts where only the trimmed method succeeds, p = 0.001.

Notes on the results:

- **`world_frame`** reaches for where the cube was in the source scene, so it almost
  never grasps. The 6 held-out exceptions grasped by accident and then dropped or
  misplaced the cube.
- **`object_relative`** fails only on reachability, and only for target cubes far
  out (x ≥ 0.54 m) and yawed negative (≤ −0.26 rad). Re-anchoring the source's whole
  free-space approach on such a cube swings the approach start outside the UR5e
  workspace. The IK residual reaches 2 cm in the early object-anchored frames and
  the transition into them.
- **`object_relative_trimmed`** keeps only the object-near part of the demo, which
  removes those failures. It was designed after the dev run exposed the 9
  reachability failures, so its dev score is not an independent test. No design
  choice used the held-out seed 7 layouts.
- **Tracking.** The position servos lag a moving target. In the demo run:

  | Robot | Mean TCP tracking error | Max |
  |---|---|---|
  | Panda source | 11 mm | 35 mm |
  | UR5e retargeted runs | 23–25 mm | 53–70 mm |
  | UR5e `target_scripted` | 34 mm | 149 mm |

  The peaks fall at the fastest target motion; `target_scripted` peaks at
  0.87 m/s. Once the target holds still the error settles below 3 mm, and the
  tests assert this. Success is judged on the cube, not on tracking error.

The predicate checks lift, transport, final position, uprightness, and release. It does not
require a settle-time/velocity threshold or goal-yaw accuracy. These numbers show a
retargeting chain working in physics from clean state for one rigid cube, one task, and one
source demo. They do not show performance on video-derived poses, unreliable pose segments,
or human correction.

## Reproducibility

- **Model pinning:** models are pinned to one MuJoCo Menagerie commit
  ([`models.py`](models.py)). `fetch-models` writes a SHA-256 receipt, and every `demo`,
  `replay` and `benchmark` command verifies it offline. Calling `runner.run` directly
  from Python skips this check.
- **Manifests:** every run writes `manifest.json` with the command, the config,
  receipt and code SHA-256, the git commit and dirty flag, the Python, MuJoCo and
  NumPy versions, and the platform.
- **Determinism:** runs are deterministic on one machine, and a test checks this.
  Layouts come from the seed in the config, or from `--seed`.
- **Benchmark outputs:** each benchmark writes `layouts.json` and `trials.jsonl`
  (one row per layout × method). It also writes `failures.jsonl` and, under
  `failures/`, the episode, plan and tracking plot of every failed run.

## Status

| Part | Built | Tested | Proposed next |
|---|---|---|---|
| Episode schema and validator | yes | 15 contract tests: transforms, schema round-trip and violations, occluded example, segmentation, the three retarget methods, Wilson/McNemar, seeded layouts, config version | add camera intrinsics/extrinsics and a per-frame pose-confidence field when an extractor exists |
| Panda → UR5e retarget from clean state | yes, 3 methods + task-informed scripted reference | integration tests: source tracking, object_relative vs world_frame, determinism, occluded example, replay artifacts | collision-aware transitions; orientation-sensitive grasps; more objects and tasks |
| Physics execution and success predicate | yes | taxonomy above, 150 paired layouts | contact-force and slip logging |
| Pose quality and recovery | no | — | inject pose noise, dropouts and scale error into the source episode, then detect and repair unreliable segments |
| Segment correction (TeleGuide) | no | — | per-segment waypoint or pose-offset edits with a logged time budget |
| Video → episode extractor | separate marker/PnP prototype in `research_pipeline/`; not connected to this package | marker pipeline tests cover one synthetic calibrated clip | establish a tested adapter and real-video calibration before claiming video-to-physics integration; markerless human-video extraction remains untested |

## Files

| Path | Contents |
|---|---|
| `__main__.py` | CLI: `fetch-models`, `demo`, `replay`, `benchmark` |
| `transforms.py` | SE(3) helpers and quaternion conversion |
| `models.py` | pinned Menagerie fetch and receipt check |
| `scene.py` | the two robot scenes and the cube |
| `kinematics.py` | 6-D TCP IK |
| `execute.py` | `Plan`, closed-loop execution, minimum-jerk transitions |
| `source_demo.py` | scripted pick-and-place plan (the source demo and `target_scripted`) |
| `retarget.py` | segmentation and the three retargeting methods |
| `metrics.py` | success predicate, failure taxonomy, Wilson interval |
| `runner.py` | demo, replay and benchmark orchestration, provenance |
| `report.py` | videos, tracking plots, benchmark plot and summary |
| `../config/physics-replay/pick-place-v1.json` | every threshold, scene and sampling range |
