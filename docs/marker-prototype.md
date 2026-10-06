# Marker-assisted pose and kinematics prototype

The `research_pipeline/` package illustrates a CPU path from synthetic, calibrated ArUco pixels through marker pose estimation, Panda kinematics, and translational displacement transfer to a UR5e. It is independent of `physics_replay/`; there is no integrated video-to-physics adapter.

## Setup

Use a separate Python 3.12 environment because each reference has independent dependency pins:

```bash
python3.12 -m venv .venv-marker
source .venv-marker/bin/activate
python -m pip install -r requirements-research.txt
python tools/research-prototype.py fetch-models
python -m unittest discover -s tests/research_pipeline -v
python tools/research-prototype.py demo --output data/research-prototype/runs/my-demo
```

The output directory must be new. Models and generated runs live in ignored `data/`. The fetched robot models are pinned to the upstream Menagerie commit in `research_pipeline/models.py`, and include an upstream license/hash receipt.

A real marker clip requires correct image size, camera intrinsics/distortion, camera-to-world transform, marker ID and size, and robot-frame alignment. The default identity calibration is a synthetic fixture, not real-camera calibration. A marker is only a proxy for an object after applying a known marker-to-object transform.

## Evidence boundary

The demo used one synthetic calibrated clip and seed. It keeps evaluator ground truth separate from the pixel extractor. Robot transfer is translation-only; orientation, collision, velocity, contact, grasp, and task success are outside this experiment. Default oracle costs and budget are simulated assumptions, not measured human time.

This does not establish markerless human-video extraction, generalization, method superiority, physical-robot performance, or measured user benefit. The marker implementation has no independent inspection verdict. See [contracts](data-contracts.md) and [evidence limits](evidence-and-limitations.md).
