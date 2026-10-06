# Trajectory Retargeting Starter

This repository provides two reproducible references: clean-state Panda-to-UR5e physics replay and a separate marker-assisted pose/kinematics prototype. **They are not integrated into one video-to-physics pipeline.** IL (Invested Lamda) is the current research/prototype priority; the label is distinct from *imitation learning*.

Students and supervisors choose whether these references fit a capstone question and set its scope and assessment. This repository provides examples and evidence, not required assignments or institutional thresholds.

## Start here

- [Capstone reader and sources](docs/reader-guide.md) · [PowerPoint](docs/presentations/capstone-reader.pptx) · [PDF](docs/presentations/capstone-reader.pdf)
- [Reading guide](docs/reading-guide.md) — prior work and a conditional research question.
- [Marker prototype](docs/marker-prototype.md) — setup and evidence scope.
- [Physics replay](physics_replay/README.md) — CPU setup, demo, replay, and benchmark.
- [Data contracts](docs/data-contracts.md) · [Evidence and limitations](docs/evidence-and-limitations.md) · [Experimental protocol](docs/experimental-protocol.md)

[![Open the physics notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/colab_physics_replay.ipynb)

## Quick start: physics replay

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv-physics
source .venv-physics/bin/activate
python -m pip install -r physics_replay/requirements.txt
python -m physics_replay fetch-models
python -m pytest -q physics_replay/tests
python -m physics_replay demo
python -m physics_replay benchmark
```

The fetch command downloads models from the pinned MuJoCo Menagerie revision into ignored `data/`. See the [physics guide](physics_replay/README.md) before changing benchmark settings.

## Quick start: marker prototype

Use a separate Python environment because the two packages have independent dependency pins:

```bash
python3.12 -m venv .venv-marker
source .venv-marker/bin/activate
python -m pip install -r requirements-research.txt
python tools/research-prototype.py fetch-models
python -m unittest discover -s tests/research_pipeline -v
python tools/research-prototype.py demo --output data/research-prototype/runs/my-demo
```

The marker example uses one synthetic calibrated ArUco clip and transfers translation displacement only. Its oracle costs are simulated assumptions. It does not establish markerless human-video extraction, physical task success, or measured user benefit.

## Evidence in one sentence

The physics benchmark reports 100 paired layouts for one cube task and one Panda source demonstration at seed 7; trimmed object-relative replay succeeded in 100/100 layouts under the current predicate. These are not 100 demonstrations or physical robot trials, and a new held-out set is needed after tuning against seed 7. The marker experiment is separate. See [evidence and limitations](docs/evidence-and-limitations.md).

## License and provenance

Starter-authored code is under the MIT License. [`NOTICE.md`](NOTICE.md) records source provenance and third-party boundaries. Robot models are fetched from their pinned upstream source and retain their own licenses; research-paper PDFs are linked, not redistributed.
