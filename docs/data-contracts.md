# Data contracts

The two packages use separate inputs and schemas. Their executable definitions are in `physics_replay/episode.py`, `research_pipeline/schema.py`, and their configuration files.

## Physics episode

`physics_replay.episode.v1` records timestamps, robot and frame conventions, a TCP definition, scene transforms, object poses, gripper commands, and joint positions. Units are metres, radians, and seconds; quaternions are `(w, x, y, z)`. `T_A_B` maps coordinates from frame B into frame A. A missing object pose is JSON `null` (NaN in memory) and remains in the time series.

The included 10 Hz example has 102 frames, with object pose missing in 36. The current retargeter uses a valid source object anchor before lift; later object-pose frames do not affect its present plan.

## Marker observations

The marker output `il-marker-poses.v1` records timestamps, estimated `T_world_marker` transforms, and a validity mask. Missing estimates remain invalid. Camera matrix, distortion, resolution, marker ID/size, and `T_world_camera` are specified in the validated configuration. Extractor/ranking inputs do not use evaluator ground truth. Marker pose is not automatically object pose without a calibrated marker-to-object transform.

## Provenance

Physics result manifests retain code, config, and model-receipt hashes. Absolute workstation `config_path` values were replaced with repository-relative paths and recorded in `provenance_redactions`. [`provenance.json`](../provenance.json) lists hashes for copied runtime source and configuration files.
