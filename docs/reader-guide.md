# Fall 2026 Capstone: slide explanations and sources

Academic English reader edition, v2.0, 6 October 2026. This companion provides explanations and sources for the 31-slide PPTX/PDF. It offers technical orientation for senior CS students and supervisors, who retain responsibility for capstone scope and assessment.

The shared research core supports IL (Invested Lamda), CS reference materials, and future research. Implementation, testing, and empirical evaluation are distinguished throughout.

## 1. Trajectory Extraction, Cross-Embodiment Retargeting & TeleGuide

This pack is intended for senior CS students and research supporters. Students and supervisors determine the capstone topic, scope, and assessment. IL (Invested Lamda) is the supporter’s current research/prototype priority; this project label is distinct from imitation learning. The update retains the design of the 25 September 2026 reader deck and incorporates evidence available by 6 October 2026. Simulation results do not establish novelty, human benefit, or performance on physical robots.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/README.md

## 2. Purpose and use of this reference pack

Read the pack in two passes: first examine scope, pipeline contracts, and results; then consult code, notes, and primary papers for experimental decisions. Reproducing every paper or training a policy is unnecessary for the initial feasibility study. Physics replay begins with clean simulator state. The marker prototype estimates pose from calibrated synthetic pixels. Their integration is incomplete. Implementation, contract tests, pilot observations, and research evaluations represent different levels of evidence.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/README.md

## 3. Shared technical core and distinct objectives

The shared core comprises video-derived object poses, Panda/UR5e retargeting, segment correction, and budget evaluation. IL receives reusable pipelines, measured capabilities, and engineering decisions. CS students receive reference materials for feasibility assessment. Future research receives testable questions, controls, and an evidence trail. Preserve the label Invested Lamda. This core does not define business requirements, authorize deployment, or impose student assignments.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/evidence-and-limitations.md

## 4. Pipeline contracts and levels of evidence

Pose accuracy, kinematic tracking, physics task success, and human benefit are distinct endpoints. Human video additionally requires grasp/contact, TCP orientation, phase, goal, and scale estimation. Robot logs with TCP/gripper state offer a more controlled starting point. The physics reference currently uses a clean TCP trajectory and known target layout. The marker prototype transfers translational displacement through kinematics. Estimated data and evaluator ground truth must remain on separate data paths.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/experimental-protocol.md

## 5. Simulation reference

The current baseline uses MuJoCo on a CPU. Select a simulator according to the research question, available inputs, and reproducibility requirements. Changing the physics engine or controller across methods introduces potential confounding. Isaac may be considered when rendering or parallel learning requirements exceed the current baseline. Such adoption is a proposed extension, not an outcome of a simulator-comparison experiment.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/README.md

## 6. Input assumptions and evaluation objectives

Metric scale and coordinate frames require an identifiable reference. A marker requires known size, camera intrinsics/extrinsics, and a marker-to-object transform. A CAD-based method requires consistent geometry and object-frame conventions. Human video does not directly provide robot actions. Describe source and target observations separately: using a ground-truth target pose can isolate source estimation, but it does not evaluate complete perception. Prespecify the primary endpoint and retain unsuccessful trials rather than substituting tracking accuracy for task success.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/README.md

## 7. Implemented MuJoCo reference

Execution uses MuJoCo mj_step, a free-joint cube, and contacts rather than teleporting or welding the object during motion. Resetting the initial state is a separate operation. Menagerie models are pinned at f054586a8e90465d49ee5be15335c4a0c7f57caf. TCP conventions are defined in scene.py. CLI commands verify the model receipt offline; a direct runner.run call does not perform that verification. The review reran 20 tests without skips. This establishes bounded simulation feasibility, not physical-robot safety.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/README.md

## 8. Isaac and larger vision models

These are adoption criteria, not measured superiority of a simulator or an updated minimum-GPU specification. Consult documentation for the exact chosen version and record rendering, controller, asset, and hardware provenance. A learned vision extractor may require a GPU. The marker/PnP and physics references do not. No cloud budget or current rental price is specified here. Retain a CPU replay path to diagnose failures independently of a costly vision stack.
Sources: https://developer.nvidia.com/isaac/lab
Current reference: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/requirements.txt

## 9. Evidence-based expansion of the reference

| Stage | Experimental basis | Required evidence | Current status |
| --- | --- | --- | --- |
| P0 · Clean | MuJoCo on CPU | Grasp, transport, release, and logs | Measured: cube / 1 demo |
| P1 · Noisy | State noise + calibrated video | Noisy inputs affect plan and success | Integration proposed |
| P2 · Learning | ACT / Diffusion / Isaac as needed | Defined endpoint and usable data | Optional future stage |

P0 clean replay has been measured for one task and source demonstration. P1 combines noisy-input sensitivity and calibrated video. Phase-local correction requires a representation that actually consumes the relevant intervals. P2 learning or scale is optional and should serve a defined endpoint. The stages are technical recommendations, not official student gates or approval decisions. Do not present an arbitrarily selected threshold as an institutional requirement.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/experimental-protocol.md

## 10. Literature and research gap

The literature findings arise from a bounded primary-source audit through 6 October 2026, not an exhaustive systematic review. Citation indexes may lag recent publications. The proposed combined question remains conditional on reviewed-source coverage. Combining existing modules, selecting Panda/UR5e, or using object-relative transforms does not alone establish a research contribution.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/reading-guide.md

## 11. Replay and imitation learning

Retargeting and replay do not train a policy or establish imitation-learning performance. A training export must define observations, action semantics, requested versus executed actions, and sampling rate. A training export would be separate from the physics JSON schema. ACT and Diffusion Policy become relevant when the chosen endpoint includes policy learning. MimicGen is a close precedent for object-relative robot-demonstration transfer. Its noise at subtask initialization differs from drift or missing intervals during motion.
Sources: https://arxiv.org/abs/2304.13705
https://diffusion-policy.cs.columbia.edu/
https://arxiv.org/abs/2310.17596

## 12. Evidence cards for primary papers

An evidence card should record the exact paper version, section/table/page, code revision, license status, and affected design decision. Separate author statements from interpretations. Rounded success percentages do not identify the sample size without a protocol or raw counts. AUROC/AUPRC measure discrimination; risk–coverage evaluates selective ranking. Probability calibration requires additional analysis. Consult the corrected synthesis and audits before citing values from earlier reports.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/reading-guide.md

## 13. Object-relative transfer and morphology

A common transform is T_target_object × inverse(T_source_object) × T_source_TCP. Panda/UR5e transfer is an experimental setting with precedents. The current physics retargeter segments motion using gripper commands and TCP height, anchors the earlier phase to the first observed object pose before grasp, and anchors the later phase to the goal. Reachability, temporal constraints, collision, and task success require separate reporting. The reference is not a general collision planner, and the cube task does not evaluate orientation-sensitive placement.
Sources: https://arxiv.org/abs/2310.17596
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/retarget.py

## 14. Automatic recovery: PSI and RIGVid

PSI §3.3 describes rejection when all candidate grasps fail, whereas §4.3 describes any grasp failing. The reviewed sources do not resolve this discrepancy in code; a reproduction must disclose its chosen interpretation. Appendix C’s 2 cm/0.2 rad threshold concerns ICP, not a calibrated FoundationPose tracking score. RIGVid uses a generated video and an initial RGB-D frame for depth alignment, with runtime backtracking at 3 cm or 20° deviation. These task-specific values are not universal thresholds. An internally implemented filter should retain its own baseline name.
Sources: https://arxiv.org/pdf/2602.13197v2
https://arxiv.org/abs/2507.00990

## 15. Reliability and budgeted human intervention

ThriftyDAgger provides a precedent for supervision budgets and autonomous versus assisted endpoints. VR-DAgger proposes three timesteps from MC-dropout action variance; the operator selects one and controls the remainder of the episode. Its correction-sample budget differs from a seconds budget. Meta Pose evaluates image-level rotation failures. SPARC evaluates annotation reliability using bounding-box IoU and downstream VLA reasoning. These targets differ from object-pose segment correction. Failure to find the full combined protocol in inspected sources does not guarantee field-wide novelty.
Sources: https://proceedings.mlr.press/v164/hoque22a.html
https://arxiv.org/abs/2605.27114
https://arxiv.org/abs/2608.03516
https://arxiv.org/abs/2606.13497

## 16. Reading sequence and design decisions

| Design question | Priority reading | Evidence to inspect |
| --- | --- | --- |
| Representation | MimicGen | Object anchors, Panda/UR5e, pose-noise Appendix U |
| Automatic recovery | PSI · RIGVid | Rejection/grasp rules, depth anchors, backtracking |
| Human budget | ThriftyDAgger · VR-DAgger | Actions/samples/seconds, review cost, controls |
| Reliability | Meta Pose · SPARC · ProxyPose | Failure labels, scale, and task sensitivity |
| Learning (optional) | ACT · Diffusion Policy | Observations/actions, loaders, policy endpoint |

Prioritize experiments, appendices, and executable code over abstracts. For robot logs, begin with MimicGen. For video-derived object trajectories, examine PSI/RIGVid and DemoBridge. For correction allocation, read ThriftyDAgger, VR-DAgger, and FlowCorrect. Examine the actual reliability targets of Meta Pose and SPARC. UMI and Human2Sim2Robot are additional readings when their inputs and embodiments match the project. Human2Sim2Robot is arXiv 2504.12609 and uses an Allegro hand; the UMI paper names UR5. Learning architectures remain optional after a suitable data contract exists.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/reading-guide.md

## 17. Experiments and reference results

The clean-state benchmark establishes feasibility of a particular replay/controller pipeline on a cube task. It does not test reliability-guided correction. Oracle pilots can validate data plumbing and input sensitivity. A subsequent human study is required to measure correction-time allocation or operator benefit. Distinguish those proposed experiments from existing results throughout the presentation.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/README.md

## 18. Candidate research question

The proposed primary estimand is the task-success difference between ranked and random correction under a common cost definition. Automatic recovery uses no human editing but incurs runtime costs; report its cost and coverage separately. A ground-truth-assisted oracle is an empirical comparator, not a guaranteed success upper bound. Pose-error ranking need not optimize task success. Prespecify whether B covers active editing or total review plus editing. Select budgets and sample size from pilot variability and required precision. Fit scores/thresholds on development data and evaluate on independent episodes, corruptions, and layouts.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/reading-guide.md

## 19. Pose inputs and data contracts

physics_replay.episode.v1 requires increasing timestamps, TCP/object position_m and wxyz quaternions, gripper commands, scene conventions, and robot joints. Missing object poses are complete null/NaN rows; the robot’s own TCP log cannot omit frames. The marker package stores observations, evaluator GT, and correction receipts separately. Their adapter is not implemented. Training export requires observation/action definitions, requested versus executed actions, synchronization, authority labels, and correction provenance. Avoid exposing evaluator GT as an observation. An estimator with unresolved scale needs an independent metric reference.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/episode.py
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/marker-prototype.md
https://arxiv.org/html/2607.06555v1

## 20. Clean-state physics: held-out results

| Method | Success / N | Wilson 95% CI | Failures |
| --- | --- | --- | --- |
| World-frame | 0 / 100 | 0.00–3.70% | 94 grasp, 2 drop, 4 place |
| Object-relative | 89 / 100 | 81.37–93.75% | 11 IK failures |
| Object-relative trimmed | 100 / 100 | 96.30–100.00% | None in this sample |
| Target scripted reference | 100 / 100 | 96.30–100.00% | None in this sample |

The committed held-out log contains 100 paired target layouts at seed 7, four methods, one source demonstration, and one cube task. Development results on 50 layouts were 0/41/50/50; the trimmed variant was refined using that set. Object-relative held-out execution has 11 IK failures. World-frame has 94 grasp failures, two drops, and four placement misses. The scripted target reference has task-specific knowledge. Wilson 95% intervals use z=1.96 under the binary-trial assumption within the sampled domain. The predicate includes lift ≥5 cm, final XY error ≤2 cm, z error ≤5 mm, uprightness, and release with failure flags. It lacks a velocity/dwell criterion and goal-yaw requirement. The review recomputed logs and verified code hashes without rerunning all 600 trials.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/tree/main/physics_replay/results/bench-heldout-seed7-n100
https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm

## 21. Module contracts and integration

| Module | Principal contract | Implementation status |
| --- | --- | --- |
| Physics replay | TCP/object/goal, grasp command, task criteria | Clean demonstration and benchmark |
| Video / marker | Pixels, calibration, estimates, validity mask | Synthetic/PnP pilot; translation only |
| Retarget input | Initial object anchor + TCP/goal trajectory | 65 edited pose frames leave plan unchanged |
| Correction / cost | Edited artifacts, candidates, time receipts | Assumed oracle cost; terminal editor |
| Bridge / export | Consequential pose inputs, physics, labels | End-to-end integration remains absent |

The table defines technical boundaries and status, not student ownership assignments. Physics and marker packages are separate. A bridge must reconcile TCP/object frames, events, and observable inputs. In the example episode, adding 0.05 m to x in 65 observed object frames after the first observed frame leaves the object_relative_trimmed TCP plan exactly unchanged. This was a plan-sensitivity probe, not a physics run or benefit measurement. The static-anchor path does not consume those later poses. Phase-local constraints are a potential representation extension requiring independent validation.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/retarget.py
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/evidence-and-limitations.md

## 22. TeleGuide and correction budgets

TeleGuide here denotes research support for trajectory review and correction. The starter contains standalone physics and marker references. A web interface and a validated pose/physics user study are outside its current scope. An interface or pause/takeover demonstration cannot establish usability or labour savings. Simulation alone cannot establish safety of physical intervention.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/marker-prototype.md

## 23. Review, correction, and reusable data

The terminal editor saves a new NPZ and a monotonic-clock receipt. The candidate segment is selected through the CLI before the clock starts. The response interval includes reading, waiting, and input, so session elapsed time does not isolate active editing. Overruns are rejected. A constant offset changes valid frames only and does not recover missing poses. A user study requires matched primitives, costs, observable information, and counterbalanced condition order. A future authority-label adapter requires its own contract before research data can train a policy.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/marker-prototype.md
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/evidence-and-limitations.md

## 24. Manipulation demonstration in MuJoCo

The images are frames from existing simulator recordings, not physical-robot photographs. Both runs execute manipulation through contact physics. Images illustrate behaviour; quantitative conclusions depend on trial logs. The README records a user-run Colab CPU evaluation on 6 October with 20 passing tests and success/failure agreement for 200 development and 400 held-out trials. This review did not independently rerun Colab. Integrated video correction and human benefit remain unevaluated.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/README.md
Image provenance: frames at t=6s from the source and target MuJoCo recordings. The source revision and associated reference results are identified in provenance.json. The videos can be recreated with the physics demo.

## 25. A protocol with matched correction budgets

Prespecify B, candidate intervals, lengths, edit primitives, replay/feedback access, start/stop events, and overrun handling. If B counts active editing, also report review, waiting, and total supervision. Active-human-time reduction can be defined as 1 - T_active_assisted/T_active_fullteleoperation, requiring a measured baseline with matched task quality. Define system-latency endpoints explicitly, for example edit submission to plan preview or controller acknowledgement, and report p50/p95. Evaluate pose-quality signals on held-out errors before studying downstream failures. Retain rejects, timeouts, and failures. Account for episode/operator dependence rather than treating frames as independent samples. Future tuning requires a new held-out set because seed 7 outcomes are already known.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/experimental-protocol.md
https://proceedings.mlr.press/v164/hoque22a.html

## 26. Resources, risks, and contingency measures

| Risk | Contingency / diagnostic | Retained evidence | Status |
| --- | --- | --- | --- |
| Poor poses / occlusion | Validity mask, filtering, manual edit | Original/estimated/corrected data + reasons | Proposed physics integration |
| Clean replay failure | Frame/TCP, grasp, controller, IK | Failure logs and reduced task scope | Reference available for diagnosis |
| Idle GPU / cost | Heartbeat, checkpoint, then stop | Runtime/cost ledger and receipt | Operational measures proposed |
| Data / checkpoints | Hashed local cache; selective sync | Manifest, version, retention policy | Storage procedure proposed |

The reference workflows run on a CPU. Idle-timeout control is proposed, not implemented. Use a workload heartbeat and active-job check, save checkpoints before termination, and log stop events. Low GPU utilization alone may reflect CPU preprocessing. Local dataset/checkpoint caches require manifests, hashes, versions, selective synchronization, and explicit retention. Diagnose clean replay failures through frames/TCP/grasp/controller/IK residuals and reachability before adding perception. Retain missing masks and correction provenance under occlusion. Kalman or particle filtering requires a defensible measurement model. No cloud budget or current GPU price is asserted.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/README.md
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/README.md

## 27. An experimental sequence

The sequence reflects dependencies, not mandatory deadlines or assignments. Sensitivity experiments must perturb planner inputs that are actually consumed. A static anchor is insufficient for allocation over intervals that cannot influence the output. A phase-dependent representation needs its own tests. Following oracle plumbing, a human pilot estimates variability and cost. The current video importer assumes constant frame rate through frame_index/fps and does not handle variable-rate timestamps. Synthetic-fixture calibration is not calibration of a physical camera. Assembly, deformation, or bimanual manipulation are optional later stress tests, not additions required for the first baseline.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/experimental-protocol.md

## 28. Evidence checklist for the reference pack

Passing implementation tests is distinct from empirical method superiority. Physics has 20 tests, committed trial logs, and model receipts. The marker package has seven tests and a synthetic pilot, with no independent inspection verdict. Model caches, generated runs, and canonical literature PDFs are excluded from Git and require recreation or separate retrieval. Do not generalize from 100 layouts of one demonstration. No human study has been conducted. A null result should report failures, coverage, time costs, and sensitivity checks so rival explanations can be assessed.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/evidence-and-limitations.md
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/evidence-and-limitations.md

## 29. Repository execution paths

Run physics commands from the repository root in an activated environment. Fetch models once, then execute the demo, example replay, benchmark, and pytest. The CPU Colab notebook clones this public starter without requiring a code upload or GitHub authentication. The marker package uses a separate Python 3.12 environment with requirements-research.txt. Run unittest discovery in tests/research_pipeline and demo --output data/research-prototype/runs/my-demo, where the output directory must not exist. Ground truth, observations, and receipts are separate. The default 20-second budget and 10-second oracle edit cost are assumptions, not observed human times. Do not combine the differently pinned environments without resolving dependencies.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/physics_replay/README.md
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/marker-prototype.md

## 30. Resources: repository and documentation

The public starter offers a selected snapshot with a fresh Git history. The imported runtime code and benchmark manifests retain their provenance and hashes in provenance.json. The repository contains the reader deck, its PDF, and this companion guide. Model caches, generated runs, and scientific-paper PDFs are excluded from Git. Match runtime hashes when comparing reruns with the reference outcomes.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/provenance.json
https://github.com/loversky02/trajectory-retargeting-starter/blob/main/README.md

## 31. Resources: primary papers

Primary references include MimicGen 2310.17596, PSI 2602.13197v2, RIGVid 2507.00990v3, ThriftyDAgger 2109.08273, VR-DAgger 2605.27114v2, Meta Pose 2608.03516v1, SPARC 2606.13497v1, and ProxyPose 2607.06555v1. ProxyPose aligns benchmark depth to GT in the first frame; robot replay needs an independent scale reference. Rounded percentages do not identify Mirage’s N. Compatibility with k/186 does not establish one execution per CosmoH2G case. Cases sharing an object may be dependent. Consult the synthesis for ACT, Diffusion Policy, UMI, and Human2Sim2Robot.
Sources: https://github.com/loversky02/trajectory-retargeting-starter/blob/main/docs/reading-guide.md
