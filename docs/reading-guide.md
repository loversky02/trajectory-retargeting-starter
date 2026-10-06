# Reading guide and research question

Use this guide with the [capstone reader](reader-guide.md). It helps students and supervisors judge whether the starter is relevant; it does not prescribe a project or assign work.

## Prior work to compare

- [MimicGen](https://arxiv.org/abs/2310.17596): object-relative transfer, subtask anchors, and initialization noise. Panda-to-UR5e transfer and object-relative transforms already have precedent.
- [PSI](https://arxiv.org/abs/2602.13197v2) and [RIGVid](https://arxiv.org/abs/2507.00990v3): inspect the precise recovery, grasp validation, and rejection procedures before using either name for a baseline.
- [ThriftyDAgger](https://arxiv.org/abs/2109.08273), [VR-DAgger](https://arxiv.org/abs/2605.27114v2), and [FlowCorrect](https://arxiv.org/abs/2602.22056v3): distinguish sample or action counts from elapsed human seconds; budget definitions and endpoints differ.
- [Meta Pose](https://arxiv.org/abs/2608.03516), [SPARC](https://arxiv.org/abs/2606.13497), and [ProxyPose](https://arxiv.org/abs/2607.06555): compare failure targets, ranking versus calibrated confidence, and metric-scale assumptions.

These references do not constitute an exhaustive systematic review. Read primary papers, appendices, and available code. Rounded percentages do not identify Mirage's trial count. A ratio compatible with *k/186* does not prove that CosmoH2G ran each case once. Do not make a blanket novelty claim from combining existing components.

## Conditional question

Under a fixed correction budget of **B seconds**, does reliability-guided selection of pose intervals improve downstream task success over random selection and automatic recovery, and how does it compare with a ground-truth-assisted oracle?

This question remains conditional on reviewed-source coverage and has not been answered by the included experiments. The oracle is a privileged comparator, not a guaranteed success upper bound. An internal filter needs its own method name unless it implements a published method faithfully.

## A useful first check

Establish that corrupted and corrected inputs affect the plan and task outcome before investing in a stronger video extractor. Current `physics_replay` retargeting consumes the first valid source object pose before grasp. In the committed example, a probe changed 65 later object frames by 5 cm without changing the TCP plan. This is a plan-sensitivity observation, not a task-success evaluation.

A correction-allocation study needs a representation that consumes the candidate intervals. Separate global scale/frame bias from local drift, jumps, and missing data. Human video also requires grasp/contact, TCP orientation, phase, goal, and metric scale; an object pose alone is insufficient. See [experimental protocol](experimental-protocol.md) and [evidence limits](evidence-and-limitations.md).
