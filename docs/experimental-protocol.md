# Experimental protocol: sensitivity before correction

This is a research planning aid, not a required assignment, approved capstone scope, or grading threshold. Students and supervisors decide what is feasible and useful.

## Check plan sensitivity

1. Trace the estimated input that retargeting actually consumes. Keep evaluator ground truth separate.
2. Compare clean input against controlled corruption of that input. Distinguish global scale/frame bias, local drift or jumps, and missing intervals.
3. Hold controller and target layouts constant. Record whether the plan changes and whether task success changes; pose error alone is not task success.
4. If an interval does not affect the current plan, revise the hypothesis or representation before claiming a correction to that interval helped.

Current `physics_replay` takes its source object reference from the first valid pose before grasp. In the committed example, perturbing 65 later object frames by 5 cm left the TCP plan unchanged. This is a plan-sensitivity probe, not a physics task trial.

## Compare correction policies

A conditional question is whether reliability-guided selection improves downstream success under a fixed human-time budget **B** compared with random selection and automatic recovery, and how it compares with a ground-truth-assisted oracle.

Match candidate pools, corruption, edit primitive, controller, layouts, and budget accounting. Report review/selection, active editing, waiting, and total supervision separately. Retain failures, timeouts, and rejected edits. Call a baseline by a paper's name only when the method is reproduced faithfully. An internal filter needs its own name; the oracle has privileged information and is not guaranteed to be an upper bound.

Use pilot variation and a chosen precision target to plan sample size. Use a new held-out set after tuning. Account for dependence among trials sharing an episode, object, or operator. No official sample size or success threshold is set here.
