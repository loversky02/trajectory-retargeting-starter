# Panda -> UR5e pick-and-place replay benchmark

50 sampled target layouts, seed 20261006, config sha256 `d82af9ce26b6`.

| Method | Success | Rate | Wilson 95% CI | Median place error (success) | Failure reasons |
|---|---|---|---|---|---|
| world_frame | 0/50 | 0.00 | 0.00-0.07 | - | grasp_failed 50 |
| object_relative | 41/50 | 0.82 | 0.69-0.90 | 1.4 mm | ik_unreachable 9 |
| object_relative_trimmed | 50/50 | 1.00 | 0.93-1.00 | 1.2 mm | none |
| target_scripted | 50/50 | 1.00 | 0.93-1.00 | 0.7 mm | none |

Paired object_relative vs world_frame: 50 layouts, only world_frame succeeds 0, only object_relative succeeds 41, exact McNemar p = 9.09e-13.

Paired object_relative_trimmed vs object_relative: 50 layouts, only object_relative succeeds 0, only object_relative_trimmed succeeds 9, exact McNemar p = 0.00391.
