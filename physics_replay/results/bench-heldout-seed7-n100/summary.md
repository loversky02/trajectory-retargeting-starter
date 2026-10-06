# Panda -> UR5e pick-and-place replay benchmark

100 sampled target layouts, seed 7, config sha256 `d82af9ce26b6`.

| Method | Success | Rate | Wilson 95% CI | Median place error (success) | Failure reasons |
|---|---|---|---|---|---|
| world_frame | 0/100 | 0.00 | 0.00-0.04 | - | grasp_failed 94, dropped_in_transit 2, place_miss 4 |
| object_relative | 89/100 | 0.89 | 0.81-0.94 | 1.3 mm | ik_unreachable 11 |
| object_relative_trimmed | 100/100 | 1.00 | 0.96-1.00 | 1.2 mm | none |
| target_scripted | 100/100 | 1.00 | 0.96-1.00 | 0.8 mm | none |

Paired object_relative vs world_frame: 100 layouts, only world_frame succeeds 0, only object_relative succeeds 89, exact McNemar p = 3.23e-27.

Paired object_relative_trimmed vs object_relative: 100 layouts, only object_relative succeeds 0, only object_relative_trimmed succeeds 11, exact McNemar p = 0.000977.
