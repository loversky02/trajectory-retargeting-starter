# Evidence and limitations

Keep implemented behavior, software tests, measured simulation, and proposed work distinct.

## Physics replay

The held-out artifact reports paired methods over 100 sampled target layouts (seed 7), one cube task, and one Panda source demonstration:

| Method | Success / layouts | Wilson 95% interval |
| --- | ---: | ---: |
| World-frame replay | 0/100 | 0.00–3.70% |
| Object-relative | 89/100 | 81.37–93.75% |
| Object-relative trimmed | 100/100 | 96.30–100.00% |
| Target-scripted reference | 100/100 | 96.30–100.00% |

These are layouts, not independent demonstrations or physical robot trials. The trimmed method was selected after the 50-layout development run exposed reachability failures. Seed 7 is no longer untouched; any later tuning requires a new held-out set. The scripted comparator uses task-specific target information and is not a mathematical success upper bound.

The current predicate checks lift, transport, final position, uprightness, and release. It does not impose a settle-time/velocity threshold or goal yaw. The measured results do not establish settled or orientation-sensitive placement, new tasks, or video robustness.

## Marker prototype

The marker tests and demo exercise plumbing on one synthetic calibrated clip. Robot transfer is translation-only. No integrated video-to-physics pipeline, markerless human-video result, physical manipulation trial, human study, or independent marker-code inspection verdict is available. The simulated oracle budget does not measure human benefit.

## Interpreting extensions

A future method comparison should match candidate intervals, corruption, edit primitive, controller, and target layouts. Define whether budget B includes review, selection, active editing, waiting, and total supervision. Keep rejected edits, timeouts, and failures in the denominator. Pilot data can inform sample size and precision; frames should not be treated as independent episodes. The oracle is privileged and not guaranteed to upper-bound task success.
