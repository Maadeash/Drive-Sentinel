# DriveSentinel v5 — how far accuracy actually goes

Every number here is **leave-one-bearing-out over all 29 bearings, pooled**, unless a row says otherwise. Baseline (always predict the majority class) is **0.4153** for the 3-class task.

## The headline

## Binary healthy vs damaged — reads as a win, mostly isn't

| Metric | Value |
|---|---|
| Window accuracy | 0.7982 |
| **Majority baseline** | **0.7917** |
| Margin over baseline | **+0.0066** |
| Macro-F1 | 0.6880 |
| Per-recording | 0.8196 |

Only 6 of 29 bearings are healthy, so *always predicting damaged*
already scores 0.7917. The model beats that by
0.7 points on accuracy — which is why accuracy is the
wrong headline for this framing. Macro-F1 of 0.6880 against roughly 0.44 for the
always-damaged predictor is the honest evidence that it has learned
something; the accuracy figure mostly reflects class imbalance.

Pooling observations does help here, unlike the 3-class task:

| Observations pooled | Accuracy |
|---|---|
| 1 | 0.8197 |
| 2 | 0.8369 |
| 5 | 0.8467 |
| 10 | 0.8519 |
| 20 | 0.8571 |
| 40 | 0.8593 |

At N=40 it reaches 0.8593, genuinely above the 0.7917 baseline — so the binary signal is
real, just weak per observation.

## Why the numbers move between runs

Per-bearing accuracy has a standard deviation near 0.36 across 29
bearings, so the standard error on any pooled figure is about 0.067.
cuDNN's convolution backward is non-deterministic, so re-running an
identical config moves the result by several points on its own: the
`v50_exact` recipe scored 0.7773 once and 0.6972 on a later identical
run. **Differences smaller than roughly 8 points are not evidence.**
Only variance-reducing changes — seed ensembling, TTA — hold up on
re-running, which is why the shipped recipe uses them.

## Calibration: what a leaky split would report

Windows split at random, ignoring bearing identity, so the seven
near-identical windows of one recording land on both sides. This is
the protocol behind most published Paderborn results.

| Protocol | Window accuracy | Macro-F1 |
|---|---|---|
| Shuffled windows (**leaky — not a generalisation estimate**) | 0.9184 | 0.9131 |
## What limits it

- **29 bearings is the sample size**, not 16,127 windows. The healthy
  class has six, and leave-one-bearing-out leaves five to learn from.
- At bearing level, **12 of 29 bearings sit closest to the wrong class**
  centroid on mean spectrum alone. The classes are genuinely entangled
  for a substantial minority of specimens.
- Errors are all-or-nothing per bearing, which is why aggregation
  saturates and why the per-bearing spread is so wide.

