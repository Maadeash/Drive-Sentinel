# DriveSentinel v5 — results

Leave-one-bearing-out over all 29 bearings, pooled. Every bearing is
held out exactly once, so concatenating the folds gives one
out-of-sample prediction per window across the whole dataset.

> Per-**fold** macro-F1 and per-fold baselines are omitted deliberately:
> a single-bearing test set contains one class, so its macro-F1 caps at
> 1/3 and its majority baseline is always 1.0. Only the pooled set has
> all three classes in it.

## Headline

| Metric | Value |
|---|---|
| Window accuracy | **0.7944** |
| Macro-F1 | **0.7852** |
| Per-recording accuracy (majority vote) | **0.8356** |
| Majority-class baseline | 0.4131 |
| Windows / recordings | 16,211 / 2,318 |
| Per-bearing accuracy | 0.7945 ± 0.2836 (SEM 0.0527, n=29) |
| Sweep time | 65.5 min |

## Confusion matrix

Rows are truth, columns are prediction.

| | healthy | inner_race | outer_race | recall |
|---|---|---|---|---|
| **healthy** | 2,670 | 538 | 152 | 0.795 |
| **inner_race** | 786 | 4,879 | 489 | 0.793 |
| **outer_race** | 501 | 867 | 5,329 | 0.796 |

## By damage origin

Never pooled into one figure — v4 measured artificial→real transfer at
37.9 %. Healthy bearings join both groups as the reference class.

| Population | Bearings | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|---|
| real | 17 | 0.7938 | 0.7891 | 0.3535 |
| artificial | 18 | 0.7950 | 0.7832 | 0.3884 |

## Per-bearing

| Bearing | Class | Origin | Accuracy | Recording acc | Epochs |
|---|---|---|---|---|---|
| `KI05` | inner_race | artificial | 0.0089 | 0.0000 | 19 |
| `K002` | healthy | none | 0.0536 | 0.0000 | 19 |
| `KA30` | outer_race | real | 0.2732 | 0.1250 | 19 |
| `KA22` | outer_race | real | 0.4022 | 0.4625 | 19 |
| `KA03` | outer_race | artificial | 0.4946 | 0.3125 | 19 |
| `KI03` | inner_race | artificial | 0.6000 | 0.8125 | 19 |
| `K005` | healthy | none | 0.7625 | 0.9125 | 19 |
| `KI07` | inner_race | artificial | 0.7657 | 0.8875 | 19 |
| `KA15` | outer_race | real | 0.7670 | 0.9500 | 19 |
| `KI21` | inner_race | real | 0.8065 | 0.9875 | 19 |
| `KI14` | inner_race | real | 0.8321 | 0.9875 | 19 |
| `KA07` | outer_race | artificial | 0.8607 | 0.9500 | 19 |
| `KI04` | inner_race | real | 0.8696 | 0.9000 | 19 |
| `KI17` | inner_race | real | 0.8837 | 1.0000 | 19 |
| `KA05` | outer_race | artificial | 0.9284 | 0.9875 | 19 |
| `KA08` | outer_race | artificial | 0.9402 | 1.0000 | 19 |
| `KA04` | outer_race | real | 0.9421 | 0.9873 | 19 |
| `K004` | healthy | none | 0.9518 | 1.0000 | 19 |
| `KA16` | outer_race | real | 0.9643 | 0.9875 | 19 |
| `KI08` | inner_race | artificial | 0.9803 | 0.9875 | 19 |
| `KA06` | outer_race | artificial | 0.9804 | 1.0000 | 19 |
| `KI16` | inner_race | real | 0.9858 | 1.0000 | 19 |
| `KI01` | inner_race | artificial | 0.9875 | 1.0000 | 19 |
| `KA09` | outer_race | artificial | 0.9982 | 1.0000 | 19 |
| `K001` | healthy | none | 1.0000 | 1.0000 | 19 |
| `K003` | healthy | none | 1.0000 | 1.0000 | 19 |
| `K006` | healthy | none | 1.0000 | 1.0000 | 19 |
| `KA01` | outer_race | artificial | 1.0000 | 1.0000 | 19 |
| `KI18` | inner_race | real | 1.0000 | 1.0000 | 19 |

## Evidence accumulation

Accuracy when N four-second observations of the same bearing are
pooled by averaging their per-window softmax. Still leave-one-
bearing-out — every probability is out-of-sample. A lift makes dozens
of trips a day, so a health verdict does not have to be called from a
single look at the machine.

| Observations pooled | Accuracy |
|---|---|
| 1 | 0.8371 |
| 2 | 0.8517 |
| 5 | 0.8581 |
| 10 | 0.8579 |
| 20 | 0.8600 |
| 40 | 0.8605 |

## INT8 export

- Weights: 27,024 int8, 179 int32 biases
- INT8 vs float argmax agreement: **99.85 %** (bar: ≥ 98 %), n=2,048
- Max |logit| difference: 1.3486

| Layer | Terms | Peak \|acc\| | Bits | int32 |
|---|---|---|---|---|
| conv1 | 45 | 733,886 | 21 | ok |
| conv2 | 112 | 1,839,894 | 22 | ok |
| conv3 | 160 | 2,642,495 | 23 | ok |
| conv4 | 192 | 3,149,695 | 23 | ok |
| fc | 64 | 1,041,656 | 21 | ok |

---

Generated from `artifacts/runs/lobo_summary.json`. DSP contract: window 1.0s hop 0.5s | 5 x 512 bins over 0-16.0 orders (0.03125 orders/bin) | classes ['healthy', 'inner_race', 'outer_race']
