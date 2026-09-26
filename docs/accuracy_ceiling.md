# How far accuracy goes, and why

Written after a long attempt to reach 85–90 %. Several conclusions here reverse
earlier ones in this project; where that happens the reason is stated, because
the *reason* is more useful than the number.

---

## Final position

**Current — single run, rebuilt cache (2,318 recordings / 16,211 windows).**
Source: `artifacts/runs/lobo_summary.json`, regenerate with `scripts/02_train_lobo.py`.

| Metric | Value |
|---|---|
| Window accuracy | **0.7944** |
| Macro-F1 | **0.7852** |
| **Per-recording accuracy** | **0.8356** |
| Per-bearing mean | **0.7945 ± 0.2836** (SEM 0.0527, n=29) |
| Majority baseline | 0.4131 |

**Prior — 3-repeat mean, pre-rebuild cache (2,306 recordings / 16,127 windows).**
Source: `artifacts/runs/repeat_v2.json`, preserved at
`docs/prior_results/v5_pre_rebuild/`.

| Metric | Value | Repeats |
|---|---|---|
| Window accuracy | 0.8018 ± 0.0103 | 3 seeds |
| Macro-F1 | 0.7944 ± 0.0114 | 3 seeds |
| Per-recording accuracy | 0.8332 ± 0.0056 | 3 seeds |
| Majority baseline | 0.4153 | — |

**Neither number is deleted, and the difference is not a regression.** The cache was
rebuilt on 2026-09-18 after the repo moved machines; this copy of `data/` has KI14's
full 80 runs where the previous machine had 68, so the rebuilt cache holds 12 more
recordings and 84 more windows. Window accuracy moved −0.0035, macro-F1 −0.0024,
per-recording −0.0026 — **all inside the ±0.0103 run-to-run standard deviation** the
3-repeat study measured on the same recipe. The DSP front end is confirmed unchanged:
the speed-estimate error reproduced to four significant figures across the rebuild
(median 0.0221 %, p99 0.2083 % against 0.0221 % / 0.2085 %). See
`docs/claims_audit.md` §1.3.

The current figure is a **single run**, so it carries no error bar of its own. Use
±0.0103 from the 3-repeat study as the noise floor when comparing anything to it;
re-running `repeat.py` on this machine costs ~3¼ hours (no GPU) and has not been done.

Leave-one-bearing-out over all 29 bearings, pooled. Single model — no ensemble,
no test-time augmentation (`n_seeds: 1`, `tta_shifts: [0]` in
`artifacts/runs/lobo_summary.json`).

The exported INT8 weights were trained with the identical single-model recipe, on
all 29 bearings (`train.py:523`, `train.py:548`); 0.7944 is the leave-one-bearing-out
estimate of that recipe's accuracy on an unseen bearing. No LOBO fold model ships —
the 29 fold models each saw 28 bearings, and the quantised artefact is a 30th model
that saw all of them. The recipe is the same; the training set is not.

Corollary: the `float_accuracy: 0.9922` / `int8_accuracy: 0.9915` figures in
`artifacts/int8_export/verification_full.json` are **training-set** accuracies. They
exist to measure INT8-vs-float agreement (`argmax_agreement: 0.9988`), which is what
the export bar is about. Neither is a generalisation estimate and neither belongs on
a slide.

Against a 85 % target: per-recording is **1.4 points short**. Window accuracy is
5.6 points short.

---

## What actually worked

| Change | Effect | Why it worked |
|---|---|---|
| **Remove the inner validation split** | 0.7283 → 0.7823, **std 0.108 → 0.0018** | It consumed 3 bearings *and* injected the variance |
| **5-channel input + strong augmentation** | 0.7823 → **0.8018** | More signal, but only survivable with more regularisation |
| Drop the ensemble | 0.7837 → 0.7822, 3× faster | It was compensating for variance that no longer exists |

Total: **0.7823 → 0.8018 window, 0.8122 → 0.8332 per-recording.**

---

## The measurement problem, which came first

For most of this work the noise floor was larger than every effect being
measured. Three repeats of an identical config scored **0.6041 / 0.7790 /
0.8018** — standard deviation **0.108**.

Weight initialisation was not the cause; those runs already averaged three
models each. The base seed also chose **which bearings entered the inner
validation split**, and with only six healthy specimens, holding out a different
one changed what the model could learn about the class.

Removing the split was affordable because the schedule is a completed OneCycle
with early stopping off — there was nothing left for a validation set to decide.

| | Mean | Std |
|---|---|---|
| 1 validation bearing per class | 0.7283 | 0.1082 |
| **No validation split** | **0.7823** | **0.0018** |

### What that invalidated

Every comparison made before this point sampled from a ±0.11 distribution.
Four things reported as findings were not:

- *"patience 12 beats patience 20"* — single draws
- *"shorter schedules generalise better"* — single draws
- *"0.8063"* — a lucky draw, not a gain
- *"the 5-channel input is 10 points worse"* — **wrong twice over**: a single
  draw, *and* tested with weak augmentation. It is actually 2 points better.

**Measure the repeat-to-repeat spread before comparing anything.**
`scripts/experiments/repeat.py` exists for this and should be the first thing run after
any change to the training path.

---

## What does not work, now measured properly

With the noise floor at ±0.01, these are all within ~1 standard deviation:

| Knob | Result |
|---|---|
| Epochs 20 vs 40 vs 60 | 0.7837 / 0.7854 / — |
| Ensembling (1 vs 3 vs 5 seeds) | 0.7822 / 0.7837 |
| Test-time augmentation | +0.000 |
| `val_bearings_per_class` 1 vs 2 | starves the healthy class |

The model is **not under-tuned**. It sits where the data puts it.

---

## Three structural limits

### 1. The sample size is 29 bearings, not 16,127 windows

The seven windows of a recording are near-identical and the 560 windows of a
bearing are highly correlated. The independent unit is the **bearing**, there
are 29, and only **six are healthy** — leave-one-out leaves five to learn the
healthy class from. Any confidence interval computed on 16,127 rows is fiction.

### 2. The classes are entangled for a third of specimens

Taking each bearing's mean order spectrum and asking which class centroid it
sits closest to, excluding itself: **12 of 29 land on the wrong class**. K002
resembles nothing — its similarities are −0.121 / −0.052 / −0.082.

The CNN beats this crude test, but it is working on specimens that look
mislabelled at the level of their average spectrum.

### 3. Errors are all-or-nothing per bearing

Pooling N four-second observations of the same bearing:

| N | 3-class accuracy |
|---|---|
| 1 | 0.7629 |
| 5 | 0.7790 |
| 40 | 0.7826 |

**Saturates by N=5.** When the model is wrong about a bearing it is wrong about
*every* window of it, so more looks at the same machine cannot help. KI05 scored
0.005 and KA30 0.161 in the final run — not partially wrong, wholly wrong.

Those same bearings swing hard between runs (KA22 was 0.343 in one repeat and
0.637 in the next) while the pooled total stays put — the errors move around
among the difficult specimens rather than disappearing.

This matters for the product claim: **more observation time does not buy
accuracy on the 3-class task.** It does help the binary task (0.820 → 0.859).

### Measured evidence: the same 29 bearings, two independent runs

The 2026-09-18 cache rebuild produced an unplanned but clean natural experiment.
Two complete leave-one-bearing-out sweeps, same recipe, same code, different
machine and a slightly different cache (KI14 gained 12 recordings). Side by side:

| Bearing | Class | Origin | Pre-rebuild | Rebuilt | Δ |
|---|---|---|---|---|---|
| `KI05` | inner_race | artificial | 0.0072 | 0.0089 | +0.0018 |
| `K002` | healthy | none | 0.0625 | 0.0536 | −0.0089 |
| `KA30` | outer_race | real | 0.2411 | 0.2732 | +0.0321 |
| `KA22` | outer_race | real | 0.3537 | 0.4022 | +0.0485 |
| `KA03` | outer_race | artificial | 0.5696 | 0.4946 | **−0.0750** |
| `KI03` | inner_race | artificial | 0.5946 | 0.6000 | +0.0054 |
| `K005` | healthy | none | 0.7661 | 0.7625 | −0.0036 |
| `KI07` | inner_race | artificial | 0.7549 | 0.7657 | +0.0107 |
| `KA15` | outer_race | real | 0.8262 | 0.7670 | **−0.0591** |
| `KI21` | inner_race | real | 0.8943 | 0.8065 | **−0.0878** |
| `KI14` | inner_race | real | 0.8655 | 0.8321 | −0.0334 |
| `KA07` | outer_race | artificial | 0.8714 | 0.8607 | −0.0107 |
| `KI04` | inner_race | real | 0.9000 | 0.8696 | −0.0304 |
| `KI17` | inner_race | real | 0.8730 | 0.8837 | +0.0107 |
| `KA05` | outer_race | artificial | 0.8748 | 0.9284 | **+0.0537** |
| `KA08` | outer_race | artificial | 0.9094 | 0.9402 | +0.0308 |
| `KA04` | outer_race | real | 0.9693 | 0.9421 | −0.0271 |
| `K004` | healthy | none | 0.9214 | 0.9518 | +0.0304 |
| `KA16` | outer_race | real | 0.9589 | 0.9643 | +0.0054 |
| `KI08` | inner_race | artificial | 0.9785 | 0.9803 | +0.0018 |
| `KA06` | outer_race | artificial | 0.9857 | 0.9804 | −0.0054 |
| `KI16` | inner_race | real | 0.9822 | 0.9858 | +0.0036 |
| `KI01` | inner_race | artificial | 0.9982 | 0.9875 | −0.0107 |
| `KA09` | outer_race | artificial | 0.9946 | 0.9982 | +0.0036 |
| `K001` | healthy | none | 0.9982 | 1.0000 | +0.0018 |
| `K003` | healthy | none | 1.0000 | 1.0000 | 0.0000 |
| `K006` | healthy | none | 0.9982 | 1.0000 | +0.0018 |
| `KA01` | outer_race | artificial | 1.0000 | 1.0000 | 0.0000 |
| `KI18` | inner_race | real | 1.0000 | 1.0000 | 0.0000 |
| **Pooled** | | | **0.7979** | **0.7944** | **−0.0035** |

Three things this shows, and they are the argument:

1. **The failures are specimens, not noise.** KI05, K002, KA30 and KA22 are below
   0.41 in *both* runs. Nine bearings are at or above 0.98 in both. Neither group
   moves. Whatever the model has learned about these particular specimens, it
   learns again from scratch on a different machine.
2. **The middle is unstable.** Ten of 29 bearings move by 3 points or more, and
   KI21 moves by 8.8. Mean |Δ| across all 29 is 0.0205 — six times the pooled Δ.
3. **The pooled number barely notices.** 0.7979 → 0.7944, a 0.35-point move, well
   inside the ±0.0103 noise floor, while individual bearings swing by up to 25× that.

That combination is the signature of a model fitted to **29 specimens**, not to
16,211 windows. The per-bearing variance is real generalisation variance; the
pooled stability is an averaging artefact. Quoting the pooled number alone hides
the fact that on four of 29 machines this model is not merely wrong but confidently
wrong on every window.

**For the slide:** show the per-bearing bar chart, not the pooled bar. The honest
claim is "0.79 pooled, but four specimens out of 29 fail completely and the same
four fail every time" — which is a much more defensible thing to say to a drive
engineer than a single number.

---

## The protocol gap

Same model, same data, windows split at random instead of by bearing:

| Protocol | Accuracy | Macro-F1 |
|---|---|---|
| Stratified 5-fold over windows | 0.9865 | 0.9847 |
| Leave-one-bearing-out | 0.8018 | 0.7944 |

**18.5 points from protocol choice alone.**

Both are reported in the root `README.md`, each labelled with the protocol that
produced it. The 5-fold figure is what the Paderborn literature reports and is
the right number for comparing against published work; the leave-one-bearing-out
figure is the one that predicts behaviour on a bearing the model has never seen,
which is the only situation the deployed product is ever in.

Neither is wrong. Quoting either without naming the protocol is.

## What would close the remaining 1.7 points

1. **More bearings.** The binding constraint. Twenty more healthy specimens
   would do more than any architecture change. A data-collection decision, not
   an engineering one.
2. **A severity target instead of a location target.** Healthy / inner / outer
   asks *where*. The product asks *how bad*. The damage-profile PDFs carry an
   extent grade of 1–3, and the confusion matrix shows inner and outer race
   trading errors constantly — errors a severity ladder would not make.
3. **Understand the four failing specimens.** KI05, KA30, KA22 and K002 fail
   hard and consistently. Four bearings out of 29 is 14 % of the test set; if
   half of them are explicable, the target is met. This is more promising than
   any further sweep.

## What would not

More capacity, more epochs, longer windows, more ensembling. All tested under a
controlled noise floor; all neutral. With 29 bearings the model is not
capacity-limited, and every extra degree of freedom is something else for it to
overfit bearing identity with.
