> **Moved from the repository root on 2026-09-26.** This is the detailed v5 bearing-pipeline
> write-up that used to be the root `README.md`; line references of the form `README.md:NNN`
> in `docs/analysis_report.md` and `docs/accuracy_ceiling.md` point here. The root README is
> now a short project overview.

# DriveSentinel v5 — order-spectrum CNN

Bearing fault classification from motor current and vibration, targeting an
INT8 accelerator on a PYNQ-Z2 (Zynq XC7Z020).

Everything here stops at the **golden reference**. No RTL.

---

## What changed from v4, and why

v4 reported ~0.66 three-class accuracy using 227 hand-crafted features. Two
problems sat underneath that number:

1. **58 of the 227 features came from sensors the product will never have** — a
   Lorenz K11 load cell and a Magtrol TM305 torque/speed shaft. A PYNQ-Z2 wired
   to current transducers cannot acquire them.
2. **k=5 gave standard errors of 0.03–0.09**, larger than every difference the
   report argued about. Paired t-tests on its own checkpoints put RF vs MLP at
   p=0.979 and envelope vs raw at p=0.502. Nothing was distinguishable.

v5 replaces the scalar-feature bottleneck with an **order-normalised spectrum**
and the 5-fold split with **leave-one-bearing-out over all 29 bearings**.

| | v4 | v5 |
|---|---|---|
| Input | 227 scalars, 6 sensors | 5 × 512 order spectrum, 2 sensors |
| Sensors | current, vibration, force, torque, speed, temp | **current + vibration only** |
| Protocol | 5-fold, unweighted fold mean | 29-fold LOBO, pooled |
| Model | MLP 227-64-32-3 | 1-D CNN, 27,555 params, 1.69 M MAC |
| Speed source | Magtrol tachometer | estimated from the current spectrum |

---

## The core idea

Bearing fault lines sit at fixed multiples of shaft rotation frequency, so in
hertz they move with speed — BPFO is 76.4 Hz at 1500 rpm and 45.8 Hz at 900 rpm.
A model fed a hertz axis must first identify the operating condition, and the
cheapest cue for that is the supply fundamental. That is the confound the v3
methodology flags as untested in section III-G.

Resampling to **shaft orders** removes it: BPFO sits at 3.05 orders at every
speed, so both conditions produce the same picture and there is nothing to key
on but the fault.

Shaft speed comes from the current spectrum, never a tachometer. The drive is a
Hanning SD4CDu8S PMSM with 4 pole pairs, so `f_shaft = f_elec / 4` exactly — no
slip. Measured against the rig tachometer across all 16,127 windows: **median
error 0.022 %, p99 0.208 %**.

> On an **induction** motor this identity does not hold. Porting the pipeline
> requires a slip estimator first. See the caveat in `config.py`.

---

## Layout

```
drivesentinel/            the package -- nothing else imports data/ or torch
  config.py               paths, 6203 geometry, the DSP contract, labels, defects
  dataset.py              .mat indexing; the only module that knows the
                          dataset's defects, and it applies them by construction
  dsp.py                  decimate -> sideband fold / envelope / raw -> order axis
  features.py             parallel cache builder + contract fingerprinting
  folds.py                leave-one-bearing-out; the inner-split decision
  model.py                OrderSpectrumCNN + BatchNorm folding
  train.py                LOBO sweep, augmentation, pooled metrics
  quantize.py             per-channel INT8 PTQ + reference integer arithmetic
  export.py               .mem writers + golden-reference generator

scripts/                  THE PIPELINE, in order
  01_build_cache.py       data/ -> artifacts/order_spectra_v2.npz    (~3.5 min)
  02_train_lobo.py        29-fold sweep + deployment model           (~2.5 min)
  03_export_int8.py       quantise, write .mem, generate golden      (~1 min)
  04_verify_golden.py     golden vs float over the whole cache       (~2 min)
  05_render_results.py    lobo_summary.json -> docs/results.md

  experiments/            investigation tools, not part of the pipeline
    repeat.py             SAME config N times -- run this FIRST after any
                          change to the training path, before comparing anything
    sweep.py              compare training recipes over all 29 folds
    task_variants.py      binary framing + the leaky shuffled reference
    accuracy_report.py    collect every run -> docs/accuracy.md

tests/
  test_dsp.py             planted-fault detection, speed invariance
  test_model_export.py    folding, quantiser, .mem round-trip, golden agreement

docs/
  results.md              current numbers, generated from the run
  accuracy.md             every experiment, generated
  accuracy_ceiling.md     what limits it, and which conclusions were wrong
  data_notes.md           dataset defects, measured not assumed
  methodology_v3.md       the paper's Section III draft
  prior_results/          v4 + v5 evidence cited by the docs above

artifacts/                all generated; safe to delete and rebuild
  order_spectra_v2.npz    the cache (144 MB)
  folds_lobo.json         fold definition, persisted so runs stay comparable
  runs/                   experiment outputs + deployment_model.pt
  int8_export/            >>> THE DELIVERABLE <<<

data/                     21 GB Paderborn dataset, untouched
```

---

## How to run

Install (use the CUDA wheel matching your driver; the CPU build works but is
slower):

```bash
pip install -r requirements.txt
```

### The dashboard

```bash
streamlit run dashboard/app.py
```

Replay only — every number is read from a results JSON, nothing is computed live.
Scenarios are pre-built by `python scripts/demo/build_scenarios.py`.

Each drive stage carries a badge saying whether it may raise `Fault`:
**FAULT-CAPABLE**, **INDICATIVE** or **NOT MEASURED**, with its honest metric and
protocol underneath. Today exactly one stage of five — S5 bearing — clears the
pre-registered floor. The other four are shown anyway; an empty stage is more
honest than a diagram implying four working branches.

Bearing scenarios are replayed through the fold model that **held that bearing
out**, asserted in code (`panels.assert_out_of_sample`) rather than trusted, and
the app refuses to display a scenario that cannot prove it.

See [`docs/results_multistage.md`](docs/results_multistage.md) for the branch
numbers and [`docs/claims_audit.md`](docs/claims_audit.md) for every claim's
protocol and source.

### The technician web app

```bash
.venv/Scripts/python.exe -m webapp          # http://127.0.0.1:8000
```

This is the user-facing prototype. It has an alerts feed, lift health, a "Why?" page per
alert, replay, sensors, hardware and about pages, and it works on laptop and phone. It
replays the same recordings through the same fusion code. Each alert travels as a 21-byte
HTTP packet with store-and-forward. The dashboard above stays the engineering tool. See
[`webapp/README.md`](webapp/README.md) for how to use each page, and
[`docs/screenshots/`](docs/screenshots/) for every page at 1280 px and 390 px.

### The pipeline, in order

Build the feature cache from `data/` — ~3.5 min on 12 cores, runs once:

```bash
python scripts/01_build_cache.py
```

Train leave-one-bearing-out over all 29 bearings, then the deployment model on
all of them — ~2.5 min on an RTX 3050:

```bash
python scripts/02_train_lobo.py
```

Quantise to INT8 and write the `.mem` files plus the golden reference — ~1 min:

```bash
python scripts/03_export_int8.py
```

Check the golden reference against the float model over the whole cache — ~2 min:

```bash
python scripts/04_verify_golden.py
```

Regenerate `docs/results.md` from the run:

```bash
python scripts/05_render_results.py
```

Or all five at once:

```bash
python scripts/01_build_cache.py && python scripts/02_train_lobo.py && python scripts/03_export_int8.py && python scripts/04_verify_golden.py && python scripts/05_render_results.py
```

### The 5-fold benchmark

Produces the 0.9865 figure with per-class precision/recall/F1 and the confusion
matrix — ~90 s:

```bash
python scripts/experiments/shuffled_benchmark.py --folds 5
```

Results land in `artifacts/runs/shuffled_benchmark.json`.

### Investigation tools

Run the same config N times and report the spread. **Do this before comparing
anything** — it is what caught four wrong conclusions in this project:

```bash
python scripts/experiments/repeat.py --repeats 3
```

Compare training recipes across all 29 folds:

```bash
python scripts/experiments/sweep.py --recipes base weak_aug no_ensemble
```

Binary healthy-vs-damaged, plus the protocol-gap reference:

```bash
python scripts/experiments/task_variants.py
```

Collect every run into `docs/accuracy.md`:

```bash
python scripts/experiments/accuracy_report.py
```

### Tests

```bash
python -m pytest
```

29 tests: planted-fault detection and speed invariance in the DSP front end,
then BatchNorm folding, the quantiser, `.mem` round-trip, and agreement between
the generated golden reference and the exporter.

### Switching feature set

The cache path, channel count and model input width all follow from one
environment variable. It must be set as an environment variable rather than in
code, because joblib worker processes re-import `config` fresh:

```bash
DRIVESENTINEL_FEATURE_SET=v1 python scripts/01_build_cache.py
DRIVESENTINEL_FEATURE_SET=v1 python scripts/02_train_lobo.py
```

---

## Signal chain

```
64 kHz current ─┬─▶ decimate ÷16 ─▶ rFFT ─▶ notch supply harmonics ─▶ sideband fold ──┐  ch0
                └─▶ decimate ÷16 ─▶ rFFT ─▶ notch ─────────────────▶ order axis ─────┤  ch1
                                                                                     ├─▶ (5,512)
64 kHz vibration ┬▶ decimate ÷2 ─▶ bandpass 0.5–2.5 kHz ─▶ Hilbert ─▶ order axis ────┤  ch2
                 ├▶ decimate ÷2 ─▶ bandpass 2.5–10 kHz ──▶ Hilbert ─▶ order axis ────┤  ch3
                 └▶ decimate ÷2 ─────────────▶ rFFT ─▶ log-frequency spectrum ───────┘  ch4
```

The two branches use **different** transforms, and that is deliberate.

Hilbert demodulation assumes a carrier far above the modulation. That holds for
vibration (1–10 kHz resonance, <200 Hz modulation) and fails for current, where
the fundamental is 60–100 Hz while BPFI reaches 124 Hz — the sidebands are
comparable to the carrier and fold through DC. `tests/test_dsp.py` caught this
directly: with envelope demodulation the planted BPFO peak landed on the
fundamental instead of on BPFO.

Current instead uses a **sideband fold**: `S(o) = |X(f₀ + o·f_r)| + |X(|f₀ − o·f_r|)|`,
divided by the fundamental. Because `f₀ = 4·f_r` identically, supply harmonics
land on a fixed order grid (the fundamental reappears at order 8), so they are
notched out before folding.

Channel 4 is deliberately on an **absolute-frequency** axis, not an order axis:
structural resonances of the housing do not move with shaft speed, so putting
them on an order axis would smear a fixed resonance across different bins at 900
and 1500 rpm. Everything else is order-normalised; this one is not, on purpose.

---

## Results

Two protocols are reported. Both are real; they answer different questions, and
each number is labelled with the protocol that produced it.

### Stratified 5-fold over windows *(leaky reference)*

The standard protocol in the Paderborn literature, and what published figures on
this dataset use. **It is leaky**: windows from the same recording, and all 560
windows of one bearing, land on both sides of the split. It measures recall of
bearings the model has already seen, not generalisation to a new one.

| Metric | Value |
|---|---|
| **Accuracy** | **0.9865 ± 0.0024** |
| **Macro-F1** | **0.9847** |
| Majority baseline | 0.4153 |
| Test windows | 16,127 |

Measured on the **pre-rebuild cache** (2,306 recordings / 16,127 windows) with
feature set v2. Not re-run after the 2026-09-18 cache rebuild.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.9605 | 0.9917 | 0.9758 | 3,360 |
| inner_race | 0.9893 | 0.9783 | 0.9838 | 6,070 |
| outer_race | 0.9976 | 0.9915 | 0.9945 | 6,697 |

Comparable to the 97.9–99.9 % range reported in the literature for this dataset
(see the comparison table in Osornio-Rios et al., *Sensors* 24(8) 2653, 2024).

### Leave-one-bearing-out

Every bearing held out in turn, so the model is scored only on bearings it has
never seen. **This is the number that predicts field behaviour**, and the one to
quote.

**Current — single run, rebuilt cache (2,318 recordings / 16,211 windows):**

| Metric | Value |
|---|---|
| **Window accuracy** | **0.7944** |
| **Macro-F1** | **0.7852** |
| **Per-recording accuracy** | **0.8356** |
| Per-bearing mean | 0.7945 ± 0.2836 (SEM 0.0527, n=29) |
| Majority baseline | 0.4131 |

**Prior — 3-repeat mean, pre-rebuild cache (2,306 recordings / 16,127 windows):**

| Metric | Value (3 repeats) |
|---|---|
| Window accuracy | 0.8018 ± 0.0103 |
| Macro-F1 | 0.7944 ± 0.0114 |
| Per-recording accuracy | 0.8332 ± 0.0056 |

Both are kept. The cache was rebuilt after the repo moved machines; this copy of
`data/` has KI14's full 80 runs where the previous machine had 68, adding 12
recordings and 84 windows. **The difference (−0.0035 window accuracy) is inside
the ±0.0103 run-to-run noise floor** that the 3-repeat study measured on the same
recipe, and the DSP front end is confirmed identical — the speed-estimate error
reproduced to four significant figures across the rebuild. See
[`docs/claims_audit.md`](docs/claims_audit.md) §1.3.

The current figure is a single run and carries no error bar of its own; use
±0.0103 as the noise floor when comparing against it.

### Why they differ

**19.2 points, from protocol choice alone.** A random window split puts the
seven near-identical windows of one recording — and the 560 windows of one
bearing — on both sides of the split, so the model is partly scored on bearings
it has already memorised.

Full analysis in [`docs/accuracy_ceiling.md`](docs/accuracy_ceiling.md),
including which of this project's own earlier conclusions turned out to be
measurement noise.

### Measure the spread before comparing anything

Run-to-run standard deviation is **±0.01**. It was **±0.11** until the inner
validation split was removed — the base seed was choosing which bearings got
held out, and with six healthy specimens that choice dominated the result.

Four conclusions here were reached under the noisy regime and were wrong,
including "the 5-channel input is 10 points worse" (it is 2 points better).
`scripts/experiments/repeat.py` measures the spread and should be the first
thing run after any change to the training path.

### INT8

| | |
|---|---|
| Weights | 27,024 int8 + 179 int32 biases |
| Agreement with float, all 16,127 windows | **99.88 %** (bar: ≥ 98 %) |
| Widest accumulator | 192 terms, peak 3,151,891 — 23 bits, fits int32 |

The 19 disagreements fall on bearings the float model is already uncertain
about — rounding at a decision boundary, not a systematic error.

---

## Deployment notes

- **Normalisation is per-channel** (4 constants), not per-bin. Per-bin would
  need 2×512 floats in front of conv1. `export.py` asserts on the shape.
- **BatchNorm is folded** into the convolutions before quantisation, so it
  disappears from the deployed graph.
- **Weights are quantised per output channel**, activations per tensor at the
  99.9th percentile. Per-tensor conv weights waste most of the INT8 range once
  BatchNorm is folded in.
- **Accumulators are int32.** The widest layer sums 192 INT8 products, worst
  case 3,096,768 — 23 bits. No saturation logic needed; asserted, not assumed.
- **The golden reference reads the same `.mem` files the RTL will.** It does not
  inline weights, so a mismatch is a logic bug and cannot be a transcription
  error.

---

## Known data defects

Applied automatically by `dataset.build_index()`:

| File | Issue |
|---|---|
| `N09_M07_F10_KA04_17.mat` | Byte-identical copy of run 18. The KA04 measuring log documents the substitution; verified across all seven channels. |
| `N15_M01_F10_KA08_2.mat` | Structurally corrupt — correct size and header, unparseable variable stream. The v4 cache also holds only 79 KA08 recordings, so the old pipeline hit this and skipped it silently. |
| KI14 | 68 recordings, not 80. Twelve are absent from this copy of the dataset but present in the copy the v4 cache was built from (2,559 vs 2,548 files), so **v4 numbers are not reproducible here**. |
| KB23 / KB24 / KB27 | Excluded. Three bearings cannot support a class under leave-one-bearing-out; v4 measured RF's combined-class F1 at exactly 0.000. |

`2,546` usable recordings → `2,306` after excluding KB → **16,127 windows**.
