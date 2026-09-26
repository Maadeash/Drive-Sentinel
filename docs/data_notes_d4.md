# D4 — Thomas et al. motor fault dataset

Stage **S1** (supply input). Source:
[figshare 27216219](https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219),
paper *Sci Data* 12:1468, 2025. **Licence CC BY-NC-ND — never commit derived data.**

Local path: `data_ext/thomas_motor/` (gitignored), 11 files, 687 MB.
Prior audit: `docs/analysis_report.md` §3.4. Written before the adapter, per
`workflow_v2.md` §3.

---

## 1. Shape

| Property | Value |
|---|---|
| Files | `FILE 1.mat` … `FILE 10.mat` (72,000,184 B each) + `LABEL DATASET.mat` |
| Format | MATLAB **v5**, `Platform: posix` — `scipy.io.loadmat` reads it, no h5py |
| Variable | `data`, double, **[1000000, 9]**, column-major, uncompressed |
| Sample rate | **50 000 Hz** — verified, not assumed (§3) |
| Duration | **20.0 s** per file |
| Columns | `vib_x, vib_y, vib_z, I1, I2, I3, V1, V2, V3` |

Column identification is confirmed by physics: column 2 has mean **1.0044**, i.e.
gravity, so it is the vertical accelerometer axis. Columns 3–5 are ±2 A, columns 6–8
are ±396 V.

Each file contains **only** the data array — 184-byte prefix + 9,000,000 × 8 bytes
exactly. There are no metadata variables, no timestamps, no channel names.

## 2. Motor and scenario map — confirmed from the paper

| File | Motor | Scenario |
|---|---|---|
| `FILE 1` | healthy | normal, no load |
| `FILE 2` | healthy | phase removal during operation |
| `FILE 3` | healthy | 0.4 Nm load |
| `FILE 4` | healthy | 0.8 Nm load |
| `FILE 5` | healthy | one phase disconnected from startup |
| `FILE 6`–`10` | **faulty** (SKF 6202-Z, outer ring, drilled holes) | the same five, in the same order |

This matches the independent inference from the audit (matched scenario pairs and
systematically 1.6–2.3× higher vibration in files 6–10).

## 3. Sample rate, verified

Taking a 4 s slice of `V1` from `FILE 3` and testing the hypothesis fs = 50 kHz gives a
dominant peak at **exactly 50.00 Hz**, with 3rd/5th/7th harmonics at 0.2–0.5 % of the
fundamental. Clean mains sinusoid, rate confirmed. 1,000,000 samples = 20.0 s.

---

## 4. Is motor identity confounded with recording session?

**The question D2 forced us to ask of every dataset.** In D2, fault type turned out to
be perfectly predictable from the acquisition session and the DAQ chassis
(`data_notes_d2.md` §7A–7D). D4 gets the same four-signal treatment.

### 4.1 The file timestamps are useless here, and that matters

| File | MATLAB header `Created on` |
|---|---|
| `FILE 6` | Wed Apr 23 **21:23:48** 2025 |
| `FILE 7` | 21:23:51 |
| `FILE 8` | 21:23:56 |
| `FILE 9` | 21:24:03 |
| `FILE 10` | 21:24:08 |
| `FILE 1` | 21:24:12 |
| `FILE 2` | 21:24:18 |
| `FILE 3` | 21:24:21 |
| `FILE 4` | 21:24:24 |
| `FILE 5` | 21:24:28 |

**All ten were written within 40 seconds.** That is a batch export from MATLAB, not
acquisition time. Unlike D2 — whose FlexLogger timestamps spanned January to August
2022 and *were* acquisition times — D4's headers say only when somebody saved the
files, and the files carry no acquisition metadata of any kind.

The earlier audit read the two write blocks (6–10 then 1–5) as corroboration of the
motor split. That inference was correct about the *grouping* and wrong about the
*mechanism*: it reflects export ordering, not two recording sessions. Noted so the
argument is not reused as evidence of acquisition timing.

So the confound check has to come from the signals, exactly as `i0rel_residual` did
for D2.

### 4.2 Four signal-derived fingerprints

| File | Motor | f_mains (Hz) | DC vib_x | DC vib_y | DC vib_z | DC V1 | Noise floor V1 |
|---|---|---|---|---|---|---|---|
| 1 | A healthy | 49.97330 | −0.00253 | 0.00556 | 1.00443 | 9.6220 | 370.6 |
| 2 | A healthy | 49.95546 | 0.00752 | 0.00611 | 0.94519 | 9.5691 | 1626.1 |
| 3 | A healthy | 49.99314 | −0.00188 | 0.01174 | 1.00646 | 9.5901 | 138.0 |
| 4 | A healthy | 50.00635 | −0.00231 | 0.01306 | 1.00660 | 9.5146 | 135.8 |
| 5 | A healthy | 50.01359 | 0.00041 | 0.00029 | 1.00025 | 9.5644 | 164.3 |
| 6 | B faulty | 50.00952 | −0.00421 | 0.00508 | 1.00433 | 9.5497 | 423.2 |
| 7 | B faulty | 50.02460 | −0.00228 | 0.00808 | 1.00710 | 9.6888 | 1243.6 |
| 8 | B faulty | 50.03613 | −0.00118 | 0.00887 | 1.00752 | 9.5924 | 134.3 |
| 9 | B faulty | 50.02129 | −0.00108 | 0.00272 | 1.00103 | 9.6549 | 133.6 |
| 10 | B faulty | 49.99784 | 0.00043 | 0.00026 | 1.00025 | 9.5585 | 152.8 |

`f_mains` is a phase-slope estimate: the phase of the 50 Hz component is measured over
the first and second half of each record, and the drift between them gives the
frequency to about 0.001 Hz — far finer than a bin search over 20 s.

**Separation test, motor A against motor B:**

| Fingerprint | Motor A range | Motor B range | Separated? |
|---|---|---|---|
| mains frequency | 49.95546 – 50.01359 | 49.99784 – 50.03613 | **no** |
| DC `vib_x` | −0.00253 – 0.00752 | −0.00421 – 0.00043 | **no** |
| DC `vib_y` | 0.00029 – 0.01306 | 0.00026 – 0.00887 | **no** |
| DC `vib_z` | 0.94519 – 1.00660 | 1.00025 – 1.00752 | **no** |
| DC `V1` | 9.5146 – 9.6220 | 9.5497 – 9.6888 | **no** |
| noise floor `V1` | 135.8 – 1626.1 | 133.6 – 1243.6 | **no** |

**Not one of the six separates the motors.** Every range overlaps, and in each case the
spread *within* a motor is comparable to or larger than the gap between motors. Compare
D2, where a single scalar separated the acquisition batches perfectly (1.000 LOO
accuracy) and the batch ranges did not even touch.

### 4.3 Positive evidence for a single session

The mains frequency spread across all ten files is **0.081 Hz** (49.955 – 50.036). Grid
frequency holds within roughly ±0.05 Hz of nominal in normal operation and wanders on a
timescale of seconds to minutes. A total spread this tight is what one session looks
like; recordings days apart would scatter further.

Ordering the files and measuring the total variation of the frequency track:

| Ordering | Σ\|Δf\| | Percentile among 20,000 random orderings |
|---|---|---|
| naive `1…10` | 0.145 Hz | ~1 % |
| export order `6…10, 1…5` | 0.165 Hz | 1.6 % |
| random | mean 0.255 Hz | — |

Both sensible orderings trace a smoother frequency walk than ~98 % of random
permutations. A set of recordings made at widely separated times would not do that.

### 4.4 The strongest varying property tracks scenario, not motor

The noise floor spans 12× across the dataset (134 → 1626). Between matched scenario
pairs it barely moves:

| Scenario | Motor A | Motor B | Ratio |
|---|---|---|---|
| start-up | 370.6 | 423.2 | 1.14× |
| phase loss while running | 1626.1 | 1243.6 | 1.31× |
| steady load a | 138.0 | 134.3 | **1.03×** |
| steady load b | 135.8 | 133.6 | **1.02×** |
| single-phasing at start | 164.3 | 152.8 | 1.08× |

The one quantity that varies strongly across this dataset is explained by **what was
being done**, not by **which motor**. That is the signature of a controlled experiment,
and the opposite of what D2 shows.

### 4.5 Verdict

**No evidence of a D2-style acquisition confound in D4, and positive evidence
consistent with a single recording session.**

Stated precisely, because the distinction matters: the `.mat` files carry no
acquisition timestamps, so the recording times are genuinely **unknown**. What can be
said is that six independent signal-derived properties fail to separate the two motors,
and that the mains-frequency evidence is consistent with one session. That is absence of
evidence plus positive corroboration — **not proof**.

**This strengthens the supply branch**, and it does so structurally as well as
empirically: the three supply classes (`normal`, `phase_loss_running`,
`single_phasing_start`) each appear on **both** motors as matched pairs. Even if a
motor↔session confound were later demonstrated, the supply classes would remain
balanced across it, which is precisely the property D2's fault types lacked.

### 4.6 What is still confounded in D4, and always will be

The **bearing-fault** confound is untouched by any of the above and remains total:
there is exactly **one motor per bearing condition**, so "outer-race fault" and "motor
identity" are the same variable. Nothing distinguishes a bearing defect from any other
difference between two physical machines. That is why the bearing result on D4 is
reported only as a documented negative result and never as a capability
(`workflow_v2.md` §5, B-S1 confound demo).

The audit's quantification stands: vibration is 1.6–2.3× higher on motor B across every
matched rotating pair.

---

## 5. Consequences for the B-S1 supply branch

| Item | Value |
|---|---|
| Groups for leave-one-motor-out | **2** — below the fusion floor's minimum of 3 |
| Recordings per (motor × scenario) | **1** |
| Windows at 0.2 s / 0.1 s hop | 199 per file |
| Session confound | **none found** (§4) |
| Bearing confound | **total** — n=1 motor per condition (§4.6) |

Two groups means B-S1 is **INDICATIVE** under the pre-registered floor regardless of
what it scores, as `claims_audit.md` §1.1 anticipated before any branch was built.

Per `workflow_v2.md` §13.8 the deliverable is a **documented threshold rule** plus a
leaky reference, not a learned classifier: with one recording per motor × class, no
learned metric is defensible. The rule is built on zero-current phase detection, where
the dead phase reads 0.012 A against 0.4–1.3 A — three orders of magnitude, unambiguous
in every affected file.

## 6. Open items

| Item | Status |
|---|---|
| Label-vector alignment (1000-sample window, step 500, over the ten files merged in order) and validation against the measured phase-current collapse | **NOT DONE** — next in P2 |
| Acquisition timestamps | **Not recoverable.** The `.mat` files carry none; headers are export times. |
