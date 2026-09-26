# Dataset notes — Paderborn (KAt) bearing data

Everything here was measured directly from the files in `data/`, not taken from
the dataset documentation. Where the documentation and the files disagree, the
files win and the disagreement is recorded.

`drivesentinel/dataset.py` applies all of this by construction, so none of it
depends on a caller remembering to.

---

## Inventory

| | Count |
|---|---|
| `.mat` files on disk | 2,548 |
| Unusable (see below) | 2 |
| Usable recordings | 2,546 |
| After excluding KB (combined class) | 2,306 |
| Windows in the v5 cache | 16,127 |
| Bearings used | 29 |

Class balance in the cache: outer_race 6,697 · inner_race 6,070 · healthy 3,360.
Majority baseline **0.4153**.

---

## Defects

### `N09_M07_F10_KA04_17.mat` — duplicate

Byte-identical to `N09_M07_F10_KA04_18.mat` across all seven channels
(verified by array comparison, not by hash). The KA04 measuring log documents
the substitution in German: run 17 contained faulty data and was replaced with a
copy of run 18 renumbered as 17.

Left in, it places the same seven windows twice inside one bearing. Under
leave-one-bearing-out that is not a train/test leak, but it does double-weight
one recording.

`dataset.verify_duplicate()` re-checks this at test time rather than trusting
the hard-coded exclusion — if a future copy of the dataset has a genuine run 17,
the self-test fails instead of silently discarding real data.

### `N15_M01_F10_KA08_2.mat` — structurally corrupt

Correct file size (8,714,872 B, versus 8,714,904 for its neighbour) and a valid
`MATLAB 5.0 MAT-file, Platform: PCWIN, Created on: Mon Dec 29 11:48:23 2014`
header, but the variable stream does not parse: scipy raises
`TypeError: Expecting matrix here`.

Independently corroborated: the v4 feature cache holds only **79** KA08
recordings, so the previous pipeline hit the same file and skipped it without
reporting it.

### KI14 — 68 recordings, not 80

Missing runs:

| Condition | Absent runs |
|---|---|
| `N09_M07_F10` | 9, 10 |
| `N15_M01_F10` | 4, 7, 17, 19 |
| `N15_M07_F04` | 1, 10 |
| `N15_M07_F10` | 2, 5, 12, 18 |

These twelve **are** present in the copy of the dataset the v4 cache was built
from, which held 2,559 recordings against this copy's 2,548. The v4 numbers are
therefore not reproducible on this machine, and the v5 cache records its own
count so the two cannot be silently confused.

### KB23 / KB24 / KB27 — excluded, not deleted

Three physical bearings carry combined inner + outer damage. Under
leave-one-bearing-out at most two are ever available for training, and v4
measured Random Forest's combined-class F1 at exactly **0.000** — the model
never predicted the class at all.

This is a data-availability ceiling, not a modelling failure, and it is not
fixable by architecture. Reported separately; never pooled into a headline
figure. `build_index(include_excluded=True)` brings them back for that
reporting.

### Variable record lengths

The 64 kHz channels run **256,000 to 289,013** samples against a nominal 256,001;
the 4 kHz channels 16,001 to 16,008; the temperature channel holds 4 or 5 points.
Nothing in the pipeline assumes a fixed length.

The four physically smallest files, all in KA04, are compression artefacts
rather than truncation — they were loaded and carry full 4-second records.

---

## Speed estimation, validated

The order axis depends entirely on shaft speed, and deployment has no
tachometer. `dsp.estimate_shaft_speed()` recovers it from the current spectrum
as `f_elec / 4` — exact for this PMSM, which has no slip.

Checked against the rig's Magtrol TM305 on every window in the cache:

| | Error vs tachometer |
|---|---|
| Median | **0.022 %** |
| p99 | **0.208 %** |

That validates `POLE_PAIRS = 4` on real data and is the evidence the order axis
rests on.

> **This does not transfer to an induction motor.** Rotor speed there lags the
> field by the slip, so `f_elec / p` overestimates shaft speed by 1–5 % under
> load — enough to walk BPFI by several order bins. A slip estimator is required
> before this pipeline moves to an induction machine.

---

## Sensors used, and deliberately not used

| Channel | Rate | Used? |
|---|---|---|
| `phase_current_1`, `phase_current_2` | 64 kHz | **yes** — core |
| `vibration_1` | 64 kHz | **yes** — core |
| `speed` | 4 kHz | validation of the speed estimator only, never a feature |
| `force`, `torque` | 4 kHz | **no** — Lorenz K11 load cell and Magtrol shaft, rig-only |
| `temp_2_bearing_module` | 1 Hz | **no** — the v5 ablation found it worth ~0 |

The mechanical channels were 58 of v4's 227 features and scored 0.53 accuracy on
their own against a 0.42 baseline. That is real signal, but it comes from
instruments a lift will not have, so it is excluded rather than quietly
inflating the headline.
