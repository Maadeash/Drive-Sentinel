# D3 — Bacha PMSM inverter fault dataset

Stages **S2 / S3**. Source:
[github.com/bachaabdelkabir/PMSM-inverter-fault-diagnosis](https://github.com/bachaabdelkabir/PMSM-inverter-fault-diagnosis)
(mirror of [10.5281/zenodo.13974503](https://doi.org/10.5281/zenodo.13974503)),
paper [10.1016/j.dib.2025.111286](https://doi.org/10.1016/j.dib.2025.111286), **CC-BY-4.0**.

Local path: `data_ext/bacha_inverter/` (gitignored). 30 files, 8.2 MB.
Inspection pass: `docs/analysis_report.md` §3.3. Adapter:
`drivesentinel/adapters/bacha_inverter.py`. Tests: `tests/test_adapter_bacha.py`.

---

## 1. What this dataset is

10 Hz Arduino telemetry from a three-phase MOSFET inverter (IRF540N) driving a PMSM
converted from a DENSO car alternator at a constant 10 rad/s. Eight 10-bit ADC channels,
one continuous run of ~5 minutes per condition, nine conditions, 10,892 samples.

**There is no waveform here.** At 10 Hz the Nyquist frequency is 5 Hz, so no harmonic
analysis of any kind is possible. The plan's §0 already corrected this ("There is no THD");
it is restated because the dataset's stage assignment (S2/S3) otherwise invites the
assumption.

| Property | Value |
|---|---|
| Sample rate | **10 Hz** — measured inter-sample interval, median 0.101 s in all nine files |
| ADC | 10-bit, 0–1023, `VREF` 5.0 V |
| Channels | `Ia, Ib, VDC, IDC, T1, T2, T3, VD` |
| Conditions | 9 (`F0`–`F8`) |
| Total samples | **10,892** ✅ matches the published count |
| Runs per condition | **1** — there is no group axis |

Sensors, per the dataset README: ACS712-20A (100 mV/A) for `Ia`, `Ib`, `IDC`; ACS712 with a
100 Ω resistor as a voltage divider for `VDC` and `VD`; 10 kΩ NTC with a 10 kΩ divider for
`T1`–`T3`; SCANCON 100 PPR encoder (not logged in these files).

Note there is **no `Ic`** — only two of the three phase currents are measured.

## 2. File inventory and label map

Verified by row count against the published label counts. All nine match **exactly**.

| F | File | Family | Location | Rows (published = measured) | Duration |
|---|---|---|---|---|---|
| F0 | `NORMAL_OP.txt` | `normal` | — | 4,295 | 439.0 s |
| F1 | `HB2_HIGH_SIDE_OC.txt` | `open_circuit` | HB2 high side | 692 | 70.6 s |
| F2 | `HB3_LOW_SIDE_OC.txt` | `open_circuit` | HB3 low side | 1,122 | 114.6 s |
| F3 | `HB1_LOW_SIDE_SC.txt` | `short_circuit` | HB1 low side | 407 | 41.5 s |
| F4 | `HB2_HIGH_SIDE_SC.txt` | `short_circuit` | HB2 high side | 341 | 34.8 s |
| F5 | `HB3_HIGH_SIDE_SC.txt` | `short_circuit` | HB3 high side | 412 | 42.0 s |
| F6 | `HB1_OVER_TEMP.txt` | `over_temp` | HB1 | 854 | 87.2 s |
| F7 | `HB1&2_OVER_TEMP.txt` | `over_temp` | HB1 + HB2 | 1,735 | 177.3 s |
| F8 | `HB3_OVER_TEMP.txt` | `over_temp` | HB3 | 1,034 | 105.6 s |
| | | | | **10,892** | |

Raw format: `HH:MM:SS.mmm -> 8 space-separated ints`, trailing whitespace on every line,
**no trailing newline** on the last line (which is why `wc -l` undercounts each file by one).

`dataset CSV.csv` is the same data with an `FDD` label column; its label census reproduces
the table above exactly. It is a valid alternative source. The adapter reads the per-condition
`.txt` files so that each condition keeps its own identity and timestamps.

### Files excluded from classification

`SETPOINT30.txt`, `SETPOINT40.txt`, `SETPONT50.txt` (sic), `SETPOINT60.txt`,
`temperature_data.txt`, `thermistor_calib_data.txt`.

These use a different 4-column format: **`[setpoint °C, heater PWM duty 0–255, measured °C,
raw ADC]`**. They are thermistor / heater-PID calibration runs, not inverter conditions.

- `temperature_data.txt` is **byte-identical** to `SETPOINT30.txt` (same MD5).
- `thermistor_calib_data.txt` (3,067 rows) is the four setpoint runs concatenated
  (906 + 640 + 808 + 709 = 3,063).
- `SETPONT50.txt` has `50` misspelled in its name **and** `40.00` in its setpoint column,
  but its measured temperature settles at 49.85 °C — it is the 50 °C run.

They are excluded from classification and **kept** as the source of the temperature
calibration (§4).

---

## 3. The unit conversions are wrong as shipped

### 3.1 `VDC` and `VD` — a mid-rail offset applied to a unipolar divider

`data_conversion_script_v4.py` applies `adc_to_voltage_acs712_100ohm()` to both:

```python
def adc_to_voltage_acs712_100ohm(adc_value):
    voltage = (adc_value / 1023) * 5
    current = (voltage - 2.5) / 0.100     # <-- ACS712 bipolar mid-rail offset
    return current * 100
```

Subtracting 2.5 V is correct for a bipolar current sensor and wrong for a DC-bus divider.
Verified: ADC 507 → −21.994 V, exactly the value in `converted_dataset.csv`.

Two consequences:
1. **`VDC` reads about −22 V**, which is not a DC link.
2. The gain is **4.888 V per ADC count**. `VDC` spans only 24 counts across the whole
   dataset, so the channel is quantised into ~24 five-volt steps.

**The divider ratio is not in any machine-readable file.** It is in
`Sensor_raw_data_conversion_formulas.pdf`, which has **not yet been transcribed** — `pypdf` is
now installed but the PDF has not been read. `adc_to_bus_voltage()` therefore returns the ADC
*pin* voltage and takes an optional `divider_ratio`; the correct form is written down next to
the incorrect one. **UNVERIFIED — transcribe before any use of these channels.**

### 3.2 `converted_dataset.csv` — unreproducible, do not use

Its temperature column cannot be reproduced from `dataset CSV.csv` by any formula in the
shipped script (row 1: raw ADC 515 → −7.236 °C, which no coefficient set here produces). It is
the output of an earlier script that is not in the repository.

### 3.3 `converted_dataset-2.csv` — header/content mismatch

Its `Ia`, `Ib` and `IDC` columns hold **raw ADC** (532.0, 388.0, 506.0) while the converted
values live in `Ia_original`, `Ib_original`, `IDC_original`. Reading it the obvious way
silently yields ADC counts labelled as amps.

**Decision: neither `converted_dataset*.csv` is used.** The adapter starts from raw ADC.
(`workflow_v2.md` §13.5.)

---

## 4. The temperature calibration — a second, larger bug

The shipped script's Steinhart–Hart conversion **is** what produced `converted_dataset-2.csv`
(ADC 515 → 10.577 °C reproduces exactly, pinned in `tests/test_adapter_bacha.py`). So the CSV
is self-consistent with the script.

**But the script disagrees with the dataset's own thermistor calibration runs by 13–20 °C.**

Extracting the settled points from the four `SETPOINT*` runs — the tails where heater PWM has
reached zero and measured temperature has converged on setpoint:

| Run | n settled | Measured °C | ADC (mean) | ADC s.d. |
|---|---|---|---|---|
| `SETPOINT30.txt` | 458 | 30.13 | 462.3 | 1.9 |
| `SETPOINT40.txt` | 217 | 40.19 | 386.8 | 4.3 |
| `SETPONT50.txt` | 205 | 49.85 | 300.1 | 6.4 |
| `SETPOINT60.txt` | 113 | 59.97 | 211.6 | 7.9 |

Against the script's conversion:

| ADC | Script says | Dataset's own calibration says | Error |
|---|---|---|---|
| 462 | 14.3 °C | 30.1 °C | **−15.8** |
| 387 | 21.2 °C | 40.2 °C | **−19.0** |
| 300 | 29.9 °C | 49.9 °C | **−20.0** |
| 212 | 41.8 °C | 60.0 °C | **−18.2** |

The consequence is physically absurd: an idle inverter reads **11 °C**, and a deliberately
induced MOSFET over-temperature fault peaks at **28 °C**.

### The fix

Steinhart–Hart refitted on the four settled points:

```
A =  5.197222e-03
B = -4.958087e-04
C =  3.504369e-06     R1 = 10 kΩ
```

Result across the D3 data range (ADC 281–554): **17.8 °C to 52.4 °C**, monotonic throughout
(asserted in the tests), with idle at ~24 °C and the over-temp faults at 42–48 °C.

Only the transient samples are excluded, and for a reason: during a ramp the reference probe
and the NTC are not thermally coupled, which makes the pooled relation non-monotonic (the
ADC ≈ 508 bin reads 46 °C while the ADC ≈ 460 bin reads 30 °C). Fitting the full 3,067 pairs
gives a mean absolute error of 2.9 °C and a maximum of 22.8 °C; fitting the settled points
gives 0.36 °C maximum.

**Limits of this fit, stated plainly:** three parameters fitted to four points. The 0.36 °C
residual therefore measures almost nothing — it is close to an exact fit, not a validated one.
What can be said is that the curve is smooth and monotonic across the whole range the fault
data occupies, and is anchored to four independently measured, well-separated points.
Calibration covers 30–60 °C (ADC 212–462); D3's idle readings sit at ADC ≈ 509, so the
baseline is a mild **extrapolation** below the calibrated range. Treat absolute idle
temperatures as approximate. Temperature **differences** and **rates** — which is what the
branch actually uses — are unaffected.

Pass `coefficients=SH_SCRIPT` to reproduce the published CSV.

---

## 5. Which channels carry information

Per-channel raw ADC statistics over all 10,892 rows:

| Channel | min | max | mean | std | distinct values | Kept? |
|---|---|---|---|---|---|---|
| `Ia` | 350 | 651 | 503.61 | 72.87 | 281 | ✅ |
| `Ib` | 355 | 647 | 489.44 | 79.99 | 285 | ✅ |
| `VDC` | 494 | 518 | 507.46 | **1.22** | 24 | ❌ |
| `IDC` | 488 | 517 | 506.52 | **1.36** | 27 | ❌ |
| `T1` | 281 | 554 | 468.02 | 66.17 | 165 | ✅ |
| `T2` | 357 | 542 | 479.34 | 33.29 | 169 | ✅ |
| `T3` | 317 | 533 | 467.51 | 31.87 | 195 | ✅ |
| `VD` | 494 | 523 | 509.33 | **1.17** | 28 | ❌ |

Per-class mean ADC — the table that decides the feature set:

| FDD | n | Ia | Ib | VDC | IDC | T1 | T2 | T3 | VD |
|---|---|---|---|---|---|---|---|---|---|
| F0 | 4295 | 514.76 | 475.15 | 507.41 | 506.50 | 510.24 | 500.92 | 475.49 | 509.36 |
| F1 | 692 | 519.20 | 474.05 | 507.43 | 506.46 | 496.43 | 497.27 | 480.68 | 509.29 |
| F2 | 1122 | 497.26 | 500.92 | 507.42 | 506.49 | 497.98 | 496.41 | 476.52 | 509.34 |
| F3 | 407 | 453.45 | 541.74 | 507.72 | 506.68 | 496.59 | 473.30 | 486.15 | 509.35 |
| F4 | 341 | 497.58 | 520.31 | 508.13 | 506.91 | 495.55 | 481.19 | 493.03 | 509.44 |
| F5 | 412 | 477.84 | 475.27 | 508.00 | 506.83 | 494.67 | 485.04 | 495.85 | 509.44 |
| F6 | 854 | 497.91 | 500.48 | 507.40 | 506.46 | **319.19** | 489.60 | 474.68 | 509.18 |
| F7 | 1735 | 498.88 | 498.93 | 507.38 | 506.50 | **368.61** | **418.40** | 471.76 | 509.30 |
| F8 | 1034 | 498.43 | 496.52 | 507.35 | 506.46 | 499.85 | 452.48 | **375.72** | 509.34 |

**`VDC`, `IDC` and `VD` are dropped.** Across all nine classes their means span **0.78, 0.45
and 0.26 ADC counts** against per-channel standard deviations of 1.2–1.4 — inside their own
quantisation noise. They carry no diagnostic content.

This also removes `Vdc·Idc`, `dVdc/dt` and `dIdc/dt` from the plan's §5 feature list: they
would be products and derivatives of a constant. (`workflow_v2.md` §13.5.)

### What the kept channels do

**Temperature is clean, physical and matches the filenames exactly.** After the recalibration,
per-condition channel means (°C):

| F | T1 | T2 | T3 | Source file says |
|---|---|---|---|---|
| F0 | 23.9 | 25.1 | 28.5 | normal |
| F6 | **47.9** | 26.7 | 28.6 | HB1 over-temp ✅ |
| F7 | **42.0** | **35.8** | 29.0 | HB1 **&2** over-temp ✅ |
| F8 | 25.3 | 31.5 | **41.2** | HB3 over-temp ✅ |

This is an independent validation of the whole conversion chain: the three over-temp
conditions heat exactly the half-bridges their filenames name, and nothing else.

**The electrical channels are weak.** `F3` (HB1 low-side short) is clear — `Ia` 453 / `Ib` 542
against F0's 515 / 475. `F1` (HB2 high-side open) is nearly indistinguishable from `F0` by
channel mean.

---

## 6. Consequences for the B-S2/S3 branch

1. **The over-temp family is a thermal threshold, not machine learning.** F6/F7/F8 (3,623
   samples, 33 % of the data) are separable by a single NTC comparison. A headline 4-class
   accuracy would be a thermometer wearing a classifier costume. The metric card must say so.
2. **The electrical-only ablation is the headline**, not the appendix. "Can open- and
   short-circuit faults be detected without temperature?" is the question that matters for a
   drive-health story, and on these channel means the honest answer is likely "weakly".
   Report it first, whatever it says.
3. **No 9-class metric.** Under a 70/30 block split with a purge, `F4` yields ~8 test windows,
   `F3` and `F5` ~10 each. A per-class accuracy on 8 samples has a 95 % CI of roughly
   ±35 points. Confusion matrix as a qualitative figure only. (`workflow_v2.md` §13.8.)
4. **Everything here is within-run, and the confound is worse than "one run per condition".**
   Each condition is a single contiguous file recorded at a distinct wall-clock time
   (13:24 → 14:31), and temperature drifts monotonically within a run. A block split separates
   early-in-run from late-in-run, **not** condition from condition.
5. **Fusion authority: indicative only.** One run per condition means zero independent
   validation groups, which fails the pre-registered `min_validation_groups = 3` test in
   `config.FUSION_CONFIG`. This branch may raise `Warning` and contribute evidence; it may
   never raise `Fault` alone. (`docs/claims_audit.md` §1.1.)

Planned windowing: 5 s (50 samples), hop 1 s (10 samples), purge ≥ 1 window at every block
boundary — `splits.contiguous_block_per_group` enforces the purge and refuses `purge=0`,
because adjacent windows at this hop share 80 % of their samples.

---

## 7. Open items

| Item | Status |
|---|---|
| Transcribe `Sensor_raw_data_conversion_formulas.pdf`; record the divider ratio and confirm the 2.5 V offset bug from the author's own formulas | **UNVERIFIED** — `pypdf` installed, PDF not yet read |
| Confirm the recalibrated Steinhart–Hart against the paper's stated temperature ranges | **UNVERIFIED** — worth one check; the physical evidence in §5 is already strong |
| Encoder / speed channel | Mentioned in the dataset README, **not present** in any logged file |
