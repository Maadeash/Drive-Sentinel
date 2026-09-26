# Claims audit

Every headline number, its protocol, the file that produced it, and the command that
regenerates it. Per `docs/workflow_v2.md` §9 rule 5.

Leaky numbers are labelled **(leaky reference)**. Anything not yet run reads `NOT RUN`.

---

## 1. Pre-registered decisions

Decisions fixed **before** the runs they govern, so that they cannot be chosen after seeing
results. Changing any of these later invalidates the metric it governs.

### 1.1 Fusion branch confidence floor

| Field | Value |
|---|---|
| **Fixed at** | **2026-09-18T07:19:23Z** |
| Repo state | `7344a436b724a08058ca6496d9d1d5936e17226b` (`main`) |
| Branches trained at this time | **none** — S4, S1, S2/S3 had not been implemented |
| Authority | `docs/workflow_v2.md` §13.6 |

A branch may raise `Fault` on its own **only if both** conditions hold:

1. it has **≥ 3 independent validation groups**, and
2. its honest (non-leaky) **macro-F1 ≥ 0.75**.

Otherwise the branch is **indicative**: it may raise `Warning` and contribute to the evidence
string, but may never raise `Fault`.

Implemented in `drivesentinel/config.py` as `FUSION_CONFIG["fault_authority"]`. The dashboard
renders evidence tiers and keeps waveform and spectrum panels live for indicative branches.

**This floor was set before any multi-stage branch was trained and must not be changed after
seeing branch results.** If it is ever changed, the change, its date and its reason go here,
and every metric measured under the old floor is re-labelled.

Expected consequences at the time of fixing (recorded so that the outcome cannot be
retrofitted):

| Branch | Groups available | Expectation |
|---|---|---|
| S5 bearing | 29 bearings | meets the floor |
| S4 winding | 3 motors | meets the group test; macro-F1 unknown |
| S1 supply | 2 motors | **fails the group test** → indicative |
| S2/S3 telemetry | 1 run per condition | **fails the group test** → indicative |

### 1.2 Declared selection-on-test

`artifacts/runs/sweep4_v1.json` and `artifacts/runs/sweep5_v2.json` are recipe sweeps scored
**on the leave-one-bearing-out folds themselves**. The configuration in
`drivesentinel/config.py:110-144` (feature set v2, `val_bearings_per_class: 0`, strong
augmentation, `n_seeds: 1`) was chosen from those results.

Under `docs/workflow_v2.md` §9 rule 3 this is **selection on test**, and it is declared as
such here rather than presented as an independent validation.

**Mitigation, and why the effect is believed small:** the differences the sweep was used to
choose between sit inside the measured run-to-run noise floor. `drivesentinel/config.py:131-137`
records the three-way comparison (v2+strong 0.8018, v1+strong 0.7823, v2+weak 0.7611) against
a ±0.0103 three-repeat standard deviation, and `docs/accuracy_ceiling.md` records that an
earlier two-draw comparison against a ±0.11 noise floor reached the **opposite** conclusion.
The one change that is larger than the noise floor — removing the inner validation split
(0.7283 → 0.7823, std 0.108 → 0.0018) — is a variance fix with a mechanical explanation
(`drivesentinel/folds.py:64-80`), not a tuned hyperparameter.

**How to read the LOBO number in light of this:** 0.8018 ± 0.0103 is an optimistic estimate
by an amount not separately measured. A clean estimate would need an outer loop, or a
recipe frozen before any sweep. Neither was done.

### 1.3 Cache rebuild, 2026-09-18 — the numbers moved and why

The repo was relocated to `C:\kone` and the Python environment did not survive.
`artifacts/order_spectra_v2.npz`, its meta parquet, and
`artifacts/runs/deployment_model.pt` were all absent, so the feature cache and every
model were rebuilt from `data/`.

**The rebuilt cache is not identical to the one behind the recorded numbers.**

| | Pre-rebuild | Rebuilt | Δ |
|---|---|---|---|
| Recordings | 2,306 | **2,318** | +12 |
| Windows | 16,127 | **16,211** | +84 |
| `inner_race` windows | 6,070 | 6,154 | +84 |
| KI14 recordings | **68** | **80** | +12 |
| All other bearings | unchanged | unchanged | — |

The entire difference is **KI14**. The machine that produced the recorded numbers held
68 of its 80 runs; this copy of `data/` has all 80. `KA04` (79, byte-identical
run-17/18 duplicate dropped) and `KA08` (79, one structurally corrupt `.mat`) are
short in both. The stale note in `dataset.py` that said "KI14 68 = absent from this
copy" was corrected at the same time.

**The DSP front end is confirmed unchanged.** The speed-estimate error — computed from
the phase current and validated against the tachometer on every recording — reproduced
to four significant figures across the rebuild:

| | Pre-rebuild | Rebuilt |
|---|---|---|
| Median speed error | 0.02214 % | **0.02210 %** |
| p99 speed error | 0.20849 % | **0.20829 %** |

Two independent runs of the whole front end over an overlapping but not identical set
of 2,300+ recordings agreeing to that precision means the signal processing, the order
resampling and the speed estimation are bit-stable across the move. The contract
fingerprint `4fdbe9910a513c40` is also unchanged.

**Effect on the headline number:**

| Metric | Pre-rebuild (3 repeats) | Rebuilt (single run) | Δ | Noise floor |
|---|---|---|---|---|
| Window accuracy | 0.8018 ± 0.0103 | **0.7944** | −0.0035 | ±0.0103 |
| Macro-F1 | 0.7944 ± 0.0114 | **0.7852** | −0.0024 | ±0.0114 |
| Per-recording accuracy | 0.8332 ± 0.0056 | **0.8356** | −0.0026 | ±0.0056 |
| Per-bearing mean | 0.7983 ± 0.2852 | **0.7945 ± 0.2836** | −0.0038 | — |

Every difference is inside the run-to-run standard deviation the 3-repeat study
measured on the same recipe. **This is not a regression and not an improvement — it is
the same result measured again.**

**Both numbers stay visible.** Neither is deleted. The rebuilt single-run figures are
current; the 3-repeat figures are labelled "pre-rebuild cache" wherever they appear.
`repeat.py` has **not** been re-run — it costs ~3¼ hours on this machine, which has no
GPU. The pre-rebuild artefacts are preserved at
`docs/prior_results/v5_pre_rebuild/`.

**Caveat on the current figure:** it is a single run, so it has no error bar of its
own. Use ±0.0103 as the noise floor when comparing anything against it. Do not compare
a future configuration to 0.7944 and call a 0.005 difference an improvement.

### 1.4 INT8 export re-run against the rebuilt model

The shipped `.mem` files were exported from the **pre-rebuild** deployment model. That
model's checkpoint no longer existed after the relocation, so the export could not have
been re-verified against its own float parent. Both were regenerated on 2026-09-18 from
`deployment_model.pt` as rebuilt.

All ten `.mem` files changed (MD5 compared before and after). The previous export's
weights are superseded; anything already flashed to an FPGA from them corresponds to a
model trained on the pre-rebuild cache.

| | Pre-rebuild export | Rebuilt export |
|---|---|---|
| argmax agreement, n=2,048 | 99.90 % | **99.85 %** |
| argmax agreement, all windows | 99.882 % (16,127) | **99.877 %** (16,211) |
| Disagreements, all windows | 19 | **20** |
| Max \|logit\| difference | 4.0409 | **1.3486** |
| Mean \|logit\| difference | 0.0811 | **0.0754** |
| Weights / biases | 27,024 / 179 | 27,024 / 179 |
| Accumulator bits (conv1–4, fc) | 21/22/23/23/21 | 21/22/23/23/21, all fit int32 |
| Methodology bar ≥ 98 % | PASS | **PASS** |

Both exports clear the bar comfortably. The max logit difference improved threefold,
which is a quantisation-calibration detail and not a claim about accuracy.

Disagreements concentrate on the known-hard specimens — `KI05` 6, `K002` 3, `KA03` 3,
`KA04` 2, `KA05` 2, `KA30` 2, `KA22` 1, `KI14` 1. Eight bearings account for all 20.
These are the same bearings the LOBO sweep fails on, which is what you would expect:
INT8 and float disagree where the float model's own logits are closest together.

### 1.5 Post-hoc protocol change: V5 leave-one-session-out (B-S4)

| Field | Value |
|---|---|
| **Decided at** | **2026-09-18**, after the `i0rel_residual` probe |
| Repo state at decision | `3c2a6d0` (the commit that recorded the probe) |
| Protocol added | **V5 — leave-one-session-out** over the three D2 acquisition days |
| Authority | user instruction, this session |

**This is a protocol chosen after seeing data, and it is declared as such.** The
pre-registered B-S4 protocol set (`workflow_v2.md` §5) was V1 leave-one-motor-out,
V2 lowest-severity holdout, V3 shuffled reference. V5 was added later.

**The reason it was added.** A 1-NN probe on `i0rel_residual` — a scalar that is zero
by Kirchhoff in a three-wire machine and therefore cannot contain winding information —
recovered the acquisition session at 1.000 on the 1000/1500 W recordings, and recovered
70 % of the fault label with it. Leave-one-motor-out does not hold session out, so V1
alone cannot distinguish winding diagnosis from session identification.

**Why adding it is not the same as tuning on test.** V5 was defined *before* any B-S4
model was fitted — no fault-class number existed when the protocol was chosen. The
grouping is also not a free parameter: it is the acquisition date read from the TDMS
`Date Created` property, corroborated independently by DAQ chassis, recording-duration
discipline and root-name convention (four signals, one partition, `data_notes_d2.md`
§7D). Nothing about the split was selected to improve a score.

**What would make this illegitimate, and did not happen:** choosing the grouping to
maximise a metric, trying several groupings and keeping the best, or defining V5 after
seeing V1's result. V1 and V5 were run in the same invocation of
`scripts/branches/10_winding.py`, from one cached feature matrix.

**How to read V1 and V5 together.** V1 answers "does this transfer to a new motor",
V5 answers "does it transfer to a new acquisition session". Both are reported; the gap
between them is itself the result (§3.2).

### 1.6 Correction: five residual groups nesting in three days, not six sessions

An earlier statement in this session — and in an instruction issued from it — held that
the `i0rel_residual` analysis resolved **six** acquisition sessions. **That was wrong.**

The correct structure is **five residual-derived groups nesting inside three
acquisition days**:

| Day | Residual-derived groups | n |
|---|---|---|
| 2022-01-25 | `cDAQ1Mod2 (+,+,+)`, `cDAQ1Mod2 (+,-,-)` high-residual | 7 + 11 |
| 2022-03-08 | `cDAQ1Mod2 (+,+,-)`, `cDAQ1Mod2 (+,-,-)` low-residual | 8 + 12 |
| 2022-08-11 | `cDAQ5Mod1 (+,+,+)` | 7 |

The error was double-counting: the 12/11 split inside the `(+,-,-)` polarity group was
counted as a sixth group, when the five batch keys already encoded it through the
root-name style.

**The metadata verification is what caught it**, which is the argument for doing the
verification rather than trusting the probe. V5 groups by the three **days**, not by the
five residual groups — the days are what the metadata independently establishes.

### 1.7 B-S4 fusion status: INDICATIVE

| Field | Value |
|---|---|
| Protocol used for the decision | **V5** (leave-one-session-out) |
| Validation groups scored | **2** — floor is **3** |
| Honest macro-F1 | **0.6250** — floor is **0.75** |
| **Status** | **INDICATIVE — no `Fault` authority** |

**Fails both tests of the pre-registered floor** (§1.1, fixed 2026-09-18T07:19:23Z at
commit `7344a436`, before any multi-stage branch existed). The floor has **not** been
adjusted, and must not be.

Groups scored is 2 rather than 3 because the 2022-08-11 session contains only
`inter_coil` recordings (7/0) and cannot score a two-class task. It remains in training
for the other folds.

Consequence for `drivesentinel/fusion.py` and the dashboard: B-S4 may raise `Warning`
and contribute to the evidence string, but may never raise `Fault` on its own. Its
waveform and spectrum panels stay live — an indicative branch is displayed, not hidden.

### 1.8 Fusion authority as measured (P4, 2026-09-18)

Which branches may raise `Fault`, and why. **Read from each branch's results JSON at
load time by `fusion.load_branch_metrics()`, never hardcoded** -- so the table cannot
drift away from the measured numbers.

Floor: **>= 3 independent validation groups AND honest macro-F1 >= 0.75**,
pre-registered 2026-09-18T07:19:23Z at commit `7344a436`, before any multi-stage branch existed (1.1).
**Not adjusted.**

| Branch | Stage | Protocol | Groups | Macro-F1 | Tier |
|---|---|---|---|---|---|
| `bearing` | S5 | leave-one-bearing-out | 29 | 0.7852 | **Fault-capable** |
| `winding` | S4 | V5 (leave-one-session-out) | 2 | 0.6250 | **INDICATIVE** |
| `supply` | S1 | — | — | — | **NOT MEASURED** |
| `inverter_telemetry` | S3 | — | — | — | **NOT MEASURED** |

- **`bearing` holds Fault authority.** 29 bearings and macro-F1 0.7852 clear both
  tests. It is the only branch that can condemn a drive on its own.
- **`winding` is INDICATIVE and fails both tests** -- 2 scored sessions against 3, and
  0.6250 against 0.75. It may raise `Warning` and contribute evidence. Its panels stay
  live; it is capped, not hidden.
- **`supply` and `inverter_telemetry` are NOT MEASURED** -- no results JSON yet.
  Absence of a metric earns the same lack of trust as a bad one, so they have no
  authority either. They are re-read automatically once their branches run.

Asserted in `tests/test_fusion.py`: an INDICATIVE branch held at `p_fault = 0.99` for
50 consecutive updates reaches `Warning` and stops. If that test ever passes as
`Fault`, the floor has been defeated.

### 1.9 Trip gating is implemented and disabled

`config.TRIP_CONFIG["enabled"] = False`. No real dataset in the roster has a speed
ramp:

| Dataset | f_e | Ramps? |
|---|---|---|
| D1 Paderborn | 900 / 1500 rpm, constant within each 4 s recording | no |
| D2 KAIST | 200.00 Hz in all 48 recordings | no |
| D3 Bacha | 10 rad/s constant, 10 Hz telemetry | no |
| D4 Thomas | mains-fed, 49.955-50.036 Hz | no |

Segmentation on any of them returns `cruise` for every window, so gating changes
nothing. Enabling it by default would ship a code path no data exercises and a claim
nothing tests. **No claim is made anywhere that trip gating gates real data.**

It is unit-tested against synthetic ramps with known ground truth: a trapezoidal
profile segments into accel/cruise/decel in order, with cruise boundaries within
0.25 s of truth, and a linear chirp tracks within 2 Hz median error.

The one real-data application is **envelope** segmentation of the D4 start-up
transients, where f_e is pinned by the grid and the transient lives in the current
envelope. Reported as segmentation only, never as frequency tracking. Measured cruise
onsets: FILE 1 at 8.83 s, FILE 6 at 9.17 s -- consistent with the per-2 s RMS blocks
in `data_notes_d4.md`. For FILE 5 and 10 the machine never rotates, so `cruise` there
means steady **current**, not steady rotation; stated so the label is not over-read.

The order-resolution table in `docs/results_multistage.md` is **analytic
extrapolation**, labelled as such per row. It is arithmetic about what a window length
can resolve, not a claim about detection performance at low speed.

### 1.10 Dashboard: what it is allowed to display (P5, 2026-09-19)

`dashboard/app.py` is **replay only**. Every number it shows is read from a results
JSON at display time; nothing is computed live, nothing is typed in, and no metric
is passed to it as a literal.

**The out-of-sample rule is enforced in code, not trusted.**
`workflow_v2.md` section 8 requires a replayed unit's predictions to come from the
fold model that held that unit out. `panels.assert_out_of_sample()` raises if a
bearing scenario is not marked out-of-sample, or if the replayed bearing appears in
its own fold's training-set list, and `app.py` renders the error instead of the
panel.

This costs real compute and is worth it. `scripts/02_train_lobo.py` keeps only the
deployment model, which trained on all 29 bearings, so each bearing scenario
**retrains its own leave-one-bearing-out fold** (about 2.8 min per bearing on this
CPU). Reusing the deployment model would have been one line and would have made
every bearing replay in-sample -- the exact failure the rule exists to prevent.

**Authority is displayed per stage, not in a legend.** Each stage light carries a
badge -- FAULT-CAPABLE / INDICATIVE / NOT MEASURED -- with the branch's honest
metric and protocol underneath, so a viewer can see that a light can never go red
without looking anywhere else. Tiers come from `fusion.load_branch_metrics()`.

**No stage is hidden.** `supply` and `inverter_telemetry` have no results JSON and
render as NOT MEASURED with live panels. An empty stage is more honest than a
diagram implying four working branches.

**As displayed today: exactly one stage of five can raise Fault.**

| Stage | Branch | Tier |
|---|---|---|
| S1 Supply | `supply` | NOT MEASURED -- branch not built |
| S2 DC link | `inverter_telemetry` | NOT MEASURED -- branch not built |
| S3 Inverter | `inverter_telemetry` | NOT MEASURED -- branch not built |
| S4 Winding | `winding` | INDICATIVE -- capped at Warning |
| S5 Bearing | `bearing` | **FAULT-CAPABLE** |

**Scenario (a) is a severity ramp, not a healthy-to-fault ramp.** All three D2
healthy recordings are from 2022-01-25, so healthy is not measurable under a
session-aware split (1.5, 1.7, data_notes_d2.md 7D). The original scenario (a) in
workflow_v2.md section 8 called for "healthy trip -> gradual S4 winding fault".
That cannot be built from real data and **is not synthesised**. The replacement
walks real inter_turn recordings from the lowest to the highest severity and is
named as such everywhere it appears.

**Scenario (b) includes a failure on purpose.** `KI05` scores 0.0089 under
leave-one-bearing-out and is one of the three bearings shipped as a scenario,
alongside `KA04` (works) and `K001` (healthy specimen). A demo that can only show
successes is a worse demo, and the per-bearing spread is already the headline
finding of `accuracy_ceiling.md`.

**Scenarios (c) and (d) are absent and say why.** (c) S1 phase loss is SKIPPED --
the B-S1 branch does not exist: no adapter, no threshold rule, no results JSON.
(d) S2/S3 open circuit is NOT MEASURED -- no
`artifacts/multistage/inverter_telemetry/*.json`. Both appear in the manifest with
a stated reason, and the dashboard lists them in the sidebar rather than omitting
them.

**Stored scenario size.** Capped at 120 windows per bearing, evenly spaced across
the fold so a replay spans the whole recording set rather than its first seconds.
Metrics shown on the cards come from the **full** fold, not from the stored subset;
`n_windows_in_fold` and `n_windows_stored` are both recorded in each NPZ.

### 1.11 B-S1 supply: a rule, not a model, and why (P2, 2026-09-19)

**The deliverable is a documented threshold rule.** D4 has one recording per
(motor × class) and two motors, so under leave-one-motor-out a learned model sees
one training example per class — it separates two 20 s captures rather than
learning phase loss. Liu et al. measure exactly that on this dataset: macro-F1
0.9682 under a random split, 0.5856 under a within-label block split.

**The rule:** a phase is LOST in a window when its 0.2 s RMS falls below 5 % of
the median of the other two. A window is OFF when every phase is below 0.05 A, and
OFF windows are excluded from scoring. `phase_loss_running` is separated from
`single_phasing_start` by **rotation** (vib_x RMS above 0.005), not by current —
the two look nearly identical electrically, and measured vibration separates them
by a factor of ~20 (stalled 0.0008–0.0012, rotating 0.017–0.041).

| Protocol | Scheme | Accuracy | Macro-F1 |
|---|---|---|---|
| **R2 — the rule** | leave-one-motor-out | **1.0000** | **1.0000** |
| L1 — learned | leave-one-motor-out | 0.6593 | 0.6394 |
| L3 — learned **(leaky reference)** | shuffled windows | 0.9983 | 0.9981 |

**34 points** separate the learned model's leaky and honest splits, on the same
features. The rule is unaffected because nothing in it is fitted.

Detection latency per event: FILE 2 **0.10 s**, FILE 5 **0.30 s**, FILE 7
**0.00 s**, FILE 10 **1.50 s**. Measured against an amplitude crossing independent
of the rule's own threshold, and floored by the 0.1 s hop.

#### Correction: the threshold's sensitivity is narrower than first claimed

The first draft of `branches/supply.py` asserted that "anything between 1 % and
30 % gives the same answer". **That was wrong**, and measuring it is what caught
it:

| Level | Safe range | Evidence |
|---|---|---|
| **Recording verdict** (what the branch reports) | **0.03 – 0.40**, 10/10 throughout | at 0.02 it loses FILE 2; at 0.01 it loses FILE 2 and FILE 7 |
| Window classification | **0.048 – 0.053** only | worst real lost-phase ratio 0.0476 (FILE 10); tightest normal-window ratio 0.0526 (FILE 5) |

Both window-level bounds come from the same few windows — those straddling the
instant a phase dies. A 0.2 s window spanning the transition contains both states
and its RMS lands between them. That is an artefact of the window length, not a
property of the fault, and it is why the branch reports a recording-level verdict
requiring a ≥ 0.5 s event rather than trusting individual windows.

**Do not quote the window-level margin as if it had the recording-level margin.**
Both numbers are in the module docstring and asserted in
`tests/test_supply_features.py`.

### 1.12 D4 label vector: reconstructed, not validated, not used

The paper describes a 1000-sample window at step 500 over the ten files merged in
order. That reconstruction was tested against the phase-current collapse
boundaries measured here:

| Check | Result |
|---|---|
| Label-run boundaries coinciding with a **file** boundary at 1998 windows/file | **6 of 14** |
| Same, at 1999 windows/file | 1 of 14 |
| Label-run boundaries coinciding with a **measured phase-loss event** | **0 of 4** |

So the per-file window count is probably 1998 and the file structure
reconstructs — but the within-file event structure does not match anything
measurable in the currents.

**The label vector is therefore not used.** Labels come from the current-collapse
rule, which was primary either way. The disagreement is recorded rather than
resolved by assumption, and `adapters/thomas_motor.py` sets
`provenance["label_vector_used"] = False` on every recording.

### 1.13 B-S1 fusion status: INDICATIVE, at macro-F1 1.0000

| Field | Value |
|---|---|
| Protocol | R2, threshold rule, leave-one-motor-out |
| Validation groups | **2** — floor is **3** |
| Honest macro-F1 | **1.0000** — floor is 0.75 |
| **Status** | **INDICATIVE — no `Fault` authority** |

**A perfect score still does not earn `Fault` authority**, because two motors is
below the pre-registered group minimum. The floor was fixed 2026-09-18T07:19:23Z
at commit `7344a436`, before any multi-stage branch existed, and §1.1 recorded the
expectation "S1 supply — 2 motors — **fails the group test** → indicative" at that
time. It has not been adjusted.

This is the floor working as intended: a branch can be perfectly right about the
recordings it has and still not be trusted to condemn a drive alone, because two
machines cannot tell you how the rule behaves on a third.

### 1.14 D4 bearing confound: a documented negative result

Predicting **motor identity** from the same features scores **0.9983** against a
0.5049 baseline under a shuffled split.

That number is not a bearing-fault capability and must never be quoted as one.
D4 has exactly one motor per bearing condition, so "outer-race fault" and "motor
identity" are the same variable; nothing distinguishes a bearing defect from any
other difference between two physical machines — winding tolerances, mounting,
alignment, age. Vibration is measurably 1.6–2.3× higher on the faulty motor across
every matched scenario pair, and that measurement cannot be attributed to the
bearing.

Stated once in `adapters/thomas_motor.py::bearing_confound_note()` and quoted from
there everywhere it appears, so the wording cannot drift.

### 1.15 B-S2/S3 inverter telemetry: the ablation is the finding (P3, 2026-09-19)

| Protocol | Split | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|---|
| **V3 electrical-only** | contiguous block | **0.8503** | **0.8229** | 0.4045 |
| V3 electrical-only **(leaky ref)** | random windows | 0.9229 | 0.9106 | 0.4048 |
| V1 with temperature | contiguous block | 1.0000 | 1.0000 | 0.4045 |
| V2 with temperature **(leaky ref)** | random windows | 1.0000 | 1.0000 | 0.4048 |

**V1 and V2 agree exactly.** The leaky split gains nothing — not because the block
split is safe, but because with a thermometer in the feature set the task is
already saturated. A 4-class score that includes `over_temp` is substantially a
temperature reading: F6 heats HB1, F7 heats HB1 and HB2, F8 heats HB3, each
matching its filename.

#### The measurement that says how much to trust any of it

Per-class F1 under the **electrical-only** block split:

| Class | F1 | Test windows |
|---|---|---|
| `normal` | 0.984 | 127 |
| **`open_circuit`** | **0.511** | 51 |
| `short_circuit` | 1.000 | 31 |
| **`over_temp`** | **0.796** | 105 |

**`over_temp` scores F1 0.796 with no temperature sensor in the feature set.**

A thermal fault is not physically detectable from two 10 Hz phase currents. That
number is the model identifying **which run** a window came from, not what
condition the inverter was in. Each condition is one contiguous file recorded at a
distinct wall-clock time (13:24 → 14:31), so anything that drifts with time
carries run identity, and the currents drift.

**Treat every 4-class number on this dataset as an upper bound contaminated by run
identification.**

And the class the electrical channels *should* see is the weakest:
`open_circuit` at F1 **0.511**. F1 (HB2 high-side open) has almost the same
channel means as F0 — measured Ia/Ib 519/474 against 515/475.

#### 9-class location view: confusion matrix only

Under the block split the smallest classes get 9–11 test windows (`F4` 9, `F3` 11,
`F5` 11). A per-class figure on nine samples has a 95 % CI of roughly ±35 points.
The matrix is rendered qualitatively in `results_multistage.md`; **no per-class
number is reported from it**, and none should be quoted.

#### Calibration independence, by construction

`data_notes_d3.md` §4 refits the NTC Steinhart–Hart coefficients because the
shipped ones under-read by 13–20 °C. That refit is three parameters on four points
and is an extrapolation below 30 °C.

**No feature in this branch depends on it.** Temperature enters only as raw ADC
statistics and as ADC *differences* between channels, both monotone in temperature
under any calibration. `tests/test_inverter_telemetry.py` asserts that a constant
shift applied to every NTC reading — which is what a different calibration does to
first order — leaves every difference feature unchanged.

If the refit is wrong, none of these numbers move.

#### Dropped channels

`VDC`, `IDC`, `VD` are not used. Per-class means span 0.78 / 0.45 / 0.26 ADC counts
against per-channel std 1.2–1.4 — inside their own quantisation noise. That also
removes `Vdc·Idc`, `dVdc/dt` and `dIdc/dt` from the original plan: products and
derivatives of a constant.

### 1.16 B-S2/S3 fusion status: INDICATIVE at zero validation groups

| Field | Value |
|---|---|
| Protocol | V1, contiguous block split, within-run |
| Validation groups | **0** — floor is **3** |
| Honest macro-F1 | 1.0000 — floor is 0.75 |
| **Status** | **INDICATIVE — no `Fault` authority** |

**Zero**, not two: one run per condition means there is nothing independent to hold
out at all. The block split separates early-in-run from late-in-run within the same
recording.

§1.1 recorded the expectation "S2/S3 telemetry — 1 run per condition — **fails the
group test** → indicative" on 2026-09-18, before this branch existed. The floor has
not been adjusted.

All three multi-stage branches now score at or above the macro-F1 floor — 1.0000,
1.0000 and 0.6250 — and **all three are INDICATIVE**, every one of them failing on
the group test. That is the floor doing the job it was written for: the binding
constraint on this project is how many independent machines each dataset contains,
not how well a model fits.

### 1.17 D2 CNN experiment — post-hoc, criteria declared before the run

**This was run AFTER the gradient-boosting result was known.** It is a post-hoc experiment and is declared as one. What makes it legitimate rather than fishing is that the recipe and the acceptance criteria were written into `scripts/experiments/d2_cnn.py` and **committed before the first run** (commit `05cf8d7`), so neither could be adjusted after seeing the numbers.

**Hypothesis:** a CNN on the full f/f_e harmonic spectrum finds inter-coil vs inter-turn structure that 23 scalar features miss.

**Recipe, fixed in advance:** OrderSpectrumCNN (16,9,2)(32,7,2)(64,5,2)(64,3,2) + GAP + linear, 512 bins over 0–20 orders, 30 epochs, lr 0.003, batch 256, OneCycle, label smoothing 0.05, class-balanced weights, seeds [0, 1, 2]. Single model, no ensemble, no TTA, **no hyperparameter search**. 10,463 windows — the same rows, task and splits as the GBM run; only the features and the model change.

| Protocol | CNN (mean ± std, 3 seeds) | Gradient boosting |
|---|---|---|
| V1 leave-one-motor-out | 0.4719 ± 0.0081 | 0.3784 |
| **V5 leave-one-session-out** | **0.5608 ± 0.0142** | **0.6251** |
| V3 shuffled **(leaky reference)** | 0.8448 ± 0.0078 | 0.9990 |

**Acceptance criteria, as declared:**

| # | Criterion | Threshold | Measured | Outcome |
|---|---|---|---|---|
| 1 | V5 beats the session-only baseline — the `i0rel_residual` 1-NN, a scalar that cannot contain winding information | > 0.750 | 0.5608 | **FAIL** |
| 2 | V5 beats the GBM by more than the seed spread | > 0.6251 + 0.0142 | margin -0.0643 | **FAIL** |

**Outcome: NEGATIVE RESULT.** Shipped branch: **gradient boosting (unchanged)**.

The hypothesis is not supported. A CNN with capacity comparable to the bearing model (26,914 parameters against 27,024), given the whole spectrum rather than 23 scalars, does not recover winding structure the scalar features missed — because on this dataset the structure that survives a session-aware split is mostly not winding structure. This is consistent with everything else measured on D2: the session is recoverable at 1.000 from a quantity that is zero by Kirchhoff, and V1 sits below its own majority baseline for both models.

**No further CNN variants were tried.** Sweeping architectures against a declared acceptance criterion until one passes is exactly the failure the criterion exists to prevent. One recipe, declared, run, reported.

**The tier does not move.** INDICATIVE -- 2 validation groups against a floor of 3, whatever this scores. The floor is on GROUPS as well as macro-F1 and was pre-registered before any branch existed.

Regenerate: `python scripts/experiments/d2_cnn.py`. Results in `artifacts/multistage/winding/winding_cnn_results.json`.

### 1.18 D3 run-identification control — the inference, measured

§1.15 argued from `over_temp` scoring F1 0.796 with no temperature sensor that the
model must be reading run identity. This measures that directly: **train the same
features, on the same block split, to predict which run a window came from.**

| Predicting | Features | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|---|
| which run (9 runs) | with temperature | **0.9713** | 0.9111 | 0.4045 |
| which run (9 runs) | electrical only | **0.7484** | 0.7277 | 0.4045 |

Beside the condition scores, same split and same features:

| Features | 4-class condition | which run | Gap |
|---|---|---|---|
| with temperature | 1.0000 | 0.9713 | +0.0287 |
| electrical only | 0.8503 | 0.7484 | +0.1019 |

**The two quantities are within a few points of each other on both feature sets**,
which is what you would expect if they are largely the same thing. D3 has exactly
one run per condition, so a model that can name the run can name the condition
without diagnosing anything.

**Consequence:** the 4-class family score is an upper bound on condition diagnosis,
and the run-identification number is how far above the truth that bound may sit.
Quote neither without the other.

This is now a measurement rather than an inference, and it is generated into
`docs/results_multistage.md` from the results JSON.

---

### 1.19 RTL spec figures — ARITHMETIC, and what is NOT measured

`docs/rtl_spec.md` (item (a) of the RTL track) is a specification, not a result. It
contains no new experiment. Every number in it is either read from
`artifacts/int8_export/` or derived from those by arithmetic, and each is verified
against the artefact by a script rather than transcribed by hand — three rounding slips
and one mis-rounded macro-F1 were caught that way before the document was committed.

**Derived from the export, exact:**

| Figure | Value | Derivation |
|---|---|---|
| int8 weights / int32 biases | 27,024 / 179 | `scales.json` |
| MACs per window | 1,691,840 | sum over layers of `O·I·K·L_out` |
| Largest activation buffer | 4,096 int8 | conv1/conv2/conv3 outputs |
| On-chip storage | < 40 KB | weights + biases + 2 buffers |
| Worst-case MAC sum | 3,121,152 (23 bits) | 192 terms of 127×128 |
| Worst-case pooled sum | 100,790,240 (28 bits) | 32 × conv4 peak |
| Requant operations per window | 12,355 | one per output element |

**ESTIMATES — no synthesis, no place-and-route, no board:**

| Figure | Status |
|---|---|
| Latency 0.13–2.11 ms at 100 MHz | **ESTIMATE.** `MACs / MACs_per_cycle / clock`. Ignores requantisation, pipeline fill, padding stalls and AXI transfer. |
| 100 MHz clock | **ASSUMED**, not constrained or closed |
| DSP / LUT / BRAM utilisation | **NOT RUN** |
| Power | **NOT RUN** |
| End-to-end latency | **NOT RUN**, and blocked: the DSP front end has never been timed on any hardware |

The latency table may be shown on the edge-feasibility slide **only** with the word
ESTIMATE attached and the front-end gap stated in the same breath. The defensible claim
is *"CNN inference is sub-millisecond by arithmetic; end-to-end latency is not yet
measured."* Anything stronger is not supported by anything in this repo.

**Two numbers that must not be confused.** `verification_full.json` reports 0.99877
argmax agreement — that is **INT8 against float**, a modelling question already settled.
The RTL acceptance bar in `rtl_spec.md` §8 is **RTL against the golden reference**, and
it is **100 %**, because the golden reference is the specification. Quoting the 98 % bar
from `methodology_v3.md` §F as the RTL bar would be quoting the wrong comparison.

**No accuracy claim changes.** The accelerator runs the frozen v5 bearing model at
macro-F1 0.7852 LOBO over 29 bearings. Hardware does not move that number, and
`rtl_spec.md` §10 lists what the hardware may not be claimed to do.

> **SUPERSEDED IN PART — see §1.20.** The RTL now exists, has been verified in
> simulation, and has been synthesised, placed and routed. Three rows of the ESTIMATE
> table above are now measurements and must be quoted as such from §1.20 rather than
> from here: latency, the 100 MHz clock, and DSP/LUT/BRAM utilisation. **Power and
> end-to-end latency remain `NOT RUN`**, and nothing has run on hardware.

---

### 1.20 RTL build — what is simulation-verified, what is synthesis-reported, and what has not run

The RTL track, items (b)–(e). Generated results: **`docs/results_rtl.md`**, from
`artifacts/rtl/verify.json`, `artifacts/rtl/synth*/synth.json` and the sweep JSONs.
Recipes were fixed in **`docs/rtl_declarations.md`** before any Verilog existed; the
acceptance bars were fixed in `rtl_spec.md` §8 before that.

**Nothing below has run on hardware.** No bitstream was written and no board was
involved (`rtl_spec.md` §9.4, answered). The three categories are kept apart on purpose,
because they are three different strengths of evidence.

#### (a) SIMULATION-VERIFIED — XSim, against the golden reference

| Figure | Value | Protocol |
|---|---|---|
| V-1 argmax agreement, all 16,211 cache windows | **COMPLETE: 16,211 / 16,211, 0 class and 0 logit disagreements, max logit difference 0 LSB** (finished 2026-09-22, 29.8 h of XSim over 65 chunks; `artifacts/rtl/verify.json`). In progress at the end of the build session | RTL vs `golden_reference.predict`, bar **100.000 %** — met |
| V-2 max abs scaled-logit difference | **0 LSB** of the exposed int32 register | over every window run so far |
| V-3 per-layer int32 accumulators, 64 windows | **917,696 compared, 0 mismatches** | bit-exact, every layer, every channel |
| V-4 round-half-to-even | **3,966 vectors, 427 exact ties, 0 mismatches** | directed, unit level (`tb/tb_requant.sv`) |
| V-5 saturation probes | **59 windows, 58,636 values clamped, 0 mismatches** | clamp fires at all four layers |
| V-6 build ID | **`0x23eb56bf`, match** | hardware CRC-32 walk vs an independent Python CRC of the `.mem` files |
| Cycles per inference | **1,786,724** | measured by the core's own counter, read over AXI-Lite; identical on every window |

#### (b) SYNTHESIS-REPORTED — Vivado 2023.2 post-route, `xc7z020clg400-1`, out of context

| Figure | Value | Source |
|---|---|---|
| Slice LUTs | **1,111** (2.09 %) | `artifacts/rtl/synth/post_route_util.rpt` |
| Slice registers | **968** (0.91 %) | same |
| Block RAM tiles | **11** (7.86 %) | same |
| DSP48E1 | **4** (1.82 %) | same |
| WNS at the 100 MHz target | **+0.615 ns**, 0 failing endpoints | `post_route_timing.rpt` |
| Achieved clock | **106.6 MHz** | derived from WNS: `1000 / (10 − 0.615)` |
| Latency at 100 MHz | **17,867 µs (17.87 ms)** | measured cycles × the target period |
| Latency at 106.6 MHz | **16,768 µs (16.77 ms)** | measured cycles × the achieved period |
| Margin against the 0.5 s hop | **30x** | 500 ms / 16.77 ms |

**Out of context is not a caveat being hidden.** The core is an AXI peripheral of the
Zynq PS with ~190 ports against 125 user I/O on the clg400 package, so a pin-constrained
build is impossible and would say nothing about the core if it were. I/O delays are left
unconstrained rather than invented (`constraints/ds_top.xdc`), so every timed path is
register-to-register and the quoted figure is the worst of those.

#### (c) NOT RUN — and not estimated either

| Figure | Status |
|---|---|
| Power | **NOT RUN.** No figure of any kind exists. |
| End-to-end system latency | **NOT MEASURED.** The DSP front end is PS software (§9.1) and has never been timed on any hardware. |
| On-board inference | **NOT RUN.** No bitstream, no board. |
| Accuracy on hardware | **Not applicable and not claimed.** Bit-exactness against the golden reference means the hardware computes the same classes the software does; it stays macro-F1 0.7852 LOBO. |

#### The two findings that came out of the build

**F = 15 would have met the argmax criterion and failed V-3's bar — in the model.**
These are statements about the bit-accurate Python model the width sweep runs, not about
the RTL; the RTL's own V-1 and V-3 results are the (a) table above. `rtl_spec.md` §9.2
left the requant
fractional width to be swept, and `rtl_declarations.md` D-2's rule — the smallest width
with 100.000 % argmax agreement — selects **F = 15**. At that width 810,265 conv2 and
2,926,089 conv3 accumulators differ from the float specification. The shipped width is
**F = 26**, the smallest at which every accumulator of every layer of all 16,211 windows
is bit-identical. Two declared criteria disagreed and the stricter bound; no bar moved,
since F = 26 also gives 100.000 % argmax agreement. Full sweep in
`artifacts/rtl/requant_sweep.json` and `artifacts/rtl/bit_exact_confirm.json`.

**The exact-tie case is unreachable from the input.** At F = 26 the exported shifts are
33–40 with odd 26-bit mantissas, so `acc * m0 == q·2^s + 2^(s-1)` has no solution inside
int32. No input window can produce an exact tie. V-4 is therefore a unit testbench
driving `ds_requant` directly, and `results_rtl.md` says so — *"we could not reach the
case"* and *"the case passed"* are different sentences.

#### What may go on a slide

The (a) and (b) tables, each labelled with which it is. The sentence that carries the
whole claim:

> *"The bearing CNN runs bit-exact against its golden reference in simulation,
> synthesises to 1,111 LUTs and 4 DSPs on a Zynq-7020, and closes timing at 100 MHz with
> a measured 17.9 ms per inference — a 30x margin on the 0.5 s hop. Nothing has run on
> hardware, and end-to-end latency is not measured, because the DSP front end has never
> been timed."*

Anything that drops the second sentence is not supported by this repository.

---

### 1.21 Board package — a bitstream exists; board execution: see §1.26

> **Updated 2026-09-24.** The notebook has since run on a PYNQ-Z2: §1.26, and (d) below.
> The text of (a)–(c) is as written before the board run, and remains accurate.

The PYNQ-Z2 overlay: `board/drivesentinel.bit` + `board/drivesentinel.hwh`, built by
`board/build_overlay.tcl` around the verified core, which is wrapped unchanged by
`board/rtl/ds_pynq_wrap.v`. When this entry was written, **no board was attached, the
bitstream had never been loaded, and the notebook had never run on a PYNQ-Z2.**

**Four categories, kept apart. The first three may be quoted, each with its label. The
fourth may not be claimed at all until `board/board_run.json` exists.**

#### (a) FULL-SYSTEM, SYNTHESIS-REPORTED — Vivado 2023.2 post-route, bitstream written

Scope: Zynq PS7 + AXI DMA + two AXI interconnects + reset + the core, one design.
Source: `artifacts/rtl/system/` (`system.json`, `post_route_util.rpt`,
`post_route_timing.rpt`).

| Figure | Value |
|---|---|
| Slice LUTs | **2,284** (4.29 %) |
| Slice registers | **2,690** (2.53 %) |
| Block RAM tiles | **12** (8.57 %) |
| DSP48E1 | **4** (1.82 %) |
| WNS / WHS at 100 MHz (`clk_fpga_0`) | **+0.777 ns / +0.050 ns**, timing met |
| Critical warnings / DRC errors | 0 / 0 (20 DRC warnings, listed in `post_route_drc.rpt`) |

**Never compare these with §1.20(b).** §1.20(b) is the **core only**, out of context,
with its I/O unconstrained. This table is the **whole system**. They measure different
things. Two statements connecting them are supported:

- **The core's instance inside the full system** (`ds_0` in
  `post_route_util_hier.rpt`) uses 1,110 LUT, 968 FF, 11 BRAM tiles and 4 DSP48. That
  matches the core-only build to within one LUT, which shows the wrapper added no
  datapath logic.
- **Both builds close at 100 MHz.** Their slacks are not the same quantity and are not
  ranked against each other.

#### (b) BUILD-VERIFIED, NOT HARDWARE-VERIFIED

| Figure | Value | How established |
|---|---|---|
| Weights in the bitstream are the export | all 11 `$readmem` images logged "read successfully" | `artifacts/rtl/system/vivado_system.log` — **the board's build-ID read is the real check and has not happened** |
| Core address window | `0x43C00000`, 4 KB | the `.hwh`, asserted in `tests/test_board_package.py` |
| Fabric clock in the handoff | 100 MHz | the `.hwh`, asserted likewise |
| Package integrity | SHA-256 of every file | `board/MANIFEST.sha256` |

#### (c) NOTEBOOK DRY RUN — against a software stand-in for `pynq`, NOT hardware

`board/dryrun/`: the notebook ran end to end, with the bit-accurate model standing in
for the fabric. Agreement with the golden reference is **true by construction** there
and is **not** a result. What the dry run does establish is that the **notebook's code**
reads the right offsets, decodes signed logits, and uses a register protocol that works
with the RTL's sticky `DONE` bit. A negative control in `tests/test_board_package.py`
shows the mock fails a notebook that polls `DONE`. **Its timing output is discarded and
must not be quoted.**

#### (d) ON HARDWARE — run 2026-09-24, §1.26

| Figure | Status |
|---|---|
| Build-ID read on a board | **`0x23eb56bf`, match** — §1.26 |
| Class agreement on silicon (120 windows) | **120 / 120** — §1.26 |
| Board inference wall-clock | **median 18.07 ms**, CNN path, DMA + inference — §1.26 |
| Power | **NOT RUN** |
| End-to-end latency | **NOT MEASURED** — the DSP front end has never been timed (`docs/frontend_timing_plan.md`) |

Until 2026-09-24 this read *"It has not been run on a board"*, and the rule was that
nothing could claim otherwise until `board/board_run.json` was committed with
`ran_on_hardware: true` and a real `pynq_version`. That file is now committed, and
`tests/test_board_package.py` still rejects one produced by the mock. What may now be
said, and what still may not, is in §1.26.

**A finding recorded here, not fixed.** In the verified RTL, `STATUS.DONE` is cleared
only by a `CTRL.start` write, and the AXI-Stream auto-start path does not clear it. So
after the first window, `DONE` cannot distinguish one inference from the next. The
notebook works around this by polling `BUSY`. Changing it is a v2 RTL change: `rtl/` was
frozen for the board package, and any change there reopens V-1 through V-6.

### 1.22 Technician web app — what it demonstrates, and what it does not

> **Presentation superseded by §1.23** (2026-09-22): no banner, a fleet of fictional
> lifts, product wording. The pipeline facts in (a) and the outcomes in (b) still hold.

`webapp/` (start: `.venv/Scripts/python.exe -m webapp`; guide: `webapp/README.md`;
screenshots: `docs/screenshots/`). **It is a replay of recorded public datasets. It is
not connected to a lift, and it measures nothing live.** Every page carries that banner.

**It introduces no new model metric.** Every figure it shows is read at runtime from the
JSON already registered above: `lobo_summary.json` (§2), the branch results JSON (§3),
and `artifacts/rtl/*` (§1.20, §1.21). `tests/test_webapp.py` asserts that the numbers
match those files and that the engineering view equals `dashboard/panels.py` output.

#### (a) What it demonstrates — measured by the tests or by a script with an output file

| Claim | Value | Source |
|---|---|---|
| Alert packet size | **21 bytes** (`struct "<BHBBBBBbIII"`), bounded at 32 by a test | `drivesentinel/alerts.py`, `tests/test_webapp.py` |
| What it replaces (bearing) | one model input window, 5 × 512 float32 = **10,240 bytes** — ARITHMETIC | `config.py` |
| ADVISORY stage never raises ALARM | both guards tested: `BranchState` caps at Warning; `build_alert` raises | `tests/test_webapp.py` |
| Store-and-forward | packets sent while the endpoint is down stay on disk, survive a sender restart, are resent, and are deduplicated on (unit, seq) | `tests/test_webapp.py` (real HTTP server) |
| Feed = fusion | every replayed alert equals a fresh `BranchState` + `AlertEmitter` replay of the same observations | `tests/test_webapp.py` |
| Supply phase named correctly | FILE 2 and FILE 5: **L2**, from the raw current, independent of the rule's code path; the alerts name L2 | `scripts/webapp/check_supply_phase.py` → `artifacts/webapp/supply_phase_check.json` |
| No winding severity band | largest within-motor \|ρ\| between the stage's signal and recorded severity: **0.34** | `scripts/webapp/measure_winding_severity.py` → `artifacts/webapp/winding_severity_check.json` |
| Layout | all 17 captures render at 1280 px and at 390 px with **0 console errors and no horizontal overflow** | `docs/screenshots/console.json` |

#### (b) Replay outcomes — what the 11 recordings do in the app, under each stage's own held-out protocol

| Recording | Outcome | Protocol behind it |
|---|---|---|
| KA04 (outer race) | **ALARM**, outer race | leave-one-bearing-out fold that held KA04 out |
| KI18 (inner race, added) | **ALARM**, inner race | fold that held KI18 out (28 training bearings, asserted) |
| KI05 (inner race) | **no alert**: the documented miss, labelled *Missed detection* on Replay and explained on About | fold 21, window accuracy 0.0089 (§2) |
| K001 (healthy) | no alert | leave-one-bearing-out |
| Supply FILE 2 / FILE 5 | **ADVISORY**, phase L2, running / stalled | R2 threshold rule, leave-one-motor-out |
| Inverter F2 / F3 / F6 | **ADVISORY**, open circuit / short circuit / overheating, "Suspected location" | V1 block split, held-out positions only |
| Winding inter-turn ramp / inter-coil | **ADVISORY**, but **with the wrong fault type in both alerts**: the inter-turn ramp is called inter-coil and vice versa. Over the whole recording the held-out windows agree with the recorded type 0.5195 / 0.5941 of the time, but the alert fires at t = 1.0 s on windows that lean the wrong way. Shown as it happened, with the note "a wrong call" | V5 leave-one-session-out (held-out rows only); `artifacts/webapp/winding_type_check.json` |

These are demonstration outcomes on single recordings, **not** accuracy figures. The
honest metric for each stage is the one registered in §2–§3, and the app shows it only on
the alert detail page, under *Technical details*.

#### (c) What it does NOT demonstrate

- **Nothing on a lift, and no live sensor.** `LiveSensorSource` is a skeleton that raises.
- **No inverter switch-leg location.** The wording is "Suspected location: inverter power
  stage", because the branch does not resolve the leg.
- **No winding phase and no winding severity band** (see (a)).
- **No end-to-end latency, and no delivery over a real network.** The HTTP POST goes to
  the same machine.
- **The composite lift is not one machine.** Each stage comes from a different test rig
  (the COMPOSITE REPLAY rule above).
- **Not FastAPI.** It is Starlette, which FastAPI is built on. Installing FastAPI needs a
  download, which was not approved in the build session.

**What may be said, exactly:** *"A technician web app replays the public recordings
through the project's fusion and alert code. Each alert is a 21-byte packet with
store-and-forward delivery. The bearing stage raises ALARMs; the other three stages raise
ADVISORY notices only."* Any accuracy figure quoted with it comes from §2–§3, with its
protocol name.

---

### 1.23 Web app, product presentation (2026-09-22): what the pages imply and what is measured

This entry **supersedes §1.22 for presentation.** §1.22(a) still holds for the pipeline: the
21-byte packet, store-and-forward, feed equals fusion, and supply phase L2. The owner asked for
this revision (2026-09-21). It changes **only wording, structure and presentation.** The
fusion, the emitter, the outbox and both ADVISORY-never-ALARM guards are unchanged, and the
tests re-assert all of them. No branch, model, result, RTL file, board file, V-1 run or fusion
floor was touched.

#### (a) What the product pages now present, and what stands behind each

The user-facing pages carry **no provenance banner and no caveats**, by the owner's decision.
Provenance lives in three places: `webapp/README.md` (first paragraph), this entry, and the
Engineering view (footer link, content unchanged).

| On the pages | What it actually is | Status |
|---|---|---|
| A fleet of **4 lifts** ("Lift 07 · Tower B" …), sites and drive tags | **Fictional names.** The 11 recorded units from §1.22(b) are placed into lift slots by `webapp/fleet.py` `LIFTS`. Each unit is used **exactly once**, in a slot of its own stage (asserted). The stages of one lift come from **different public test rigs** (COMPOSITE REPLAY). Empty slots show NO DATA | PRESENTATION |
| "Streaming", live signal strips, "Data 12 s ago" | The recorded stream (`RecordedScenarioSource`) paced by the service. The timestamps are **wall-clock display times**, not acquisition times | PRESENTATION |
| Live monitor "real time" | Observations appear at their **own data-time offsets** (bearing 0.5 s, supply 0.1 s, inverter 1 s, winding 0.5 s), and alerts are those of the precomputed fusion timeline (asserted equal to the feed) | MEASURED behaviour |
| "Winding anomaly detected" (no type) | Decision 2. The type call is **crossed on both units** (§1.22(b)). The code still travels in the packet | WORDING |
| Winding and inverter alerts have no "How bad" field | Decision 1 (winding \|ρ\| ≤ 0.34). Inverter severity is not determined, so the field is **omitted**, not explained | WORDING |
| "Location: inverter power stage" | Decision 3. The switch leg is not resolved | WORDING |
| **Verified bit-exact in simulation** | V-1…V-6: **0** mismatches (§1.20). V-1 was partial at the 2026-09-22 screenshot pass (9,352 of 16,211) and **completed the same day: 16,211 / 16,211, 0 disagreements** | TRUE |
| **Synthesised for Zynq-7020**, resources, timing met, bitstream generated | §1.20 / §1.21 post-route reports; the numbers are read from `artifacts/rtl/*` and asserted | SYNTHESIS-REPORTED |
| Fixed **1,786,724** cycles per verdict, 17.9 ms at 100 MHz | `verify.json` `cycles_max`, one value across V-1/V-3/V-5 (asserted). Simulation cycles, not silicon | SIMULATION |
| **Board deployment: next step** → **Running on PYNQ-Z2** | Changed 2026-09-24 by the owner's decision. The page now reads `board/board_run.json` (and `artifacts/webapp/board_server_check.json`): build ID matched, 120 of 120 verdicts bit-exact on the board, 18.1 ms per verdict, and 600 of 600 bit-exact as the app's bearing engine. It shows "Board deployment: next step" again if no genuine board run is committed; `tests/test_webapp.py` allows board claims on any page only then | MEASURED ON HARDWARE (§1.26, §1.27) |
| Settings: "Drive interface — PYNQ-Z2: **Ready for connection**" / **Connected · bearing engine** | Since 2026-09-24 the row says Connected while the board is the active bearing engine (§1.27), and Ready for connection otherwise. It is never the active data source: the data is the drive stream, and **`LiveSensorSource` (sensors on the drive) is still a skeleton that raises `NotImplementedError`** | ENGINE STATE, LIVE; SENSORS: SEAM ONLY |
| About roadmap: "Advisory stages are promoted to alarm-ready through calibration on fleet data, with no change to the models." | The owner's plan. Promotion requires clearing the pre-registered floor (≥3 independent validation groups **and** macro-F1 ≥ 0.75, §1.1) on data that does not exist yet | ROADMAP, not measured |
| Lift 01 bearing shows NORMAL | TRUE when written: that bearing is **KI05, damaged and missed** by the held-out model (§2). **Since 2026-09-24 it shows an ALARM**, because the web app's bearing stage now runs the deployed network, which was trained on KI05 (§1.28) — an in-sample verdict | MODEL OUTPUT; ALARM since 2026-09-24 is (leaky reference) |

#### (b) Measured by this revision

| Claim | Value | Source |
|---|---|---|
| Sign-in on every page and API | 11 page routes + 13 API routes: unauthenticated → **303 to /signin** / **401**. Page HTML is not reachable under `/static` | `tests/test_webapp.py` |
| Sign-out, open-redirect guard, cookie flags | `ds_session` HttpOnly, SameSite=Lax; `next` accepts local paths only | `tests/test_webapp.py` |
| Banned vocabulary and claims | **0** hits across 10 user page sources, 3 scripts and every user-facing API response. The rendered DOM text of all 46 page captures also has 0 hits. The Engineering view is exempt | `tests/test_webapp.py`, `docs/screenshots/console.json` |
| Layout | 46 captures at 1280 px and 390 px: **0 console errors, 0 horizontal overflow** | `docs/screenshots/console.json` |
| Navigation | back/back/forward returns /monitor → /alerts → /monitor at both widths; active nav item correct; phone menu opens; header sign-out lands on /signin | `console.json` `_navigation_*` |
| Live monitor | Lift 07 bearing **ALARM** "Outer-race bearing damage"; inverter **ADVISORY** "Open circuit in the inverter"; winding **ADVISORY** "Winding anomaly detected", at both widths | `console.json` `05_monitor_*` |

**Not verified by an agent in a browser.** Nobody typed the password into the sign-in form
during verification. The browser sessions used a cookie issued by `POST /signin` with the
demo credentials. `tests/test_webapp.py` asserts that the form posts exactly the field names
the handler reads. One manual click-through of the form is still owed.

**Device channel.** `POST /api/alerts/ingest` needs no session, because a drive's outbox
has none. It accepts only a packet of exactly the wire size, and it is **not otherwise
authenticated**.

**What may be said, exactly:** *"DriveSentinel's technician app presents four example lifts
built from public test-rig recordings. Each is streamed through the project's detection,
fusion and alert code, with sign-in, a live monitor and 21-byte store-and-forward alerts.
The bearing stage raises ALARMs; the other three raise ADVISORY notices."* If anyone asks,
say that the lifts are examples built from recorded public data. **Do not describe them as a
pilot, a deployment or connected lifts, and do not say that the accelerator has run on a
board.**

---

### 1.24 Board-to-web connection (2026-09-22): ready, NOT run on a board

`board/server.py` (new) is the PYNQ-Z2 inference server. `drivesentinel/engines.py` (new)
holds the web app's engines. `drivesentinel/sources.py` `BoardSource` fills the board
data-source stub. Settings gains a **Bearing engine** card. Steps are in `board/SERVER.md`.
**No board was attached. The server has never run on a PYNQ-Z2, and no web-app verdict has
come from an FPGA.**

#### (a) READY — code, tested against the software stand-in for `pynq`

| What | How it is established |
|---|---|
| Server reads the register map of `rtl/ds_defs.vh` | test compares the server's constants with `ds_defs.vh` (a first draft had BUILDID/CYCLES and DONE/IDLE swapped. That was caught by checking against `ds_defs.vh` before any test ran, and the test now pins them) |
| Server refuses to start on a build-ID mismatch (≠ `0x23eb56bf`) | test, mock core with a patched build ID |
| Busy-bit protocol returns fresh results window after window | test, alternating classes against the mock's sticky `DONE` |
| 120 packaged windows over HTTP: class = `golden_reference.predict`, logit registers = RTL model, cycles = 1,786,724 | test. **True by construction behind the mock** (its "core" is the bit-accurate model). It checks the server's code, not the FPGA |
| Software engine = the FPGA's expected registers, register for register | test, `SoftwareEngine` (the bit-accurate model) against `test_windows.npz` `logit_reg` |
| Fallback to software when the board is down; back to the board when it answers | test, real HTTP, mock board started and stopped on the same port |
| A board that answers with the wrong build ID is refused | test, an impostor server |
| The label "FPGA (PYNQ-Z2)" is written only by `BoardClient.infer` after a validated reply | test (runtime and static) |
| With no board, Settings shows **Software** | test, and the browser check |

#### (b) WHAT THE BOARD'S VERDICT DRIVES — a protocol decision, recorded

> **Superseded 2026-09-24 by the owner's decision, §1.28:** the engine's verdict now drives
> the bearing stage on recorded windows too. The reasoning below is why that is labelled
> (leaky reference).

The deployed INT8 network was trained on **all 29 bearings** (`train.train_deployment_model`),
so its verdict on any recorded Paderborn window is **in-sample (leaky reference)**. On the
stored demo windows: KI05 **0.85 (leaky reference)**, against **0.0167** for the held-out fold
model (`artifacts/webapp/deployed_model_in_sample.json`, from
`scripts/webapp/check_deployed_in_sample.py`). So `BoardSource` sends every recorded bearing
window to the active engine, board or software, but keeps the **held-out fold verdict** as
the bearing stage's input. The engine's verdict drives the stage only for windows flagged
`deployment_in_sample = False`, meaning live windows from a machine outside the training set.
There are none today. The brief asked for "uses the returned verdict"; letting it drive
recorded windows would turn KI05, the documented miss, into an in-sample ALARM. That is left
to the owner (`handoff.md` §6.9).

#### (c) NOT RUN when written — all since run, 2026-09-24

| | Status |
|---|---|
| `server.py` on a PYNQ-Z2 | **run** — §1.27 |
| Any verdict labelled "FPGA (PYNQ-Z2)" in the web app from real hardware | **yes**: 960 bearing windows, no fallback — §1.27 |
| Board wall-clock per window, network latency | on-board 18.07 ms (§1.26); HTTP round trip median 40.9 ms (§1.27) |
| `board_run.json` | **committed** — §1.26 |

**What may be said:** *"The web app can route each bearing window to the FPGA on a PYNQ-Z2
through an inference server, and falls back to the bit-identical software model when the
board is absent. Tested against a software stand-in; not yet run on a board."*

**What may not be said** (updated 2026-09-24): that the FPGA's verdict drives the alerts on
recorded data. The server and the web app have since run against the board (§1.27);
until then this line said they had not.

### 1.25 First real board run: `ol.ds_0` is not the register file (2026-09-24)

The notebook (§1.21) was opened on an actual PYNQ-Z2 for the first time. Cell 1 failed:

```
IP blocks: ['axi_dma_0', 'ds_0/s_axi']
...
core = ol.ds_0      # succeeds -- returns a hierarchy proxy, not the register file
...
AttributeError: Could not find IP or hierarchy read in overlay
```

**Cause, most likely.** `board/build_overlay.tcl` adds the accelerator with
`create_bd_cell -type module -reference ds_pynq_wrap ds_0` — a bare RTL module
reference, never packaged as an IP-XACT core (no `package_ip` / `ipx::package_project`
step). PYNQ's overlay parser folded the module's one AXI4-Lite interface into
`ip_dict['ds_0/s_axi']` instead of `ip_dict['ds_0']`, and `ol.ds_0` came back as a bare
hierarchy container with no `.read()`/`.write()` of its own — the register file sits one
level down, at `ol.ds_0.s_axi`. `axi_dma_0`, a properly packaged Xilinx IP, was
unaffected. **This diagnosis is inferred from the traceback and the build script; it has
not been independently confirmed by inspecting the actual `.hwh` on that board.**

**The fix, applied only in Python.** `board/mmio_resolve.py` (new) resolves the core's
registers by trying, in order: the cell's own attribute if it already has read/write, its
child matching whatever `ip_dict` key actually carries the address (`ds_0.s_axi` here),
and finally a raw `pynq.MMIO` built straight from that `ip_dict` entry's `phys_addr` /
`addr_range`, independent of attribute naming. `board/server.py` and the regenerated
`board/drivesentinel_overlay.ipynb` (via `board/make_notebook.py`) both use it instead of
`ol.ds_0` directly. **No RTL, bitstream, or block-design file changed** — this is
Python-side wiring only, exactly as scoped.

**The dry-run mock did not catch this, and now does.** `board/dryrun/pynq/__init__.py`
previously exposed `ol.ds_0` as the register file directly (flat), which is *not* what
the real board did. It has been corrected to reproduce the observed shape —
`ip_dict['ds_0/s_axi']`, `ol.ds_0` a bare proxy — so a notebook or server that assumes a
flat `ol.ds_0` now fails the dry run instead of the board. `tests/test_board_package.py`
pins this shape (`test_mock_replicates_the_observed_hardware_naming_quirk`) so it cannot
quietly regress to being more lenient than the hardware again.

**Status: CONFIRMED ON HARDWARE, 2026-09-24.** On the next board run cell 1 printed
`hierarchies: ['ds_0']` and resolved the core to PYNQ's own `DefaultIP` at `0x43c00000`
(range `0x1000`) — the `ol.ds_0.s_axi` path — and the run completed (§1.26). The
diagnosis in this entry was inferred from the traceback and the build script; the `.hwh`
itself was never inspected on the board, and did not need to be once the fix worked.
Two further board-only failures followed before the run completed; §1.26 lists them.

---

### 1.26 On the board: the accelerator ran on a PYNQ-Z2 (2026-09-24)

`board/drivesentinel_overlay.ipynb` (as committed at `88256a0`) was run top to bottom on
a PYNQ-Z2 by the board owner, and its output file was copied back unedited:
**`board/board_run.json`**. It is rendered in `docs/results_rtl.md` §4c.
`tests/test_board_package.py` checks the committed file says `ran_on_hardware: true`
with a real `pynq_version`, which only the real `pynq` package produces.

It completed after four board-only failures. Each was a difference between the board
and the build machine that the 64-bit dry run could not reproduce by executing code;
each is fixed, and each now has a test (§1.25 for the first):

| # | Failed at | Cause | Fix |
|---|---|---|---|
| 1 | cell 1 | `ol.ds_0` is a hierarchy proxy, registers at `ol.ds_0.s_axi` | `board/mmio_resolve.py` (§1.25) |
| 2 | import of that fix | board Python is **3.6**; `from __future__ import annotations` needs 3.7 | line removed from board-side files |
| 3 | cell 3 | board is **32-bit ARM**; numpy refuses `np.bincount` on int64 | `.astype(np.intp)` |
| 4 | cell 5 | board numpy is **1.13.3**; `np.pad` needs an explicit `mode` before 1.17, and `golden_reference.py` (generated, frozen) omits it | default supplied in notebook section 3 |

A fifth, in `board/server.py` (`http.server.ThreadingHTTPServer` is 3.7+), was found by
reading the file against the board's Python before the server was ever run there, and
fixed with the standard fallback.

#### (a) ON HARDWARE — measured on the board (board run, 120 packaged windows)

| Figure | Value | Protocol |
|---|---|---|
| Build-ID register | **`0x23eb56bf`, match** | read on the board; MAGIC and CONFIG also as built |
| Class vs `golden_reference.predict` (run on the board's ARM) | **120 / 120** | board run, `board/test_windows.npz` |
| Logit registers vs the RTL model's predicted registers | **120 / 120 bit-exact** | board run |
| CYCLES register vs simulation (1,786,724) | **120 / 120 equal** | board run |
| CNN path, DMA + inference, PS wall clock | **median 18.07 ms**, p95 18.11 ms, max 18.32 ms, 360 windows | board run; Python polling STATUS |
| of which the DMA transfer | median 0.209 ms | board run |
| Same INT8 model in numpy on the Cortex-A9 | 40.9 ms per window | board run, **for scale only** |
| Software | PYNQ 2.5, Python 3.6, numpy 1.13.3, armv7l | `board_run.json` |

#### (b) What it is not

- **Not an accuracy figure.** The 120 windows are drawn from all 29 bearings, which the
  deployed network was trained on. The notebook printed 0.975 against ground truth on
  them; that is **(leaky reference)** and is deliberately not in `board_run.json`. The
  model's honest figure stays **macro-F1 0.7852, leave-one-bearing-out** (§2).
- **Not V-1 on silicon.** V-1 is all 16,211 windows, in simulation (§1.20). This is 120.
- **Not end-to-end latency.** The DSP front end is not in the timing:
  `docs/frontend_timing_plan.md`, still **NOT MEASURED**.
- **Not a speed-up claim against the ARM.** The numpy reference is an unoptimised,
  float-requantising implementation; it is there for scale.
- **Not power.** **NOT RUN.**
- **Not the server or the web app.** `board/server.py` shares the notebook's register
  access, protocol and quantiser, but has not itself run on a board (§1.24). No web-app
  verdict has come from an FPGA.

**The date.** The board had no time source (direct cable), so `board_run.json` reads
`date_utc: 2026-08-31T17:37:03Z`. The run was on **2026-09-24**, from this repository's
session record; the file is left unedited because it is the evidence.

**What may be said, exactly:** *"The accelerator has run on a PYNQ-Z2. On 120 real bearing
windows it matched the golden reference and the verified RTL model bit for bit, in a
fixed 1,786,724 cycles; CNN inference measured on the board takes a median 18.07 ms per
window (DMA plus inference). End-to-end latency is not yet measured."*

**What may not be said:** any accuracy from the board run; "real-time on hardware"
(the front end is untimed); "V-1 on silicon"; any power figure; that the web app shows
FPGA verdicts; that anything runs in a lift.

---

### 1.27 The board serves the web app (2026-09-24)

`board/server.py` (as committed at `fc67f0d`) was started on the PYNQ-Z2 by the board
owner (`board/SERVER.md`), and the running web app was pointed at it from Settings. The
record is `artifacts/webapp/board_server_check.json`, from
`scripts/webapp/check_board_server.py`, run on the laptop against the live board and the
live web app.

#### (a) ON HARDWARE — board server, measured

| Figure | Value | Protocol |
|---|---|---|
| Board `/health` | build ID `0x23eb56bf`, MAGIC and CONFIG as built | the server's own identity check |
| Web app, Bearing engine counters | **FPGA 960, Software 0**, board "ready" | read from the running app's `/api/engine`; two drive streams of 480 bearing windows, no fallback |
| Board-server answers vs the bit-accurate software model | **600 / 600** class, **600 / 600** logit registers bit-exact | the web app's path (`BoardClient`, int8 over HTTP); the 480 demo bearing windows + the 120 board test windows |
| CYCLES | always **1,786,724** | as reported per request |
| HTTP round trip, laptop → board → laptop | median **40.9 ms**, p95 44.7 ms, max 50.0 ms | direct cable; upload, FPGA inference, JSON reply |

The web-app counters show that the FPGA served every bearing window; the check shows the
answers were right. The round trip is about 23 ms more than the 18.07 ms the notebook
measured on the board (§1.26): the network, the HTTP handling, and Python on the ARM.

#### (b) What it is not

- **Not a change to any alert, as run.** During this run the bearing stage's status came
  from the held-out fold model and the FPGA's verdict was only counted. The owner has since
  let the engine's verdict drive the stage (§1.28); the alerts that produces are the same
  whichever engine computes it, and are (leaky reference) on this data.
- **Not an accuracy figure** (in-sample windows), **not end-to-end latency** (the DSP front
  end runs on the laptop's recorded data, untimed on the board), **not a lift.**

**What may be said, exactly:** *"The web app can use the FPGA on a PYNQ-Z2 as its bearing
engine: in a live run every bearing window went to the board, and on 600 windows the
board's answers matched the bit-accurate model exactly, at about 41 ms per
window round trip."*

**What may not be said:** that the board's verdict drives the alerts (it does not, for
recorded data); any accuracy from this run; end-to-end latency; that the system runs in a
lift.

---

### 1.28 The deployed network drives the web app's bearing alerts (2026-09-24)

**The owner's decision**, taken after §1.24(b) set out its cost: the web app's bearing
stage now takes its verdict from the deployed INT8 network, through the active engine
(`drivesentinel/sources.py` `BoardSource(verdict="engine")`, set in `webapp/server.py`).
Until then it took the held-out fold model's verdict for recorded windows.

**What changes, in the web app, on the recorded demo data** (the Streamlit dashboard and
every result in this register are unchanged -- they still use the held-out models):

| Bearing | Held-out verdict (before) | Deployed network (now) |
|---|---|---|
| KA04 (outer race) | ALARM, outer race | ALARM, outer race |
| KI18 (inner race) | ALARM, inner race | ALARM, inner race |
| K001 (healthy) | no alert | no alert |
| **KI05 (inner race)** | **no alert — the documented miss** | **ADVISORY, then ALARM, inner race** |

**These bearing alerts are (leaky reference).** The deployed network was trained on all 29
Paderborn bearings, these four included. On KI05's stored windows it is right 85 %
(leaky reference), where the model that never saw KI05 is right 1.67 %
(`artifacts/webapp/deployed_model_in_sample.json`). So Lift 01's ALARM shows what the
network learned about KI05, not that it would catch a KI05 it had never seen. **The honest
figure for a new bearing is unchanged: macro-F1 0.7852, leave-one-bearing-out (§2)**, and
the stage's rating (ALARM-READY) rests on that figure, not on these alerts.

**What does not change.** Software and FPGA are bit-identical, so which engine computes the
verdict never changes an alert: `tests/test_board_server.py` streams the fleet once in
software and once through `board/server.py` and requires identical bearing alerts. The
Why? pages' traces are built from the same verdicts (software engine, no board calls), so
every alert still finds its evidence. The other three stages are untouched.

**What may be said:** *"In the web app the bearing stage runs the network the FPGA
executes; on the demo data it flags all three damaged bearings. That demo data was part of
its training, so this shows the path works, not how well it generalises: on bearings it
has never seen the model scores macro-F1 0.7852."*

**What may not be said:** that the system detects KI05 (or any bearing) it has not seen;
any detection rate from the demo; that the web app's alerts are an evaluation.

### 1.29 The DSP front end on the board, raw signal in (2026-09-24): ready, NOT RUN on a board

**What exists.** `board/frontend.py` ports `drivesentinel/dsp.py` (`process_recording`,
feature set v2) to the board's stack. The filters are designed on the laptop and shipped
in `board/frontend_params.json` (`board/make_frontend_params.py`), so the board needs only
`scipy.signal.sosfilt` and numpy. `board/server.py` gains `POST /process`, which takes 4 s
of raw 64 kHz current ×2 and vibration and runs DSP → quantise → FPGA on the board. The web
app's **Board live** page (`/board`, `webapp/boardlive.py`) streams the raw data files
behind the four bearing lifts to it and shows the signal sent, the spectrum the board
computed, its verdicts, cycles and per-stage timings. For every data file it checks the
board's int8 input against the laptop's `dsp.py` + quantiser, byte for byte, and the FPGA's
logits against the bit-accurate model.

**What is tested, on the laptop** (`tests/test_board_frontend.py`). The port's output equals
`dsp.py`'s: bitwise with `scipy.fft`, and with identical int8 network inputs with the
`numpy.fft` fallback the board uses if its scipy predates 1.4. This holds on a synthetic
signal and on a real test-rig data file. `POST /process` on the mock pynq returns int8
inputs equal to the laptop's and logits equal to the model's. Board live streams through
it, checks every window, raises alerts, and leaks no file name or specimen ID to the page.
All of this runs on the laptop's CPU behind the mock. **It says nothing about the board's
ARM, the board's scipy, or timing.**

| | Status |
|---|---|
| `frontend.py` on the PYNQ-Z2 (does the board's scipy have `sosfilt`?) | **NOT RUN** |
| Board int8 input identical to the laptop's, on the board | **NOT RUN**: `scripts/webapp/check_board_frontend.py` → `artifacts/webapp/board_frontend_check.json` → `docs/results_rtl.md` §4d |
| Board DSP time per 4 s file / amortised per window | **NOT RUN** |
| `docs/frontend_timing_plan.md` §3.3 bar (p95 ≤ 250 ms per window, single-core, pinned) | **NOT RUN**, and the check script's amortised whole-file figure is **not** that protocol. It is reported beside the bar, not as a verdict on it |

**First attempt on the board (2026-09-24).** The server started with the front end
available: Python 3.6.5, numpy 1.13.3, scipy 0.19.1, so the FFT is `numpy.fft` (no
`scipy.fft` before 1.4). One 4 s data file sent to `POST /process` **did not return in
280 s**. Cause: the Hilbert envelope transforms the whole decimated vibration signal,
128,001 samples = 3 × 42,667 with 42,667 prime. numpy 1.13's fftpack has no fast path
for a large prime factor and is O(n × p) there, about 5e9 operations per transform, four
per file. scipy.fft (pocketfft) on the laptop uses Bluestein's algorithm and takes
milliseconds. Fix, in `board/frontend.py` only: on the numpy path, lengths with a prime
factor above 64 go through Bluestein (the same DFT via power-of-two FFTs). On the laptop
that path gives network inputs identical to `dsp.py`'s on 280 / 280 windows from 40 data
files across the four bearings (an ad-hoc check, not a committed artifact;
`tests/test_board_frontend.py` pins the synthetic and one real file). Its board time is
**NOT RUN**.

**Second attempt: the full Lift 07 bearing on the board (2026-09-24), with Bluestein.**
Observed on the Board live page (a screenshot, **not a committed artifact**), 79 data files
= 553 windows, one DSP process:

| | Observed |
|---|---|
| FPGA logits vs the bit-accurate model, on the board's own input | 553 / 553 exact |
| Board int8 input identical to the laptop's | **65 / 553** |
| Board time for the last 4 s file | 5.23 s: decimate 0.71 s, envelopes 3.26 s, spectra 1.06 s, quantise 7.4 ms, FPGA 157 ms (7 windows) — **behind real time** |
| Laptop ⇄ board network for 6.1 MB | 589 ms |

An ad-hoc diagnostic on 5 of those files (35 windows, not committed) located every
differing byte in **channel 5, the raw log spectrum**; the other four channels were
identical. Typically 1 byte of 2,560 per window, 20–60 in four windows, by up to 17
steps; the verdict on the board's input equalled the verdict on the laptop's in 34 / 35.
Cause: numpy 1.13's `geomspace` does not pin its endpoints, so the first band edge was
200.00000000000003 Hz and dropped the exact 200 Hz FFT bin; when that band crossed the
channel's median, the whole channel shifted. Fix: the edges are computed on the laptop and
shipped in `frontend_params.json` (a test forbids `geomspace` in `frontend.py`). And
`server.py --workers 2` (default) splits the two envelopes and the seven windows across the
Zynq's two Cortex-A9 cores, with output identical to one process (tested). **Both fixes are
NOT RUN on the board**: the identity and the time per 4 s after them are unmeasured.

**Third run, both fixes, two workers (2026-09-24).** Observed on the Board live page (a
screenshot, **not a committed artifact**), Lift 07, 28 data files = 196 windows: board input
identical to the laptop's on **195 / 196** (114 of 501,760 bytes differ); verdict on the
laptop's input = the board's on 196 / 196; FPGA exact 196 / 196; board time per 4 s file
2.08–4.44 s in the last dozen (one 7.43 s), "keeps up" 1.02× overall. The one differing
window was in file 30, where the board's shaft-speed estimate read 293 rpm in the stream;
the same file sent to the board again afterwards gave 7 / 7 identical windows at 1499 rpm.
**Not reproduced, cause unknown** -- recorded as observed.

**Since 2026-09-24 the app's bearing stage comes from the board (owner's decision).** When
the board answers with its DSP front end, `webapp/boardlive.py` `BoardFeed` is the source
of every lift's bearing observations -- raw files to `POST /process` in turn, 7 windows
each -- for Fleet, Alerts, Live monitor and the Why? pages; Board live is a view of that
feed, not a second stream. The power supply, inverter and motor winding stages stay on
the laptop. Every page now shows what the stream computed (`service.live_records`),
labelled with the engine and where the DSP ran; the precomputed timelines remain for stage
metadata only. Without the board, the bearing runs on its stored spectra as before
(tested).

**Board live's verdicts and alerts are (leaky reference)**, exactly as §1.28: the deployed
network was trained on these bearings. The page says so in its own words.

**What may be said, once §4d exists:** *"The board takes raw current and vibration and
computes the network's input itself, identical byte for byte to the laptop's on N of N
windows, in X ms per 4 s of signal on its ARM cores."* Until then: *"The board-side signal
processing is written and matches the laptop exactly in testing; it has not yet run on the
board."*

**What may not be said:** that the front end meets the timing plan's bar (a different
protocol); that the board processes live sensor data (the samples are stored test-rig
data; no sensor is wired to the board); any accuracy from Board live.

---

## 2. Bearing pipeline (S5, frozen)

**Cache generation matters for every row.** Rows marked **[rebuilt]** were measured on
the 2026-09-18 rebuilt cache (2,318 recordings / 16,211 windows); rows marked
**[pre-rebuild]** were measured on the earlier cache (2,306 / 16,127) and have not been
re-run. See §1.3.

| Claim | Value | Cache | Protocol | Produced by | Recorded in | Regenerate |
|---|---|---|---|---|---|---|
| **Window accuracy** | **0.7944** | **[rebuilt]** | LOBO, 29 folds, pooled, **single run** | `scripts/02_train_lobo.py` | `artifacts/runs/lobo_summary.json` → `pooled.window_acc` | `python scripts/02_train_lobo.py` |
| **Macro-F1** | **0.7852** | **[rebuilt]** | same | same | same → `pooled.macro_f1` | same |
| **Per-recording accuracy** | **0.8356** | **[rebuilt]** | same | same | same → `pooled.recording_acc` | same |
| **Per-bearing mean** | **0.7945 ± 0.2836** (SEM 0.0527) | **[rebuilt]** | same | same | same → `per_bearing_acc_mean` / `_std` / `_sem` | same |
| Majority baseline | 0.4131 | [rebuilt] | — | same | same → `pooled.majority_baseline` | same |
| Aggregation @ 40 windows | 0.8626 | [pre-rebuild] | LOBO + pooling | same | `docs/prior_results/v5_pre_rebuild/lobo_summary_pre_rebuild.json` | same |
| Window accuracy | 0.8018 ± 0.0103 | [pre-rebuild] | LOBO, 29 folds, pooled, **3 repeats** | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_v2.json` → `summary.window_acc` | `python scripts/experiments/repeat.py` (~3¼ h, no GPU) |
| Macro-F1 | 0.7944 ± 0.0114 | [pre-rebuild] | same | same | same → `summary.macro_f1` | same |
| Per-recording accuracy | 0.8332 ± 0.0056 | [pre-rebuild] | same | same | same → `summary.recording_acc` | same |
| **Run-to-run noise floor** | **± 0.0103** | [pre-rebuild] | 3 repeats, same recipe, different base seeds | same | same → `summary.window_acc.std` | same |
| Pooled window accuracy | 0.7979 | [pre-rebuild] | LOBO, single run | `scripts/02_train_lobo.py` | `docs/prior_results/v5_pre_rebuild/lobo_summary_pre_rebuild.json` | superseded |
| Per-bearing mean | 0.7983 ± 0.2852 | [pre-rebuild] | same | same | same | superseded |
| v1 (2-channel) comparison | 0.7823 | [pre-rebuild] | LOBO, 3 repeats, feature set v1 | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_noval.json` | `DRIVESENTINEL_FEATURE_SET=v1 python scripts/experiments/repeat.py` |
| Accuracy **(leaky reference)** | **0.9865 ± 0.0024** | [pre-rebuild] | stratified 5-fold over **windows**, feature set **v2** | `scripts/experiments/shuffled_benchmark.py` | `artifacts/runs/shuffled_benchmark.json` → `accuracy` | `python scripts/experiments/shuffled_benchmark.py` |
| Macro-F1 **(leaky reference)** | 0.9847 | [pre-rebuild] | same | same | same → `macro_f1` | same |
| Window accuracy **(leaky reference)** | **0.9184** | [pre-rebuild] | shuffled windows, feature set **v1** | `scripts/experiments/task_variants.py:38` | `artifacts/runs/task_variants.json` → `3class_shuffled_LEAKY.window_acc` | `DRIVESENTINEL_FEATURE_SET=v1 python scripts/experiments/task_variants.py` |
| Binary LOBO window accuracy | 0.7982 | [pre-rebuild] | LOBO, binary healthy/damaged | `scripts/experiments/task_variants.py` | `artifacts/runs/task_variants.json` → `binary_lobo.pooled.window_acc` | same |
| Speed-estimate error | median **0.0221 %**, p99 **0.2083 %** | [rebuilt] | current-derived vs tachometer, all recordings | `scripts/01_build_cache.py` | `artifacts/cache_report.json` | `python scripts/01_build_cache.py` |
| INT8 vs float argmax agreement | **99.85 %** (n=2,048) | [rebuilt] | calibration-disjoint windows | `scripts/03_export_int8.py` | `artifacts/int8_export/verification.json` → `agreement.argmax_agreement` | `python scripts/03_export_int8.py` |
| INT8 vs float agreement, full cache | **99.877 %** | [rebuilt] | all 16,211 windows | `scripts/04_verify_golden.py` | `artifacts/int8_export/verification_full.json` | `python scripts/04_verify_golden.py` |
| INT8 max \|logit\| difference | 1.3486 | [rebuilt] | n=2,048 | `scripts/03_export_int8.py` | `artifacts/int8_export/verification.json` | same |

### 2.1 Reconciling 0.9865 and 0.9184

**Both are random window splits. Both are leaky by the same mechanism.** Windows from one
recording — and all 560 windows of one bearing — appear on both sides of the split.

The ~6.8-point gap between them is **not** a protocol difference. It is the feature set:

| | 0.9865 | 0.9184 |
|---|---|---|
| Feature set | **v2 — 5 channels** | **v1 — 2 channels** |
| Split | stratified random over windows | unstratified random permutation |
| Evidence | `artifacts/runs/shuffled_benchmark.json` → `"feature_set": "v2"` | `artifacts/runs/task_variants.log` line 1: `cache (16127, 2, 512) \| feature set v1` |

Corroborated by the LOBO pair measured on the same two feature sets: v2 0.8018 vs v1 0.7823.

Honest presentation:

> Under a random window split — the protocol behind most published Paderborn results — the
> 5-channel model scores 0.9865 ± 0.0024 and the 2-channel model 0.9184. Both are **leaky
> references**. Under leave-one-bearing-out the same 5-channel model scores 0.8018 ± 0.0103
> window / 0.8332 ± 0.0056 per recording. The LOBO figure is the honest one.

### 2.2 Known reporting gaps

| Gap | Status |
|---|---|
| `artifacts/runs/shuffled_benchmark.json` records `feature_set` but **not** `n_seeds` or `tta_shifts`, and there is no `shuffled_benchmark.log`. The 0.9865 headline is therefore not fully reproducible from recorded artefacts. | **OPEN** — re-run with the config dumped into the JSON |
| `artifacts/runs/task_variants.json` records no `train_config` or `feature_set`; the feature set is known only from the `.log`. | **OPEN** — same fix |
| The INT8 calibration and verification windows are drawn from the same cache the deployment model trained on (`scripts/03_export_int8.py:35-38`). Disjointness between calibration and verification is real; neither set is unseen data. | **DECLARED** — the figure quoted is agreement, not accuracy |

---

## 3. Multi-stage branches

| Branch | Dataset | Groups | Honest metric | Fault authority (§1.1) | Status |
|---|---|---|---|---|---|
| S5 `bearing` | D1 Paderborn | 29 bearings | **0.7852 macro-F1** (LOBO, rebuilt cache) | **may raise Fault** (29 ≥ 3 groups, 0.7852 ≥ 0.75) | frozen |
| S4 `winding` | D2 KAIST | **2 sessions scored** (3 days, 1 degenerate) | **0.6250** macro-F1 (V5, leave-one-session-out) | **INDICATIVE** — fails both floor tests | inter_coil vs inter_turn only; `healthy` NOT MEASURABLE |
| S1 `supply` | D4 Thomas | **2 motors** | **1.0000** macro-F1 (R2, threshold rule) | **INDICATIVE** — 2 groups < 3, as predicted | threshold rule, not a learned model |
| S2/S3 `inverter_telemetry` | D3 Bacha | **0 groups** (1 run/condition) | 1.0000 macro-F1 with temperature; **0.8229 electrical-only** | **INDICATIVE** — 0 groups < 3, as predicted | ablation leads; 9-class is qualitative only |
| B-SIM `inverter_waveform` | simulator | — | NOT RUN | out of MVP (§13.8) | NOT RUN |

### 3.1 Required companion measurements

| Measurement | Why | Status |
|---|---|---|
| **B-S4 batch control** — train a classifier to predict the D2 acquisition batch from the same features | The polarity-corrected negative-sequence ratio separates acquisition batches perfectly at 1000 W (batch A max 0.0775 < batch B min 0.0875) and is non-monotonic in severity on 5 of 6 motor × fault-type combinations. If batch is predicted better than fault class, the fault metric is not measuring what it claims. | NOT RUN — **required beside every B-S4 metric** (§13.8) |
| **B-S1 threshold rule** — documented zero-current phase detection + its detection latency | One recording per motor × class makes a learned metric indefensible. The rule is the deliverable; the learned model is the leaky reference. | NOT RUN |
| **B-S2/S3 electrical-only ablation** — OC/SC classification with the temperature channels removed | `over_temp` is separable by a single NTC threshold, so a 4-class headline would be a thermometer in a classifier costume. The electrical-only number is the one that matters. | NOT RUN |
| **D4 label-alignment validation** — reconstructed 1000/500 sliding-window labels vs measured phase-current collapse boundaries | §13.3. If validation fails, the current-collapse rule stands alone and that must be stated. | NOT RUN |

---

### 3.2 B-S4 winding — every number

Task is **inter_coil vs inter_turn**. `healthy` is **NOT MEASURABLE** on this dataset
under any session-aware split (all three healthy recordings are from 2022-01-25) and no
number is reported for it anywhere.

| Claim | Value | Protocol | Recorded in |
|---|---|---|---|
| **V5 accuracy** | **0.6251** | leave-one-session-out, 2 scored folds | `artifacts/multistage/winding/winding_results.json` -> `V5.pooled` |
| **V5 macro-F1** | **0.6250** | same | same |
| V5 majority baseline | 0.5975 | same | same |
| V1 accuracy | 0.3784 | leave-one-motor-out, 3 folds | -> `V1.pooled` |
| V1 macro-F1 | 0.3456 | same | same |
| V1 majority baseline | 0.5020 | same | same |
| Accuracy **(leaky reference)** | 0.9990 | shuffled windows | -> `V3.pooled` |
| **Session-only baseline** | **0.750** | 1-NN on `i0rel_residual`, 1000/1500 W | -> `V4_probe.subsets.1000W_1500W_only` |
| Session from features **(leaky reference)** | 0.9996 | shuffled windows | -> `V4.targets` |

Regenerate: `python scripts/branches/10_winding.py --rebuild`
(~2 min), then `python scripts/60_render_multistage_results.py`.

**How to read these.** V1 (0.3784) is **below its 0.5020
majority baseline** — nothing transfers across motors. V5 (0.6251) is barely
above its 0.5975 baseline, and **below the 0.750 that a single
winding-independent scalar achieves**. Under the reporting rule agreed for this branch, a
fault-class score that does not clearly beat the session-only baseline is **not
demonstrating winding diagnosis**, and V5 does not beat it.

The comparison is directional rather than exact: the session-only probe is per-recording
(n=28) and V5 is per-window (n=10,463), so they are not a like-for-like contest. The
conclusion does not rest on the margin — it rests on V1 sitting below chance and V5
sitting below a scalar that cannot contain winding information.

## 4. Claims that must never appear

| Number | Why not |
|---|---|
| 99.2 % / 0.9922 / 0.9915 as an **accuracy** | `verification_full.json` figures are training-set accuracies — the deployment model trained on all 16,211 windows. Quote `argmax_agreement` instead. |
| 0.9865 or 0.9184 without the **(leaky reference)** label | Both are random window splits over near-identical windows of the same recordings. |
| "Verdict within a single AC cycle" | Replaced by "per-trip verdict with evidence accumulation" (`docs/workflow_v2.md` §12). |
| Any D2 claim of f/f_e **invariance** | f_e ≡ 200.00 Hz across all 48 D2 recordings; the axis is a fixed rescale on this dataset (§13.8). |
| Any D2 **absolute current amplitude** in amps | `NI_SensorSensitivity = 1.0` with the channel sensor declared `Voltage`: the `A` unit is a label on an identity scale. Ratios only. |
| Any D3 claim resting on `VDC`, `IDC` or `VD` | Per-class means span < 0.8 ADC counts against std 1.2–1.4. Dropped in §13.5. |
| A D3 **9-class** accuracy | Three classes have ~8–10 test windows under the block split (§13.8). |
| A D4 **bearing-fault** capability claim | n = 1 motor per class; bearing fault is perfectly confounded with motor identity. Documented negative result only. |
