# DriveSentinel v2 — Multi-Stage Drive Health Monitoring Workflow
### KONE Elevate'26 · PS-06: Real-time drive health monitoring from various-stage voltage & current waveforms

Status: supersedes `DriveSentinel_Integration_Workflow.md`. The existing Paderborn bearing pipeline (`drivesentinel/`, scripts 01–05) is **complete and frozen**. Everything below extends it, and nothing replaces it.

---

## 0. Corrections to the previous workflow

| Previous claim | Correct fact | Action |
|---|---|---|
| Paderborn is an induction lab rig | Hanning SD4CDu8S **8-pole PMSM**, fed by a KEB inverter (see `config.py`) | Fix the wording everywhere, including the slides |
| Bacha inverter dataset = waveforms, so an RMS/THD branch | **10 Hz** telemetry, 8 raw channels, about 5 min of continuous run per condition | Reframe as slow inverter telemetry. There is no THD. |
| Bacha is only on Zenodo (blocked) | Files are mirrored on the author's **GitHub** | `git clone` it |
| Bacha supports leave-one-severity-out | There is no severity axis, only one run per class | Use a contiguous block split with a purge gap |
| `winding_branch` reuses `f_shaft = f_elec/4` | That identity is Paderborn-only. Stator faults sit at **electrical** harmonics. | Use an f_elec-normalised axis |
| Huawei door feeds a classifier | Unlabelled; the target is vibration regression; no V/I | **Dropped** |
| Thomas et al. validates bearing faults | Mains-fed induction motor, **2 motors**; the bearing fault equals motor identity | Use it for supply faults (phase loss) with leave-motor-out. Report bearing only as a confound demo. |
| CORA RDR gives speed-ramp validation | The full download is ~38 GB, it has no voltage channel, and its speed encoder logs at 1/30 Hz | Dropped; speed-ramp validation moves to simulation |
| Fusion combines branch outputs on one machine | No dataset shares a physical machine across branches | Rule-based fusion only; the demo is labelled **composite replay** |

---

## 1. Drive stage map (what "various stages" means in this project)

```
Grid ──▶ [S1 Supply input] ──▶ Rectifier ──▶ [S2 DC link] ──▶ [S3 Inverter] ──▶ [S4 Motor winding] ──▶ [S5 Bearing/mechanical]
          V,I 3-phase                        Vdc, Idc          switch health      3-phase I              I (+vib)
          Thomas et al.                      Bacha (10 Hz)     Bacha + SIM*       KAIST PMSM             Paderborn (done)
```
\*SIM = gym-electric-motor simulation. It is a stretch goal, always labelled `SIMULATED`, and never mixed into real-data metrics.

---

## 2. Dataset roster (all verified freely downloadable)

| ID | Dataset | Stage | Signals / rate | Access link | Paper | Licence note |
|---|---|---|---|---|---|---|
| D1 | Paderborn KAt bearing | S5 | 2× phase I + vib, 64 kHz; PMSM + inverter | https://groups.uni-paderborn.de/kat/BearingDataCenter/ | Lessmeier et al., PHM Europe 2016 | Existing, already in `data/` |
| D2 | KAIST PMSM stator faults | S4 | 3-phase I **100 kHz**, vib 25.6 kHz; 1.0/1.5/3.0 kW; 3000 rpm; ITSC + ICSC, multiple severities; `.tdms` | https://data.mendeley.com/datasets/rgn5brrgrn/5 | Data in Brief — https://pmc.ncbi.nlm.nih.gov/articles/PMC9957734/ | Check the licence shown on the Mendeley page |
| D3 | Bacha PMSM inverter faults | S2/S3 | Ia, Ib, Vdc, Idc, T1–T3, Vd @ **10 Hz**; F0 normal, F1–2 open-circuit, F3–5 short-circuit, F6–8 over-temp | https://github.com/bachaabdelkabir/PMSM-inverter-fault-diagnosis (mirror of https://doi.org/10.5281/zenodo.13974503) | https://doi.org/10.1016/j.dib.2025.111286 | CC-BY-4.0 |
| D4 | Thomas et al. motor fault data | S1 | 3-phase **V + I** + 3-axis vib, **50 kHz**, 20 s/scenario, 10 CSVs; 0.2 kW induction, mains-fed; healthy motor + outer-race-fault motor; phase loss running/at start; 0 / 0.4 / 0.8 Nm | https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219 (DOI https://doi.org/10.6084/m9.figshare.27216219) | https://www.nature.com/articles/s41597-025-05437-3 | **CC BY-NC-ND**: never commit derived data |
| T1 | gym-electric-motor (simulator) | S2/S3 waveforms (stretch) | PMSM + B6 bridge converter | https://github.com/upb-lea/gym-electric-motor | https://arxiv.org/abs/1910.09434 | Open source; output is **simulated** |

**Validation reference to cite:** Liu et al., *Validation Study on Fault Diagnosis of Elevator Traction Drive Systems Based on Public Motor-Drive Data and Multimodal Signal Analysis*, Machines 14(8):903, 2026. https://doi.org/10.3390/machines14080903. It reports Macro-F1 on D4 of 0.9682 under a random split and 0.5856 under a within-label block split. It used only a random split on D3.

**Verified facts about D3 (checked against the live repo, 2026-09-18):**
- Data sits in the repo root, not in the `raw_data/`, `processed_data/` folders its README describes.
- Per-condition Arduino logs `*.txt` use the format `HH:MM:SS.mmm -> 8 ADC ints`; `dataset CSV.csv` holds raw ADC + `FDD` label; `converted_dataset.csv` and `converted_dataset-2.csv` hold converted units (the latter with timestamps).
- Label counts: F0 4295, F1 692, F2 1122, F3 407, F4 341, F5 412, F6 854, F7 1735, F8 1034 (total 10,892).
- `converted_dataset.csv` shows implausible units (VDC about -22 V, temperatures below 0 °C). Start from raw ADC + `Sensor_raw_data_conversion_formulas.pdf` and sanity-check ranges.
- `SETPOINT30/40/50/60.txt` use a different 4-column format. Identify their purpose; exclude from classification unless clearly fault runs.

**Excluded (do not scaffold):**
- **CORA RDR (D5, dropped):** the full download is about 38 GB, dominated by 45-minute stationary tests and thermal images. It also has no voltage channel, and its speed encoder logs at 1/30 Hz, so it cannot provide ground truth for frequency tracking. Speed-ramp validation moves to simulation (see B-SIM and P8).
- Zenodo 15613954 elevator PMSM: restricted files, and its healthy/faulty data is confounded (field vs lab bench).
- Huawei elevator door (Kaggle): unlabelled and has no V/I.
- NPC inverter (IEEE DataPort): gated.
- ziya07 Kaggle elevator dataset: provenance problems.

---

## 3. Download and folder layout

```
data/                       # D1 Paderborn — existing, untouched
data_ext/                   # env DRIVESENTINEL_EXT_DATA (default ./data_ext), gitignored
  kaist_pmsm/               # D2: all .tdms files, flat
  bacha_inverter/           # D3: git clone of the GitHub repo
  thomas_motor/             # D4: 10 CSVs + label .mat from Figshare
```
Manual steps (PowerShell):
```powershell
mkdir data_ext\kaist_pmsm, data_ext\thomas_motor
git clone https://github.com/bachaabdelkabir/PMSM-inverter-fault-diagnosis data_ext\bacha_inverter
# D2, D4: download via browser ("Download All") and extract into the folders above
```
The first job for each dataset is to **inspect the real files** and record the findings in `docs/data_notes_<id>.md`, covering file list, columns, units, sample rate, durations, label mapping and defects. This must happen before any adapter is written. Measured facts override this document.

---

## 4. Common schema (`drivesentinel/common/schema.py`)

```python
@dataclass(frozen=True)
class Recording:
    source: str              # "kaist_pmsm" | "bacha_inverter" | "thomas_motor" | "sim_gem"
    stage: str               # "S1".."S5"
    signals: dict            # name -> np.ndarray (float32, 1-D)
    fs: dict                 # name -> Hz
    label: str               # branch-level class name
    group: str               # physical unit / run id used for splitting (motor id, run id)
    condition: dict          # load, speed, severity, direction...
    simulated: bool = False
```
Adapters are the **only** code that knows about dataset quirks. Branches only ever see `Recording`.

---

## 5. Branch specifications

### B-S5 `bearing` (D1): FROZEN
A thin wrapper `branches/bearing.py` that loads the existing deployment model and calls the existing DSP. It gets no retraining and no behaviour change. Its metrics come from `artifacts/runs/lobo_summary.json`.

### B-S4 `winding` (D2): priority 1
- **Preprocessing:** 3-phase current, anti-alias decimate 100 kHz → 10 kHz, windows 1.0 s / hop 0.5 s.
- **Features:**
  - (a) spectrum on an **electrical-harmonic axis** (f/f_e over 0–20, 512 bins, log-magnitude, median-normalised), with f_e estimated from the spectral peak;
  - (b) negative-sequence ratio |I₂|/|I₁| (Fortescue at f_e);
  - (c) Park's-vector ellipse eccentricity;
  - (d) 3rd-harmonic ratio.
- **Models:** baseline gradient boosting on the scalars plus harmonic band energies. Main model is a small 1-D CNN on the spectrum, with the scalars concatenated before the FC layer.
- **Classes:** `healthy`, `inter_turn`, `inter_coil`, plus a binary healthy/fault view.
- **Validation:**
  - V1 **leave-one-motor-out** (3 folds by kW rating);
  - V2 **lowest-severity holdout** (train on higher severities, test on the lowest; this is the early-warning claim);
  - V3 shuffled windows, reported only as the leaky reference.
- **Report:** macro-F1, per-motor and per-severity accuracy, confusion matrix, and fault-detection recall at the lowest severity.

### B-S1 `supply` (D4): priority 2
- **Preprocessing:** 3-phase V and I, windows of 0.2 s (≥10 cycles), hop 0.1 s.
- **Features:** per-phase RMS, voltage and current unbalance (negative sequence), THD up to the 40th harmonic, V–I phase angle / power factor per phase, crest factor, zero-current phase detection.
- **Classes:** `normal`, `phase_loss_running`, `single_phasing_start`, with load as a condition (not a class). Labels come from the label `.mat` if present; otherwise derive event boundaries from the phase-current collapse and **document the rule**.
- **Validation:**
  - V1 **leave-one-motor-out** (train healthy motor, test fault motor, and the reverse);
  - V2 contiguous block split with a purge gap;
  - V3 random split (leaky reference, to reproduce the Liu et al. gap).
- **Confound demo:** show that a "bearing fault" classifier on D4 cannot be separated from motor identity (n=1 motor per class). Present this as a documented negative result, not as a capability.

### B-S2/S3 `inverter_telemetry` (D3): priority 3
- **Preprocessing:** convert ADC values to physical units using the author's conversion script and formulas PDF, then rolling windows of 5 s (50 samples), hop 1 s.
- **Features:** window mean/std/min/max per channel, Ia–Ib imbalance, Vdc·Idc power, dVdc/dt, dIdc/dt, T-differences, dT/dt.
- **Classes:** a 4-class family (`normal`, `open_circuit`, `short_circuit`, `over_temp`), plus a 9-class location view.
- **Validation:**
  - V1 **contiguous block split** per condition file (first 70% train, last 30% test, purge gap ≥ one window);
  - V2 random split (leaky reference);
  - V3 **ablation without temperature channels** (electrical-only diagnosis of OC/SC).
- **Known limit:** one run per condition, so any result is within-run. State this on the metric card.

### B-SIM `inverter_waveform` (T1): stretch, gated
- **Model:** PMSM + B6 bridge under an elevator trip profile (jerk-limited S-curve speed; accel, cruise, decel; load torque from car/counterweight imbalance).
- **Faults:** only those that can be modelled honestly:
  - single-switch open circuit (force the switch state off);
  - current-sensor gain and offset fault;
  - DC-link degradation, **only if** the supply/DC-link model supports ripple (otherwise skip it and document why).
- **Signals:** Vdc, Idc (if available), phase voltages, and phase currents.
- **Validation:** hold out load and speed setpoints unseen in training.
- **Rules:** every artefact is tagged `simulated=True`, and its metrics live in a separate table.

### R-SIM `speed_tracking` (T1 + D4): stretch, replaces CORA
- Generate elevator S-curve trips in the simulator (accel, cruise, decel) across several speed and load setpoints. Measure `trip.py` f_e(t) tracking error and segmentation accuracy against the simulator's exact ground truth. Tag `SIMULATED`.
- Real-data sanity check: run `trip.py` on the D4 start-up segments. D4 is mains-fed, so the supply frequency is fixed; report segmentation only, never frequency tracking.
- No fault-classifier headline comes from this.

---

## 6. Trip gating (`drivesentinel/trip.py`)

1. Estimate instantaneous electrical frequency f_e(t) from phase current (STFT ridge, refined with Hilbert phase).
2. Segment each trip into `accel`, `cruise` and `decel` using |df_e/dt| with a threshold and a minimum duration.
3. Order-/harmonic-normalised branches (S4, S5) run **only on cruise windows**. Telemetry and supply branches run on all windows.
4. Document the low-speed limit. A gearless sheave turns far slower than the 25 Hz Paderborn shaft, so order resolution at a fixed window length gets coarser. Report resolution against shaft speed as a table, not as a claim of field performance.

---

## 7. Fusion (`drivesentinel/fusion.py`): rule-based, no learning

- **Per branch:** evidence accumulation. Keep a rolling mean of softmax over the last N gated windows (N from config, default 10). Output is `{p_fault, top_class, confidence, n_obs}`.
- **Per-branch status:**
  - `Fault` if p_fault ≥ τ_F for K consecutive updates;
  - `Warning` if p_fault ≥ τ_W;
  - clearing requires p_fault < τ_W − h for K updates (hysteresis).
- **System status:** the worst stage. The evidence string reads like: `"WARNING · S4 winding · 82% · 8 pooled obs · negative-sequence ↑ · branch LOMO macro-F1 0.xx"`.
- A branch whose validated metric is below the configured floor displays `LOW CONFIDENCE BRANCH` and **cannot raise Fault on its own**.
- Unit tests cover threshold crossing, hysteresis, the K-consecutive rule, the low-confidence cap, and worst-stage aggregation.

---

## 8. Dashboard and demo (`dashboard/`, Streamlit)

- **Replay mode only.** Scenarios are pre-built into `artifacts/demo/*.npz` by `scripts/demo/build_scenarios.py`.
- **Out-of-sample rule:** a replayed unit's predictions must come from the fold model that **held that unit out**. The demo never uses a model that trained on the replayed unit.
- **Panels:**
  1. Stage diagram S1–S5 with status lights.
  2. Per-stage waveform view.
  3. Spectrum with markers: BPFO/BPFI for S5, f_e harmonics for S4.
  4. Fused status and evidence string.
  5. Metric cards per branch (protocol name and honest metric, read from results JSON).
  6. Badges: `COMPOSITE REPLAY`, and `SIMULATED` where applicable.
- **Scenarios:**
  - (a) healthy trip → gradual S4 winding fault;
  - (b) S5 bearing fault on a held-out bearing;
  - (c) S1 phase loss on the held-out motor;
  - (d) S3 open circuit (telemetry);
  - (e, stretch) simulated open switch across a full trip with gating visible.

---

## 9. Reporting rules

1. Every number carries its protocol name. Leaky numbers are labelled `(leaky reference)`.
2. Results markdown is **generated from JSON run outputs**, never hand-typed. Anything not run shows `NOT RUN`.
3. No tuning against test folds. Declare the recipe before the run. Any sweep uses an inner grouped split or is reported as selection on test.
4. Measure run-to-run spread (≥3 repeats) before comparing configurations. Differences inside the spread are not reported as improvements.
5. `docs/claims_audit.md` lists every headline number with its protocol, source file and command.
6. Reconcile the existing inconsistency: README 0.9865 (`shuffled_benchmark.py`) vs `accuracy.md` 0.9184 (`task_variants.py`). Explain both; delete neither. **Corrected 2026-09-18:** these are **both** random window splits, and both are leaky by the same mechanism. The gap is the feature set — **v2 (5-channel) 0.9865 vs v1 (2-channel) 0.9184** — not the protocol. Evidence: `shuffled_benchmark.json` records `"feature_set": "v2"`; `task_variants.log` line 1 reads `cache (16127, 2, 512) | feature set v1`. Corroborated by the LOBO pair on the same two feature sets, 0.8018 (v2) vs 0.7823 (v1). See `docs/claims_audit.md` §2.1.

---

## 10. Target repo structure

```
drivesentinel/                  # existing modules unchanged
  common/schema.py  common/splits.py  common/metrics.py  common/signal.py
  adapters/kaist_pmsm.py  bacha_inverter.py  thomas_motor.py
  branches/bearing.py  winding.py  supply.py  inverter_telemetry.py  inverter_sim.py
  trip.py  fusion.py
scripts/
  branches/10_winding.py  20_supply.py  30_inverter_telemetry.py  40_inverter_sim.py
  demo/build_scenarios.py
  60_render_multistage_results.py
dashboard/app.py  panels.py
tests/test_schema.py  test_splits.py  test_winding_features.py  test_supply_features.py  test_fusion.py  test_trip.py
docs/workflow_v2.md  data_notes_<id>.md  results_multistage.md  claims_audit.md
artifacts/multistage/<branch>/  # json + small figures; large arrays gitignored
requirements.txt  requirements-sim.txt  CLAUDE.md
```

---

## 11. Phases, priority and acceptance criteria

| Phase | Deliverable | Done when | Tier |
|---|---|---|---|
| P0 | Recon, CLAUDE.md, gitignore, schema, split utilities | Existing 29 tests pass; split tests prove zero group overlap | MVP |
| P1 | D2 data notes + winding branch | LOMO + severity-holdout JSON written; results rendered | MVP |
| P2 | D4 data notes + supply branch | Leave-motor-out JSON + leaky reference + confound note | MVP |
| P3 | D3 data notes + inverter telemetry | Block-split + ablation JSON | MVP |
| P4 | Trip gating + fusion + tests | Fusion/trip tests pass | MVP |
| P5 | Scenarios + Streamlit dashboard | `streamlit run dashboard/app.py` replays (a)–(d) out-of-sample | MVP |
| P6 | Results, claims audit, README section | Every number traceable | MVP |
| P7 | Simulation branch | Held-out-setpoint JSON, `SIMULATED` everywhere | Stretch |
| P8 | Simulated speed-ramp / trip-gating validation | Tracking-error + segmentation table (SIMULATED) | Stretch |

---

## 12. Pitch fixes (outside the repo)

- Replace "verdict within a single AC cycle" with "per-trip verdict with evidence accumulation".
- Change Paderborn's description to PMSM + inverter.
- Show voltage only where it is real (S1, S2).
- Add an honesty slide (leaky vs honest per branch, plus the Liu et al. citation), a per-bearing chart that shows the failures (KI05, K002), and one INT8 edge-feasibility slide.
- Prepare answers on low sheave speed and speed ramps (section 6).

---

## 13. Decisions after the pre-build audit (2026-09-18)

Recorded after `docs/analysis_report.md`. These supersede anything above that conflicts with
them. Measured facts still override this document (§3), but items 13.1–13.3 are resolved
against the dataset publications and are **not** to be re-litigated from file metadata.

### 13.1 D2 `0_00` is healthy — RESOLVED

The Mendeley file-format description states that `1000W_0_00_current_interturn.tdms` means
"healthy ... motor current data", and defines the `bb_cc` filename token as **percent
severity**, not fault resistance. The `0ohm` / `0mohm` strings in the TDMS root `name`
property are unreliable FlexLogger test names and carry no meaning.

Consequences for B-S4:
- Classes stay `healthy` / `inter_turn` / `inter_coil`, plus the binary view.
- The six `*_0_00_*` files are **three unique recordings** (byte-identical pairs, one per kW
  rating). The adapter must deduplicate them; loading all six double-counts the healthy class.
- Severity is a percentage and is monotonic in fault extent, so the V2 lowest-severity holdout
  remains meaningful in principle — but see 13.8 on the batch confound.

### 13.2 D4 motor/scenario map — RESOLVED

Confirmed against the paper (*Sci Data* 12:1468, Data Records):

| File | Motor | Scenario |
|---|---|---|
| `FILE 1.mat` | healthy | normal, no load |
| `FILE 2.mat` | healthy | phase removal during operation |
| `FILE 3.mat` | healthy | 0.4 Nm load |
| `FILE 4.mat` | healthy | 0.8 Nm load |
| `FILE 5.mat` | healthy | one phase disconnected from startup |
| `FILE 6–10.mat` | **faulty** (SKF 6202-Z, outer ring, drilled holes) | the same five scenarios, in the same order |

This matches the audit's independent inference from file mtimes, 1:1 scenario pairing and
1.6–2.3× higher vibration in files 6–10. Leave-one-motor-out is constructible: 2 groups.

### 13.3 D4 label alignment — reconstruct, then validate

The paper's labels come from a **sliding window of 1000 samples, step 500**, applied to the
ten files **merged in order**, yielding 19,982 labelled windows in 13 categories. Categories
exceed files because a single file spans several states (off, startup, running, fault).

Procedure:
1. Reconstruct the alignment on that basis: window *i* covers merged samples
   `[500i, 500i + 1000)` across the concatenated 10 × 1,000,000 sample stream.
2. **Validate** the reconstructed class boundaries against the phase-current collapse
   boundaries measured in `docs/analysis_report.md` §3.4 (per-file 2 s segment RMS; a phase is
   lost when its short-window RMS falls below 5 % of the median of the other two).
3. If validation fails, fall back to the current-collapse rule and **say so** in
   `docs/data_notes_d4.md`.

**The current-collapse rule is primary either way.** The 13-class vector is a cross-check and
a stretch target, never the sole source of truth.

### 13.4 B-S5 bearing branch — rebuild the models

`artifacts/runs/deployment_model.pt` and the feature caches did not survive the relocation.
Rebuild: `scripts/01_build_cache.py`, then `scripts/02_train_lobo.py`. Save pooled
predictions — the demo's out-of-sample rule (§8) needs per-bearing predictions from the fold
model that held that bearing out.

If the rebuild fails, fall back to `artifacts/int8_export/` + `golden_reference.py` and
**flag the bearing demo panel as in-sample**.

### 13.5 D3 feature set — drop the dead channels

`VDC`, `IDC` and `VD` are dropped from the B-S2/S3 feature set. Across all nine classes their
per-class means span 0.78, 0.45 and 0.26 ADC counts against per-channel std of 1.2–1.4 —
they carry no diagnostic information. This also removes `Vdc·Idc`, `dVdc/dt` and `dIdc/dt`
from §5's feature list.

`Sensor_raw_data_conversion_formulas.pdf` is to be read once a PDF library is installed, and
the divider ratio plus the 2.5 V ACS712 mid-rail offset bug documented in
`docs/data_notes_d3.md` as the reason for the drop.

**Never use `converted_dataset.csv` or `converted_dataset-2.csv`.** Start from raw ADC in
`dataset CSV.csv` (or the per-condition `*.txt`).

### 13.6 Fusion confidence floor — fixed before any branch runs

A branch may raise `Fault` on its own only if **both** hold:

- it has **≥ 3 independent validation groups**, and
- its honest (non-leaky) **macro-F1 ≥ 0.75**.

Otherwise the branch is **indicative**: it may raise `Warning` and contribute evidence, but
never `Fault`.

This floor was fixed **before any multi-stage branch was trained**, and is recorded with its
timestamp in `docs/claims_audit.md`. It must not be changed after seeing branch results.

The dashboard shows **evidence tiers**, and keeps waveform and spectrum panels live for
indicative branches — a greyed status light over a live signal, not a hidden panel.

### 13.7 Documentation corrections

- `docs/accuracy_ceiling.md` — the claim that the exported INT8 weights "are the model these
  numbers describe" is corrected. The weights share the recipe (single model, no ensemble, no
  TTA) but are trained on **all 29 bearings**; 0.8018 is the leave-one-bearing-out estimate of
  that recipe on an unseen bearing.
- `docs/claims_audit.md` — declares the recipe sweeps (`sweep4_v1.json`, `sweep5_v2.json`) as
  **selection on test, inside the measured noise floor**.
- §9 rule 6 of this document is corrected: README 0.9865 and `accuracy.md` 0.9184 are **both**
  random window splits. The gap is feature set **v2 (5-channel) vs v1 (2-channel)**, not
  protocol. Corroborated by the LOBO pair 0.8018 (v2) vs 0.7823 (v1).

### 13.8 Scope cuts and B-S4 framing

Adopted from the audit:

| Item | Decision |
|---|---|
| B-S1 learned classifier | **Dropped.** Documented threshold rule (zero-current phase detection) + leaky reference. One recording per motor × class makes any learned metric indefensible. |
| D3 9-class location view | **Not a reported metric.** Three classes have ~8–10 test windows. Confusion matrix as a qualitative figure only. |
| D2 f/f_e invariance | **Not claimed.** f_e ≡ 200.00 Hz across all 48 files, so the axis is a fixed rescale here. Implement for the API; do not defend it as invariance. |
| P7 / P8 simulation | **Out of the MVP.** |
| B-S4 headline | **Binary healthy/fault leads; 3-class is secondary.** Three unique healthy recordings under a 3-fold group split will not support a 3-class headline. |
| B-S4 batch control | **Required.** Run a "predict the acquisition batch" experiment and report it **beside every B-S4 metric**. If batch is predicted better than fault class, that goes on the metric card. |

D2 adapter obligations (from §3.2 of the audit): read the channel list per file (8 files use
`ai1` instead of `ai2`), detect and correct per-file current-probe polarity by minimising
|I₀|, apply a per-file channel gain calibration for the 3–7 % residual zero-sequence,
normalise the `coil` → `intercoil` token in 8 vibration filenames, and deduplicate the three
`0_00` recordings.

### 13.9 Phase order

**P0 (env + bearing rebuild) → P3 (D3, dress rehearsal) → P1 (D2) → P2 (D4) → P4 → P5 → P6.**

P3 runs before P1 deliberately: D3 is 8 MB and processes in seconds, so it exercises
`schema.py`, `splits.py`, the adapter pattern, the metric-card format and the results renderer
at negligible cost before the same plumbing meets 15 GB of TDMS.

---

## 14. P5 dashboard — as built (2026-09-19)

`streamlit run dashboard/app.py`. Replay only; every number is read from a results
JSON, nothing is computed live, nothing is typed in.

### Panel layout

| # | Panel | What it shows |
|---|---|---|
| 1 | Drive stages S1–S5 | Status light per stage, **tier badge on the stage itself** (FAULT-CAPABLE / INDICATIVE / NOT MEASURED), and the branch's honest metric with its protocol underneath. No legend-only design — a viewer must see that a light can never go red without looking elsewhere. |
| 2 | Signal | Order spectrum per window for bearing scenarios; measured negative-sequence ratio for the winding ramp. |
| 3 | Trip gating | Real D4 start-up segmentation, labelled **envelope-based**, plus the order-resolution table labelled **extrapolation** below 900 rpm. |
| 4 | Fused status | Status, evidence string, and the rolling `p_fault` trace with the thresholds stated. For bearing scenarios it also shows the held-out bearing, how many bearings the fold trained on, and an **OUT-OF-SAMPLE verified** badge. |
| 5 | Metric cards | One per branch, with the floor alongside each figure. The winding card carries V1, V5, the session-only baseline, the leaky reference, and `healthy: NOT MEASURABLE`. |

### Rules the UI enforces rather than assumes

**Out-of-sample.** `panels.assert_out_of_sample()` raises if a bearing scenario is
not marked out-of-sample, or if the replayed bearing appears in its own fold's
training set. `app.py` shows the error instead of the panel. Because
`02_train_lobo.py` keeps only the deployment model, each bearing scenario
**retrains its own fold** (~2.8 min on this CPU) rather than reusing the
all-bearings model, which would make every replay in-sample.

**Authority.** Tier comes from `fusion.load_branch_metrics()`, read from each
branch's results JSON at display time. An INDICATIVE stage keeps live waveform and
spectrum panels and is capped at `Warning`. **No stage is ever hidden** — an empty
stage is more honest than a diagram implying four working branches.

### Scenarios

| # | Scenario | Status |
|---|---|---|
| (a) | S4 `inter_turn` **severity ramp** | Built. **Not** healthy→fault: all three D2 healthy recordings are from one acquisition session, so healthy is not measurable under a session-aware split and is **not synthesised**. |
| (b) | S5 bearing on a **held-out** bearing | Built for `KA04` (works), `K001` (healthy specimen), `KI05` (**fails**, 0.0089). A demo that only shows successes is a worse demo. |
| (c) | S1 D4 phase loss | **SKIPPED** — the B-S1 branch does not exist: no adapter, no threshold rule, no results JSON. |
| (d) | S2/S3 D3 open circuit | **NOT MEASURED** — no `artifacts/multistage/inverter_telemetry/*.json`. |

Scenario NPZs are capped at 120 windows each, evenly spaced across the fold so the
replay spans the whole recording set. Metrics on the cards come from the **full**
fold, not from the stored subset.

### What the dashboard currently cannot show

Two of five stages have no branch behind them, and one of the two that do is
capped at `Warning`. **Exactly one stage — S5 bearing — can raise `Fault`.** That
is the honest state of the project and the dashboard displays it rather than
padding it.
