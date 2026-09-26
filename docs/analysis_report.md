# DriveSentinel v2 — pre-build audit

Read-only audit of `C:\kone` against the build plan, run 2026-09-18.
Nothing was modified, installed, trained or cached. This file is the only one created.

**Note on the plan's location:** the build plan is at `workflow_v2.md` (repo root), **not**
`docs/workflow_v2.md` as the audit brief and the plan's own §10 state. It is untracked
(`git status` shows `?? workflow_v2.md`).

Every claim about code cites `file:line`. Anything I could not execute or confirm is marked
**unverified**.

---

# 1. Repo and environment

## 1.1 Directory tree (depth 3, excluding `data/`, `data_ext/`, `third_party/`, `.git/`, `__pycache__`)

```
.
├── .gitignore
├── README.md
├── pytest.ini
├── requirements.txt
├── workflow_v2.md              <- the plan, untracked, at root not docs/
├── artifacts/
│   ├── cache_report.json
│   ├── folds_lobo.json
│   ├── int8_export/            (14 files: 10 .mem, 3 .json, golden_reference.py)
│   └── runs/                   (7 .json, 8 .log)
├── docs/
│   ├── accuracy.md
│   ├── accuracy_ceiling.md
│   ├── data_notes.md
│   ├── methodology_v3.md
│   ├── results.md
│   └── prior_results/
│       ├── README.md
│       ├── v4/                 (12 files)
│       └── v5_sensor_ablation/ (2 files)
├── drivesentinel/
│   ├── __init__.py  config.py  dataset.py  dsp.py  export.py
│   ├── features.py  folds.py   model.py    quantize.py  train.py
├── scripts/
│   ├── 01_build_cache.py  02_train_lobo.py  03_export_int8.py
│   ├── 04_verify_golden.py  05_render_results.py
│   └── experiments/
│       ├── accuracy_report.py  repeat.py  shuffled_benchmark.py
│       ├── sweep.py  task_variants.py
└── tests/
    ├── __init__.py  test_dsp.py  test_model_export.py
```

Directories the plan's §10 expects but which **do not exist yet**: `drivesentinel/common/`,
`drivesentinel/adapters/`, `drivesentinel/branches/`, `drivesentinel/trip.py`,
`drivesentinel/fusion.py`, `scripts/branches/`, `scripts/demo/`, `dashboard/`,
`artifacts/multistage/`, `requirements-sim.txt`, `CLAUDE.md`. That is expected — P0 has not run.

## 1.2 Modules — purpose and entry point

| Module | Purpose | Entry point |
|---|---|---|
| `drivesentinel/config.py` | Single source of truth for paths, physical constants, DSP + training config | library |
| `drivesentinel/dataset.py` | Indexes Paderborn recordings on disk and loads their channels | library |
| `drivesentinel/dsp.py` | Signal-processing front end, in the order the FPGA implements it | library |
| `drivesentinel/features.py` | Builds the order-spectrum cache: 21 GB `.mat` in, ~150 MB `.npz` out | library |
| `drivesentinel/folds.py` | Leave-one-bearing-out CV + the grouped inner split | library |
| `drivesentinel/model.py` | 1-D CNN over an order spectrum → health verdict | library |
| `drivesentinel/train.py` | LOBO training loop, and the deployment model that ships | library |
| `drivesentinel/quantize.py` | Post-training INT8 PTQ + reference integer arithmetic | library |
| `drivesentinel/export.py` | Writes INT8 weight memories, scale manifest, golden reference | library |
| `scripts/01_build_cache.py` | Build the order-spectrum cache from `data/` | `__main__` |
| `scripts/02_train_lobo.py` | 29-fold LOBO sweep, then the deployment model | `__main__` |
| `scripts/03_export_int8.py` | Quantise the deployment model, write FPGA artefacts | `__main__` |
| `scripts/04_verify_golden.py` | Verify the golden reference against float on the full cache | `__main__` |
| `scripts/05_render_results.py` | Render `lobo_summary.json` → `docs/results.md` | `__main__` |
| `scripts/experiments/accuracy_report.py` | Collect every run into `docs/accuracy.md` | `__main__` |
| `scripts/experiments/repeat.py` | Same config, several base seeds (noise floor) | `__main__` |
| `scripts/experiments/shuffled_benchmark.py` | Stratified 5-fold over **windows** (leaky reference) | `__main__` |
| `scripts/experiments/sweep.py` | Compare training recipes over the full 29-fold protocol | `__main__` |
| `scripts/experiments/task_variants.py` | Binary LOBO + 3-class shuffled leaky reference | `__main__` |

## 1.3 `artifacts/` — everything, with sizes

```
    344  artifacts/cache_report.json
   3929  artifacts/folds_lobo.json
   4937  artifacts/int8_export/golden_reference.py
    356  artifacts/int8_export/input_norm.json
   2880  artifacts/int8_export/layer_0_W.mem        160  layer_0_b.mem
  14336  artifacts/int8_export/layer_1_W.mem        320  layer_1_b.mem
  40960  artifacts/int8_export/layer_2_W.mem        640  layer_2_b.mem
  49152  artifacts/int8_export/layer_3_W.mem        640  layer_3_b.mem
    768  artifacts/int8_export/layer_4_W.mem         30  layer_4_b.mem
   9236  artifacts/int8_export/scales.json
    931  artifacts/int8_export/verification.json
    400  artifacts/int8_export/verification_full.json
   1419  artifacts/runs/export.log
   4470  artifacts/runs/lobo_run.log
  12981  artifacts/runs/lobo_summary.json
   1875  artifacts/runs/repeat_noval.json           9984  repeat_noval.log
   1852  artifacts/runs/repeat_v2.json              9960  repeat_v2.log
   1308  artifacts/runs/shuffled_benchmark.json     (no .log)
  71743  artifacts/runs/sweep4_v1.json             16835  sweep4_v1.log
  28630  artifacts/runs/sweep5_v2.json              6901  sweep5_v2.log
  13929  artifacts/runs/task_variants.json          4017  task_variants.log
    682  artifacts/runs/verify.log
```
Total 379 KB.

**Results JSON present:** `lobo_summary`, `repeat_v2`, `repeat_noval`, `shuffled_benchmark`,
`task_variants`, `sweep4_v1`, `sweep5_v2`, `cache_report`, `folds_lobo`,
`int8_export/verification{,_full}`.

**Absent (gitignored, regenerable — but see §6):**
`artifacts/order_spectra_v2.npz`, `artifacts/order_spectra_v2_meta.parquet`,
`artifacts/order_spectra_v1.npz`, `artifacts/runs/deployment_model.pt`,
`artifacts/runs/pooled_predictions.npz`.

## 1.4 Git

| Item | Value |
|---|---|
| Branch | `main`, ahead of `origin/main` by 1 |
| Remote | `https://github.com/Maadeash-K/kone` (fetch + push) |
| Uncommitted | one untracked file: `workflow_v2.md` |
| Commits | only 2: `7344a43 chore: ignore data_ext and third_party`, `b036525 Final` |

`.gitignore` correctly excludes all three data trees — verified with `git check-ignore -v`:

```
.gitignore:6:data/       data
.gitignore:18:data_ext/  data_ext
.gitignore:21:third_party/  third_party
```

`git status --porcelain` lists **no dataset files**. Clean. ✅

## 1.5 Python environment — **BLOCKING**

| Item | Value |
|---|---|
| Python on PATH | 3.12.5 (`C:\Users\maade\AppData\Local\Programs\Python\Python312\python.exe`) |
| Also installed | 3.13 (the `py` launcher default) |
| Virtualenv | **none anywhere** (`pyvenv.cfg` search on C: and D: found nothing) |
| `pip list` (3.12) | `pip 24.2` — nothing else |
| `pip list` (3.13) | `pip 24.2` — nothing else |
| torch | **not installed** |
| CUDA | **cannot be determined** — no torch to ask |

**Missing from `requirements.txt`: every single package.** `numpy`, `scipy`, `pandas`,
`pyarrow`, `joblib`, `torch`, `pytest` are all absent.

The environment the pipeline was built in did not survive the move. `artifacts/runs/export.log`
line "wrote `D:\kone\artifacts\int8_export`" shows the repo previously lived at `D:\kone`;
`D:\kone` no longer exists and no venv was found there.

Additional packages the plan needs but which `requirements.txt` does not yet list:
`nptdms` (D2), `matplotlib` (figures), `streamlit` (P5), and a gradient-boosting library for
the B-S4/B-S1 baselines (`scikit-learn` or `lightgbm`). `scipy.io.loadmat` covers D4 — the
`.mat` files are MATLAB **v5**, not v7.3, so `h5py` is not needed (verified from the file
headers, §3.4).

## 1.6 Disk and OneDrive

| Item | Value |
|---|---|
| C: free | **37.1 GB** (39,866,642,432 B of 223 GB) |
| D: free | 145.9 GB (156,648,357,888 B of 253 GB) — available if needed |
| Repo path | `C:\kone` |
| OneDrive | exists at `C:\Users\maade\OneDrive`; repo is **outside it** ✅ |

Repo footprint today: `data/` 21 GB, `data_ext/` 16 GB, `third_party/` 14 MB, `artifacts/` 379 KB.

---

# 2. Existing bearing pipeline — verifying the claims

## 2.1 Test suite — **could not run**

```
python -m pytest -q
→ No module named 'pytest'
```

**Unverified.** pytest is not installed, so the plan's P0 gate ("Existing 29 tests pass")
cannot be checked. This is an environment failure, **not** a relocation failure — see §2.2.

What I could verify statically: `tests/test_dsp.py` defines 12 test functions and
`tests/test_model_export.py` defines 14 → 26 functions. Three are parametrised with 2 cases
each (`tests/test_dsp.py:46`, `tests/test_dsp.py:92`, `tests/test_model_export.py:127`), so
pytest would collect **26 + 3 = 29 tests**. The "29 tests" figure is arithmetically consistent.
Whether they *pass* is unverified.

Two of the 29 will likely need data or artefacts. `tests/test_model_export.py` exercises
`.mem` round-trips in a temp dir (self-contained), and `tests/test_dsp.py` builds synthetic
signals. Neither obviously needs `data/`, so I expect a clean run once dependencies are
installed — but that is a prediction, not a measurement.

## 2.2 Hardcoded paths — **clean**

I grepped `drivesentinel/`, `scripts/`, `tests/`, `docs/`, `README.md`, `pytest.ini`,
`requirements.txt` for drive letters, `Desktop`, `/home/`, `/mnt/`, and `../..`.

**Zero hardcoded absolute paths in any code or doc.** Every path derives from `__file__`:

- `drivesentinel/config.py:42` — `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
- `drivesentinel/config.py:43` — `DATA_ROOT = os.path.join(PROJECT_ROOT, "data")`
- `drivesentinel/config.py:44,51-55` — all artifact paths derive from `PROJECT_ROOT`
- `drivesentinel/export.py:131` — `_HERE = os.path.dirname(os.path.abspath(__file__))`
- `scripts/01_build_cache.py:3`, `scripts/02_train_lobo.py:16`, `scripts/03_export_int8.py:14`,
  `scripts/04_verify_golden.py:21` — `sys.path.insert` from `__file__`

The **only** surviving `D:\kone` reference in the whole repo:

- `artifacts/runs/export.log` — `wrote D:\kone\artifacts\int8_export`

That is a historical log line, not executable. **The relocation to `C:\kone` broke nothing in
the code.** The single relocation-adjacent issue is `drivesentinel/dataset.py:175-177`, which
name-checks two specific Paderborn files (`KA04/N09_M07_F10_KA04_{17,18}.mat`) — both are
present, so it is fine.

The plan's `DRIVESENTINEL_EXT_DATA` env var (§3) is **not implemented**; the only env var read
is `DRIVESENTINEL_FEATURE_SET` (`drivesentinel/config.py:144`).

## 2.3 Headline numbers — provenance table

| Number | Where claimed | Protocol | Produced by | Recorded in |
|---|---|---|---|---|
| **0.9865 ± 0.0024** accuracy | `README.md:262` | Stratified 5-fold over **windows**, random split — **leaky** | `scripts/experiments/shuffled_benchmark.py` | `artifacts/runs/shuffled_benchmark.json` → `accuracy` |
| **0.9847** macro-F1 | `README.md:263` | same | same | same → `macro_f1` |
| per-class P/R/F1 (`README.md:269-271`) | same | same | same | same → `per_class` |
| **0.9184** window acc | `docs/accuracy.md:57` | Shuffled windows — **explicitly labelled leaky** | `scripts/experiments/task_variants.py:38` `shuffled_split_reference()` | `artifacts/runs/task_variants.json` → `3class_shuffled_LEAKY.window_acc` (0.9183977…) |
| **0.9131** macro-F1 | `docs/accuracy.md:57` | same | same | same |
| **0.8018 ± 0.0103** window acc | `README.md:283`, `docs/accuracy_ceiling.md:12` | LOBO, 29 folds, pooled, 3 repeats | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_v2.json` → `summary.window_acc.mean/std` |
| **0.7944 ± 0.0114** macro-F1 | `README.md:284` | same | same | same → `summary.macro_f1` |
| **0.8332 ± 0.0056** per-recording | `README.md:285` | same | same | same → `summary.recording_acc` |
| **0.7983** per-bearing mean | (used by plan §5 B-S5) | LOBO single run | `scripts/02_train_lobo.py` | `artifacts/runs/lobo_summary.json` → `per_bearing_acc_mean` |
| **0.7979** pooled window acc | `docs/results.md` | same | same | same → `pooled.window_acc` |
| **0.8626** aggregation @40 | `docs/results.md:90-94` | LOBO + pooling | same | same → `aggregation_curve` |
| **0.022 % / 0.208 %** speed error | `README.md:50` | median / p99, cache build | `scripts/01_build_cache.py` | `artifacts/cache_report.json` → `speed_err_median_pct`, `speed_err_p99_pct` |
| **99.90 %** INT8 argmax agreement | `README.md:309ff`, `docs/results.md:98` | 2,048 held-out-of-calibration windows | `scripts/03_export_int8.py` | `artifacts/int8_export/verification.json` → `agreement.argmax_agreement` |
| **0.7823** (v1) | `docs/accuracy_ceiling.md:30` | LOBO, 3 repeats, feature set **v1**, n_seeds=3, TTA | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_noval.json` |

**Reporting gap:** `artifacts/runs/shuffled_benchmark.json` records `feature_set` but **not**
`n_seeds` or `tta_shifts`, and there is **no `shuffled_benchmark.log`**. The 0.9865 headline
therefore cannot be fully reproduced from recorded artefacts — the model config behind it is
partly unrecorded. This violates the plan's own §9 rule 2. Fix by re-running with the config
dumped into the JSON.

## 2.4 Is the shipped INT8 model the single model the LOBO number describes?

**Partly. The recipe matches; the training data does not.** Three separate facts:

**(a) Single model, no ensemble, no TTA — confirmed.** `artifacts/runs/lobo_summary.json`
records the exact config used:

```json
"train_config": { "n_seeds": 1, "tta_shifts": [0], "epochs": 20,
                  "val_bearings_per_class": 0, "early_stop": false, ... }
```

`artifacts/runs/repeat_v2.json` records the same. So the 0.8018 / 0.7983 numbers are from a
single model with no test-time augmentation. ✅

**(b) The export is forced single regardless of config.** Deciding code,
`drivesentinel/train.py:531-533`:

> ```
> Always a SINGLE model regardless of cfg["n_seeds"] -- the FPGA runs one
> network, so shipping an ensemble would make the exported weights stop
> matching the accuracy that was reported for them.
> ```

✅ — no ensemble or TTA variant is shipped.

**(c) But the shipped model saw every bearing.** `drivesentinel/train.py:523`:

> ```
> Trained on every bearing, no holdout.
> ```

`drivesentinel/train.py:548` calls `inner_split(sorted(set(bearings)), ...)` over **all**
bearings — no test set. `scripts/02_train_lobo.py:92-94` saves that model as
`deployment_model.pt`, and `drivesentinel/export.py:255` defaults to exporting exactly that
checkpoint.

So: 29 LOBO fold models (each trained on 28 bearings) produce the 0.8018 figure; a **30th**
model trained on all 29 bearings is what gets quantised. Same architecture, same
hyperparameters, same single-model recipe — different training set.

**Consequence — a real misstatement in the docs.** `docs/accuracy_ceiling.md:18-19` says:

> "Single model — no ensemble, no test-time augmentation — so the exported INT8 weights are
> the model these numbers describe."

The first clause is true; the conclusion is not. The exported weights are *the same recipe*,
not *the model these numbers describe*. Suggested honest wording: *"the exported INT8 weights
were trained with the identical single-model recipe, on all 29 bearings; 0.8018 is the
leave-one-bearing-out estimate of that recipe's accuracy on an unseen bearing."*

**Related trap:** `artifacts/int8_export/verification_full.json` reports
`float_accuracy: 0.9927`, `int8_accuracy: 0.9920` over all 16,127 windows. Because the
deployment model trained on all of them, **those are training-set accuracies.** Their purpose
is INT8-vs-float agreement (`argmax_agreement: 0.9988`), which is legitimate. The docs get
this right — `README.md:309ff` and `docs/results.md:95-98` quote only the agreement figure,
never 99.2 % as an accuracy claim. ✅ Keep it that way; never let 99.2 % onto a slide.

## 2.5 Leakage hunt

| Vector | Verdict | Evidence |
|---|---|---|
| Scaler / normalisation fit outside the fold | **Clean** | `drivesentinel/train.py:353` — `mean, std = channel_stats(X[i_fit])`, computed on fit indices only, then applied to val/test (`train.py:354-359`). Same pattern in `shuffled_benchmark.py:79` and `task_variants.py:66`. |
| Train/test bearing overlap | **Clean, asserted** | `drivesentinel/train.py:341` — `assert not (set(bearings[is_train]) & set(bearings[is_test])), "BEARING LEAK"` |
| Inner (early-stopping) split leaking windows | **Clean** | `drivesentinel/folds.py:54-100` `inner_split()` holds out whole **bearings**, never windows. Documented at `folds.py:21-31` as the specific v4 bug being fixed. Moot anyway: `val_bearings_per_class: 0` and `early_stop: false` in the shipped config. |
| Windows of one recording split across train/test | **Clean under LOBO** | Folds are per bearing (`folds.py:45-50`); a recording belongs to exactly one bearing, so all 7 of its windows stay on one side. **Deliberately violated** in the two leaky references (`shuffled_benchmark.py:38-47`, `task_variants.py:55-57`) — both labelled. |
| Speed estimation calibrated on test data | **Clean** | `drivesentinel/dsp.py:120-127` — shaft speed comes from the recording's **own** phase current (`estimate_electrical_fundamental`) ÷ `POLE_PAIRS`. The tachometer is used only in `validate_speed_estimate` (`dsp.py:131-142`), documented as "Diagnostic only", called at cache-build time. No label or fold information enters. |
| File / bearing ID reaching the features | **Clean** | `drivesentinel/features.py` writes only spectra; `bearing` and `filename` live in the meta parquet and are consumed solely by `folds.py` and the reporting code. |
| Quantisation calibration | **Acceptable, worth a footnote** | `scripts/03_export_int8.py:35-38` draws `calib_idx` and a disjoint `verify_idx` from the same 16,127-window cache the deployment model trained on. Disjointness is real, but neither set is unseen data. Fine for measuring INT8-vs-float agreement; not an accuracy estimate. |
| Tuning against test folds | **One soft spot** | `artifacts/runs/sweep4_v1.json` / `sweep5_v2.json` are recipe sweeps scored on the LOBO folds themselves. `drivesentinel/config.py:131-137` shows configs being chosen from those results. Under the plan's §9 rule 3 this is *selection on test*. The saving grace is that `config.py:137` and `docs/accuracy_ceiling.md:164-169` explicitly note the differences were inside the noise floor. **Declare this in `claims_audit.md`** rather than leaving it implicit. |

Net: the LOBO pipeline is honestly constructed. I found **no true leakage**. The two things
to fix are documentation claims (§2.4) and disclosure of the sweep-on-test selection.

## 2.6 Reconciling README 0.9865 vs accuracy.md 0.9184

The plan's §9 rule 6 assumes these differ by *protocol*. **They do not — they differ by
feature set.** Decisive evidence, `artifacts/runs/task_variants.log` line 1:

```
cache (16127, 2, 512) | feature set v1 | cfg epochs=20 seeds=3 tta=(-3, 0, 3)
```

versus `artifacts/runs/shuffled_benchmark.json`: `"feature_set": "v2"`.
And `drivesentinel/config.py:151` — `_CHANNELS = {"v1": 2, "v2": 5}`.

| | 0.9865 | 0.9184 |
|---|---|---|
| Script | `shuffled_benchmark.py` | `task_variants.py::shuffled_split_reference` |
| Split | stratified random over windows (`shuffled_benchmark.py:37-47`) | unstratified random permutation (`task_variants.py:55-57`) |
| Feature set | **v2 — 5 channels** | **v1 — 2 channels** |
| Models/fold | `--seeds` default 1 (**not recorded in the JSON**) | 1 |
| TTA | `cfg["tta_shifts"]` (`shuffled_benchmark.py:90`) | none (`task_variants.py:71`) |

Both are random window splits. Both are leaky by exactly the same mechanism. The ~6.8-point
gap is dominated by the v2→v1 feature-set difference, which the LOBO numbers corroborate:
v2 0.8018 vs v1 0.7823 (`repeat_v2.json` vs `repeat_noval.json`).

**How to present both honestly:**

> Under a random window split — the protocol behind most published Paderborn results — the
> 5-channel model scores **0.9865 ± 0.0024** and the 2-channel model **0.9184**. Both are
> **leaky references**: windows from the same recording, and from the same bearing, appear on
> both sides of the split, so they measure recall of bearings the model has already seen.
> Under leave-one-bearing-out the same 5-channel model scores **0.8018 ± 0.0103** window /
> **0.8332 ± 0.0056** per recording. The honest figure is the LOBO one.

Concrete edits:
1. `README.md:258` — retitle "Stratified 5-fold cross-validation" to
   "Stratified 5-fold over windows **(leaky reference)**". The section text already explains
   the leak; the heading does not, and headings are what get screenshotted.
2. `docs/accuracy.md:57` — add "feature set v1 (2-channel)" next to 0.9184, and add the v2
   0.9865 row beside it so the comparison is visible in one place.
3. `workflow_v2.md` §9 rule 6 — correct "stratified 5-fold vs shuffled windows" to
   "v2 5-channel vs v1 2-channel, both shuffled".
4. Re-run `shuffled_benchmark.py` with `n_seeds`/`tta_shifts` dumped into the JSON.

## 2.7 Per-bearing failures

From `artifacts/runs/lobo_summary.json` `folds[]` (also rendered at `docs/results.md:46-58`).
All 29, worst first:

| Bearing | Class | Origin | Window acc | Recording acc |
|---|---|---|---|---|
| **KI05** | inner_race | artificial | **0.0072** | **0.0000** |
| **K002** | healthy | none | **0.0625** | **0.0000** |
| **KA30** | outer_race | real | **0.2411** | 0.1000 |
| **KA22** | outer_race | real | **0.3537** | 0.4375 |
| KA03 | outer_race | artificial | 0.5696 | 0.4750 |
| KI03 | inner_race | artificial | 0.5946 | 0.7375 |
| KI07 | inner_race | artificial | 0.7549 | 0.8375 |
| K005 | healthy | none | 0.7661 | 0.9500 |
| KA15 | outer_race | real | 0.8262 | 0.9750 |
| KI14 | inner_race | real | 0.8655 | 1.0000 |
| KA07 | outer_race | artificial | 0.8714 | 0.9500 |
| KI17 | inner_race | real | 0.8730 | 0.9750 |
| KA05 | outer_race | artificial | 0.8748 | 0.9875 |
| KI21 | inner_race | real | 0.8943 | 1.0000 |
| KI04 | inner_race | real | 0.9000 | 0.9375 |
| KA08 | outer_race | artificial | 0.9094 | 1.0000 |
| K004 | healthy | none | 0.9214 | 1.0000 |
| KA16 | outer_race | real | 0.9589 | 1.0000 |
| KA04 | outer_race | real | 0.9693 | 0.9873 |
| KI08 | inner_race | artificial | 0.9785 | 0.9875 |
| KI16 | inner_race | real | 0.9822 | 1.0000 |
| KA06 | outer_race | artificial | 0.9857 | 1.0000 |
| KA09 | outer_race | artificial | 0.9946 | 1.0000 |
| K001 | healthy | none | 0.9982 | 1.0000 |
| K006 | healthy | none | 0.9982 | 1.0000 |
| KI01 | inner_race | artificial | 0.9982 | 1.0000 |
| K003 | healthy | none | 1.0000 | 1.0000 |
| KA01 | outer_race | artificial | 1.0000 | 1.0000 |
| KI18 | inner_race | real | 1.0000 | 1.0000 |

Pooled confusion (rows = true, cols = healthy / inner / outer):

```
healthy      2658   550   152
inner_race    764  4878   428
outer_race    558   808  5331
```

### Does anything in the data explain them?

I measured raw-channel statistics directly from `data/` — three recordings per bearing at the
common operating point `N09_M07_F10`, channels `vibration_1` and `phase_current_1/2` at 64 kHz
(parsed with a stdlib MATLAB-v5 reader, no scipy available):

| Bearing | LOBO acc | vib RMS | vib peak-to-peak | **crest (p2p/RMS)** |
|---|---|---|---|---|
| K001 | 0.998 | 0.362 | 9.44 | 26.1 |
| **K002** | **0.063** | 0.178 | 7.80 | **43.9** |
| K003 | 1.000 | 0.334 | 8.87 | 26.5 |
| K004 | 0.921 | 0.106 | 2.97 | 27.9 |
| K005 | 0.766 | 0.114 | 2.64 | 23.2 |
| K006 | 0.998 | 0.358 | 9.01 | 25.2 |
| **KI05** | **0.007** | 0.177 | 7.57 | **42.9** |
| KI01 | 0.998 | 0.414 | 9.70 | 23.4 |
| KI03 | 0.595 | 0.108 | 2.13 | 19.7 |
| KI04 | 0.900 | 0.207 | 6.70 | 32.3 |
| KI07 | 0.755 | 0.187 | 6.85 | 36.7 |
| KI08 | 0.979 | 0.196 | 6.48 | 33.1 |
| KI14 | 0.866 | 0.114 | 2.68 | 23.5 |
| KI16 | 0.982 | 0.168 | 3.10 | 18.5 |
| KI17 | 0.873 | 0.119 | 2.49 | 21.0 |
| KI18 | 1.000 | 0.161 | 3.50 | 21.7 |
| KI21 | 0.894 | 0.140 | 2.99 | 21.4 |

Phase-current RMS is essentially identical across all 32 bearings (1.61–1.75 A, ratio
I1/I2 = 0.992–0.996), so **current amplitude explains nothing**. Two things in the vibration
data do point somewhere:

1. **K002 has the highest crest factor of any healthy bearing (43.9 vs 23–28 for the other
   five).** A high crest factor with moderate RMS is the textbook signature of a localised
   defect. K002's vibration genuinely *looks* impulsive, i.e. damaged. That is a plausible
   physical reason the model calls it damaged — not a modelling bug.
2. **KI05 has the highest crest factor of the inner-race group (42.9 vs 18–37).** This one
   cuts the other way and is *not* explained: an unusually impulsive inner-race bearing should
   be easier, not harder.

**Caveat, stated plainly:** the model does not see RMS or crest factor. It sees a
5-channel, 512-bin order spectrum over 0–16 shaft orders (`drivesentinel/config.py:105-109`).
These broadband statistics are suggestive, not decisive. The definitive test — comparing
BPFI/BPFO band energy in the model's own order spectra for KI05 and K002 against their class
siblings — needs the feature cache rebuilt and is **unverified**.

Also worth noting: the three healthy bearings with *high* vibration RMS (K001 0.362,
K003 0.334, K006 0.358) all score ≥0.998, while the three with low RMS (K002 0.178,
K004 0.106, K005 0.114) score 0.063 / 0.921 / 0.766. That ordering suggests the model has
partly learned "broadband vibration energy ⇒ healthy" from a sample of six specimens. With
n=6 healthy bearings that is exactly the overfitting-to-specimen-identity that
`docs/accuracy_ceiling.md:164-169` already warns about. It is worth one plot.

`docs/accuracy_ceiling.md:159-162` already lists "understand the four failing specimens" as
the highest-value next step. The measurements above are a partial start on K002.

---

# 3. Data readiness

## 3.0 Summary

| Set | Path | Files | Size | Extensions | Incomplete downloads |
|---|---|---|---|---|---|
| D1 Paderborn | `data/` | 2,624 | 21 GB | 2,560 `.mat`, 64 `.pdf` | none |
| D2 KAIST | `data_ext/kaist_pmsm/` | 96 | 15 GB | 96 `.tdms` | none |
| D3 Bacha | `data_ext/bacha_inverter/` | 30 | 8.2 MB | 15 `.txt`, 3 `.csv`, 3 `.pdf`, 3 `.png`, 2 `.py`, 2 `.ino`, 1 `.tex`, 1 `.md` | none |
| D4 Thomas | `data_ext/thomas_motor/` | 11 | 687 MB | 11 `.mat` | none |
| T1 simulator | `third_party/gym-electric-motor/` | — | 14 MB | source tree | none |

No `.zip`, `.part`, `.crdownload` or `.tmp` anywhere. ✅

Two deviations from the plan's §3 layout:
- D2 is **not flat** — it is split into `current/` (48 files) and `vibration/` (48 files).
- D3 is **not a git clone** — `data_ext/bacha_inverter/.git` does not exist; the files were
  copied. Provenance is therefore not recorded in-tree. `third_party/gym-electric-motor` is
  likewise a copy, not a clone (no `.git`).

## 3.1 D1 — Paderborn (`data/`)

32 bearing folders × 80 `.mat` + 2 `.pdf` each. 29 used; KB23/KB24/KB27 excluded
(`README.md:348`).

One file, `KI05/N09_M07_F10_KI05_1.mat`:

| Property | Value |
|---|---|
| Format | MATLAB 5.0, `Platform: PCWIN`, created 2015-03-26 |
| Structure | one struct named after the file, fields `Info`, `X`, `Y`, `Description` |
| Channels (`Y`, 7) | `force`, `phase_current_1`, `phase_current_2`, `speed`, `temp_2_bearing_module`, `torque`, `vibration_1` |
| Fast raster | 256,081 samples @ 64 kHz = **4.001 s** (`phase_current_1/2`, `vibration_1`) |
| Mech raster | 16,005 samples @ 4 kHz (`force`, `speed`, `torque`), raster string `Mech_4kHz` |
| Temperature | 5 samples per recording |
| dtype | float64 throughout |
| Stop condition | `Time limit: 4.000000` |

Measured ranges across 32 bearings: speed 899.3–900.1 rpm (nominal 900), torque 1.16–1.27 Nm
(nominal 0.7 Nm setting `M07`… the channel reads mechanical load), bearing-module temperature
29.3–54.4 °C, phase current RMS 1.61–1.75 A. **All physically plausible.** ✅

One outlier worth a glance: KI18 runs at 29.3 °C while every other bearing sits at 41–54 °C.
KI18 scores 1.0000, so it is not causing a failure, but it is a 12 °C gap from the next
coldest and hints at a different test session.

`drivesentinel/config.py:74` records the machine correctly: `POLE_PAIRS = 4 # Hanning
SD4CDu8S, 8-pole PMSM`, matching the plan's §0 correction. ✅

## 3.2 D2 — KAIST PMSM (`data_ext/kaist_pmsm/`) — the most important section

### Full parsed grid

All 96 filenames parse. Severity appears as `0_68` **and** `1.01` — and the two formats appear
in the *same* logical recording: `current/1000W_1.01_current_intercoil.tdms` vs
`vibration/1000W_1_01_vibration_intercoil.tdms`. A parser must accept both and normalise
(replace `_` with `.` in the severity token).

**96 files = 48 recordings × 2 modalities.** The complete grid, 3 ratings × 2 fault types ×
8 severities = 48:

| Rating | Fault type | Severities (from filename) |
|---|---|---|
| 1000 W | intercoil | 0.00, 0.68, 0.81, 1.01, 1.34, 2.00, 3.93, 7.56 |
| 1000 W | interturn | 0.00, 2.26, 2.70, 3.35, 4.41, 6.48, 12.17, 21.69 |
| 1500 W | intercoil | 0.00, 4.79, 5.70, 7.02, 9.15, 13.12, 23.20, 37.66 |
| 1500 W | interturn | 0.00, 1.57, 1.88, 2.34, 3.10, 4.57, 8.74, 16.08 |
| 3000 W | intercoil | 0.00, 2.49, 2.98, 3.69, 4.86, 7.12, 13.30, 23.48 |
| 3000 W | interturn | 0.00, 1.78, 2.13, 2.65, 3.50, 5.16, 9.81, 17.86 |

Count per combination: exactly **1 current file + 1 vibration file**, for every cell. No cell
has more than one run.

**Fault-type token is inconsistent.** In `vibration/`, the 3000 W intercoil files use
`_vibration_coil` (8 files), not `_vibration_intercoil`. Token census:
- `current/`: `intercoil` 24, `interturn` 24
- `vibration/`: `intercoil` 16, `interturn` 24, **`coil` 8**

A naive `split('_')[-1]` parser creates a phantom fourth class. **Normalise `coil` → `intercoil`.**

### One file read in full: `current/1000W_0_68_current_intercoil.tdms`

| Property | Value |
|---|---|
| Size | 293.3 MB, TDMS version 4713, 371 segments |
| Producer | NI FlexLogger 2021 R1.1, chassis cDAQ1, module `cDAQ1Mod2` = **NI 9775** |
| Groups | `/`, `/'Test Information'`, `/'Log'` |
| Channels | `/'Log'/'cDAQ1Mod2/ai0'`, `ai2`, `ai3` — **3 phase currents** |
| dtype | float64 |
| `wf_increment` | 9.999999999999972e-06 → **fs = 100.0 kHz** ✅ |
| Samples/channel | 12,219,901 → **122.2 s** |
| `unit_string` | `A` |
| `DAC~Channel~Type` | `Current`; `DAC~Channel~Sensor` = `Voltage` |
| Range | `LowRange -10.0` / `HighRange 10.0` |
| Root `name` prop | `1kw_6ohm_current_coil` |

Vibration files: 1 channel (`cDAQ1Mod1/ai0` or `ai3`), float64, `wf_increment` → **25.6 kHz**,
`unit_string` = `g`, ~3.1 M samples ≈ **121 s**. (Note: vibration segments contain ~121
*chunks* each; a reader that assumes one chunk per segment will report 1.3 s instead of 121 s.
`nptdms` handles this correctly.)

Measured values, first 1 s of `1000W_0_68`: ai0 min −6.09 / max 6.77, ai2 −7.09 / 7.20,
ai3 −7.63 / 7.77. Whole-file RMS 0.49–0.60.

**Plausibility flag — amplitude scale is not trustworthy.** `NI_SensorSensitivity = 1.0` and
`NI_SensorScale~ScaleType = 'InverseLinear'` with `Y-Intercept = 0.0`, and the channel sensor
is declared `Voltage`. The `A` unit is therefore a label on an identity scale: these are
**probe output volts, not amps**, and the probe's V/A factor is not in the file. An RMS of
0.5 "A" for a 1 kW PMSM at 3000 rpm is not physical. **Any absolute-amplitude feature on D2 is
meaningless.** The plan's B-S4 features are all ratios (negative-sequence ratio, ellipse
eccentricity, 3rd-harmonic ratio, median-normalised spectrum) so they survive this — that is
lucky, and it should be a stated design constraint, not an accident.

### Fundamental frequency

Measured by DFT peak: **f_e = 200.00 Hz exactly**, in every file checked. At the documented
3000 rpm that is 4 pole pairs / 8 poles — the same pole count as the Paderborn machine
(`config.py:74`). Convenient, and it means `POLE_PAIRS = 4` transfers.

**But:** f_e is a *constant* across all 48 recordings. The plan's "electrical-harmonic axis
(f/f_e), with f_e estimated from the spectral peak" is therefore a fixed rescaling that adds
no invariance on this dataset. It is still worth implementing for the API and for the
simulator branch, but it buys nothing for D2 accuracy. Do not claim it does.

### ⚠️ Finding 1 — the three "healthy" recordings are duplicates

`*_0_00_*_intercoil` and `*_0_00_*_interturn` are **byte-identical** within each rating.
Verified by size + MD5 of the first 4 MB:

```
current/1000W_0_00_current_intercoil.tdms    288031292  6f2b7967fd353488
current/1000W_0_00_current_interturn.tdms    288031292  6f2b7967fd353488   <- identical
current/1500W_0_00_current_intercoil.tdms    288031293  4ead7984449a9a16
current/1500W_0_00_current_interturn.tdms    288031293  4ead7984449a9a16   <- identical
current/3000W_0_00_current_intercoil.tdms    288031293  7a5c558e81c1823e
current/3000W_0_00_current_interturn.tdms    288031293  7a5c558e81c1823e   <- identical
(and the same for all six vibration counterparts)
```

So there are **45 unique recordings, not 48**, and **3 healthy recordings, not 6** — one per
motor. Loading all 96 files naively double-counts the healthy class.

### ⚠️ Finding 2 — "is `0_00` healthy?" — probably, but the file metadata contradicts it

The audit brief asks to confirm from the data. Here is what the data says, both ways.

**For healthy:**
- The file is shared between both fault-type names — exactly what a common baseline looks like.
- At 1000 W interturn the polarity-corrected negative-sequence ratio rises monotonically with
  severity and `0.00` sits at the bottom: 0.00→**0.0236**, 3.35→0.0433, 6.48→0.0526,
  21.69→0.0775.
- In the `0_00` files the 200 Hz fundamental carries **96.9 %** of the signal power, versus
  62–72 % in the batch-B fault files — a markedly cleaner waveform.

**Against healthy:**
- The TDMS root `name` property reads `1kW_0mohm_Current` (1000 W), `1.5kW_0ohm_current`
  (1500 W). A **0 Ω fault path is a dead short — maximum severity**, not healthy.
- The severity number tracks fault current, which scales as 1/R. Measured mapping for
  1000 W intercoil: 6 Ω→0.68, 5 Ω→0.81, 4 Ω→1.01, 3 Ω→1.34, 2 Ω→2.00, 1 Ω→3.93, 0.5 Ω→7.56.
  Halving R doubles the number (3 Ω→1.34 vs 6 Ω→0.68 ✓; 1 Ω→3.93 vs 4 Ω→1.01 ✓). Extrapolating
  to R=0 gives **∞**, not 0.00. So "0.00" more likely means "not measured" than "no fault".
- At 1500 W and 3000 W, `0.00` does **not** sit at the bottom of the unbalance ordering.

**Also: the root names are demonstrably unreliable.** `3000W_0_00_current_*` has root
`3kW_0.5ohm_current` — the *same* root as `3000W_17_86_current_interturn`. And several roots
read `0.5mohm` / `2mohm` where the severity ordering requires `0.5 Ω` / `2 Ω`, i.e. a stray
"m". These are FlexLogger test names, copy-pasted between runs.

**Verdict: unresolved from the files alone.** My reading is that `0_00` *is* the shared healthy
baseline — the duplication across both fault types is hard to explain otherwise, and a dataset
described in Data in Brief as healthy + ITSC + ICSC must contain healthy data. But the
evidence is only consistent at 1000 W. **This needs confirmation against the paper
(PMC9957734) before B-S4 is built.** It is question 1 in §6.

### ⚠️ Finding 3 — channel identity is not constant across files

Most files use `ai0, ai2, ai3`. But:
- The eight `3000W_*_current_intercoil.tdms` files use **`ai0, ai3, ai1`** — a different
  channel set (`ai1`, not `ai2`).
- `1500W_16_08_current_interturn.tdms` writes its segments in the order `ai0, ai3, ai2`.

An adapter that hardcodes `ai0/ai2/ai3` silently drops a phase on 8 of 48 recordings. **Read
the channel list per file; never assume it.**

### ⚠️ Finding 4 — per-file current-probe polarity varies, and it dominates the physics feature

Channel timestamps confirm the three phases **are** simultaneous (`wf_start_time` identical,
`wf_start_offset = 0.0` for all three), so phase relationships are meaningful. But taking
channels at face value gives nonsense: zero-sequence |I₀|/|I₊| ≈ 1.0 in most files, which is
impossible for a three-wire machine.

Searching the 8 sign combinations to minimise |I₀| resolves it exactly — three distinct probe
wirings are present:

| Polarity of (ch1, ch2, ch3) | Files |
|---|---|
| `(+,+,+)` | 1000 W roots `1kW_*`; all 3000 W intercoil `Current_*` |
| `(+,+,−)` | 1000 W roots `1kw_*_coil` / `1kw_*_interturn` |
| `(+,−,−)` | all 1500 W; all 3000 W interturn |

After correction |I₀|/|I₊| drops to 0.006–0.079 — physical.

**This is the single most dangerous thing in D2.** The uncorrected negative-sequence ratio —
the plan's headline physics feature for B-S4 — separates *acquisition batch*, not fault
severity. At 1000 W the two batches do not even overlap:

| 1000 W, polarity-corrected unbalance |I₋|/|I₊| |
|---|
| Batch A (roots `1kW_<R>ohm_current`): 0.0236, 0.0258, 0.0261, 0.0353, 0.0433, 0.0526, 0.0775 |
| Batch B (roots `1kw_<R>ohm_current_coil/interturn`): 0.0875, 0.0899, 0.0923, 0.0925, 0.0936, 0.1014, 0.1086, 0.1117 |

Batch A max 0.0775 < batch B min 0.0875. A single threshold separates them perfectly. The
batches also differ in duration (A: exactly 120.0 s; B: 122–149 s) and segment count, and
**batch alternates with severity** (A holds R ∈ {0, 4, 2, 0.5} Ω, B holds R ∈ {6, 5, 3, 1} Ω).

Worse, the severity trend only behaves at 1000 W interturn. At 1500 W interturn it runs
**backwards**: 1.57→0.0299, 1.88→0.0234, 3.10→0.0190, 8.74→0.0096. At 1500 W and 3000 W
intercoil it is flat.

So: even with polarity corrected, negative-sequence ratio is **not monotonic in severity** on
5 of 6 (rating × fault type) combinations. Treat the "early-warning at lowest severity" claim
as unproven until measured, not as a design assumption.

Residual |I₀|/|I₊| after correction is still 0.03–0.07 (should be ~0 for a three-wire machine),
implying per-channel gain mismatch of a few percent. A per-file gain calibration — scale each
channel to make |I₀| minimal — would be a cheap, defensible normalisation. Recommend it.

## 3.3 D3 — Bacha inverter (`data_ext/bacha_inverter/`)

### Inventory and `*.txt` → F-label mapping, by row count

The plan's expected counts match **exactly**. (`wc -l` undercounts by 1 where the file has no
trailing newline; the `dataset CSV.csv` label census below is authoritative.)

| F-label | File | Expected | `dataset CSV.csv` census | Match |
|---|---|---|---|---|
| F0 normal | `NORMAL_OP.txt` | 4295 | 4295 | ✅ |
| F1 open-circuit | `HB2_HIGH_SIDE_OC.txt` | 692 | 692 | ✅ |
| F2 open-circuit | `HB3_LOW_SIDE_OC.txt` | 1122 | 1122 | ✅ |
| F3 short-circuit | `HB1_LOW_SIDE_SC.txt` | 407 | 407 | ✅ |
| F4 short-circuit | `HB2_HIGH_SIDE_SC.txt` | 341 | 341 | ✅ |
| F5 short-circuit | `HB3_HIGH_SIDE_SC.txt` | 412 | 412 | ✅ |
| F6 over-temp | `HB1_OVER_TEMP.txt` | 854 | 854 | ✅ |
| F7 over-temp | `HB1&2_OVER_TEMP.txt` | 1735 | 1735 | ✅ |
| F8 over-temp | `HB3_OVER_TEMP.txt` | 1034 | 1034 | ✅ |
| | | **10,892** | **10,892** | ✅ |

Raw format confirmed: `HH:MM:SS.mmm -> 8 space-separated ints`. Timestamp deltas are
100–104 ms → **10 Hz** ✅. `NORMAL_OP.txt` spans 13:24:31 → 13:31:49 = 7 min 18 s.
ADC is **10-bit (0–1023)**, all channels biased near mid-rail ~500.

### `SETPOINT30/40/50/60.txt` — identified: thermistor calibration, not fault runs

Different 4-column format, e.g. `12:45:07.736 -> 30.00  255.00  26.00  509`. Columns are
**[setpoint °C, heater PWM duty 0–255, measured °C, raw ADC]**. Evidence:
- `SETPOINT30.txt` and `temperature_data.txt` are **byte-identical** (same MD5).
- `thermistor_calib_data.txt` (3067 rows) is the concatenation of the four setpoint runs
  (906 + 640 + 808 + 709 = 3063).
- The calibration pairs are internally consistent and give a usable NTC curve:
  ADC 509 ↔ 26 °C, ADC 464 ↔ 30 °C, ADC 360 ↔ 40.5 °C, ADC 211 ↔ 60 °C.
- Note the filename typo: `SETPONT50.txt`, and its first column reads 40.00 not 50.00.

**Exclude all of `SETPOINT*`, `temperature_data.txt`, `thermistor_calib_data.txt` from
classification** — they are exactly what the plan suspected. But **keep them**: they are the
correct source for the temperature conversion (below).

### Units — you are right, `converted_dataset.csv` is wrong

Per-channel raw ADC statistics over all 10,892 rows of `dataset CSV.csv`:

| Channel | min | max | mean | std | distinct values |
|---|---|---|---|---|---|
| Ia | 350 | 651 | 503.61 | 72.87 | 281 |
| Ib | 355 | 647 | 489.44 | 79.99 | 285 |
| **VDC** | **494** | **518** | **507.46** | **1.22** | **24** |
| **IDC** | **488** | **517** | **506.52** | **1.36** | **27** |
| T1 | 281 | 554 | 468.02 | 66.17 | 165 |
| T2 | 357 | 542 | 479.34 | 33.29 | 169 |
| T3 | 317 | 533 | 467.51 | 31.87 | 195 |
| **VD** | **494** | **523** | **509.33** | **1.17** | **28** |

The shipped `data_conversion_script_v4.py` applies `adc_to_voltage_acs712_100ohm()` to both
`VDC` and `VD`. That function subtracts the ACS712 **2.5 V bipolar mid-rail offset** and
divides by 100 mV/A before multiplying by 100 Ω:

```python
def adc_to_voltage_acs712_100ohm(adc_value):
    voltage = (adc_value / 1023) * 5
    current = (voltage - 2.5) / 0.100
    return current * 100
```

That offset is correct for a bipolar current sensor and wrong for a DC-bus measurement.
Verified: ADC 507 → −21.994 V, exactly the value in `converted_dataset.csv`. The gain is
4.888 V **per ADC count**, so with VDC spanning only 24 counts the whole DC-link channel is
quantised into 24 steps of ~5 V. **Do not use `converted_dataset.csv`.**

Temperature is worse: `converted_dataset.csv` shows T1 = −7.24 °C for raw ADC 515, a value
I **could not reproduce** from `dataset CSV.csv` with any formula in the shipped script.
`converted_dataset.csv` appears to be the output of an earlier, different script.

`converted_dataset-2.csv` is better but has its own trap:
- Its temperatures **are** correct Steinhart–Hart: ADC 515 → 10.577 °C, which I reproduced
  exactly with the script's `adc_to_temperature_ntc()` (A=1.2666e-3, B=2.3661e-4, C=9.6094e-8,
  R1=10 kΩ). ✅
- But its **`Ia`, `Ib`, `IDC` columns contain raw ADC** (532.0, 388.0, 506.0), while the
  converted values live in `Ia_original`, `Ib_original`, `IDC_original`. Header/content
  mismatch.
- `VDC` and `VD` are still wrong (−21.99 V), same bug.

**Recommended conversion, derived here:**

| Channel | Source | Formula |
|---|---|---|
| Ia, Ib, IDC | ACS712-20A, bipolar | `((adc/1023)*5 − 2.5)/0.100` A — the script's own function, correct |
| T1, T2, T3 | 10k NTC | Steinhart–Hart per the script, **cross-checked** against the calibration pairs in `thermistor_calib_data.txt` |
| **VDC, VD** | voltage divider | **Do not subtract 2.5 V.** Use `(adc/1023)*5 * divider_ratio`. With ratio ≈10 this gives ≈24.8 V, a plausible DC link. The exact ratio is in `Sensor_raw_data_conversion_formulas.pdf`, which I could not read (no PDF library installed) — **unverified**, confirm before use. |

### The finding that matters most for B-S2/S3

Per-class mean ADC, all nine classes:

| FDD | n | Ia | Ib | **VDC** | **IDC** | T1 | T2 | T3 | **VD** |
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

**VDC, IDC and VD are dead channels.** Across all nine classes their means span 0.78, 0.45 and
0.26 ADC counts respectively — well inside their own noise (std 1.2–1.4). They carry
**no diagnostic information whatsoever.**

**The temperature channels are perfect and trivial.** T1 collapses for F6 (HB1 over-temp), T1
and T2 for F7 (HB1&2), T3 for F8 (HB3) — exactly matching the filenames, with the NTC falling
as temperature rises. Physically consistent ✅, and a thermal threshold classifies F6/F7/F8
with no machine learning at all.

**Ia/Ib carry the only OC/SC signal, and it is weak.** F3 (HB1 low-side SC) is clear
(453/542 vs F0's 515/475). F1 (HB2 high-side OC) is nearly indistinguishable from F0.

Plausibility: an inverter DC link that never moves while the motor runs is itself suspicious —
either the divider is badly scaled (likely, given the VDC bug) or the bus is stiff enough that
a 10 Hz sampler sees nothing. Either way the channel is unusable as delivered.

`README.md` of the dataset confirms: three-phase MOSFET (IRF540N) inverter driving a PMSM
converted from a DENSO alternator, constant 10 rad/s, ~5 min per condition, ACS712-20A current
sensors, 10k NTC temperature. Only **Ia and Ib** are measured — there is no Ic.

## 3.4 D4 — Thomas motor (`data_ext/thomas_motor/`)

### Format — MATLAB, as you said, not the Figshare CSVs

| Property | Value |
|---|---|
| Files | `FILE 1.mat` … `FILE 10.mat`, each **72,000,184 B**, plus `LABEL DATASET.mat` (160,112 B) |
| Format | MATLAB **5.0**, `Platform: posix`, created 2025-04-23 (data) / 2025-04-26 (labels) |
| Variable | `data`, class double, dims **[1000000, 9]**, column-major, uncompressed |
| Labels | struct `data` with one field `label`, class **int64**, dims **[1, 19982]**, values **1–13** |

72,000,184 = 184-byte prefix + 9,000,000 × 8 B. ✅ Matches your expectation exactly.

### Column identification — confirmed from the physics

| Col | Content | Evidence |
|---|---|---|
| 0 | vibration x (g) | ±0.045, zero mean |
| 1 | vibration y (g) | ±0.060, zero mean |
| 2 | **vibration Z (g)** | mean **1.0044** — that is **gravity**, so Z is the vertical axis ✅ |
| 3–5 | I1, I2, I3 (A) | ±2 A, RMS 0.28–0.72 |
| 6–8 | V1, V2, V3 (V) | ±396 V, RMS 90–277 |

**Sample rate verified, not assumed.** Taking a 4 s slice of V1 from `FILE 3.mat` and testing
frequencies on the hypothesis fs = 50 kHz gives a dominant peak at **exactly 50.00 Hz**, with
3rd/5th/7th harmonics at 0.2–0.5 % of the fundamental. **fs = 50 kHz confirmed**, 1,000,000
samples = **20.0 s** per file ✅, clean mains sinusoid.

Plausibility: V1 peak 383 V → 271 V RMS, currents 0.28–0.72 A RMS. 3 × 271 × 0.28 ≈ 228 VA for
a 0.2 kW motor — plausible ✅. Nothing impossible.

### Which files belong to which motor — **recoverable, and I recovered it**

Per-file RMS (full 20 s):

| File | vib_x | vib_y | vib_z | I1 | I2 | I3 | V1 | V2 | V3 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0123 | 0.0155 | 1.0045 | 0.277 | 0.279 | 0.287 | 210.9 | 213.4 | 213.8 |
| 2 | 0.0173 | 0.0152 | 0.9461 | 0.382 | **0.177** | 0.385 | 245.3 | 222.9 | 274.9 |
| 3 | 0.0189 | 0.0242 | 1.0066 | 0.312 | 0.316 | 0.329 | 271.5 | 275.0 | 275.4 |
| 4 | 0.0198 | 0.0264 | 1.0067 | 0.508 | 0.515 | 0.528 | 270.3 | 273.9 | 274.3 |
| 5 | **0.0008** | 0.0006 | 1.0002 | 0.840 | **0.012** | 0.837 | 102.3 | 100.6 | 202.6 |
| 6 | 0.0203 | 0.0181 | 1.0045 | 0.392 | 0.400 | 0.406 | 203.6 | 206.1 | 206.2 |
| 7 | 0.0408 | 0.0251 | 1.0074 | 0.764 | **0.402** | 0.770 | 237.5 | 237.1 | 275.7 |
| 8 | 0.0301 | 0.0272 | 1.0079 | 0.479 | 0.491 | 0.498 | 273.8 | 277.1 | 277.1 |
| 9 | 0.0384 | 0.0382 | 1.0018 | 0.699 | 0.712 | 0.720 | 272.5 | 275.9 | 275.9 |
| 10 | **0.0012** | 0.0008 | 1.0002 | 0.741 | **0.012** | 0.739 | 92.9 | 90.7 | 183.2 |

Per-file current RMS in ten 2 s sub-segments makes the scenarios unambiguous:

| File | Scenario | Signature |
|---|---|---|
| 1 | **start-up, light load** | I≈0.014 (off) for 6 s, inrush 0.63, settles 0.255 |
| 6 | **start-up, heavier load** | off for 8 s, inrush 0.77, settles 0.44 |
| 2 | **phase loss while running** | runs at 0.253; **I2 → 0.012** for 8 s while I1/I3 rise to 0.47; recovers |
| 7 | **phase loss while running, higher load** | runs at 0.41; I2 → 0.012; I1/I3 rise to 1.14; recovers |
| 5 | **single-phasing at start** | I2 = 0.012 throughout; I1/I3 ramp to 1.28 then decay; **vib_x 0.0008 → the motor never rotates**; V1≈V2≈102 V, V3≈203 V (winding divider) |
| 10 | **single-phasing at start** | same, **vib_x 0.0012 → not rotating** |
| 3, 4 | **normal running**, two load levels | flat 0.315 and 0.515, balanced |
| 8, 9 | **normal running**, two load levels | flat 0.49 and 0.71, balanced |

**Healthy motor vs faulty-bearing motor — three independent lines of evidence agree:**

1. **File mtimes form two blocks.** FILE 6–10 created 21:23:48–21:24:08; FILE 1–5 created
   21:24:12–21:24:28. Two contiguous sessions of five.
2. **Scenarios pair 1:1** across the blocks: 1↔6 (start-up), 2↔7 (phase loss running),
   3/4↔8/9 (steady load), 5↔10 (single-phasing at start).
3. **Vibration is systematically 1.6–2.3× higher in 6–10** for every rotating pair:
   1→0.0123 vs 6→0.0203; 2→0.0173 vs 7→0.0408; 3→0.0189 vs 8→0.0301; 4→0.0198 vs 9→0.0384.
   Exactly what an outer-race defect does, and absent in the two non-rotating files (5, 10).

**Conclusion: FILE 1–5 = healthy motor; FILE 6–10 = outer-race-fault motor.** This is an
inference from three converging signals, not a label in the files — but it is a strong one and
I would build on it. **Confirm against the paper before it reaches a slide.**

### Label alignment — **cannot be recovered from the files**

`LABEL DATASET.mat` contains a single flat int64 vector, 19,982 entries, values 1–13, in
**13 contiguous runs** — one run per class, no interleaving:

| Label | start | length | end |
|---|---|---|---|
| 1 | 0 | 1698 | 1698 |
| 2 | 1698 | **99** | 1797 |
| 3 | 1797 | 2197 | 3994 |
| 4 | 3994 | 799 | 4793 |
| 5 | 4793 | 1999 | 6792 |
| 6 | 6792 | 1999 | 8791 |
| 7 | 8791 | 1999 | 10790 |
| 8 | 10790 | **99** | 10889 |
| 9 | 10889 | 2037 | 12926 |
| 10 | 12926 | 1059 | 13985 |
| 11 | 13985 | 1999 | 15984 |
| 12 | 15984 | 1999 | 17983 |
| 13 | 17983 | 1999 | 19982 |

**What one label covers:** 10 files × 1,000,000 samples ÷ 19,982 labels = **500.4 samples per
label ≈ 10 ms at 50 kHz**. A 500-sample non-overlapping window over 1,000,000 samples gives
2000 windows/file, 20,000 total — 18 more than we have. So the intended windowing is almost
certainly **500 samples (10 ms), hop 500, ~1998 per file**.

**Why it still cannot be recovered:**
- There is **no file index** in the label file. The vector is flat.
- There are **no class names** anywhere — only integers 1–13. `LABEL DATASET.mat` contains
  exactly one struct with exactly one field (`label`); I enumerated it.
- The run boundaries do **not** land on consistent per-file multiples. At ~1998/file the file
  boundaries would be 1998, 3996, 5994, … Observed boundaries include 1698, 1797, 3994, 4793,
  6792, 8791, 10790 — some near a boundary (3994 vs 3996) and some nowhere near (1698, 4793).
- Six runs are exactly 1999 long (labels 5, 6, 7, 11, 12, 13), which would mean six
  single-class files. But only **four** files (3, 4, 8, 9) are eventless from the signal
  evidence above. The two models contradict each other.

**Stating it clearly, as the brief asks: the mapping from the 13 label integers to the paper's
named scenarios, and the alignment of the label vector to specific files and sample offsets,
cannot be recovered from the files alone.** Both require the paper
(*Sci Data* 12, 2025, s41597-025-05437-3) or the authors.

**Practical consequence:** the 13-class label vector is **not usable** as delivered. But the
B-S1 branch does not need it. The plan's §5 B-S1 already provides the fallback: *"otherwise
derive event boundaries from the phase-current collapse and document the rule."* The
phase-current evidence above gives clean, reproducible boundaries for all three planned
classes (`normal`, `phase_loss_running`, `single_phasing_start`) plus start-up. **Take that
path, and treat the 13-class vector as a stretch item pending the paper.**

---

# 4. Spec versus reality, branch by branch

## 4.1 B-S4 `winding` (D2) — priority 1

### Group counts — the real numbers

| Quantity | Plan assumes | Reality |
|---|---|---|
| Motors (LOMO groups) | 3 (by kW) | **3** ✅ — but 3 folds is the floor for a group split |
| Recordings | 48 | **45 unique** (3 healthy duplicated across fault-type names) |
| Healthy recordings | implied ≥1/motor | **1 per motor, 3 total** |
| Fault recordings | — | 42 (7 intercoil + 7 interturn per motor) |
| Runs per (motor, type, severity) | — | **exactly 1** — no repeat runs anywhere |
| Severity levels | multiple | 7 non-zero per (motor, fault type) |

Windows at the planned 1.0 s / 0.5 s hop over ~120 s: **≈239 per recording**.

Per-fold class balance under leave-one-motor-out (train on 2 motors):

| Split | healthy | inter_coil | inter_turn |
|---|---|---|---|
| Train (2 motors) | **2 recordings** (478 windows) | 14 (3,346) | 14 (3,346) |
| Test (1 motor) | **1 recording** (239 windows) | 7 (1,673) | 7 (1,673) |

Healthy is **6.7 %** of training windows and comes from **two continuous recordings**.

### Are the planned features computable?

| Feature | Computable? | Note |
|---|---|---|
| 3-phase current for Fortescue | **Yes** — 3 channels present, verified time-aligned | but channel *names* vary per file (§3.2 Finding 3) and **polarity varies per file** (Finding 4) |
| Negative-sequence ratio \|I₂\|/\|I₁\| | Yes | **but not monotonic in severity on 5 of 6 motor×type combinations** |
| Park's-vector ellipse eccentricity | Yes | same polarity dependency — a sign flip turns a circle into a line |
| 3rd-harmonic ratio | Yes | scale-invariant, unaffected by the amplitude problem |
| Electrical-harmonic axis (f/f_e) | Yes | but **f_e ≡ 200 Hz for all 48 files**, so it is a fixed rescale that adds no invariance here |
| Absolute RMS / current magnitude | **No** — meaningless | `NI_SensorSensitivity = 1.0`: values are probe volts, not amps |

### Sample rate and window length

- Decimation 100 kHz → 10 kHz: Nyquist 5,000 Hz. Plan wants f/f_e up to order 20 = **4,000 Hz**.
  Fits, with 20 % margin ✅. The anti-alias filter must be genuinely steep — 4 kHz to 5 kHz is a
  narrow transition band. `drivesentinel/dsp.py:171-181` `decimate_signal()` already uses an
  order-8 Chebyshev-I zero-phase decimator and is directly reusable.
- Window 1.0 s → **1 Hz resolution = 0.005 orders** at f_e = 200 Hz. Ample ✅ (the existing
  bearing pipeline runs at 0.031 orders/bin, `config.py:156`).
- 120 s per recording at hop 0.5 s → 239 windows ✅ plenty.

### Biggest technical risk, and what I would change

**Risk: acquisition-batch signature, not fault severity, is what the model will learn.**
§3.2 Finding 4 shows negative-sequence ratio separates batch A from batch B perfectly at
1000 W, and batch correlates with which resistance values were used. The 3-fold LOMO split
does not protect against this — each motor has its own polarity and its own batch mix, so the
model must transfer across *both* the fault and the wiring, and will likely fail at both
without being able to tell you which.

**Changes I would make:**
1. **Detect and correct per-file channel polarity in the adapter**, by minimising |I₀|. Log the
   detected sign vector per file and put the table in `docs/data_notes_d2.md`. Non-negotiable.
2. **Add a per-file channel gain calibration** (scale to minimise residual |I₀|), since 3–7 %
   residual zero-sequence remains after sign correction.
3. **Deduplicate the healthy files** — load the three `0_00` recordings once, not six times.
4. **Add a batch covariate** (derived from the root-name style and duration) and report a
   "predict the batch" control experiment. If a model predicts acquisition batch better than it
   predicts fault class, say so on the metric card. This single control is worth more than any
   accuracy number here.
5. **Demote the 3-class framing.** With 3 healthy recordings and one run per cell, `healthy`
   vs `inter_turn` vs `inter_coil` under leave-one-motor-out rests on **n=3 healthy specimens**.
   Lead with the **binary healthy/fault** view and report 3-class as secondary. The bearing
   pipeline already learned this lesson at n=6 (`docs/accuracy_ceiling.md:60-64`).
6. **Do not build V2 (lowest-severity holdout) as an early-warning claim.** The lowest
   severities are concentrated in batch B, so holding them out holds out a batch. Run it, report
   it, and label it "confounded with acquisition batch" — that is honest and still interesting.
7. **Resolve the `0_00` question before writing the adapter** (§6 question 1). If `0_00` is a
   dead short rather than healthy, **there is no healthy class in D2** and the branch must be
   reframed as severity regression or fault-type discrimination only. This is the single
   decision that most changes the shape of P1.

**Verdict on the plan: survives contact, but the "early warning at lowest severity" headline
does not.** Everything else is implementable.

## 4.2 B-S1 `supply` (D4) — priority 2

### Group counts

| Quantity | Plan assumes | Reality |
|---|---|---|
| Motors (LOMO groups) | 2 | **2** ✅ (FILE 1–5 / FILE 6–10, inferred — §3.4) |
| Folds | 2 | 2 — the minimum possible |
| Recordings per class per motor | — | **exactly 1** |
| Labels from `.mat` | "if present" | present but **unusable** (§3.4) |

Windows at 0.2 s / 0.1 s hop over 20 s → **199 per file**. Per fold:

| Class | Train (1 motor) | Test (1 motor) |
|---|---|---|
| normal | 2 files ≈ 398 windows | 2 files ≈ 398 |
| phase_loss_running | 1 file, ~80 windows in the event | 1 file |
| single_phasing_start | 1 file ≈ 199 | 1 file |
| (start-up) | 1 file | 1 file |

**One recording per class per motor.** Every class is a single continuous 20 s capture.

### Are the planned features computable?

| Feature | Computable? |
|---|---|
| Per-phase V and I RMS | **Yes** — 3 V + 3 I channels, all present ✅ |
| Voltage/current unbalance (negative sequence) | **Yes** — genuine 3-phase set |
| THD to the 40th harmonic | **Yes** — 40 × 50 Hz = 2 kHz vs 25 kHz Nyquist, huge margin ✅ |
| V–I phase angle / power factor per phase | **Yes** ✅ |
| Crest factor | Yes |
| Zero-current phase detection | **Yes, and trivially reliable** — the dead phase reads 0.012 A against 0.4–1.3 A ✅ |

Window length 0.2 s = **10 mains cycles → 5 Hz resolution**. The plan's "≥10 cycles" is met
exactly ✅. For 40th-harmonic THD, 5 Hz resolution against 50 Hz spacing is comfortable.

D4 is the **only** dataset in the roster with real voltage channels. The plan's §12 pitch fix
("show voltage only where it is real") is correct and D4 is where.

### Biggest technical risk, and what I would change

**Risk: class and recording are the same thing, so there is nothing to generalise over.** With
one file per (motor × class), leave-one-motor-out means **training on exactly one example per
class**. Any classifier that scores well is separating two 20 s captures, not learning phase
loss. Liu et al.'s 0.9682 → 0.5856 collapse under a block split is the published version of
this same problem, on this same dataset.

**Changes I would make:**
1. **Do not build a learned classifier here at all.** Zero-current phase detection is a
   *threshold*: I2 = 0.012 A vs 0.4–1.3 A is three orders of magnitude, unambiguous in every
   affected file. Ship a documented rule, report its exact detection latency, and present the
   learned model as the leaky comparison. This is both more honest and more defensible under
   questioning than a 2-fold macro-F1.
2. **Derive labels from the current collapse, not from `LABEL DATASET.mat`.** Rule:
   a phase is "lost" when its 0.2 s RMS falls below 5 % of the median of the other two. That
   rule reproduces every event I measured. Document it in `docs/data_notes_d4.md` with the
   per-file segment table from §3.4.
3. **Distinguish `phase_loss_running` from `single_phasing_start` by rotation, not by current.**
   vib_x = 0.0008–0.0012 in FILE 5/10 versus 0.017–0.041 in FILE 2/7 is a 20× separation: the
   motor is stalled. That is a genuinely useful, physically-grounded discriminator and a good
   demo moment.
4. **Keep the confound demo, and use my numbers.** The bearing-fault/motor-identity confound is
   now quantified: vibration is 1.6–2.3× higher across every matched pair, with n=1 motor per
   class. That is a clean, specific negative result — much stronger than asserting it.
5. **Be careful with absolute voltage RMS as a feature.** It ranges 91–277 V across files,
   mostly because the start-up files include a de-energised period. It will separate files
   trivially and for the wrong reason. Prefer unbalance ratios and per-phase relative measures.

**Verdict: the plan's three classes are derivable and its features are all computable, but the
learned-classifier framing cannot produce a defensible number.** Reframe as a rule + a leaky
reference.

## 4.3 B-S2/S3 `inverter_telemetry` (D3) — priority 3

### Group counts

| Quantity | Plan assumes | Reality |
|---|---|---|
| Runs per condition | 1 (plan already says so) | **1** ✅ correctly anticipated |
| Group axis | none available | **none** — confirmed |
| Samples | 10,892 | 10,892 ✅ |

Windows at 5 s (50 samples) / 1 s hop (10 samples):

| Class | Samples | Windows | Test windows at 70/30 with a 1-window purge |
|---|---|---|---|
| F0 | 4295 | ~425 | ~127 |
| F1 | 692 | ~65 | ~19 |
| F2 | 1122 | ~108 | ~32 |
| F3 | 407 | ~36 | **~10** |
| F4 | 341 | ~30 | **~8** |
| F5 | 412 | ~37 | **~10** |
| F6 | 854 | ~81 | ~24 |
| F7 | 1735 | ~169 | ~50 |
| F8 | 1034 | ~99 | ~29 |

**Three classes get fewer than 11 test windows.** A per-class accuracy on 8 samples has a
95 % CI roughly ±35 points. The 9-class location view is not measurable.

### Are the planned features computable?

| Feature | Computable? | Useful? |
|---|---|---|
| Window mean/std/min/max per channel | Yes | for Ia, Ib, T1–T3 only |
| Ia–Ib imbalance | Yes | **the only OC/SC signal**, and weak |
| Vdc·Idc power | Yes | **worthless** — both channels are constant (§3.3) |
| dVdc/dt, dIdc/dt | Yes | **worthless and worse** — VDC has std 1.22 counts over 24 distinct values; a derivative of that is pure quantisation noise |
| T-differences, dT/dt | Yes | **excellent** — cleanly separates F6/F7/F8 |

Sample rate for harmonics: **10 Hz, Nyquist 5 Hz.** No harmonic analysis of any kind is
possible. The plan already says "There is no THD" ✅ — correct.

### Biggest technical risk, and what I would change

**Risk: the 4-class result will be a temperature thermometer wearing a classifier costume.**
`over_temp` (F6–F8, 3,623 samples = 33 % of the data) is separable by a single NTC threshold.
`normal` vs `open_circuit` vs `short_circuit` rests entirely on two 10 Hz current channels,
and F1 is nearly indistinguishable from F0 by mean. A headline 4-class accuracy will look
strong and mean almost nothing.

**Changes I would make:**
1. **Promote the plan's V3 ablation to the headline.** "Electrical-only OC/SC detection" is the
   question that matters for a drive-health story. Report it first, whatever it says. My
   prediction from the class means is that it will be weak — **and that is the finding**.
2. **Drop `Vdc·Idc`, `dVdc/dt`, `dIdc/dt` and the `VD` channel from the feature set**, and say
   why in the metric card: they are constant to within their own quantisation noise. Keeping
   them adds three ways to overfit run-specific drift.
3. **Do not ship the 9-class location view as a metric.** Three classes have ~8–10 test windows.
   Show the confusion matrix as a qualitative figure and report only the 4-class family.
4. **Fix the units yourself** — do not use either shipped `converted_dataset*.csv`. Use raw ADC
   from `dataset CSV.csv` with the conversions in §3.3, and **sanity-check VDC against the
   divider ratio in the formulas PDF** before trusting it.
5. **State the confound precisely.** Each class is one contiguous file recorded at a distinct
   wall-clock time (13:24 → 14:31), and temperature drifts monotonically within runs. A block
   split separates early-in-run from late-in-run, **not** condition from condition. The plan's
   "within-run" caveat is right but understates it.

**Verdict: the plan is implementable and correctly scoped, but the DC-link half of the S2/S3
story does not exist in this data.** Present it as inverter **thermal + phase-current**
telemetry, and drop "DC link" from the stage map claim unless the simulator supplies it.

## 4.4 Trip gating (`trip.py`)

**The plan's §6 asks for f_e(t) tracking and accel/cruise/decel segmentation. No dataset in the
roster contains a speed ramp.**

| Dataset | Speed | Ramp? |
|---|---|---|
| D1 Paderborn | 900 or 1500 rpm, constant within a 4 s recording (measured: 899.3–900.1 rpm) | No |
| D2 KAIST | 3000 rpm constant, **f_e ≡ 200.00 Hz measured in every file** | No |
| D3 Bacha | 10 rad/s constant | No |
| D4 Thomas | mains-fed, **f_e ≡ 50.00 Hz measured** | No — but start-up transients exist |

Consequences:
- Step 3 ("order-normalised branches run only on cruise windows") is a **no-op** for D2 and
  D1 — every window is cruise. Implement it, unit-test it, but do not claim it gates anything
  on real data.
- Step 1 (f_e estimation) is trivially exact on every real dataset, so measuring "tracking
  error" on real data is meaningless. The plan's R-SIM already routes this to the simulator ✅.
- The only real segmentation data is **D4 FILE 1, 5, 6, 10** — four motor-energisation
  transients. The plan already scopes this correctly ("report segmentation only, never
  frequency tracking") ✅.

**Biggest risk: building `trip.py` to a spec that no real data exercises, then discovering at
demo time that the simulator branch (P7/P8, both Stretch) never got built.** Then trip gating
is untested machinery in the critical path.

**What I would change:** make `trip.py` a thin, well-tested module with synthetic-signal unit
tests (a chirp with known f_e(t) — the existing `tests/test_dsp.py` already builds synthetic
signals this way and is a good template), and make the gating **optional and off by default**
for D1/D2. Keep the low-speed resolution table from §6.4 — it is a good honest artefact and
costs nothing to compute analytically.

## 4.5 Fusion (`fusion.py`)

The plan's §7 is **purely rule-based with no learning and no cross-dataset data requirement**,
so nothing in the data can break it. It is the lowest-risk item in the plan ✅. The unit tests
it specifies (threshold crossing, hysteresis, K-consecutive, low-confidence cap, worst-stage
aggregation) are all testable against synthetic probability streams.

**One risk, and it is a presentation risk.** §7 says a branch below the configured metric floor
shows `LOW CONFIDENCE BRANCH` and cannot raise Fault alone. On my reading of the data:

| Branch | Likely honest metric | Above a sane floor? |
|---|---|---|
| S5 bearing | 0.7983 per-bearing LOBO | Yes |
| S4 winding | unknown; 3 folds, 3 healthy recordings, batch confound | **Doubtful** |
| S1 supply | 2 folds, n=1 per class | **No** |
| S2/S3 telemetry | within-run only, ~8 test windows in 3 classes | **No** |

So the dashboard may end up with **three of four branches greyed out** and only the frozen
bearing branch able to raise a Fault. That is the honest outcome, and it is defensible — but it
is a very different demo from the one §8 describes, and you should decide now how to frame it
rather than discovering it in P5.

**What I would change:** set the confidence floor **before** running any branch, write it into
config, and record it in `claims_audit.md`. Choosing the floor after seeing the numbers is the
same sin as tuning on test. And build the dashboard so a `LOW CONFIDENCE` branch still shows
its waveform and spectrum panels — a greyed-out status light with live signal behind it tells a
better story than a hidden panel.

---

# 5. Effort and compute

Machine: no GPU confirmable (no torch installed); assume the RTX 3050 the docs reference
(`drivesentinel/folds.py:19` "on a 3050 is minutes"). **37 GB free on C:.**

| Phase | Implementation | Run time (this machine) | Peak disk | Peak RAM | Notes |
|---|---|---|---|---|---|
| **P0** env + schema + splits | 0.5–1 day | `pip install` ~10–20 min (torch CUDA wheel ≈2.5 GB download) | **+3–4 GB** (site-packages) | <1 GB | Blocking. Also rebuild the D1 cache if S5 metrics must be re-derived: reads **21 GB**, writes 165 MB, `cache_report.json` says 182.6 s but `README.md` says 30–60 min for a cold run. Budget **45 min**. |
| **P1** D2 winding | 2–3 days (the adapter is the work, not the model) | **⚠ slow — see below** | **+0.7 GB** decimated cache | **~350 MB** per file | |
| **P2** D4 supply | 1–1.5 days | **<10 min** total | +0.2 GB | **~600 MB** if a whole file is loaded at once | |
| **P3** D3 telemetry | 0.5–1 day | **seconds** | negligible | <50 MB | Smallest, fastest, do it first to warm up |
| **P4** trip + fusion | 1–1.5 days | seconds (unit tests) | negligible | negligible | Pure logic |
| **P5** scenarios + dashboard | 1.5–2 days | seconds per replay | +0.1–0.5 GB `artifacts/demo/*.npz` | <1 GB | Keep scenario NPZs small and decimated |
| **P6** results + claims audit | 0.5 day | seconds | negligible | negligible | |
| **P7/P8** simulation (stretch) | 2–3 days | minutes/episode | +0.5 GB | <2 GB | `gym-electric-motor` is present but its deps are not installed |

### Things that will be slow, specifically

**P1 D2 decimation is the one real compute cost.** Concretely:
- **14 GB of TDMS to read** (48 current files × ~290 MB). Cold-cache disk read alone is
  **8–15 min**.
- Each file is 3 channels × 12.2 M float64 samples. `scipy.signal.decimate(..., ftype="iir",
  zero_phase=True)` on 12.2 M samples costs roughly 2–5 s per channel → **48 × 3 × ~3 s ≈ 7 min**
  of pure filtering, plus float64→float32 conversion.
- **Total realistic budget: 25–45 min for a one-time D2 cache build.** Do it once, write
  float32, never re-read the TDMS.
- Output size: 48 × 3 × 1.22 M samples × 4 B ≈ **700 MB**. Acceptable.
- **Do not cache at full 100 kHz** — that would be 48 × 3 × 12.2 M × 4 B = **7 GB**, and there
  is no reason to keep it.
- RAM: read and decimate **one file at a time**; peak ≈ 3 × 12.2 M × 8 B = **293 MB**. If you
  load all three channels at float64 *and* keep the decimated copy, ~350 MB. Fine. Do **not**
  parallelise across files with `joblib` at default `n_jobs=-1` — 8 workers × 350 MB = 2.8 GB
  and heavy disk contention on a single drive. Use `n_jobs=2` or 3.

**P2 D4 `.mat` loading.** `scipy.io.loadmat` on a 72 MB v5 file materialises the full
1,000,000 × 9 float64 array = **72 MB**, fast (<2 s). Ten files sequentially is trivial. The
trap is loading all ten at once (720 MB) — process one at a time. Note the array is
**column-major**, so a phase is a contiguous 8 MB block; slicing one channel is cheap.

**Disk headroom check.** Worst case additions: torch+deps 4 GB, D2 decimated cache 0.7 GB, D1
order-spectrum cache 0.17 GB, demo scenarios 0.5 GB, simulator deps 0.5 GB → **≈6 GB**.
Against 37 GB free that is comfortable. **No phase needs more than you have** — provided you do
not cache D2 at full rate (7 GB) and do not keep both v1 and v2 D1 caches plus intermediates.
If you later want headroom, D: has 146 GB free and `DRIVESENTINEL_EXT_DATA` (planned, not yet
implemented) is the right hook for moving `data_ext/` there.

**Critical-path estimate: P0→P6 is ~8–11 working days.** P1 is the long pole, and most of it
is adapter correctness (polarity detection, channel-name variance, dedup), not compute.

---

# 6. Verdict

## NOT READY

Four blocking items. Three are cheap; one is a decision only you can make.

| # | Blocker | Fix | Cost |
|---|---|---|---|
| **B1** | **No Python environment.** No venv anywhere; both Pythons have only `pip`. torch, numpy, scipy, pandas, pyarrow, joblib, pytest all missing. `python -m pytest -q` fails with `No module named pytest`, so the plan's P0 gate ("existing 29 tests pass") is **unverified**. | Create a venv and install `requirements.txt`, plus `nptdms`, `matplotlib`, `streamlit`, and a GBM library. Then run the tests. | 20 min |
| **B2** | **`artifacts/runs/deployment_model.pt` does not exist**, nor do the feature caches (`order_spectra_v*.npz`, `*_meta.parquet`). The plan's B-S5 wrapper (§5) says it "loads the existing deployment model" — that file is gone. The INT8 `.mem` files survive but their float parent does not, so the export cannot be re-verified. | Either (a) rebuild: `01_build_cache.py` (~45 min, reads 21 GB) then `02_train_lobo.py`; or (b) point `branches/bearing.py` at `artifacts/int8_export/` + `golden_reference.py`, which is self-contained. **(b) is faster and is what actually ships.** | (a) ~1 h, (b) 1 h of code |
| **B3** | **D2's healthy class is unconfirmed.** `0_00` files are byte-identical duplicates (3 healthy recordings, not 6), and their TDMS root names say `0mohm`/`0ohm` — a dead short, i.e. maximum severity. Unbalance evidence supports "healthy" at 1000 W only. If `0_00` is not healthy, **D2 has no healthy class** and B-S4's entire class scheme is wrong. | Check the Data in Brief paper (PMC9957734) for the healthy-condition description. 20 minutes of reading decides the shape of P1. | 20 min |
| **B4** | **D2's negative-sequence feature tracks acquisition batch, not fault severity.** Per-file current-probe polarity varies in three patterns; after correction, unbalance still separates acquisition batches perfectly at 1000 W (A max 0.0775 < B min 0.0875) and is non-monotonic in severity on 5 of 6 motor×type combinations. | Polarity + gain correction in the adapter (minimise \|I₀\|), plus a "predict the batch" control experiment reported alongside every B-S4 metric. | 1 day, inside P1 |

Everything else checked out. The frozen bearing pipeline is **relocation-clean** (zero
hardcoded absolute paths; the only `D:\kone` string is a log line in
`artifacts/runs/export.log`), **leakage-clean** (normalisation fit on fit-indices only at
`train.py:353`; an explicit `assert ... "BEARING LEAK"` at `train.py:341`; speed from the
recording's own current at `dsp.py:120-127` with the tachometer diagnostic-only), and
`.gitignore` correctly excludes all three data trees with no dataset files in `git status`.
All four datasets are complete — no `.zip`, `.part` or `.crdownload` anywhere — and every unit
I checked is physically plausible except D3's DC-link channels.

## The three highest-value things to do first

**1. Install the environment and run the test suite (20 minutes).** Everything downstream is
blocked, and the plan's own P0 gate is currently unverifiable. Do this before anything else.

**2. Settle the D2 healthy question, then write the D2 adapter with polarity correction
(1 day).** B3 and B4 together decide whether B-S4 is a 3-class problem, a binary problem, or a
severity regression. Writing features before this is rework. The adapter must: read the channel
list per file (8 files use `ai1` instead of `ai2`), detect polarity per file by minimising
|I₀|, normalise `coil` → `intercoil` in the 8 mislabelled vibration filenames, and deduplicate
the three `0_00` recordings. Nothing else in P1 is hard; this is the whole job.

**3. Do P3 (D3) end to end in half a day, as the pipeline's dress rehearsal.** It is the
smallest dataset (8 MB), the labels are already confirmed exactly against your spec, the
conversion bug is fully diagnosed in §3.3, and it exercises `schema.py`, `splits.py`, the
adapter pattern, the metric-card format and the results renderer. Getting the plumbing right on
a dataset that takes seconds to process is far cheaper than debugging it against 15 GB of TDMS.

## What to drop from the spec to save time

| Drop | Why | Saves |
|---|---|---|
| **The 13-class D4 label vector** | Cannot be aligned to files or mapped to scenarios from the data (§3.4). Derive the three classes from the phase-current collapse instead, as the plan's own fallback allows. | 0.5–1 day of dead-end work |
| **The learned classifier for B-S1** | With 1 recording per (motor × class) and 2 folds, no learned metric is defensible. Ship a documented threshold rule + the leaky reference. | ~1 day, and produces a *better* artefact |
| **D3's 9-class location view as a reported metric** | Three classes get ~8–10 test windows. Show the confusion matrix qualitatively; report only the 4-class family. | 0.5 day |
| **`Vdc`, `Idc`, `VD` features in B-S2/S3** | Constant to within their own quantisation noise across all nine classes (§3.3). Three more ways to overfit. | Small, but removes a real risk |
| **P7/P8 simulation** | Already Stretch. With B1–B4 outstanding, treat as out of scope for the MVP and cut cleanly rather than half-building. | 2–3 days |
| **The f/f_e electrical-harmonic axis as a *claimed* invariance** | f_e ≡ 200.00 Hz across all 48 D2 files; it is a fixed rescale here. Implement it for the API, but do not spend time tuning or defending it. | 0.5 day |

Keep the vibration channels in D2 even though only current is in the plan's feature list —
they are only 25 MB each and they give you a second modality for the confound control.

## Questions I need answered before the build starts

1. **Is D2 `0_00` the healthy baseline, or a 0 Ω dead short?** The file is duplicated across
   both fault-type names (→ shared baseline), but its TDMS root name says `0mohm`/`0ohm` (→
   maximum severity), and the severity number tracks fault current, which diverges as R→0.
   What does PMC9957734 say? **This decides B-S4's class scheme.** (§3.2 Finding 2)

2. **Do you accept my D4 motor split — FILE 1–5 healthy, FILE 6–10 outer-race fault?** Inferred
   from mtime blocks, 1:1 scenario pairing, and systematically 1.6–2.3× higher vibration in
   6–10. Strong, but it is inference. Can you confirm from the Sci Data paper? **This decides
   whether leave-one-motor-out is even constructible.** (§3.4)

3. **What is the D3 voltage-divider ratio?** It is in `Sensor_raw_data_conversion_formulas.pdf`,
   which I could not read (no PDF library installed). Without it, VDC/VD remain unconverted.
   Given both channels are diagnostically dead, I would simply drop them — but tell me if you
   want them fixed for the stage-map story. (§3.3)

4. **Do you want the B-S5 branch to load the INT8 export, or to rebuild the float deployment
   model?** The `.pt` is gone (B2). The INT8 path is self-contained and is what ships; the float
   path costs ~1 h of compute and 21 GB of reads. My recommendation: INT8.

5. **What is the branch confidence floor for fusion, and are you willing to demo with three of
   four branches showing `LOW CONFIDENCE`?** Set it now, in config, before any branch runs —
   picking it afterwards is selection on test. (§4.5)

6. **Should I correct `docs/accuracy_ceiling.md:18-19`?** It says the exported INT8 weights "are
   the model these numbers describe". They are the same *recipe* (single model, no ensemble, no
   TTA — confirmed at `train.py:531-533` and in `lobo_summary.json`), but trained on **all 29
   bearings** (`train.py:523`, `train.py:548`), so no LOBO fold model ships. This is the one
   claim in the docs I would call wrong rather than merely unlabelled. (§2.4)

7. **Is the recipe sweep (`sweep4_v1.json`, `sweep5_v2.json`) declared as selection-on-test?**
   Configs were chosen from LOBO-fold scores (`config.py:131-137`). The differences were inside
   the noise floor, which mitigates it, but under the plan's §9 rule 3 it should be stated in
   `claims_audit.md`. Do you want me to write that entry? (§2.5)

---

## Appendix — method notes

No Python packages were available, so all data inspection used **stdlib-only parsers written
for this audit** and kept in the session scratchpad (outside the repo):

- **TDMS reader** — lead-in/ToC parsing, per-segment metadata, raw-data indices, repeated
  chunks, `struct` + `array` for float64 blocks. Cross-checked against declared
  `wf_increment`, `wf_samples` and file sizes.
- **MATLAB v5 reader** — tag/element walk, `miCOMPRESSED` via `zlib`, struct/cell/char/numeric
  classes. Used for both D1 (nested struct, 7 channels) and D4 (flat 1e6 × 9 double).
- **Goertzel DFT** — single-frequency amplitude and phase, used for f_e estimation, Fortescue
  sequence components and THD spot-checks.

Where a claim needed a library I do not have (PDF text extraction, and re-running the test
suite), it is marked **unverified** rather than assumed.

Statistics computed on subsampled strides where noted (typically every 4th–8th sample); this
is sufficient for RMS, min/max and spectral peaks at the frequencies of interest, and is not
sufficient for anything claimed about individual samples. No claim here depends on that.
