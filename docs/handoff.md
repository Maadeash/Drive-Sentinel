# DriveSentinel — context handoff

> **START HERE.** Read these, in this order, before touching anything:
> **1.** `docs/handoff.md` (this file — current state and what is blocked)
> **2.** `CLAUDE.md` (the non-negotiables and the dataset traps)
> **3.** `docs/results_rtl.md` (generated — every RTL measurement)
> **4.** `docs/rtl_declarations.md` (the RTL recipes, fixed before the RTL existed)

State document for a session that knows nothing about this project.
Updated 2026-09-21 at the end of the RTL build session. The software track is complete
and frozen. The RTL track is built, verified in simulation on every bar except the
long one, synthesised, placed, routed and timing-closed at 100 MHz. **V-1 has since
finished: all 16,211 windows, 0 disagreements** (§6.1). A PYNQ-Z2 package (bitstream, handoff, notebook) is
in `board/`, and **on 2026-09-24 the notebook ran on a real PYNQ-Z2**: build ID matched,
120/120 windows bit-exact against the reference and the RTL model, median 18.07 ms per
window (§6.7.1, `claims_audit.md` §1.26). The inference server and the web app have not
run on a board. A **technician web app** (`webapp/`, §6.8) now sits on top of the
software track as the user-facing prototype. It replays the recordings and is not
connected to a lift.

---

## 1. What this is

**KONE Elevate'26, PS-06: real-time drive health monitoring from various-stage voltage
and current waveforms.** The drive is modelled as five stages — S1 supply input, S2 DC
link, S3 inverter, S4 motor winding, S5 bearing — with one diagnostic branch per stage,
fused by a rule-based layer, and shown in a replay dashboard.

| | |
|---|---|
| Repo | `C:\kone` — remote `https://github.com/Maadeash/kone`, branch `main` |
| Build plan | `docs/workflow_v2.md` — §13 holds post-audit decisions and **supersedes** anything above it |
| Claims register | `docs/claims_audit.md` — every number with its protocol and source |
| Environment | `.venv/`, Python 3.12, **CPU torch 2.14.0+cpu**. No NVIDIA GPU (Intel UHD only) |
| Tests | **473 collected**, all pass (1 skipped: no board run yet); `tests/test_webapp.py` and `tests/test_board_server.py` start real servers |
| Datasets | `data/` (D1 Paderborn, 21 GB), `data_ext/` (D2 KAIST, D3 Bacha, D4 Thomas, 16 GB). All gitignored |

No dataset shares a physical machine across stages, so the demo is labelled
**COMPOSITE REPLAY** and fusion is rule-based rather than learned.

### Phase status

| Phase | Deliverable | Status |
|---|---|---|
| P0 | Env, schema, split utilities, bearing rebuild | **done** |
| P1 | D2 winding branch | **done** — the confound analysis is the deliverable |
| P2 | D4 supply branch | **done** — threshold rule |
| P3 | D3 inverter telemetry branch | **done** |
| P4 | Trip gating + fusion | **done** |
| P5 | Scenarios + Streamlit dashboard | **done** |
| P6 | Results, claims audit, README | **done**, rolling |
| | **software track, P0–P6** | **COMPLETE** |
| P7/P8 | Simulation branch, simulated speed-ramp validation | **out of scope** (§5) |
| — | **RTL** (a) — the spec | **done** — `docs/rtl_spec.md`, §9 now closed |
| — | **RTL** (b) — the Verilog | **done** — `rtl/`, 16 modules, LANES = 1, F = 26 |
| — | **RTL** (c) — verification | V-1..V-6 **PASS** — V-1 16,211/16,211, 0 disagreements (§6.1) |
| — | **RTL** (d) — synthesis | **done** — post-route, **106.6 MHz**, timing met |
| — | **RTL** (e) — docs | **done** — `results_rtl.md` generated, claims audit §1.20 |
| — | **RTL** (f) — extras | timing closure **done**; front-end timing **plan** written; panel 3 table **fixed** |
| — | **RTL** board package | **built, and the notebook RAN ON A PYNQ-Z2 (2026-09-24)** — 120/120 bit-exact, `board/board_run.json` (§6.7.1). Server and web app: not run on a board |
| — | **Technician web app** | **done, FROZEN** — `webapp/`, sign-in + 10 pages, fleet of 4 example lifts from 11 recorded units, 21-byte store-and-forward alerts (§6.8) |

---

## 2. Branches

Authority is read from the results JSON at runtime by `fusion.load_branch_metrics()` —
never hardcoded. Regenerate this table with that function, don't retype it.

| Branch | Stage | Protocol | Groups | Macro-F1 | Tier | Results JSON |
|---|---|---|---|---|---|---|
| `bearing` | S5 | leave-one-bearing-out | **29** | **0.7852** | **FAULT-CAPABLE** | `artifacts/runs/lobo_summary.json` |
| `winding` | S4 | V5 leave-one-session-out | 2 | 0.6250 | INDICATIVE | `artifacts/multistage/winding/winding_results.json` |
| `supply` | S1 | R2 threshold rule, leave-one-motor-out | 2 | 1.0000 | INDICATIVE | `artifacts/multistage/supply/supply_results.json` |
| `inverter_telemetry` | S2/S3 | V1 contiguous block, within-run | **0** | 1.0000 | INDICATIVE | `artifacts/multistage/inverter_telemetry/inverter_telemetry_results.json` |

**Exactly one stage of five can raise `Fault`.** Three branches sit at or above the
macro-F1 floor and all three are INDICATIVE — every one fails on the *group* test. The
binding constraint on this project is how many independent machines each dataset
contains, not how well a model fits.

Headline numbers, all from the JSON above:

| Branch | Honest | Leaky reference | Baseline / control |
|---|---|---|---|
| bearing | 0.7944 window, 0.8356 per-recording | 0.9865 *(stratified 5-fold over windows)* | 0.4131 majority |
| winding | V5 0.6251; **V1 0.3784, below chance** | V3 0.9990 | **session-only probe 0.750** |
| supply | R2 rule **1.0000**, 10/10 recordings | L3 learned 0.9983 | L1 learned honest 0.6593 |
| inverter | V3 electrical-only 0.8503 / F1 0.8229 | V2 1.0000 | V1 with temperature 1.0000 |

---

## 3. Standing rules

These are not stylistic. Several were violated once already and the corrections are in
`claims_audit.md`.

1. **The fusion floor is pre-registered and must never be adjusted.**
   `config.FUSION_CONFIG["fault_authority"]`: ≥ 3 independent validation groups **and**
   honest macro-F1 ≥ 0.75, fixed **2026-09-18T07:19:23Z** at commit `7344a436`, before
   any multi-stage branch existed. If a branch fails it, the branch is INDICATIVE — the
   floor does not move. `tests/test_fusion.py` asserts an INDICATIVE branch held at
   `p_fault = 0.99` for 50 updates reaches `Warning` and stops.
2. **No invented numbers.** Every figure comes from a results JSON. Results markdown is
   *generated* (`scripts/60_render_multistage_results.py`), never hand-typed. Anything
   not run reads `NOT RUN`.
3. **Honest validation only.** Group splits, routed through
   `common.splits.iter_checked`, which asserts group separation and refuses an
   undeclared leak. No tuning against test folds — the one historical violation is
   declared in `claims_audit.md` §1.2.
4. **Leaky references are labelled `(leaky reference)` in the heading, not just the
   body.** Headings are what get screenshotted.
5. **The v5 bearing pipeline is frozen.** `drivesentinel/{config,dataset,dsp,export,
   features,folds,model,quantize,train}.py` and `scripts/01..05` get no behaviour
   changes. Extend around them.
6. **Commit and push after every phase.** 15 commits, all pushed.
7. **Measure claims before writing them down.** Two docstring claims in this project
   turned out false when measured (§5). Assert the property in a test.

---

## 4. Key findings

| Finding | Where it is documented |
|---|---|
| **D2 three-day acquisition confound.** The KAIST dataset was recorded on 2022-01-25, 2022-03-08 and 2022-08-11. Four independent metadata signals agree on the partition — `Date Created`, DAQ chassis, duration discipline, root-name style. The 3000 W motor's seven inter-coil faults are on a *different chassis and module* (`cDAQ5Mod1`) from everything else, so within that motor fault type is perfectly predictable from the instrument. | `docs/data_notes_d2.md` §7A |
| **The `i0rel_residual` probe.** Residual zero-sequence after the polarity fix is zero by Kirchhoff in a three-wire machine, so it cannot contain winding information. A 1-NN on that single scalar recovers acquisition session at **1.000** on the two clean motors and still predicts **0.750** of the fault class (baseline 0.467) — beating the full feature set under an honest split. | `docs/data_notes_d2.md` §7B |
| **All three D2 healthy recordings are from one day**, so `healthy` is not measurable under any session-aware split. | `docs/data_notes_d2.md` §7D |
| **D4 single-session verification.** Six signal-derived fingerprints (mains frequency to 0.001 Hz, DC offsets, noise floor) fail to separate the two motors; mains frequency spans only 0.081 Hz across all ten files. No D2-style confound. Stated as absence of evidence plus corroboration, **not proof** — the `.mat` headers are export times, not acquisition times. | `docs/data_notes_d4.md` §4 |
| **D3 over_temp run-identification.** The electrical-only ablation scores `over_temp` at **F1 0.796 with no temperature sensor in the feature set**. A thermal fault is not detectable from two 10 Hz phase currents, so that is the model identifying *which run* a window came from. Treat every 4-class number on D3 as contaminated by run identification. | `claims_audit.md` §1.15, `results_multistage.md` |
| **Order-resolution table.** `Δorder = 60/(T·rpm)`. At 900 rpm a 1 s window resolves 0.0667 orders; at 20 rpm it resolves 3.00, wider than the 1.89-order BPFO/BPFI gap, so those lines merge. Holding 900 rpm resolution at 20 rpm needs a **45 s** window. Arithmetic, labelled extrapolation. This is the answer to the low-speed sheave question. | `trip.resolution_report()`, `results_multistage.md`, dashboard panel 3 |
| **D3 run-identification control.** Training the same features on the same block split to predict *which run* a window came from succeeds at **0.9713** (with temperature) and **0.7484** (electrical only) against a 0.4045 baseline over 9 runs — within a few points of the condition scores themselves. One run per condition means naming the run names the condition. Converts the `over_temp` inference into a measurement. Also on the inverter metric card, beside the 4-class score. | `claims_audit.md` §1.18, `results_multistage.md`, `panels._inverter_extras()` |
| **D2 CNN negative result.** A CNN of comparable capacity (26,914 params vs the bearing model's 27,024) on the full harmonic spectrum scores V5 **0.5608 ± 0.0142** against the GBM's 0.6251 — both pre-declared criteria fail. On the **leaky** split it scores 0.8448 against the GBM's 0.9990, which says that score was largely per-recording memorisation: a **fourth independent line** on the D2 confound. | `claims_audit.md` §1.17, `results_multistage.md` |
| **KI05 missed detection.** A genuinely faulty inner-race bearing the model scores **0.0089** on. It is shipped as a demo scenario on purpose, and panel 4 raises a MISSED DETECTION banner rather than letting "Normal" pass as a healthy machine. In the web app it is Lift 01's bearing. Since 2026-09-24 that shows an **ALARM**: the web app's bearing stage runs the deployed network, which was trained on KI05 -- an in-sample verdict (`claims_audit.md` §1.28). The held-out miss is still the honest finding, shown in the Engineering view. | `docs/accuracy_ceiling.md`, `artifacts/demo/bearing_KI05.npz` |

---

## 5. Closed decisions — do not re-litigate

| Decision | Why |
|---|---|
| **D2 gain calibration is computed but OFF by default** | Zeroes \|I₀\| by construction (25/45 came out exactly 0.0000), hits its ±15 % clamp on 20/45 files, and moves the negative-sequence ratio by up to 0.0504 on a dataset whose whole fault range is 0.01–0.11. `calibrate_gains()` docstring. |
| **D2 `healthy` is NOT MEASURABLE** | All three healthy recordings are from 2022-01-25. No session-aware protocol can train and test the class. **No caveated number for it anywhere.** |
| **D2 task is `inter_coil` vs `inter_turn` only** | Follows from the above. |
| **D4 label vector is not used** | The 1000/500 reconstruction matches 6/14 *file* boundaries but **0/4** measured phase-loss events. Labels come from the current-collapse rule. |
| **No CNN for D2** | Originally: 30 fault recordings across 2 usable folds. **Now also measured** — a CNN was tried once with criteria declared in advance and FAILED both (§4). GBM ships. **No further architectures**; sweeping until one passes is the failure the criteria exist to prevent. |
| **Metric-card rows render as markdown, never `st.dataframe`** | `st.dataframe` paints to a `<canvas>`: its text cannot be selected, copied, found with ctrl-F, or read by a screen reader. The card rows *are* the argument — session-only baseline, NOT MEASURABLE, the CNN negative result, the run-identification control — and a number a judge cannot select is a number they cannot check. `panels.markdown_table()` / `panels.extras_table()`; guarded by `test_card_extras_are_not_rendered_with_st_dataframe`. |
| **Demo NPZs are tracked in git** (3.3 MB) | A fresh clone must run the dashboard without a 7-minute retrain. |
| **P7/P8 simulation out of scope** | Stretch, cut for the MVP. |
| **Trip gating implemented but disabled** | No dataset in the roster has a speed ramp; gating would mark every window `cruise` and change nothing. |
| **B-S1 is a threshold rule, not a learned model** | One recording per (motor × class); a learned metric separates two 20 s captures. |
| **D3 drops `VDC`, `IDC`, `VD`** | Per-class means span < 0.8 ADC counts against std 1.2–1.4. |

**Two claims that were measured and found false** — both corrected, both now asserted in
tests. Do not reintroduce either:

- "D2 channels within one file differ by up to a second" — wrong, an artefact of a
  hand-written TDMS parser. All 48 files have equal-length channels.
- B-S1's "any threshold between 1 % and 30 % gives the same answer" — wrong. The
  *recording-level* verdict holds across 0.03–0.40, but the *window-level* safe band is
  only (0.048, 0.053).

---

## 6. The RTL track — what exists, what is verified, what is not

### 6.1 V-1 — complete

**V-1 (RTL argmax vs `golden_reference.predict`, all 16,211 windows, bar 100.000 %)
finished on 2026-09-22: 16,211 / 16,211, 0 class and
0 logit disagreements, max logit difference 0 LSB, every window
1,786,724 cycles.** 29.8 hours of XSim across 65 chunks, two at a time, survived
a laptop sleep mid-run. Recorded in `artifacts/rtl/verify.json`; rendered in
`results_rtl.md` §3. What follows is how it ran, kept for a re-run.

| | |
|---|---|
| How it runs | 65 chunks of ~250 windows, 2 XSim processes at a time, shipped config (`DEBUG_TRACE=0`) |
| Expected duration | **~17 hours** on this machine (measured throughput 0.26 windows/s, `artifacts/rtl/bench_workers.json`) |
| Progress so far | read `artifacts/rtl/sim/v1_progress.log` — one line per finished chunk, with its mismatch count |
| Resume / finish | **`scripts/rtl/run_tb.sh v1`** (or `.\scripts\rtl\run_tb.ps1 v1`). Finished chunks are skipped. |
| Then | `.venv/Scripts/python.exe scripts/rtl/render_results.py`, commit |

Until every chunk has run, `results_rtl.md` reports V-1 as **PARTIAL with the exact
count** — never as V-1 (`rtl_declarations.md` D-6). If a disagreement appears, the
chunk's report names the window; diagnose it with V-3's accumulator monitor on that
window (put it in a one-window vector set with `acc` enabled), which reports the layer,
channel and position where divergence starts.

**Evidence already in hand, and where it stops.** Every other bar passes on the final
RTL, and V-3 is the stronger check on the datapath: 917,696 accumulators over 64
windows, bit-exact, plus V-5's 59 saturating windows. An earlier, killed V-1 attempt
reached window 768 of each of its four shards with zero mismatches — that is *reported
here as an observation from the log*, not as a result: its report files were never
written (the flaw that made V-1 resumable), so it is in no JSON and must not be quoted.

### 6.2 What was built

| | |
|---|---|
| RTL | `rtl/` — 16 Verilog-2001 modules, one per file, every port commented, widths parameterised |
| Generated | `rtl/ds_gen.vh`, `rtl/mem/requant_params.mem` — from `scales.json` by `scripts/rtl/gen_rtl_params.py`; guarded by `tests/test_rtl_generated.py` |
| Weights / biases | the exported `.mem` files, **read unchanged** via `$readmemh` |
| Testbenches | `tb/tb_ds_top.sv` (buses, V-1/2/3/5/6), `tb/tb_requant.sv` (V-4) |
| Bit-accurate model | `drivesentinel/rtl/model.py` — asserted byte-identical to the golden reference |
| Constraints | `constraints/ds_top.xdc` — 100 MHz, out of context |
| One-command re-run | `scripts/rtl/run_tb.sh` / `run_tb.ps1` |

### 6.3 Measured results — `docs/results_rtl.md` is the authority

**Simulation-verified (XSim):**

| | |
|---|---|
| V-2 max logit difference | **0 LSB** |
| V-3 accumulators, 64 windows | **917,696 compared, 0 mismatches** |
| V-4 round-half-to-even | **3,966 vectors, 427 exact ties, 0 mismatches** |
| V-5 saturation | **59 windows, 58,636 values clamped at all four layers, 0 mismatches** |
| V-6 build ID | **`0x23eb56bf`, match** — computed by the hardware from the loaded memories |
| Cycles per inference | **1,786,724** (identical on every window) |

**Synthesis-reported (Vivado 2023.2, xc7z020clg400-1, post-route, out of context):**

| | |
|---|---|
| Resources | **1,111 LUT (2.09 %), 968 FF, 11 BRAM tiles, 4 DSP48** |
| Clock | **106.6 MHz achieved**, WNS +0.615 ns at the 100 MHz target |
| Latency | **17,867 µs at 100 MHz; 16,768 µs at 106.6 MHz** |
| Margin vs the 0.5 s hop | **30x** |

**NOT RUN:** power; end-to-end latency (the DSP front end has never been timed);
anything on hardware. No bitstream exists.

### 6.4 Decisions taken this session, and why

| Decision | Why |
|---|---|
| **LANES = 1** | `rtl_declarations.md` D-1: smallest lane count with a measured 10x margin at the achieved clock. One lane gives 30x. The fabric is better left for the front end. |
| **F = 26, not 15** | D-2's rule gives F = 15 (smallest width with 100 % argmax). F = 15 leaves 810,265 conv2 and 2.9 M conv3 accumulators differing from the specification. F = 26 is the smallest width bit-exact on every accumulator of all 16,211 windows. The stricter of two declared criteria bound; no bar moved. |
| **Input quantisation is PS-side** | D-4. `rtl_spec.md` §2 and §7 contradicted each other; §7 wins now §9.1 is answered PS software. Narrows the verified boundary — stated wherever the results are. |
| **Build ID walked, not baked** | D-3. The walk caught a real bug (every byte read twice, `0x8a36ab8f`); a constant would have agreed with itself. |
| **Out-of-context implementation** | ~190 ports against 125 package I/O. I/O delays left unconstrained rather than invented; the quoted clock is the worst register-to-register path. |
| **Two timing-closure iterations** | Pipelined requantiser (−8.253 → −0.560 ns), then registered MAC product (→ +0.615 ns). Constraint never edited. Both earlier builds kept under `artifacts/rtl/`. |

### 6.5 Findings worth a slide

- **The tie case is unreachable from the input.** At F = 26 the shifts are 33–40 with
  odd mantissas, so no int32 accumulator produces an exact tie. V-4 therefore tests the
  rounder at the unit level, and says so.
- **V-6 caught a real bug.** Argument for computing provenance in hardware, in one line.
- **The pool layer's clamp needed a search to reach.** The obvious extremes top out at
  121; a fixed-seed hill climb found a probe that reaches 129.
- **17.9 ms is not sub-millisecond.** The spec's arithmetic assumed ≥ 8 MACs/cycle; the
  hop budget justified one. The measured 1.056 cycles/MAC matches the arithmetic well —
  the *architecture choice* is what moved the number, not an error in the estimate.

### 6.7 The board package — built, and run on a board (§6.7.1)

`board/` holds everything needed to run the accelerator on a PYNQ-Z2 in a few minutes,
following `board/README.md`. When this section was written no board was attached; the
notebook has since run on one (§6.7.1).

| | |
|---|---|
| Bitstream + handoff | `board/drivesentinel.bit`, `board/drivesentinel.hwh` |
| Block design | PS7 + AXI DMA (MM2S only, simple mode) → core `s_axis`; PS `M_AXI_GP0` → DMA and core registers; FCLK_CLK0 at 100 MHz |
| Core | `rtl/ds_top.v` **unchanged**, `DEBUG_TRACE=0`, wrapped by `board/rtl/ds_pynq_wrap.v` (interface metadata only; a test asserts it has no logic) |
| Addresses | core `0x43C00000` (4 KB), DMA `0x40400000` — from the `.hwh`, nothing hard-coded |
| **Full system**, post-route | **2,284 LUT, 2,690 FF, 12 BRAM, 4 DSP; WNS +0.777 ns, WHS +0.050 ns at 100 MHz** — `artifacts/rtl/system/` |
| Core share of it | 1,110 LUT / 968 FF / 11 BRAM / 4 DSP — matches the core-only build to within one LUT |
| Weights | all 11 `$readmem` images logged "read successfully"; the board's build-ID read (`0x23eb56bf`) is the real check, **not done** |
| Notebook | `board/drivesentinel_overlay.ipynb` (generated by `make_notebook.py`): load, build-ID check, 120 real windows through the DMA, compare with `golden_reference.py` run on the ARM, timing, writes `board_run.json` |
| Dry run | `board/dryrun/` — the notebook ran end to end against a software stand-in for `pynq`. Tests the notebook's code, not the hardware. |

**Keep the two resource figures apart.** The core-only numbers (§6.3) are out of
context and cover the accelerator alone. The full-system numbers cover the PS
boundary, the DMA and the interconnects. `claims_audit.md` §1.21 says what may be said
about the package, and the exact sentence for a slide.

**Build problems hit and how they were handled** (all in `board/build_overlay.tcl`):

- **Vivado's IP-integrator init failed intermittently** with "couldn't read file" on
  its own scripts. It was a different script each run, all of them present, and
  `create_bd_design` did not raise when it happened. Fix: call it with the working
  directory set to Vivado's `scripts/ipintegrator`, restore the directory straight
  after, judge success by whether the init actually defined its commands, and retry at
  most 3 times with logging (`artifacts/rtl/system/create_bd_attempts.log`).
- **In-session `synth_design` could not see the generated block-design sources.**
  Synthesis now runs in a fresh in-memory project (`read_bd` + explicit sources), from
  the repo root, so the core's relative `$readmemh` paths still resolve.
- **The core first got a 512 MB address window**, because the wrapper presents a 32-bit
  address. It is now pinned to 4 KB at `0x43C00000`.

**The one thing to know before running it.** The verified RTL's `STATUS.DONE` is sticky:
it is cleared only by a `CTRL.start` write, and a DMA frame auto-starts without one. So
from the second window on, polling `DONE` returns stale, cleared registers. The notebook
polls `BUSY` instead and checks `CYCLES`. A negative-control test shows the dry-run mock
catches a notebook that polls `DONE`. **v2 RTL change:** clear `DONE` on the auto-start
path too. This was deliberately not done here, because `rtl/` is frozen and any change
there reopens V-1 through V-6.

#### 6.7.1 The board run (2026-09-24)

The owner ran `board/drivesentinel_overlay.ipynb` on a PYNQ-Z2 (PYNQ 2.5, Python 3.6,
numpy 1.13.3, 32-bit ARM) and copied back `board/board_run.json`, unedited. Rendered in
`docs/results_rtl.md` §4c; registered in `claims_audit.md` §1.26, with what may and may
not be said.

| | |
|---|---|
| Build ID | `0x23eb56bf`, match |
| 120 windows | class 120/120 vs golden reference; logit registers 120/120 bit-exact; cycles 120/120 = 1,786,724 |
| CNN path, DMA + inference | median 18.07 ms, p95 18.11 ms (360 windows, PS wall clock) |
| Not | an accuracy figure (in-sample windows), V-1 on silicon, end-to-end latency, power |

It took four board-only fixes, none visible to the 64-bit dry run, each now tested:
register naming (`mmio_resolve.py`), Python 3.6 (`from __future__ import annotations`),
32-bit numpy (`np.bincount`), numpy 1.13 (`np.pad` mode). A fifth, `server.py`'s 3.7-only
`ThreadingHTTPServer`, was fixed before the server was ever run. The board's clock was
unsynchronised: `board_run.json` says 2026-08-31; the run was 2026-09-24.

**Board layout that worked:** everything in one folder,
`/home/xilinx/jupyter_notebooks/drivesentinel/` (notebook, `.bit`, `.hwh`,
`mmio_resolve.py`, `test_windows.npz`, `golden/`), copied with `scp` as user `xilinx`.
Avoid folder names with spaces, and create folders from the shell rather than the Jupyter
UI (which runs as root and leaves them unwritable by `xilinx`).

### 6.8 The technician web app — `webapp/` — FROZEN (2026-09-22)

**Frozen after the product-presentation revision, and re-frozen 2026-09-22 after the polish pass and the bearing-engine card (§6.9).** Only fixes, no features. Guide:
`webapp/README.md`. Register: `claims_audit.md` §1.23, which supersedes §1.22 for
presentation. Screenshots: `docs/screenshots/`, 46 captures plus the phone menu, with the
record in `console.json`.

| | |
|---|---|
| Start | `.venv/Scripts/python.exe -m webapp` → <http://127.0.0.1:8000>, sign in **`technician` / `demo`** |
| Pages | Sign in, **Fleet** (landing), Lift health `/lifts/<lift>`, **Alerts**, Why? `/alerts/<id>`, **Live monitor**, Sensors, Hardware, About, Settings. The **Engineering view** is a footer link only and its content is unchanged. Every page has its own URL; back/forward and bookmarks work; each page fades in |
| Sign-in | `webapp/auth.py`: signed HttpOnly cookie, a gate on every route (303 to /signin or 401). Pages live in `webapp/pages/`, which is not under `/static`. `/api/alerts/ingest` is the device channel and needs no session; it checks the size only |
| Presentation | `webapp/fleet.py` is the only code that shapes page data. It places the 11 recorded units into **4 fictional lifts**, each unit exactly once (asserted), and strips every internal field (paths, IDs, fault codes, dataset names). The pages have no banner and no caveats, by the owner's decision |
| Wording decisions | 1. Winding severity is **omitted**. 2. Winding headline is **"Winding anomaly detected"**, never a type (the type call is crossed on both units). 3. Inverter: **"Location: inverter power stage"**. 4. Stay on Starlette |
| Behaviour | Unchanged: fusion, emitter, outbox, both ADVISORY-never-ALARM guards. The only service change is display bookkeeping (`live_unix`) |
| Tests | `tests/test_webapp.py`, **58 tests** (plus 13 in `tests/test_board_server.py`): sign-in on every route, sign-out, redirect guard, cookie flags, form field names; banned vocabulary and claims on every user page, script and API; winding and inverter wording; fleet placement; every number against its JSON |
| Browser check | `scripts/webapp/screenshots.py` (Windows Edge over DevTools), re-run 2026-09-22 after the polish and engine card: 46 captures, **0 console errors, 0 overflow, 0 banned words** at 1280 and 390 px. The Live-monitor runs gave a bearing **ALARM**, an inverter **ADVISORY** and a winding **ADVISORY**. The script's one-session navigation check stalls in headless Edge (the page script never runs; the server answered in <0.2 s). The same sequence, including back/forward, sign-out from Settings and 375 px, passed by hand in the browser pane with 0 console errors (`console.json` `_navigation_manual_browser_pane`). Settings shows **Software** with no board |

**Still owed:**

- **One manual sign-in click-through.** Verification sessions got their cookie by POSTing
  the demo credentials, and no password was typed into the form by the agent. A test
  asserts that the form's field names match the handler.
- **"Connected · bearing engine"** on Settings (since 2026-09-24) means the PYNQ-Z2 is
  serving bearing verdicts now. **"Ready for connection"** otherwise means the `DataSource` seam and the bitstream
  exist. Sensor acquisition (`LiveSensorSource`) is still a skeleton that raises. The
  bearing inference path to the board is built (§6.9).
- **If a judge asks about the fleet,** it is example lifts built from recorded public data.
  The wording is in §1.23 "What may be said".

### 6.9 Board-to-web connection: built, NOT run on a board (2026-09-22)

The web app can send each bearing window to the FPGA on a PYNQ-Z2. Register: `claims_audit.md`
§1.24. Steps: `board/SERVER.md`.

| | |
|---|---|
| Board side | `board/server.py` (new file; nothing existing in `board/` changed). Loads the overlay, **refuses to start unless BUILD_ID = `0x23eb56bf`**, then `GET /health` and `POST /infer` (2,560 int8 bytes or 10,240 float32 bytes) → class, logit registers, cycles. Uses the notebook's busy-bit protocol |
| Web side | `drivesentinel/engines.py`: `SoftwareEngine` (the bit-accurate model), `BoardClient`, and `EngineRouter` (board when it answers, software otherwise, retries every 10 s). `drivesentinel/sources.py` `BoardSource` sends each bearing window through the router, lazily as the stream reaches it. Other stages stay in software |
| Settings | **Bearing engine** card: active engine (**Software** or **FPGA (PYNQ-Z2)**), board state, windows served per engine, board address (saved in `webapp/var/engine.json`; `DS_BOARD_ADDRESS` takes precedence) |
| Tests | `tests/test_board_server.py`, 13 tests against the mock `pynq` (see §1.24(a)); +3 in `tests/test_board_package.py` after §6.9.1 |
| **On the board (2026-09-24)** | server started on the PYNQ-Z2; web app counters FPGA 960 / Software 0; 600/600 answers bit-exact vs the software model; median 40.9 ms round trip (`claims_audit.md` §1.27). The deployed network's verdict now drives the bearing alerts (§1.28) |

#### 6.9.1 First real board attempt failed at cell 1: `ol.ds_0` is not the register file (2026-09-24)

Register: `claims_audit.md` §1.25; finding and fix: `board/mmio_resolve.py`. The owner
opened the notebook on an actual PYNQ-Z2. `ol.ip_dict` came back
`{'axi_dma_0': ..., 'ds_0/s_axi': ...}` -- no `'ds_0'` key -- because `ds_0` was added to
the block design as a bare RTL module reference (`board/build_overlay.tcl`), never
packaged as an IP-XACT core. `ol.ds_0` itself did not error (it's a hierarchy proxy);
`ol.ds_0.read(...)` did. Fixed in Python only, no RTL/bitstream/block-design change:
`board/mmio_resolve.py` (new) resolves the registers by hierarchy-chase then raw-MMIO
fallback; `board/server.py` and the regenerated `drivesentinel_overlay.ipynb` (via
`board/make_notebook.py`) both use it. The dry-run mock
(`board/dryrun/pynq/__init__.py`) previously exposed a flat `ol.ds_0` -- unlike the real
board -- and has been corrected to match, with a test pinning the corrected shape so it
cannot quietly go back to being more lenient than the hardware.

**Confirmed on the board the same day** (§6.7.1): the core resolved to `ol.ds_0.s_axi`
and the run completed. `board/README.md`, `board/SERVER.md` and `MANIFEST.sha256` add
`mmio_resolve.py` to what gets copied to the board.

**Decided 2026-09-24 (owner): the deployed network drives the bearing stage.** It was
trained on every bearing, so on the recorded windows its verdict is in-sample: KI05 is 0.85
(leaky reference), against 0.0167 held out, and Lift 01 now raises an ALARM the held-out
evaluation says the model would miss on a new bearing. `BoardSource(verdict="engine")`,
set in `webapp/server.py`; `verdict="held_out"` restores the old behaviour. Registered in
`claims_audit.md` §1.28, with what may and may not be said. The alerts are identical
whether software or the FPGA computes them (tested).

**`board/README.md` was not edited in the original board-to-web connection work** (that
task's guardrail said to add only new files in `board/`; the steps went into the new
`board/SERVER.md` instead). It **was** edited in §6.9.1 above, under a later, different
guardrail that explicitly asked for the notebook fix and its consequences.

#### 6.9.2 The DSP front end on the board, and the Board live page (2026-09-24): ready, NOT RUN on a board

Register: `claims_audit.md` §1.29. Steps: `board/SERVER.md` §4.

| | |
|---|---|
| Board side | `board/frontend.py` (`dsp.py` ported: Python 3.6, numpy 1.13, only `scipy.signal.sosfilt` needed) with `board/frontend_params.json` (filters designed on the laptop by `board/make_frontend_params.py`; a test pins the file to the generator). `board/server.py` gains `POST /process` (raw 4 s, current ×2 + vibration, float64 → DSP → quantise → FPGA, per-stage timings) and reports `python`/`numpy`/`scipy`/`frontend`/`machine` on `/health` |
| Web side | `drivesentinel/engines.py` `encode_raw`, `process_raw`, `SoftwareEngine.infer_int8`. `webapp/boardlive.py` streams the raw data files behind a bearing lift to the board and checks every window against the laptop. Routes `/board`, `GET /api/board/live`, `POST /api/board/live/start` `{lift, pace}`, `POST /api/board/live/stop`. Page `webapp/pages/board.html`; "Board live" in the nav |
| Tests | `tests/test_board_frontend.py` (11): port = `dsp.py` (bitwise with scipy.fft; identical int8 with the numpy.fft fallback), synthetic and real data; `/process` on the mock pynq; Board live end to end, error paths, no leaks. `test_webapp.py` scans the new page and API for banned vocabulary |
| Record | `scripts/webapp/check_board_frontend.py` → `artifacts/webapp/board_frontend_check.json` → `docs/results_rtl.md` §4d (`NOT RUN` until it exists and says `ran_on_hardware`) |
| **On the board, first attempt** | server up with the front end available (Python 3.6.5, numpy 1.13.3, scipy 0.19.1 → `numpy.fft`). One 4 s file did not return in 280 s: numpy 1.13's FFT is O(n × p) on the envelope's length 128,001 = 3 × 42,667. Fixed with Bluestein in `frontend.py` (`claims_audit.md` §1.29) |
| **On the board, full run** | 553 windows, FPGA exact 553 / 553; input identical only 65 / 553 (channel 5, numpy 1.13 `geomspace` endpoint), 5.23 s per 4 s on one core. Fixed: shipped band edges; `--workers 2` (`claims_audit.md` §1.29) |
| **Third run** | 195 / 196 identical, 196 / 196 same verdict, 2.1–4.4 s per 4 s on two cores (§1.29) |
| **The app's bearing from the board** | `BoardFeed` supplies every lift's bearing observations (raw → board → FPGA) to the one fleet stream; pages show `service.live_records`; Live monitor polls the stream; Board live is a view of the feed; Sensors page removed (owner, 2026-09-24) |
| **Not yet done** | run `check_board_frontend.py` against the board for the committed record (§4d) |

### 6.6 Also outstanding, smaller

| Item | Note |
|---|---|
| **V-1** | **done** — §6.1 |
| Front-end timing | `docs/frontend_timing_plan.md` — method and bar (p95 ≤ 250 ms single-core) fixed; **blocked on having a board** |
| Power | NOT RUN; needs a board or at minimum `report_power` with real activity — neither done |
| **Board run** | **done**: notebook (§6.7.1) and the web app's board server (§6.9). Not done: power, and timing the DSP front end on the board |
| `STATUS.DONE` semantics | v2 RTL change (§6.7); not made, `rtl/` frozen |
| LANES > 1 | not built; `ds_weight_mem.v` header records what banking would cost |
| `repeat.py` not re-run after the cache rebuild | unchanged from before — `claims_audit.md` §1.3 |
| `Sensor_raw_data_conversion_formulas.pdf` | unchanged — moot, those channels are dropped |
| Slide deck | not started; `results_rtl.md` §1 is slide-ready, `claims_audit.md` §1.20 says what may go on it |

## 7. How to run things

```bash
.venv/Scripts/python.exe -m pytest -q                        # 401 tests, ~50 s (includes the notebook dry run)
.venv/Scripts/python.exe -m pytest -q -m "not slow"          # skip full-dataset reads
```

```bash
streamlit run dashboard/app.py                               # replay dashboard
.venv/Scripts/python.exe -m webapp                           # technician web app, http://127.0.0.1:8000
```

Branch scripts — each writes its own results JSON:

```bash
.venv/Scripts/python.exe scripts/branches/10_winding.py --rebuild
.venv/Scripts/python.exe scripts/branches/20_supply.py
.venv/Scripts/python.exe scripts/branches/30_inverter_telemetry.py
```

Declared post-hoc experiment (negative result, ~43 min, GBM still ships):

```bash
.venv/Scripts/python.exe scripts/experiments/d2_cnn.py
.venv/Scripts/python.exe scripts/_claims_cnn.py          # regenerates claims_audit §1.17
```

Bearing pipeline (frozen) and demo scenarios:

```bash
.venv/Scripts/python.exe scripts/01_build_cache.py --jobs 4  # ~6 min, reads 21 GB
.venv/Scripts/python.exe scripts/02_train_lobo.py            # ~66 min on this CPU
.venv/Scripts/python.exe scripts/03_export_int8.py
.venv/Scripts/python.exe scripts/04_verify_golden.py
.venv/Scripts/python.exe scripts/demo/build_scenarios.py     # ~8 min (retrains 3 folds)
.venv/Scripts/python.exe scripts/demo/build_scenarios.py --skip-bearings   # ~10 s
```

RTL track (run from the repository root — `$readmemh` paths are relative to it):

```bash
.venv/Scripts/python.exe scripts/rtl/sweep_requant.py       # D-2 width sweep, ~8 min
.venv/Scripts/python.exe scripts/rtl/confirm_bit_exact.py   # full-cache check, ~7 min
.venv/Scripts/python.exe scripts/rtl/gen_rtl_params.py      # ds_gen.vh + requant .mem
.venv/Scripts/python.exe scripts/rtl/make_vectors.py --sets v3 v4 v5
.venv/Scripts/python.exe scripts/rtl/make_vectors.py --sets v1 --shards 65
scripts/rtl/run_tb.sh v4 v3 v5                              # ~12 min
scripts/rtl/run_tb.sh v1                                    # ~17 h, resumable
.venv/Scripts/python.exe scripts/rtl/render_results.py      # docs/results_rtl.md
```

```bash
D:/Vivado/2023.2/bin/vivado.bat -mode batch -nojournal \
  -log artifacts/rtl/synth/vivado_synth.log \
  -source scripts/rtl/synth.tcl -tclargs artifacts/rtl/synth 0 1   # ~4 min, core only, no bitstream
D:/Vivado/2023.2/bin/vivado.bat -mode batch -nojournal \
  -log artifacts/rtl/system/vivado_system.log \
  -source board/build_overlay.tcl                                   # ~12 min on 1 thread, full system + bitstream
.venv/Scripts/python.exe board/dryrun/run_dryrun.py                 # notebook vs mock pynq (NOT hardware)
```

Regenerate the results docs (never edit them by hand):

```bash
.venv/Scripts/python.exe scripts/05_render_results.py                 # docs/results.md
.venv/Scripts/python.exe scripts/60_render_multistage_results.py      # docs/results_multistage.md
```

**Timings are for this machine** — 4 threads, i3-1115G4, no GPU. The "minutes on a 3050"
figures in older docs and `folds.py:19` do not apply; re-time rather than quoting them.

---

## 8. Where to read next

`CLAUDE.md` first — it is short and holds the non-negotiables and the dataset traps.
Then `docs/workflow_v2.md` §13 for the decisions, `docs/claims_audit.md` for any number
you intend to quote, and the relevant `docs/data_notes_d*.md` before touching a dataset.
`docs/analysis_report.md` is the original pre-build audit; it is superseded where the
data notes disagree, and each disagreement is listed in the notes.
