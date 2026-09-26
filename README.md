# DriveSentinel

**Real-time health monitoring for industrial drive systems, with AI in PYNQ-Z2 FPGA hardware at the edge.**

DriveSentinel listens to a drive's motor current and vibration (64 kHz), recognises early fault
fingerprints, and raises short, explained alerts before a failure stops the machine. It monitors
four stages of a drive:

| Stage | Signal | Method |
|---|---|---|
| Power supply | current in the three supply phases | fixed RMS threshold rule (lost phase) |
| Inverter | output phase currents, module temperature | gradient-boosted trees |
| Motor winding | the three motor phase currents | current-balance features + gradient-boosted trees |
| Bearing | motor current + vibration | DSP front end + INT8 1-D CNN on an FPGA |

Every stage feeds one evidence-fusion layer (rolling 10-window evidence, hysteresis, a
pre-registered rule for which stages may raise a full alarm) and a 21-byte, store-and-forward
alert packet.

## How the bearing stage works

```
raw current + vibration (64 kHz)
  -> anti-alias filter and decimate           ┐
  -> band-pass + Hilbert envelope             │  ARM Cortex-A9 × 2
  -> shaft speed from the current spectrum    │  (board/frontend.py)
  -> order spectra, 5 × 512, quantised to INT8┘
  -> INT8 CNN accelerator in FPGA fabric         Verilog RTL (rtl/)
  -> verdict: healthy / outer race / inner race
```

Fault frequencies are fixed multiples of shaft speed, so the spectra are resampled to **shaft
orders**: a fault appears at the same place at any speed. Speed is estimated from the current
itself; no tachometer.

## Results

Each figure is named with how it was measured; the full register is `docs/claims_audit.md`.

| | Result | Protocol |
|---|---|---|
| Bearing fault type | macro-F1 **0.785** | leave-one-bearing-out, 29 bearings (group holdout) |
| Shaft speed from current | median error **0.022 %** | vs the rig tachometer, all windows |
| FPGA vs reference model | **16,211 / 16,211** windows bit-exact | cycle-accurate simulation (XSim), V-1 |
| On the PYNQ-Z2 | **600 / 600** bit-exact, served over HTTP | board run |
| Latency per verdict | **1,786,724 cycles**, fixed; **18.07 ms** measured | PYNQ-Z2, DMA + inference |
| Same network on the board's ARM CPU | 40.89 ms (FPGA **2.3× faster**) | PYNQ-Z2 |
| Full system on the Zynq-7020 | 2,284 LUT (4.29 %), 12 BRAM, 4 DSP48 | Vivado 2023.2, post-route |
| Raw signal processed on the board | verdict = laptop verdict **196 / 196** | board DSP front end, observed |

## Hardware

PYNQ-Z2 (Xilinx Zynq-7020): the ARM cores run the signal processing, the FPGA fabric runs the
network. The accelerator, bitstream and board package are in `rtl/` and `board/`
(`board/README.md`, `board/SERVER.md`).

## Repository layout

```
drivesentinel/   the package: DSP, models, fusion, alerts, inference engines, data sources
rtl/             Verilog for the INT8 CNN accelerator
board/           PYNQ-Z2 package: bitstream, notebook, inference server, DSP front end
webapp/          technician web app (Starlette): fleet, alerts, live monitor, board live
scripts/         training, export, verification and synthesis pipelines
tests/           pytest suite
docs/            design, results and the claims register
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q            # test suite
.venv/Scripts/python.exe -m webapp               # web app on http://127.0.0.1:8000
```

The raw datasets are not in the repository (`data/`, `data_ext/`; see `.gitignore`). The board is
started with `sudo -E python3 server.py --host 0.0.0.0 --port 8765` on the PYNQ-Z2 and its address
is set on the web app's Settings page.

## Documentation

- `docs/claims_audit.md` — every headline number, its protocol and its source
- `docs/results_rtl.md` — accelerator results, generated from the run JSONs
- `docs/README_v5_pipeline.md` — the detailed bearing-pipeline write-up
- `docs/handoff.md` — project state and history
