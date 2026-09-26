# Board package — DriveSentinel INT8 bearing CNN on a PYNQ-Z2

> **The notebook has been run on a PYNQ-Z2 (2026-09-24).** Its result is
> `board_run.json` in this folder, rendered in `docs/results_rtl.md` §4c and registered in
> `docs/claims_audit.md` §1.26: build ID matched, 120/120 windows agreed with the golden
> reference, logit registers bit-exact, cycle counts equal to simulation. `server.py` has
> since served the web app from the board too (`SERVER.md`, `docs/claims_audit.md` §1.27).
> The first attempt found three board-only problems, each fixed and listed under "Things
> that will look like bugs and are not" below.

---

## What is here

| File | What it is | Copy to board? |
|---|---|---|
| `drivesentinel.bit` | the PL bitstream: Zynq PS7 + AXI DMA + the verified core | **yes** |
| `drivesentinel.hwh` | hardware handoff; PYNQ reads the IP names, addresses and the 100 MHz fabric clock from it. **Must sit next to the `.bit` with the same base name.** | **yes** |
| `drivesentinel_overlay.ipynb` | load, build-ID check, 120 real windows through the DMA, compare with the golden reference, timing | **yes** |
| `mmio_resolve.py` | finds the core's registers wherever PYNQ exposed them (`ds_0` was never packaged as an IP-XACT core -- see the note below). The notebook imports it | **yes** |
| `test_windows.npz` | 120 real cache windows (40 per class, all 29 bearings) + expected results, 1.3 MB | **yes** |
| `golden/` | `golden_reference.py` + `scales.json` + `input_norm.json` + the ten `.mem` files, copied unchanged from `artifacts/int8_export/` | **yes, as a folder** |
| `MANIFEST.sha256` | checksums of everything above | **yes** |
| `README.md` | this file | no |
| `rtl/ds_pynq_wrap.v` | the wrapper that puts `rtl/ds_top.v` (unchanged) into the block design | no |
| `build_overlay.tcl` | rebuilds `.bit` + `.hwh` from source | no |
| `make_test_windows.py`, `make_notebook.py` | regenerate the test set and the notebook | no |
| `dryrun/` | the software stand-in for `pynq` and its runner | no |

## What the board needs

- A **PYNQ-Z2** running an official **PYNQ image**. The notebook is written against the PYNQ 2.7+ API and is untested on any version (it uses
  `pynq.allocate`, `Overlay`, and the simple-mode AXI DMA driver, all long-standing parts of the API).
- Network access to the board's Jupyter server (default `http://pynq:9090`, or the
  board's IP), or an SMB share (`\\pynq\xilinx`), or `scp` (user `xilinx`).
- `numpy` — already on every PYNQ image. Nothing else is installed.

## Order of operations

**1. Check the files before copying** (on the development machine, from `board/`):

```bash
sha256sum -c MANIFEST.sha256
```

**2. Make a folder on the board** and copy the seven items into it, keeping `golden/` as
a folder. With `scp` (password `xilinx` on a stock image):

```bash
ssh xilinx@pynq "mkdir -p ~/jupyter_notebooks/drivesentinel"
scp drivesentinel.bit drivesentinel.hwh drivesentinel_overlay.ipynb mmio_resolve.py test_windows.npz MANIFEST.sha256 xilinx@pynq:~/jupyter_notebooks/drivesentinel/
scp -r golden xilinx@pynq:~/jupyter_notebooks/drivesentinel/
```

Or drag the same files into `jupyter_notebooks/drivesentinel/` through the Jupyter
upload button or the SMB share. **The `.bit` and `.hwh` must be in the same folder,
with the same base name**, or PYNQ will not find the handoff file and will not know the
fabric clock or the addresses.

**3. Check them on the board:**

```bash
ssh xilinx@pynq "cd ~/jupyter_notebooks/drivesentinel && sha256sum -c MANIFEST.sha256"
```

**4. Open `drivesentinel_overlay.ipynb` in the board's Jupyter and run the cells in
order, top to bottom.** Each section stops with an assertion if its check fails. Do
not skip past a failure:

| Section | Checks | If it fails |
|---|---|---|
| 1 | the overlay loads; `ds_0` and `axi_dma_0` are present; `mmio_resolve.resolve_mmio` finds the core's registers | wrong PYNQ version, `.hwh` not next to `.bit`, or `mmio_resolve.py` not copied alongside the notebook |
| 2 | `MAGIC` = `0x44533031`, **`BUILD_ID` = `0x23eb56bf`**, `CONFIG` = F 26 / 1 lane / shift 16 | **Stop.** The weights in the fabric are not the export this package claims (`claims_audit.md` §1.4). |
| 3 | the PS-side quantiser reproduces the packaged int8 bytes | numpy differs from the build machine's; report it |
| 4 | one window through the DMA; `CYCLES` = 1,786,724 | DMA or register-protocol problem; see the protocol note below |
| 5 | 120 windows: class vs `golden_reference`, logit registers vs the model, cycles | any disagreement names the cache window |
| 6 | timing, as distributions | — |
| 7 | writes `board_run.json` | — |

**5. Copy `board_run.json` back** into this folder and commit it. That file is the
only thing that can move any hardware claim in this project from `NOT RUN` to a
result.

```bash
scp xilinx@pynq:~/jupyter_notebooks/drivesentinel/board_run.json board/
```

## Things that will look like bugs and are not

- **The core's register map repeats inside its window.** The core decodes 8 address
  bits (a 256-byte map); the block design gives it the minimum 4 KB window at
  `0x43C00000`, so offsets `0x100`, `0x200`, ... alias the same registers.
- **`STATUS.DONE` stays set after the first inference.** In the verified RTL, `DONE` is
  cleared only by a write to `CTRL.start`, and a DMA frame starts an inference without
  one. The notebook therefore waits for `BUSY` to rise and fall and checks `CYCLES`,
  rather than polling `DONE`. This is a property of `rtl/ds_axil_ctrl.v`, recorded in
  `docs/handoff.md` as a v2 change; `rtl/` was not modified for the board package.
- **`ol.ds_0` is not itself the register file.** `ds_0` was added to the block design as
  a bare RTL module reference (`create_bd_cell -type module -reference`), never packaged
  as an IP-XACT core, so PYNQ is not guaranteed to expose its registers at `ol.ds_0`
  directly. A real board run on 2026-09-24 showed `ol.ip_dict['ds_0/s_axi']` instead of
  `ol.ip_dict['ds_0']`, with `ol.ds_0` a bare hierarchy proxy (`AttributeError` on
  `.read()`). Cell 1 now calls `mmio_resolve.resolve_mmio(ol, 'ds_0')`, which finds the
  registers by whichever path PYNQ actually used; on the board that was
  `ol.ds_0.s_axi`. `board/mmio_resolve.py` has the full finding.
- **The board's Python stack is older than the build machine's.** PYNQ 2.5 ships Python
  3.6 and numpy 1.13 on 32-bit ARM. Three consequences, all hit on the first run and
  fixed: `from __future__ import annotations` (3.7+) removed from the files that run on
  the board; `np.bincount` given an explicit `np.intp` array; and `np.pad`'s `mode`,
  optional only from numpy 1.17, supplied in section 3 of the notebook, because
  `golden/golden_reference.py` is generated and frozen. The 64-bit dry run cannot
  reproduce these by executing them, so `tests/test_board_package.py` checks each one.
- **The interrupt is not wired.** The core's `irq` is a one-cycle pulse; a pulse into a
  level-sensitive GIC input can be missed, so the notebook polls. The cost is
  microseconds against a 17.9 ms inference.
- **The PS settings in the bitstream are not what configures the PS.** No PYNQ-Z2 board
  file was installed on the build machine, so the PS7 was configured property by
  property (all 23 applied — `artifacts/rtl/system/ps7_config.log`). A PYNQ overlay does
  not reconfigure DDR or MIO — the image's boot loader already did — so the values that
  matter from this bitstream are the 100 MHz fabric clock and the address map, both of
  which are in the `.hwh` and were checked.

## What the numbers will mean

- **Class agreement** with `golden_reference` on 120 windows is a spot check of V-1 on
  silicon. V-1 proper is all 16,211 windows in simulation (`docs/results_rtl.md` §3).
- **`fabric` timing** (`CYCLES` / 100 MHz) is the accelerator alone and should read
  17.87 ms — the simulated figure.
- **DMA + inference wall-clock** is measured on the PS and includes Python and the
  register polling. Report the distribution the notebook prints, not a single number.
- **None of it is end-to-end latency.** The DSP front end (FFT, envelope, order
  resampling) is not in this notebook. Its method and bar are in
  `docs/frontend_timing_plan.md`, and it remains `NOT MEASURED`.
- **Accuracy** printed in section 5 is on in-sample windows and is **not** a
  generalisation estimate. The model's honest figure stays macro-F1 0.7852,
  leave-one-bearing-out.

## Rebuilding the package

From the repository root, with Vivado 2023.2:

```bash
D:/Vivado/2023.2/bin/vivado.bat -mode batch -nojournal -log artifacts/rtl/system/vivado_system.log -source board/build_overlay.tcl
.venv/Scripts/python.exe board/make_test_windows.py
.venv/Scripts/python.exe board/make_notebook.py
.venv/Scripts/python.exe board/dryrun/run_dryrun.py
cd board && sha256sum drivesentinel.bit drivesentinel.hwh drivesentinel_overlay.ipynb mmio_resolve.py test_windows.npz golden/* > MANIFEST.sha256
```

The build took about 12 minutes on one thread on the build machine. It writes a bitstream **only** if the
full system closes timing at 100 MHz; the constraint is the PS's own `FCLK_CLK0` period
and is never relaxed.
