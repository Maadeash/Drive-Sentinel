# Board inference server — connecting the PYNQ-Z2 to the web app

> **RUN ON THE BOARD, 2026-09-24.** Started on the PYNQ-Z2 exactly as below and pointed
> at by the web app: the FPGA served every bearing window (960, no fallback), and
> on 600 windows its answers matched the bit-accurate model exactly, at a median
> 40.9 ms round trip (`artifacts/webapp/board_server_check.json`,
> `docs/claims_audit.md` §1.27). The command that worked on PYNQ 2.5 was
> `sudo -E python3 server.py --host 0.0.0.0 --port 8765`, run from the folder that holds the
> notebook's files; the board's hardware evidence is still the notebook's `board_run.json`
> (§1.26).
>
> **A real board run of the notebook on 2026-09-24 found `ol.ds_0` is not itself the
> register file** (`ds_0` was never packaged as an IP-XACT core, so PYNQ exposed it as
> `ol.ip_dict['ds_0/s_axi']`, not `ol.ip_dict['ds_0']`). `server.py` and the notebook now
> both resolve the registers through `mmio_resolve.py` instead of assuming `ol.ds_0`
> directly. That fix **has now run on the board** -- the notebook completed on the PYNQ-Z2
> on 2026-09-24 (`board_run.json`, `docs/claims_audit.md` §1.26), and the server
> after it (§1.27). It also needed a Python 3.6 fallback for `ThreadingHTTPServer` (3.7+), added
> before it was ever run there. See `README.md` "Things that will look like bugs and are
> not" for everything the board's older Python stack changes.

`server.py` loads the overlay and checks that the core's build-ID register reads
`0x23eb56bf`. If it doesn't, the server refuses to start. It then serves one bearing window
per HTTP request through the AXI DMA and returns the class, the three scaled-logit registers
and the measured cycle count. It uses the notebook's BUSY-rise-then-fall protocol, because
`DONE` is sticky after the first window (`README.md`, protocol note).

It needs `numpy` and `pynq`, both already on a PYNQ image, and the Python standard library.
Nothing else is installed, other than `mmio_resolve.py` itself (copied alongside it below).

## 1. Copy the files to the board

From `board/` on the development machine. The server needs the same bitstream, handoff,
`golden/` folder and `mmio_resolve.py` as the notebook, plus `server.py`:

```bash
sha256sum -c MANIFEST.sha256
ssh xilinx@pynq "mkdir -p ~/drivesentinel"
scp server.py mmio_resolve.py drivesentinel.bit drivesentinel.hwh MANIFEST.sha256 xilinx@pynq:~/drivesentinel/
scp -r golden xilinx@pynq:~/drivesentinel/
ssh xilinx@pynq "cd ~/drivesentinel && sha256sum -c --ignore-missing MANIFEST.sha256"
```

The `.bit` and `.hwh` must sit in the same folder with the same base name. Replace `pynq`
with the board's address if the name doesn't resolve. A PYNQ-Z2 on a direct cable defaults to
`192.168.2.99`.

## 2. Start the server on the board

Loading an overlay needs root and the PYNQ Python environment. On PYNQ 2.7 and later, that
environment is sourced from `/etc/profile.d/pynq_venv.sh`:

```bash
ssh xilinx@pynq
cd ~/drivesentinel
sudo bash -c "source /etc/profile.d/pynq_venv.sh && python3 server.py --host 0.0.0.0 --port 8765"
```

(On images without that file, `sudo -E python3 server.py --host 0.0.0.0 --port 8765`.)
**Untested**: which of the two applies depends on the image version.

Expected output:

```
DriveSentinel core ok: {'magic': '0x44533031', 'build_id': '0x23eb56bf', 'config': {...}}
python 3.6.x, numpy 1.13.3, scipy <version>; DSP front end: available, FFT from numpy.fft
serving on http://0.0.0.0:8765  (GET /health, POST /process, POST /infer)
```

The second line is new with §4. If it says `DSP front end: NOT AVAILABLE -- ...`, `/infer`
still works (the web app's fleet is unaffected) and §4 says what to fix.

If it prints `REFUSED: BUILD_ID 0x…, expected 0x23eb56bf` and exits with status 2, the
weights in the fabric are not the export this package claims (`claims_audit.md` §1.4).
Stop there.

Check it from the development machine:

```bash
curl http://192.168.2.99:8765/health
```

## 3. Point the web app at it

Either:

* **Settings → Bearing engine → Board address**: enter `192.168.2.99:8765` and press
  **Save**. The address is kept in `webapp/var/engine.json` (not committed). Or
* start the app with `DS_BOARD_ADDRESS=192.168.2.99:8765`. The environment variable takes
  precedence over the saved address.

The Settings page shows which engine is active:

| Board | Active engine | Board line |
|---|---|---|
| no address set | **Software** | No address set |
| address set, board not answering | **Software** | Unreachable · using Software |
| board answering with the wrong build ID | **Software** | Build ID mismatch · using Software |
| board answering, build ID `0x23eb56bf` | **FPGA (PYNQ-Z2)** | Ready · build ID 0x23eb56bf |

The fallback is automatic and per window. A failed request is answered by the software
engine, the bit-accurate model whose register values the FPGA must reproduce. The router
retries the board every 10 s and returns to it when it answers. The Windows counter shows how
many bearing windows each engine served. Only the bearing stage uses an engine. The supply,
inverter and winding stages run in software either way.

## 4. Raw signal: the board does the signal processing (`POST /process`)

With `/infer` the laptop computes the spectra and sends the board a finished 5 × 512
int8 picture. With `/process` the laptop sends the **raw** 64 kHz signal: 4 s of motor
current (two phases) and vibration, 6.1 MB, exactly as the data file holds it. The board
does the rest itself: anti-alias filtering and decimation, band-pass and Hilbert
envelopes, shaft speed from the current, the five spectra, median and `log1p`, the
quantiser, then the FPGA for each of the 7 windows. It returns the verdicts, its int8 input
and a per-stage timing. The web app's **Board live** page drives this and shows it.

`frontend.py` is `drivesentinel/dsp.py` ported for the board's stack. The filters are
designed on the laptop and shipped in `frontend_params.json`
(`make_frontend_params.py`), so the board needs only `scipy.signal.sosfilt` (scipy ≥ 0.16)
plus numpy. It uses `scipy.fft` when the board's scipy has it (1.4+), else `numpy.fft`. On
the laptop both give network inputs identical to `dsp.py`'s (`tests/test_board_frontend.py`).
**On the board this has not run yet.** Whether the board's scipy has `sosfilt` shows on the
start-up line above.

Copy the two new files and the updated server next to the existing ones. On the board used
on 2026-09-24 that folder is `~/jupyter_notebooks/drivesentinel/`. From `board/` on the
laptop:

```bash
scp server.py frontend.py frontend_params.json xilinx@192.168.2.99:~/jupyter_notebooks/drivesentinel/
```

Then stop the running server (Ctrl+C in its SSH window) and start it again as in §2.

**Stop it with Ctrl+C, never with `pkill -f server.py`.** PYNQ 2.5's own PL server is
`pynq/pl_server/server.py`, so that pattern kills it too, and the next overlay load fails
with `ConnectionRefusedError: [Errno 111]` inside `pl_server/server.py` `client_request`.
It happened on 2026-09-24. Recover with `sudo systemctl restart pl_server` or, failing
that, `sudo reboot`. If there's no window to Ctrl+C, use
`sudo pkill -f "server.py --host"`, which matches only this server's command line.
`GET /health` now also reports `python`, `numpy`, `scipy`, `frontend` and `machine` (CPU
architecture, governor, clock, temperature).

`--workers 2` (the default) runs the DSP in two worker processes, one per Cortex-A9 core,
forked before the overlay loads. The output is identical to one process. On one core the
board took 5.23 s per 4 s of signal (`claims_audit.md` §1.29). `--workers 1` restores that.

To record the check (board int8 vs laptop, byte for byte, and the board's timings) from the
laptop:

```bash
.venv/Scripts/python.exe scripts/webapp/check_board_frontend.py --board 192.168.2.99:8765
```

That writes `artifacts/webapp/board_frontend_check.json`. `scripts/rtl/render_results.py`
turns it into `docs/results_rtl.md` §4d, which reads `NOT RUN` until then.

## What the board's verdict drives, and what it does not

The shipped INT8 network was trained on **all 29** Paderborn bearings
(`train.train_deployment_model`). On the recorded bearing windows the app streams, its verdict
is therefore **in-sample**. On KI05's stored windows it is right 85 % of the time. The fold
model that held KI05 out is right 1.67 % of the time
(`artifacts/webapp/deployed_model_in_sample.json`, from
`scripts/webapp/check_deployed_in_sample.py`). So for recorded windows the engine,
board or software, ran on every window while the bearing stage's status still came from the
held-out fold model. **Since 2026-09-24, by the owner's decision, the engine's verdict drives
the bearing stage** (`BoardSource(verdict="engine")` in `webapp/server.py`): on the recorded
data those alerts are in-sample -- Lift 01 (KI05) now raises an ALARM -- and are labelled
(leaky reference) in `docs/claims_audit.md` §1.28. Software and FPGA give identical alerts.
`verdict="held_out"` restores the previous behaviour.

## What this does and does not establish

| | Status |
|---|---|
| Server code against the RTL's register semantics (mock pynq): 120 windows, class = golden reference, logit registers = RTL model, cycles = 1,786,724 | tested, `tests/test_board_server.py`. **True by construction behind the mock**; it checks the server's code, not the FPGA |
| Build-ID refusal | tested (mock) |
| Web-app fallback to software and recovery | tested (real HTTP, mock board) |
| The server on a PYNQ-Z2, serving the web app | **run, 2026-09-24**: 600 / 600 bit-exact (`claims_audit.md` §1.27) |
| `board_run.json` | **exists**: the notebook on the board, 120 / 120 bit-exact (§1.26) |
| `POST /process`, the DSP front end on the board's ARM (mock pynq, laptop CPU) | tested, `tests/test_board_frontend.py`: int8 identical to `dsp.py`, logits exact |
| `POST /process` on a PYNQ-Z2 | **NOT RUN** (`claims_audit.md` §1.29) |
