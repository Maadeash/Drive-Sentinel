"""
Generate board/drivesentinel_overlay.ipynb.

Written as a generator rather than edited as JSON so the notebook's source is
reviewable in an ordinary diff, and so it cannot drift from the register map:
the offsets below are read from rtl/ds_defs.vh at generation time, not typed.

    .venv/Scripts/python.exe board/make_notebook.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def reg_map():
    txt = open(os.path.join(ROOT, "rtl", "ds_defs.vh"), encoding="utf-8").read()
    regs = {m.group(1): int(m.group(2), 16) for m in
            re.finditer(r"`define DS_REG_(\w+)\s+8'h([0-9A-Fa-f]+)", txt)}
    bits = {m.group(1): int(m.group(2)) for m in
            re.finditer(r"`define DS_ST_(\w+)\s+(\d+)", txt)}
    magic = int(re.search(r"`define DS_MAGIC\s+32'h([0-9A-Fa-f]+)", txt).group(1), 16)
    return regs, bits, magic


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def main():
    regs, bits, magic = reg_map()
    params = json.load(open(os.path.join(ROOT, "artifacts", "rtl", "rtl_params.json")))
    build_id = params["expected_build_id"]

    cells = [
        md("# DriveSentinel INT8 bearing-CNN overlay — PYNQ-Z2",
           "",
           "> ## This notebook has been run on a PYNQ-Z2 (2026-09-24).",
           "> Its result is `board_run.json`, committed at `board/board_run.json` and "
           "rendered in `docs/results_rtl.md` §4c. That run found three board-only "
           "problems -- register naming, the board's 32-bit numpy, and its older "
           "`np.pad` -- each fixed below and noted where it applies. Running the "
           "notebook again overwrites `board_run.json` on the board.",
           "",
           "**What it does, in order**",
           "",
           "1. Loads `drivesentinel.bit` / `.hwh` and finds the DMA and the core.",
           f"2. Reads the build-ID register and checks it equals `0x{build_id:08x}` "
           "— the CRC-32 the hardware computes over the weights and biases it "
           "actually loaded (`docs/rtl_declarations.md` D-3).",
           "3. Runs the PS front end's quantiser (D-4) on 120 real bearing windows "
           "and checks it reproduces the packaged int8 bytes.",
           "4. Streams each window to the core through the AXI DMA, reads the class "
           "and the three logit registers back over AXI-Lite.",
           "5. Compares every verdict against `golden_reference.py`, run here on the "
           "board's own ARM cores, and against the RTL model's predicted registers.",
           "6. Times it, and writes everything to `board_run.json`.",
           "",
           "**What it does not do.** It does not time the DSP front end (FFT, "
           "envelope, order resampling): the windows arrive already conditioned. "
           "That measurement has its own plan, `docs/frontend_timing_plan.md`, and "
           "its own bar. The timing here is the CNN path only."),

        md("## 1. Load the overlay",
           "",
           "**Register access note (found on a real board, 2026-09-24).** `ds_0` was "
           "added to the block design as a bare RTL module reference "
           "(`board/build_overlay.tcl`), never packaged as an IP-XACT core, so PYNQ "
           "is not guaranteed to expose its registers at `ol.ds_0` directly -- on that "
           "run it showed up as `ol.ip_dict['ds_0/s_axi']` instead, with `ol.ds_0` a "
           "bare hierarchy proxy. `mmio_resolve.py` (bundled next to this notebook) "
           "finds the registers wherever PYNQ put them; see it for the full story. "
           "On the board it resolved to PYNQ's own IP driver (a `DefaultIP`) at "
           "`ol.ds_0.s_axi`."),
        code("import json, os, sys, time",
             "import numpy as np",
             "from pynq import Overlay, allocate",
             "",
             "HERE = os.path.abspath('.')",
             "sys.path.insert(0, HERE)      # mmio_resolve.py sits next to this notebook",
             "from mmio_resolve import resolve_mmio",
             "",
             "ol = Overlay(os.path.join(HERE, 'drivesentinel.bit'))",
             "print('IP blocks:', sorted(ol.ip_dict))",
             "print('hierarchies:', sorted(getattr(ol, 'hierarchy_dict', {})))",
             "",
             "dma  = ol.axi_dma_0",
             "core = resolve_mmio(ol, 'ds_0')   # rtl/ds_top.v, unchanged, inside board/rtl/ds_pynq_wrap.v",
             "core_info = next((v for k, v in ol.ip_dict.items()",
             "                 if k == 'ds_0' or k.startswith('ds_0/')), None)",
             "print('core register interface:', core)",
             "if core_info:",
             "    print('  at', hex(core_info['phys_addr']), 'range', hex(core_info['addr_range']))"),

        md("## 2. Register map and the build-ID check",
           "",
           "Offsets generated from `rtl/ds_defs.vh` by `board/make_notebook.py`. "
           "The build-ID walk takes about 28,000 cycles (0.28 ms) after the PL "
           "comes out of reset; the core refuses to start until it has finished, "
           "so the notebook waits for it too."),
        code(*[f"REG_{k:8s} = 0x{v:02X}" for k, v in sorted(regs.items(), key=lambda kv: kv[1])],
             "",
             *[f"ST_{k:9s} = {v}" for k, v in sorted(bits.items(), key=lambda kv: kv[1])],
             "",
             f"MAGIC             = 0x{magic:08X}   # 'DS01'",
             f"EXPECTED_BUILD_ID = 0x{build_id:08X}",
             f"EXPECTED_CYCLES   = 1786724       # measured in simulation, every window",
             f"FRAC_W, LANES, LOGIT_SHIFT = {params['frac_width']}, 1, {params['logit_shift']}",
             "",
             "def rd(off):   return core.read(off)",
             "def wr(off, v): core.write(off, v)",
             "def signed32(v): return v - (1 << 32) if v & 0x80000000 else v",
             "",
             "t0 = time.time()",
             "while not (rd(REG_STATUS) >> ST_CRC_DONE) & 1:",
             "    if time.time() - t0 > 1.0:",
             "        raise RuntimeError('build-ID walk did not finish within 1 s')",
             "",
             "magic    = rd(REG_MAGIC)",
             "build_id = rd(REG_BUILDID)",
             "config   = rd(REG_CONFIG)",
             "print(f'MAGIC    0x{magic:08x}  ' + ('ok' if magic == MAGIC else 'WRONG'))",
             "print(f'BUILD_ID 0x{build_id:08x}  expected 0x{EXPECTED_BUILD_ID:08x}  '",
             "      + ('MATCH' if build_id == EXPECTED_BUILD_ID else 'MISMATCH'))",
             "print(f'CONFIG   frac_w={config & 0xFF} lanes={(config >> 8) & 0xFF} '",
             "      f'logit_shift={(config >> 16) & 0xFF}')",
             "",
             "assert magic == MAGIC, 'not the DriveSentinel core at this address'",
             "assert build_id == EXPECTED_BUILD_ID, (",
             "    'the loaded weights are not the export this bitstream claims '",
             "    '(claims_audit.md section 1.4) -- stop here')",
             "assert (config & 0xFF, (config >> 8) & 0xFF, (config >> 16) & 0xFF) == \\",
             "    (FRAC_W, LANES, LOGIT_SHIFT)"),

        md("## 3. Test windows, and the PS-side quantiser",
           "",
           "`test_windows.npz`: 120 real windows from the order-spectrum cache, 40 "
           "per class across all 29 bearings, with the float DSP output, the int8 "
           "bytes, and what the RTL model and the golden reference each predict. "
           "The quantiser below is the PS front end's last step "
           "(`docs/rtl_declarations.md` D-4) — it is **not** in the fabric, which is "
           "why it has to be run and checked here."),
        code("# The PYNQ-Z2 image's numpy predates 1.17, where np.pad's `mode` became",
             "# optional (default 'constant'). golden_reference.py is generated and frozen",
             "# and calls np.pad without a mode -- found on the board, 2026-09-24. Supply",
             "# the same default here; on newer numpy this changes nothing.",
             "print('numpy', np.__version__)",
             "if not getattr(np.pad, '_ds_default_mode', False):",
             "    _np_pad = np.pad",
             "    def _pad_constant_default(array, pad_width, mode='constant', **kwargs):",
             "        return _np_pad(array, pad_width, mode, **kwargs)",
             "    _pad_constant_default._ds_default_mode = True",
             "    np.pad = _pad_constant_default",
             "",
             "sys.path.insert(0, os.path.join(HERE, 'golden'))",
             "import golden_reference as golden      # numpy only; reads golden/*.mem",
             "",
             "T = np.load(os.path.join(HERE, 'test_windows.npz'))",
             "x_float, x_int8_ref = T['x_float'], T['x_int8']",
             "N = len(x_float)",
             "# .astype(np.intp): the PYNQ-Z2 is 32-bit ARM, whose numpy refuses to",
             "# bincount an int64 array (found on the board, 2026-09-24)",
             "print(N, 'windows; classes', np.bincount(T['label'].astype(np.intp)))",
             "",
             "def ps_quantise(x):",
             "    '''D-4: standardise, scale, round half to even, clamp -- on the PS.'''",
             "    a = (np.asarray(x, np.float64) - golden.INPUT_MEAN) / golden.INPUT_STD",
             "    return np.clip(np.rint(a / golden.LAYERS[0]['s_x']), -128, 127).astype(np.int8)",
             "",
             "x_int8 = np.stack([ps_quantise(w) for w in x_float])",
             "assert np.array_equal(x_int8, x_int8_ref), 'PS quantiser disagrees with the packaged bytes'",
             "print('PS quantiser reproduces the packaged int8 bytes on all', N, 'windows')"),

        md("## 4. One inference through the DMA",
           "",
           "**Protocol note — a property of the verified RTL, recorded rather than "
           "fixed.** A completed DMA frame starts an inference automatically. The "
           "STATUS `DONE` bit, however, is cleared only by a write to `CTRL.start`, "
           "so after the first window it stays set and cannot tell one inference "
           "from the next. This function therefore waits for `BUSY` to rise and "
           "then fall, and checks the `CYCLES` register against the simulated "
           "count as proof a whole inference ran. Changing `DONE`'s semantics is a "
           "v2 RTL change and is listed in `docs/handoff.md`; `rtl/` was not "
           "modified for the board package."),
        code("buf = allocate(shape=(2560,), dtype=np.uint8)   # one window, channel-major",
             "",
             "def infer(window_int8, timeout_s=1.0):",
             "    buf[:] = np.ascontiguousarray(window_int8).reshape(-1).view(np.uint8)",
             "    buf.flush()",
             "    t_start = time.perf_counter()",
             "    dma.sendchannel.transfer(buf)",
             "    dma.sendchannel.wait()",
             "    t_dma = time.perf_counter()",
             "    t0 = time.time()",
             "    while not (rd(REG_STATUS) >> ST_BUSY) & 1:          # wait for the start",
             "        if time.time() - t0 > 0.05:",
             "            raise RuntimeError('inference did not start after the frame')",
             "    while (rd(REG_STATUS) >> ST_BUSY) & 1:              # wait for the end",
             "        if time.time() - t0 > timeout_s:",
             "            raise RuntimeError('inference did not finish')",
             "    t_end = time.perf_counter()",
             "    return {",
             "        'class':  rd(REG_CLASS) & 0x3,",
             "        'logits': [signed32(rd(o)) for o in (REG_LOGIT0, REG_LOGIT1, REG_LOGIT2)],",
             "        'cycles': rd(REG_CYCLES),",
             "        'dma_s':  t_dma - t_start,",
             "        'total_s': t_end - t_start,",
             "    }",
             "",
             "r0 = infer(x_int8[0])",
             "print(r0)",
             "assert r0['cycles'] == EXPECTED_CYCLES, f\"cycles {r0['cycles']} != {EXPECTED_CYCLES}\""),

        md("## 5. All 120 windows against the golden reference",
           "",
           "Three comparisons per window. **Class vs `golden_reference.predict`** is "
           "the V-1 question asked on silicon. **Logit registers vs the RTL model** "
           "is V-2 — expected to be exact. **Cycles** must equal the simulated "
           "1,786,724: the engine has no data-dependent paths."),
        code("golden_pred = golden.predict(x_float.astype(np.float64))   # on the ARM cores",
             "assert np.array_equal(golden_pred, T['pred_float']), 'golden on the board differs from golden on the PC'",
             "",
             "results = [infer(w) for w in x_int8]",
             "hw_pred   = np.array([r['class'] for r in results])",
             "hw_logits = np.array([r['logits'] for r in results], dtype=np.int64)",
             "hw_cycles = np.array([r['cycles'] for r in results])",
             "",
             "class_miss = np.flatnonzero(hw_pred != golden_pred)",
             "logit_miss = np.flatnonzero((hw_logits != T['logit_reg']).any(axis=1))",
             "cycle_miss = np.flatnonzero(hw_cycles != EXPECTED_CYCLES)",
             "print(f'class vs golden_reference : {N - len(class_miss)}/{N} agree')",
             "print(f'logit registers vs model  : {N - len(logit_miss)}/{N} exact')",
             "print(f'cycles == {EXPECTED_CYCLES:,}      : {N - len(cycle_miss)}/{N}')",
             "if len(class_miss):",
             "    print('DISAGREEMENTS at cache windows', T['window'][class_miss].tolist())",
             "print('accuracy vs ground truth (in-sample windows, NOT a generalisation '",
             "      'estimate):', float((hw_pred == T['label']).mean()))"),

        md("## 6. Timing — the CNN path only",
           "",
           "Three numbers per window, reported as distributions (median and p95, "
           "not means — a real-time claim is about the tail):",
           "",
           "* **fabric** — `CYCLES / 100 MHz`: the accelerator alone. Should read "
           "17.87 ms.",
           "* **DMA + inference, wall clock** — from `transfer()` to `BUSY` falling, "
           "on the PS. Includes the DMA, the MMIO polling and Python.",
           "* **golden_reference on the ARM** — the same model in numpy on the "
           "Cortex-A9, for scale.",
           "",
           "None of these is end-to-end latency: the DSP front end is not in this "
           "notebook (`docs/frontend_timing_plan.md`)."),
        code("def dist(a):",
             "    a = np.asarray(a) * 1e3",
             "    return {'median_ms': float(np.median(a)), 'p95_ms': float(np.percentile(a, 95)),",
             "            'min_ms': float(a.min()), 'max_ms': float(a.max())}",
             "",
             "REPS = 3",
             "wall, dmas = [], []",
             "for _ in range(REPS):",
             "    for w in x_int8:",
             "        r = infer(w)",
             "        wall.append(r['total_s']); dmas.append(r['dma_s'])",
             "",
             "t = time.perf_counter()",
             "for w in x_float[:20]:",
             "    golden.predict(w[None].astype(np.float64))",
             "cpu = (time.perf_counter() - t) / 20",
             "",
             "fabric_ms = float(np.median(hw_cycles)) / 100e6 * 1e3",
             "timing = {",
             "    'fabric_ms_at_100MHz': fabric_ms,",
             "    'dma_plus_inference_wall': dist(wall),",
             "    'dma_transfer_wall': dist(dmas),",
             "    'golden_reference_on_arm_ms_per_window': cpu * 1e3,",
             "    'windows_timed': len(wall),",
             "}",
             "print(json.dumps(timing, indent=2))"),

        md("## 7. Save the evidence",
           "",
           "`board_run.json` is the only thing that can turn any hardware claim in "
           "this project from `NOT RUN` into a result. Copy it back into the "
           "repository at `board/board_run.json` and commit it; "
           "`docs/claims_audit.md` §1.21 says what may then be claimed and how."),
        code("import platform, datetime, pynq",
             "# True only when the real pynq package drove a real board. The software",
             "# stand-in used for the dry run (board/dryrun/pynq) reports 'MOCK...'.",
             "ON_HARDWARE = not str(pynq.__version__).startswith('MOCK')",
             "run = {",
             "    'ran_on_hardware': ON_HARDWARE,",
             "    'date_utc': datetime.datetime.utcnow().isoformat() + 'Z',",
             "    'board': 'PYNQ-Z2 (edit if different)',",
             "    'pynq_version': __import__('pynq').__version__,",
             "    'numpy_version': np.__version__,",
             "    'platform': platform.platform(),",
             "    'bitstream': 'drivesentinel.bit',",
             "    'build_id': f'0x{build_id:08x}',",
             "    'build_id_match': build_id == EXPECTED_BUILD_ID,",
             "    'windows': int(N),",
             "    'class_agree_with_golden': int(N - len(class_miss)),",
             "    'class_disagreement_windows': T['window'][class_miss].tolist(),",
             "    'logit_registers_exact': int(N - len(logit_miss)),",
             "    'cycles_equal_simulation': int(N - len(cycle_miss)),",
             "    'timing': timing,",
             "}",
             "with open(os.path.join(HERE, 'board_run.json'), 'w') as fh:",
             "    json.dump(run, fh, indent=2)",
             "print(json.dumps({k: v for k, v in run.items() if k != 'timing'}, indent=2))"),
    ]

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"},
                       "drivesentinel": {"board_run": "board/board_run.json",
                                         "generated_by": "board/make_notebook.py"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(HERE, "drivesentinel_overlay.ipynb")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
