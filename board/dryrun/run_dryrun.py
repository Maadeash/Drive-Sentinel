"""
Execute board/drivesentinel_overlay.ipynb against the software stand-in for pynq.

    .venv/Scripts/python.exe board/dryrun/run_dryrun.py

THIS IS NOT A HARDWARE RUN AND PRODUCES NO HARDWARE EVIDENCE. See
board/dryrun/pynq/__init__.py for exactly what it can and cannot catch. In short:
it tests the notebook's code -- offsets, decoding, the BUSY/DONE protocol, buffer
layout -- against the RTL's register semantics, with the bit-accurate model in
place of the fabric.

It runs in a temporary directory, so the notebook's final cell, which on a real
board writes `board_run.json` with `ran_on_hardware: true`, cannot drop that file
into board/. The runner rewrites the result as `board/dryrun/dryrun_result.json`
with `ran_on_hardware: false` and `mock: true`, and discards the timing figures,
which mean nothing when the "fabric" is numpy.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.dirname(HERE)


def main() -> int:
    nb = json.load(open(os.path.join(BOARD, "drivesentinel_overlay.ipynb"),
                        encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="ds_dryrun_")
    shutil.copy(os.path.join(BOARD, "test_windows.npz"), tmp)
    shutil.copy(os.path.join(BOARD, "mmio_resolve.py"), tmp)   # bundled with the notebook
    shutil.copytree(os.path.join(BOARD, "golden"), os.path.join(tmp, "golden"))

    sys.path.insert(0, HERE)            # the mock `pynq` shadows any real one
    cwd = os.getcwd()
    os.chdir(tmp)
    ns = {"__name__": "__main__"}
    log, ok = [], True
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, f"<cell {i}>", "exec"), ns)
                text = buf.getvalue()
                if "fabric_ms_at_100MHz" in src:
                    # the timing cell: numbers from numpy standing in for the
                    # fabric mean nothing, so they are not kept anywhere
                    text = "(timing output discarded: the mock does not model time)\n"
                log.append({"cell": i, "status": "ok", "stdout": text})
            except Exception:
                log.append({"cell": i, "status": "FAILED", "stdout": buf.getvalue(),
                            "error": traceback.format_exc()})
                ok = False
                break
        result = None
        if os.path.exists("board_run.json"):
            result = json.load(open("board_run.json"))
            result["ran_on_hardware"] = False
            result["mock"] = True
            result["board"] = "NONE -- software stand-in for pynq (board/dryrun/pynq)"
            result["timing"] = "discarded: the mock does not model time"
            result["pynq_version"] = "MOCK"
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    out = {
        "what_this_is": "notebook dry run against a software mock of pynq. NOT a "
                        "hardware run. Tests the notebook's code, not the FPGA.",
        "all_cells_ran": ok,
        "cells": log,
        "result": result,
    }
    with open(os.path.join(HERE, "dryrun_result.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    for entry in log:
        print(f"--- cell {entry['cell']}: {entry['status']}")
        if entry["stdout"].strip():
            print(entry["stdout"].rstrip())
        if entry["status"] != "ok":
            print(entry["error"])
    print("\nDRY RUN (mock pynq, not hardware):", "ALL CELLS RAN" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
