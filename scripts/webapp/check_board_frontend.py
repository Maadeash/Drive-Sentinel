"""
Does the board's own signal processing compute what the laptop computes, and how
long does it take on the PYNQ-Z2's ARM?

Sends raw 64 kHz test-rig data -- motor current (two phases) and vibration, 4 s
per data file, straight from data/<bearing>/*.mat -- to board/server.py's
POST /process. The board runs the DSP front end (board/frontend.py) on its ARM
cores, quantises, and runs every window on the FPGA. For every window this
compares, against the laptop:

  * the board's int8 network input with drivesentinel/dsp.py + the quantiser
    on the same samples -- byte for byte;
  * the FPGA's logit registers with the bit-accurate model on the board's input;
  * the shaft speed the board estimated from the current.

and records the board's time per stage for every data file, plus the laptop's
own time for the same processing, for the ratio (docs/frontend_timing_plan.md 4.2).

    .venv/Scripts/python.exe scripts/webapp/check_board_frontend.py --board 192.168.2.99:8765
                                                   [--per-bearing 20 | --per-bearing 0 for all]
    -> artifacts/webapp/board_frontend_check.json

The data files are the ones behind the web app's four bearing lifts, in the
order the Board live page streams them.

WHAT THE TIMING IS, AND IS NOT. The board processes a whole 4 s data file at
once, as dsp.py does: decimation and envelopes over the whole signal, then 7
windows. So the natural unit is the data file, and the real-time question is
"is 4 s of signal processed in under 4 s". The per-window figure reported is
that time divided by the windows -- AMORTISED, not the single-window, pinned-
core protocol docs/frontend_timing_plan.md 3.3 declares its bar against. The
two are reported side by side and not conflated. The server splits the two
envelopes and the windows across its DSP worker processes (--workers, default 2
= both Cortex-A9 cores; recorded as board_workers); no core pinning is applied.

WHAT THIS IS NOT. An accuracy figure: the deployed network was trained on these
bearings.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from drivesentinel import config as C          # noqa: E402
from drivesentinel import dataset as DS        # noqa: E402
from drivesentinel import dsp as D             # noqa: E402
from drivesentinel.engines import SoftwareEngine, process_raw  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "webapp", "board_frontend_check.json")
BEARINGS = ["KA04", "KI18", "K001", "KI05"]      # the web app's bearing lifts
WARMUP = 2                                        # data files left out of the timing summary
STAGES = ["parse_ms", "decimate_ms", "envelope_ms", "spectra_ms", "quantise_ms", "fpga_ms",
          "total_ms"]


def files(bearing: str):
    z = np.load(os.path.join(ROOT, "artifacts", "demo", f"bearing_{bearing}.npz"),
                allow_pickle=False)
    return list(dict.fromkeys(str(f) for f in z["filenames"]))


def dist(xs):
    a = np.asarray(xs, dtype=np.float64)
    if not a.size:
        return None
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=os.environ.get("DS_BOARD_ADDRESS", ""))
    ap.add_argument("--per-bearing", type=int, default=20, help="data files per bearing; 0 = all")
    a = ap.parse_args()
    if not a.board:
        raise SystemExit("--board ADDRESS (or DS_BOARD_ADDRESS) is required")
    base = a.board if a.board.startswith("http") else "http://" + a.board
    health = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read())
    if not (health.get("frontend") or {}).get("available"):
        raise SystemExit("the board's DSP front end is not available: %s"
                         % (health.get("frontend") or {}).get("error", "old server.py"))
    sw = SoftwareEngine()
    rows = []
    for b in BEARINGS:
        names = files(b)
        if a.per_bearing:
            names = names[:a.per_bearing]
        for name in names:
            path = os.path.join(C.DATA_ROOT, b, name)
            c1, c2, vib, _ = DS.load_analysis_channels(path)
            t0 = time.perf_counter()
            res = process_raw(base, c1, c2, vib, C.FS_FAST, timeout_s=120.0)
            rtt = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            ref, ref_diag = D.process_recording(c1, c2, vib)
            laptop_dsp_ms = (time.perf_counter() - t0) * 1e3
            ref_q = [sw.quantise(w) for w in ref]
            ident = exact = same = nbytes = maxd = 0
            rpm_diff = []
            for k, w in enumerate(res["windows"]):
                q = np.frombuffer(base64.b64decode(w["int8"]), dtype=np.int8).reshape(5, 512)
                if k < len(ref_q):
                    d = np.abs(q.astype(int) - ref_q[k].astype(int))
                    ident += int(not d.any())
                    nbytes += int(np.count_nonzero(d))
                    maxd = max(maxd, int(d.max()))
                    same += int(sw.infer_int8(ref_q[k])["class"] == w["class"])
                exact += int(sw.infer_int8(q)["logits"] == w["logits"])
                if k < len(ref_diag):
                    rpm_diff.append(abs(w["rpm"] - ref_diag[k]["rpm"]))
            rows.append({"bearing": b, "file": name, "windows": len(res["windows"]),
                         "laptop_windows": len(ref), "int8_identical": ident,
                         "int8_bytes_differing": nbytes, "int8_max_abs_diff": maxd,
                         "verdict_same_as_laptop_input": same, "logits_exact": exact,
                         "max_rpm_diff": float(max(rpm_diff)) if rpm_diff else None,
                         "board_ms": {k: float(res["timings"][k]) for k in STAGES},
                         "rtt_ms": rtt, "bytes_sent": int(res["bytes_sent"]),
                         "laptop_dsp_ms": laptop_dsp_ms})
            print(f"{b} {name}: {ident}/{len(res['windows'])} identical, "
                  f"{exact} exact, board {res['timings']['total_ms']:.0f} ms, "
                  f"round trip {rtt:.0f} ms", flush=True)
    timed = rows[WARMUP:]
    n_win = sum(r["windows"] for r in rows)
    board_dsp = [r["board_ms"]["decimate_ms"] + r["board_ms"]["envelope_ms"]
                 + r["board_ms"]["spectra_ms"] for r in timed]
    out = {
        "what": "raw test-rig data -> board DSP (ARM) -> quantise -> FPGA, "
                "checked window by window against the laptop",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board_address": base,
        "board": {k: health.get(k) for k in ("build_id", "python", "numpy", "scipy",
                                             "frontend", "machine")},
        "board_workers": (health.get("frontend") or {}).get("workers"),
        "ran_on_hardware": bool(str((health.get("machine") or {}).get("arch", "")).startswith(
            ("arm", "aarch")) and not str((health.get("machine") or {}).get("pynq") or "")
            .startswith("MOCK")),
        "laptop": {"python": sys.version.split()[0], "numpy": np.__version__},
        "data_files": len(rows), "windows": n_win,
        "int8_identical": sum(r["int8_identical"] for r in rows),
        "int8_bytes_differing": sum(r["int8_bytes_differing"] for r in rows),
        "int8_bytes_compared": 2560 * sum(min(r["windows"], r["laptop_windows"]) for r in rows),
        "int8_max_abs_diff": max((r["int8_max_abs_diff"] for r in rows), default=0),
        "verdict_same_as_laptop_input": sum(r["verdict_same_as_laptop_input"] for r in rows),
        "logits_exact": sum(r["logits_exact"] for r in rows),
        "window_count_equal": all(r["windows"] == r["laptop_windows"] for r in rows),
        "max_rpm_diff": max((r["max_rpm_diff"] or 0.0) for r in rows) if rows else None,
        "timing": {
            "warmup_files_excluded": WARMUP,
            "signal_s_per_file": 4.0,
            "board_per_file_ms": {k: dist([r["board_ms"][k] for r in timed]) for k in STAGES},
            "board_dsp_per_file_ms": dist(board_dsp),
            "board_dsp_per_window_amortised_ms": dist(
                [d / r["windows"] for d, r in zip(board_dsp, timed) if r["windows"]]),
            "round_trip_per_file_ms": dist([r["rtt_ms"] for r in timed]),
            "laptop_dsp_per_file_ms": dist([r["laptop_dsp_ms"] for r in timed]),
            "plan_bar_ms": 250.0,
            "plan_bar_note": "docs/frontend_timing_plan.md 3.3 declares p95 <= 250 ms per "
                             "window, single-core, pinned. The amortised figure here is a "
                             "whole-file measurement without pinning: comparable in scale, "
                             "not the declared protocol.",
        },
        "files": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(f"\n{out['int8_identical']}/{n_win} int8 identical, {out['logits_exact']}/{n_win} "
          f"logits exact over {len(rows)} data files; ran_on_hardware={out['ran_on_hardware']}")
    t = out["timing"]["board_per_file_ms"]["total_ms"]
    if t:
        print(f"board total per 4 s file: median {t['median']:.0f} ms, p95 {t['p95']:.0f} ms")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
