"""
Generate board/frontend_params.json: everything board/frontend.py needs to run
the bearing DSP front end on the PYNQ-Z2 exactly as drivesentinel/dsp.py runs it
on the laptop.

    .venv/Scripts/python.exe board/make_frontend_params.py

WHY A GENERATED FILE, NOT CONSTANTS TYPED INTO frontend.py
----------------------------------------------------------
Two reasons. The numbers come from drivesentinel/config.py and dsp.py's own
defaults, read here, so they cannot drift from what built the training cache
(tests/test_board_frontend.py checks the committed file equals this output).
And the FILTERS are designed here, by the laptop's scipy, not on the board. The
board runs PYNQ 2.5's older scipy; scipy's decimate() used transfer-function
(b, a) filtering in older releases and second-order sections in newer ones,
which for an order-8 Chebyshev at 0.05 of Nyquist is not the same arithmetic.
Shipping the sections and their initial states means the board only has to
run scipy.signal.sosfilt -- a routine unchanged since scipy 0.16 -- and does
the same arithmetic the laptop does.
"""

import inspect
import json
import os
import sys

import numpy as np
from scipy.signal import cheby1, sosfilt_zi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from drivesentinel import config as C  # noqa: E402
from drivesentinel import dsp as D     # noqa: E402

OUT = os.path.join(HERE, "frontend_params.json")


def _section(sos):
    """sos, the steady-state initial state per section, and sosfiltfilt's pad
    length -- computed exactly as scipy.signal.sosfiltfilt computes it."""
    sos = np.asarray(sos, dtype=np.float64)
    ntaps = 2 * sos.shape[0] + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return {"sos": sos.tolist(), "zi": sosfilt_zi(sos).tolist(), "edge": 3 * ntaps}


def params():
    if C.FEATURE_SET != "v2":
        raise SystemExit(f"frontend.py implements FEATURE_SET v2; config says {C.FEATURE_SET}")
    fund = inspect.signature(D.estimate_electrical_fundamental).parameters
    fs_vib = C.FS_FAST / C.DECIMATE_VIBRATION
    filters = {}
    for q in sorted({C.DECIMATE_CURRENT, C.DECIMATE_VIBRATION}):
        # scipy.signal.decimate(ftype="iir", zero_phase=True): an order-8
        # Chebyshev-I, 0.05 dB ripple, cutoff 0.8/q, as second-order sections.
        filters["decimate_%d" % q] = _section(cheby1(8, 0.05, 0.8 / q, output="sos"))
    filters["band_low"] = _section(D._bandpass_sos(*C.VIBRATION_BAND_LOW_HZ, fs_vib))
    filters["band_high"] = _section(D._bandpass_sos(*C.VIBRATION_BAND_HIGH_HZ, fs_vib))
    # The raw log spectrum's band edges, as the laptop's numpy computes them. numpy
    # 1.13 (the board's) does not pin geomspace's endpoints: its first edge is
    # 200.00000000000003 Hz, which drops the exact 200 Hz FFT bin from the first band
    # -- measured on the PYNQ-Z2 on 2026-09-24 as 1 differing input byte per window,
    # and up to 60 when the change moved the channel's median.
    lo, hi = C.RAW_SPECTRUM_HZ[0], min(C.RAW_SPECTRUM_HZ[1], 0.98 * fs_vib / 2)
    raw_edges = np.geomspace(lo, hi, C.N_ORDER_BINS + 1)
    return {
        "generated_by": "board/make_frontend_params.py from drivesentinel/config.py + dsp.py",
        "feature_set": C.FEATURE_SET,
        "channels": ["current sideband fold", "current raw order", "vibration envelope low",
                     "vibration envelope high", "vibration raw log spectrum"],
        "fs": C.FS_FAST,
        "decimate_current": C.DECIMATE_CURRENT,
        "decimate_vibration": C.DECIMATE_VIBRATION,
        "window_s": C.WINDOW_SECONDS,
        "hop_s": C.HOP_SECONDS,
        "n_bins": C.N_ORDER_BINS,
        "order_max": C.ORDER_MAX,
        "pole_pairs": C.POLE_PAIRS,
        "fundamental_lo_hz": fund["lo"].default,
        "fundamental_hi_hz": fund["hi"].default,
        "supply_harmonics_notched": C.SUPPLY_HARMONICS_NOTCHED,
        "supply_notch_halfwidth_hz": C.SUPPLY_NOTCH_HALFWIDTH_HZ,
        "spectrum_eps": C.SPECTRUM_EPS,
        "raw_spectrum_hz": list(C.RAW_SPECTRUM_HZ),
        "raw_spectrum_edges_hz": {"fs": fs_vib, "edges": raw_edges.tolist()},
        "fault_orders": {k: float(v) for k, v in C.FAULT_ORDERS_NOMINAL.items()},
        "filters": filters,
    }


def main():
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(params(), fh, indent=1)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
