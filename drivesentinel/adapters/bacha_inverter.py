"""
D3 -- Bacha PMSM inverter fault telemetry (stages S2/S3).

Source: https://github.com/bachaabdelkabir/PMSM-inverter-fault-diagnosis
        (mirror of https://doi.org/10.5281/zenodo.13974503), CC-BY-4.0.

WHAT THIS DATASET ACTUALLY IS
-----------------------------
10 Hz Arduino telemetry from a three-phase MOSFET inverter (IRF540N) driving a
PMSM converted from a DENSO alternator at a constant 10 rad/s.  Eight 10-bit ADC
channels, one continuous run of roughly five minutes per condition, nine
conditions, 10,892 samples total.  There is no waveform here and no harmonic
content of any kind: the Nyquist frequency is 5 Hz.

THE SHIPPED UNIT CONVERSIONS ARE WRONG. DO NOT USE THEM.
--------------------------------------------------------
`converted_dataset.csv` and `converted_dataset-2.csv` both apply
`adc_to_voltage_acs712_100ohm()` to VDC and VD, which subtracts the ACS712's
2.5 V bipolar mid-rail offset.  That offset is correct for a current sensor and
wrong for a DC-bus divider: it yields VDC = -22 V, and a gain of 4.888 V per ADC
count, so the whole DC-link channel is quantised into 24 five-volt steps.

`converted_dataset.csv` is worse still -- its temperature column cannot be
reproduced from `dataset CSV.csv` by any formula in the shipped script, so it is
the output of some earlier script that no longer exists.

`converted_dataset-2.csv` has correct Steinhart-Hart temperatures but its `Ia`,
`Ib` and `IDC` columns hold RAW ADC while the converted values sit in
`Ia_original` etc.  A header/content mismatch waiting to be read the obvious way.

So: this adapter starts from raw ADC and converts here.

WHY VDC, IDC AND VD ARE DROPPED
-------------------------------
Measured over all 10,892 rows, their per-class means span 0.78, 0.45 and 0.26 ADC
counts against per-channel standard deviations of 1.2-1.4.  They do not vary with
condition; they barely vary at all.  Keeping them would add three channels of
quantisation noise for a model to overfit, and would make `Vdc*Idc`, `dVdc/dt` and
`dIdc/dt` -- all in the original plan -- derivatives of a constant.

Decision recorded in docs/workflow_v2.md section 13.5.

WHAT IS LEFT, AND WHAT IT CAN SUPPORT
-------------------------------------
Ia, Ib (phase currents; there is no Ic) and T1-T3 (half-bridge NTCs).  The
temperature channels separate the three over-temp conditions perfectly and
physically -- F6 drops T1, F7 drops T1 and T2, F8 drops T3, exactly matching the
file names.  That is a thermal threshold, not machine learning, and the metric
card says so.  The open- and short-circuit conditions live entirely in Ia/Ib and
are weak: F1 (HB2 high-side open) has almost the same channel means as F0.

One run per condition means every result here is within-run.  There is no group
axis to hold out.
"""

from __future__ import annotations

import csv
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common.schema import Recording

FS_HZ = 10.0
ADC_MAX = 1023
VREF = 5.0

# Raw channels, in the column order of `dataset CSV.csv`.
RAW_CHANNELS = ("Ia", "Ib", "VDC", "IDC", "T1", "T2", "T3", "VD")

# Kept for modelling.  See "WHY VDC, IDC AND VD ARE DROPPED" above.
USED_CHANNELS = ("Ia", "Ib", "T1", "T2", "T3")
DROPPED_CHANNELS = ("VDC", "IDC", "VD")

# Per-condition file -> F-label.  Verified by row count against the published
# label counts: every one of the nine matches exactly (audit section 3.3).
FILE_TO_LABEL = {
    "NORMAL_OP.txt":        ("F0", "normal",         4295),
    "HB2_HIGH_SIDE_OC.txt": ("F1", "open_circuit",    692),
    "HB3_LOW_SIDE_OC.txt":  ("F2", "open_circuit",   1122),
    "HB1_LOW_SIDE_SC.txt":  ("F3", "short_circuit",   407),
    "HB2_HIGH_SIDE_SC.txt": ("F4", "short_circuit",   341),
    "HB3_HIGH_SIDE_SC.txt": ("F5", "short_circuit",   412),
    "HB1_OVER_TEMP.txt":    ("F6", "over_temp",       854),
    "HB1&2_OVER_TEMP.txt":  ("F7", "over_temp",      1735),
    "HB3_OVER_TEMP.txt":    ("F8", "over_temp",      1034),
}

# Identified from their 4-column format [setpoint C, PWM duty, measured C, ADC]:
# thermistor / heater PID calibration runs, not inverter conditions.
# `temperature_data.txt` is byte-identical to `SETPOINT30.txt`, and
# `thermistor_calib_data.txt` is the four setpoint runs concatenated.
CALIBRATION_FILES = (
    "SETPOINT30.txt", "SETPOINT40.txt", "SETPONT50.txt", "SETPOINT60.txt",
    "temperature_data.txt", "thermistor_calib_data.txt",
)

_LINE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*->\s*(.*)$")


# ===========================================================================
# unit conversion
# ===========================================================================

def adc_to_current_acs712(adc) -> np.ndarray:
    """
    ACS712-20A, 100 mV/A, bipolar about VREF/2.

    This is the author's own `adc_to_current_acs712_20a`, which IS correct for
    the three current channels -- the bug was applying the same mid-rail offset
    to the voltage dividers.
    """
    a = np.asarray(adc, dtype=np.float64)
    return ((a / ADC_MAX) * VREF - VREF / 2.0) / 0.100


# Steinhart-Hart coefficients.
#
# `SCRIPT` is what data_conversion_script_v4.py ships.  `CALIBRATED` is refitted
# here on the dataset's OWN thermistor calibration runs.  They disagree by
# 13-20 C, and the script's version is the one that is wrong -- see
# adc_to_temperature_ntc.
SH_SCRIPT = (1.2666e-3, 2.3661e-4, 9.6094e-8)
SH_CALIBRATED = (5.197222e-03, -4.958087e-04, 3.504369e-06)
NTC_SERIES_R = 10_000.0

# The four settled points the CALIBRATED coefficients were fitted to, kept so the
# fit can be reproduced and so its validity range is visible at the call site.
# (adc, celsius), from the tails of SETPOINT{30,40,50,60}.txt where the heater
# PWM has gone to zero and the measured temperature sits within 1 C of setpoint.
NTC_CALIBRATION_POINTS = ((462.3, 30.13), (386.8, 40.19),
                          (300.1, 49.85), (211.6, 59.97))
NTC_CALIBRATION_ADC_RANGE = (211.6, 462.3)


def adc_to_temperature_ntc(adc, coefficients=SH_CALIBRATED) -> np.ndarray:
    """
    10k NTC with a 10k series resistor, Steinhart-Hart.

    THE SHIPPED COEFFICIENTS ARE MISCALIBRATED. WE DO NOT USE THEM BY DEFAULT.

    `data_conversion_script_v4.py` uses A=1.2666e-3, B=2.3661e-4, C=9.6094e-8,
    and `converted_dataset-2.csv` reproduces exactly from them (ADC 515 ->
    10.577 C, verified).  So the CSV is self-consistent with the script.  But the
    script disagrees with the dataset's own thermistor calibration runs by
    13-20 C across the whole range:

        ADC    script    dataset's own calibration
        462     14.3 C    30.1 C
        387     21.2 C    40.2 C
        300     29.9 C    49.9 C
        212     41.8 C    60.0 C

    The consequence of using the script's numbers is physically absurd: an idle
    inverter reads 11 C, and a deliberately induced MOSFET over-temperature fault
    peaks at 28 C.  With the refitted coefficients the same data reads 24 C idle
    and 48 C at the over-temp fault, which is what a small bench inverter
    actually does.

    The refit uses the four settled points in NTC_CALIBRATION_POINTS -- the tails
    of the four SETPOINT runs, where the heater PWM has reached zero and the
    measured temperature has converged on setpoint.  Transient samples are
    excluded because the reference probe and the NTC are not thermally coupled
    during a ramp: including them makes the relation non-monotonic (the ADC~508
    bin reads 46 C while the ADC~460 bin reads 30 C).

    LIMITS OF THIS FIT, STATED PLAINLY
    ----------------------------------
    Three parameters fitted to four points.  The residual on those points
    (0.36 C max) therefore measures almost nothing -- it is close to an exact
    fit, not a validated one.  What can be said is that the curve is smooth and
    monotonic across the full ADC range the fault data occupies (281-554), and
    that it is anchored to four independently measured, well-separated points.

    Calibration covers 30-60 C (ADC 212-462).  D3's idle readings sit at ADC
    ~509, so the baseline is a mild EXTRAPOLATION below the calibrated range.
    Treat absolute idle temperatures as approximate; temperature DIFFERENCES and
    RATES, which is what the branch actually uses, are unaffected.

    Pass `coefficients=SH_SCRIPT` to reproduce the published CSV.
    """
    a = np.asarray(adc, dtype=np.float64)
    A, B, Cc = coefficients
    with np.errstate(divide="ignore", invalid="ignore"):
        r_th = NTC_SERIES_R / ((ADC_MAX / a) - 1.0)
        ln_r = np.log(r_th)
        kelvin = 1.0 / (A + B * ln_r + Cc * ln_r ** 3)
    return kelvin - 273.15


def adc_to_bus_voltage(adc, divider_ratio: Optional[float] = None) -> np.ndarray:
    """
    Unipolar divider -- NO mid-rail offset.  Unused by default; see module docs.

    `divider_ratio` is not in any machine-readable file in the dataset.  It is in
    `Sensor_raw_data_conversion_formulas.pdf`, which has not been transcribed.
    Until it has, this returns the ADC pin voltage and the caller must scale it.
    Kept so that the correct form is written down next to the incorrect one.
    """
    a = np.asarray(adc, dtype=np.float64)
    v_pin = (a / ADC_MAX) * VREF
    return v_pin * (divider_ratio if divider_ratio is not None else 1.0)


# ===========================================================================
# reading
# ===========================================================================

def _parse_condition_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read `HH:MM:SS.mmm -> 8 ints`.  Returns (seconds_since_start, raw[n, 8]).

    Tolerates the trailing whitespace every line carries and a missing final
    newline (which is why `wc -l` undercounts these files by one).
    """
    t: List[float] = []
    rows: List[List[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            m = _LINE.match(line)
            if not m:
                raise ValueError(f"{os.path.basename(path)}:{lineno}: "
                                 f"unparseable line {line!r}")
            hh, mm, ss, ms, rest = m.groups()
            vals = rest.split()
            if len(vals) != len(RAW_CHANNELS):
                raise ValueError(
                    f"{os.path.basename(path)}:{lineno}: {len(vals)} values, "
                    f"expected {len(RAW_CHANNELS)}"
                )
            t.append(int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0)
            rows.append([int(v) for v in vals])

    raw = np.asarray(rows, dtype=np.int16)
    tt = np.asarray(t, dtype=np.float64)
    return tt - tt[0], raw


def _to_signals(raw: np.ndarray, keep=USED_CHANNELS) -> Dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(RAW_CHANNELS)}
    out: Dict[str, np.ndarray] = {}
    for name in keep:
        col = raw[:, idx[name]]
        if name in ("Ia", "Ib", "IDC"):
            out[name] = adc_to_current_acs712(col).astype(np.float32)
        elif name in ("T1", "T2", "T3"):
            out[name] = adc_to_temperature_ntc(col).astype(np.float32)
        else:                                    # VDC / VD, pin volts only
            out[name] = adc_to_bus_voltage(col).astype(np.float32)
        out[f"{name}_adc"] = col.astype(np.float32)
    return out


def load(root: str, keep=USED_CHANNELS,
         strict_counts: bool = True) -> List[Recording]:
    """
    One Recording per condition file.

    `group` is the condition file, because that is the only unit this dataset
    has: one continuous run per condition means there is no independent group to
    hold out, and naming the group after the file makes that visible in every
    split report rather than hidden behind an index.

    `strict_counts` checks each file's row count against the published label
    counts and raises on a mismatch.  On by default: a silently short file would
    change class balance without changing anything a metric would show.
    """
    recs: List[Recording] = []
    for fname, (fcode, family, expected) in sorted(FILE_TO_LABEL.items()):
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} (D3 expects all nine condition files)")

        t, raw = _parse_condition_txt(path)
        n = len(raw)
        if strict_counts and n != expected:
            raise ValueError(
                f"{fname}: {n} rows, expected {expected}. The published label "
                f"counts are the contract; investigate before relaxing this."
            )

        dt = np.diff(t)
        recs.append(Recording(
            source="bacha_inverter",
            stage="S3" if family in ("open_circuit", "short_circuit") else "S2",
            signals=_to_signals(raw, keep),
            fs={k: FS_HZ for k in _to_signals(raw, keep)},
            label=family,
            group=fcode,
            condition={
                "f_code": fcode,
                "location": fname.replace(".txt", ""),
                "n_samples": n,
                "duration_s": float(t[-1]),
            },
            provenance={
                "source_file": fname,
                "row_count_verified": n == expected,
                "expected_rows": expected,
                "sample_interval_s": {
                    "median": float(np.median(dt)) if dt.size else None,
                    "min": float(dt.min()) if dt.size else None,
                    "max": float(dt.max()) if dt.size else None,
                },
                "channels_kept": list(keep),
                "channels_dropped": [c for c in DROPPED_CHANNELS if c not in keep],
                "drop_reason": (
                    "VDC/IDC/VD per-class means span <0.8 ADC counts against "
                    "std 1.2-1.4: no diagnostic content. workflow_v2.md 13.5"
                ),
                "conversion": {
                    "Ia/Ib/IDC": "ACS712-20A bipolar, ((adc/1023)*5 - 2.5)/0.1",
                    "T1/T2/T3": (
                        "Steinhart-Hart REFITTED on the dataset's own settled "
                        "calibration points (SETPOINT30/40/50/60 tails): "
                        f"A={SH_CALIBRATED[0]:.6e} B={SH_CALIBRATED[1]:.6e} "
                        f"C={SH_CALIBRATED[2]:.6e}, R1=10k. The shipped script's "
                        "coefficients under-read by 13-20 C and are NOT used."
                    ),
                    "T_calibration_points": list(NTC_CALIBRATION_POINTS),
                    "T_calibration_adc_range": list(NTC_CALIBRATION_ADC_RANGE),
                    "VDC/VD": "pin volts only; divider ratio not transcribed",
                },
                "converted_csv_used": False,
                "converted_csv_reason": (
                    "converted_dataset*.csv apply the ACS712 2.5 V mid-rail "
                    "offset to a DC-bus divider, giving VDC = -22 V"
                ),
            },
        ))
    return recs


def calibration_files_present(root: str) -> Dict[str, bool]:
    """
    Which thermistor-calibration files are here.  Excluded from classification;
    reported so a data-notes document can say they were seen and skipped rather
    than silently ignored.
    """
    return {f: os.path.exists(os.path.join(root, f)) for f in CALIBRATION_FILES}
