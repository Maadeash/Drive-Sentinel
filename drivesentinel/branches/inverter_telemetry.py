"""
B-S2/S3 inverter telemetry branch (D3 Bacha, stages S2 and S3).

WHAT THIS BRANCH CAN AND CANNOT BE
----------------------------------
10 Hz Arduino telemetry, one continuous run per condition, nine conditions,
10,892 samples.  Nyquist is 5 Hz, so there is no waveform and no harmonic
content of any kind.  There is also **no group axis**: one run per condition
means there is nothing independent to hold out, and every number here is a
within-run estimate.  B-S2/S3 is INDICATIVE under the pre-registered floor
regardless of what it scores, and `claims_audit.md` §1.1 said so before this
branch existed.

CALIBRATION-FREE BY CONSTRUCTION
--------------------------------
`adapters/bacha_inverter.py` refits the NTC Steinhart-Hart coefficients because
the shipped ones under-read by 13-20 C.  That refit is three parameters on four
points and its absolute accuracy below 30 C is an extrapolation
(`data_notes_d3.md` §4).

**No feature here depends on it.**  Temperature enters only as RAW ADC and as
DIFFERENCES between channels.  An NTC's ADC reading is monotone in temperature,
so a difference in ADC is monotone in a difference in temperature whatever the
Steinhart-Hart constants turn out to be.  If the refit is wrong, every number in
this branch is unaffected -- which is the point.

Currents use the ACS712 conversion, which is the author's own formula and is
correct; the bug was only ever in applying it to the voltage dividers.

THE DROPPED CHANNELS
--------------------
`VDC`, `IDC` and `VD` are not used.  Measured over all 10,892 rows, their
per-class means span 0.78, 0.45 and 0.26 ADC counts against per-channel standard
deviations of 1.2-1.4 -- inside their own quantisation noise.  That also removes
`Vdc*Idc`, `dVdc/dt` and `dIdc/dt` from the original plan: they are products and
derivatives of a constant.

THE CONFOUND, WHICH IS WORSE THAN "ONE RUN PER CONDITION"
---------------------------------------------------------
Each condition is a single contiguous file recorded at a distinct wall-clock time
(13:24 through 14:31), and temperature drifts monotonically within a run.  A
contiguous block split therefore separates **early-in-run from late-in-run**, not
condition from condition.  Stated on the metric card, not buried here.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..adapters.bacha_inverter import FS_HZ
from ..common.schema import Recording

WINDOW_S = 5.0                 # 50 samples at 10 Hz
HOP_S = 1.0                    # 10 samples

# 4-class family. The 9-class location view is produced as a qualitative
# confusion matrix only -- three of its classes get ~8-10 test windows under the
# block split, which is not enough to report a per-class number.
LABELS = ("normal", "open_circuit", "short_circuit", "over_temp")

CURRENT_CH = ("Ia", "Ib")
TEMP_ADC_CH = ("T1_adc", "T2_adc", "T3_adc")

_STATS = ("mean", "std", "min", "max")


def _stats(x: np.ndarray) -> List[float]:
    return [float(x.mean()), float(x.std()), float(x.min()), float(x.max())]


def feature_names(use_temperature: bool = True) -> Tuple[str, ...]:
    names: List[str] = []
    for c in CURRENT_CH:
        names += [f"{c}_{s}" for s in _STATS]
    names += ["Ia_Ib_imbalance", "Ia_Ib_abs_diff"]
    if use_temperature:
        for c in TEMP_ADC_CH:
            names += [f"{c}_{s}" for s in _STATS]
        # Relative, so no dependence on the Steinhart-Hart refit.
        names += ["dT12_adc", "dT13_adc", "dT23_adc",
                  "T_spread_adc", "T_min_adc_rate", "T_max_adc_rate"]
    return tuple(names)


def window_features(win: Dict[str, np.ndarray], fs: float = FS_HZ,
                    use_temperature: bool = True) -> np.ndarray:
    """
    Features for one window. `win` maps channel name -> 1-D slice.

    Temperature contributes only raw ADC statistics and ADC differences. Nothing
    here inverts the NTC curve, so the branch is immune to the calibration
    question in data_notes_d3.md §4.
    """
    f: List[float] = []
    for c in CURRENT_CH:
        f += _stats(np.asarray(win[c], dtype=np.float64))

    ia = float(np.mean(win["Ia"]))
    ib = float(np.mean(win["Ib"]))
    denom = (abs(ia) + abs(ib)) / 2.0
    f.append((ia - ib) / denom if denom > 1e-9 else 0.0)
    f.append(abs(ia - ib))

    if use_temperature:
        t = {}
        for c in TEMP_ADC_CH:
            a = np.asarray(win[c], dtype=np.float64)
            t[c] = a
            f += _stats(a)
        m = {c: float(t[c].mean()) for c in TEMP_ADC_CH}
        f.append(m["T1_adc"] - m["T2_adc"])
        f.append(m["T1_adc"] - m["T3_adc"])
        f.append(m["T2_adc"] - m["T3_adc"])
        f.append(max(m.values()) - min(m.values()))
        # ADC/s. An NTC's ADC falls as it heats, so a negative rate is warming.
        # Sign is preserved under any monotone calibration.
        n = len(t["T1_adc"])
        dur = max(n / fs, 1e-9)
        rates = [(float(t[c][-1]) - float(t[c][0])) / dur for c in TEMP_ADC_CH]
        f.append(min(rates))
        f.append(max(rates))

    return np.asarray(f, dtype=np.float32)


def recording_windows(rec: Recording, window_s: float = WINDOW_S,
                      hop_s: float = HOP_S, use_temperature: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """(features, window_start_index) for one condition recording."""
    fs = rec.fs["Ia"]
    w = int(round(window_s * fs))
    h = int(round(hop_s * fs))
    n = len(rec.signals["Ia"])
    if n < w:
        return (np.empty((0, len(feature_names(use_temperature))), np.float32),
                np.empty(0, np.int64))

    need = list(CURRENT_CH) + (list(TEMP_ADC_CH) if use_temperature else [])
    missing = [c for c in need if c not in rec.signals]
    if missing:
        raise KeyError(f"{rec.group}: missing channel(s) {missing}")

    starts = list(range(0, n - w + 1, h))
    out = np.empty((len(starts), len(feature_names(use_temperature))), np.float32)
    for i, s in enumerate(starts):
        out[i] = window_features({c: rec.signals[c][s:s + w] for c in need},
                                 fs, use_temperature)
    return out, np.asarray(starts, dtype=np.int64)


def build_dataset(recs: Sequence[Recording], use_temperature: bool = True
                  ) -> Dict[str, np.ndarray]:
    """
    Stack windows with every label view and the within-run position.

    `position` is the window's index inside its own run, which is what the
    contiguous block split orders by -- and what makes the early-run/late-run
    confound visible rather than hidden.
    """
    X, family, location, run, pos = [], [], [], [], []
    for r in recs:
        f, starts = recording_windows(r, use_temperature=use_temperature)
        if f.shape[0] == 0:
            continue
        X.append(f)
        n = f.shape[0]
        family += [r.label] * n
        location += [r.condition["f_code"]] * n
        run += [r.group] * n
        pos += list(range(n))
    return dict(
        X=np.concatenate(X) if X else np.empty((0, 0), np.float32),
        family=np.asarray(family), location=np.asarray(location),
        run=np.asarray(run), position=np.asarray(pos, dtype=np.int64),
        feature_names=np.asarray(feature_names(use_temperature)),
        uses_temperature=np.asarray(use_temperature),
    )


CONFOUND_NOTE = (
    "Each condition is ONE contiguous run recorded at a distinct wall-clock time "
    "(13:24 to 14:31), and temperature drifts monotonically within a run. A "
    "contiguous block split therefore separates EARLY-IN-RUN from LATE-IN-RUN, "
    "not condition from condition. There is no group axis on this dataset -- one "
    "run per condition means nothing independent can be held out -- so every "
    "number here is a within-run estimate and the branch is INDICATIVE whatever "
    "it scores."
)

TEMPERATURE_NOTE = (
    "The over_temp family is separable by a single NTC threshold: F6 heats HB1, "
    "F7 heats HB1 and HB2, F8 heats HB3, each matching its filename. A 4-class "
    "score that includes over_temp is therefore substantially a thermometer "
    "reading, which is why the electrical-only ablation leads the card."
)
