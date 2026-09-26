"""
B-S4 winding branch (D2 KAIST PMSM, stage S4).

SCOPE, AND WHY IT IS NARROWER THAN THE PLAN
-------------------------------------------
`workflow_v2.md` §5 specified three classes (healthy / inter_turn / inter_coil)
with binary healthy-vs-fault as the headline.  **The healthy class is not
measurable on this dataset under any session-aware split**: all three healthy
recordings were taken on 2022-01-25, so healthy-vs-faulty is perfectly confounded
with acquisition session.  Hold that day out and training contains no healthy
recordings at all; hold out either other day and the test set contains none.  See
docs/data_notes_d2.md §7D for the per-fold table.

So this branch does **inter_coil vs inter_turn only**.  No healthy number is
produced anywhere -- not in the results, not on a metric card, not as a fusion
input.  A caveated healthy number would be worse than none, because the confound
is total rather than partial.

EVERY FEATURE IS A RATIO, BY CONSTRUCTION
-----------------------------------------
D2 current is in probe volts on an identity scale (NI_SensorSensitivity = 1.0), so
absolute amplitude is meaningless and not comparable across files.  All four
feature families below are invariant to a uniform scaling of the three phases, and
`test_winding_features.py` asserts that on every one of them.

WHAT THE FEATURES ARE, AND WHAT THEY CANNOT DO
----------------------------------------------
  negative-sequence ratio   the classical stator-asymmetry indicator
  Park's-vector eccentricity  the same asymmetry seen as an ellipse
  3rd-harmonic ratio        winding distortion
  harmonic band energies    orders 1-20 on the f/f_e axis, median-normalised

The f/f_e axis is implemented because the branch API needs it and the simulator
branch will need it for real.  It buys nothing here: f_e is 200.00 Hz in all 48
recordings, so the axis is a fixed rescaling.  It is not claimed as invariance
(workflow_v2.md §13.8).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..adapters.kaist_pmsm import F_ELEC_HZ, sequence_components
from ..common.schema import Recording, windows

WINDOW_S = 1.0
HOP_S = 0.5
N_ORDERS = 20                      # harmonic band energies, orders 1..20
PHASES = ("ia", "ib", "ic")

LABELS = ("inter_coil", "inter_turn")

FEATURE_NAMES: Tuple[str, ...] = (
    "neg_seq_ratio",
    "park_eccentricity",
    "third_harmonic_ratio",
) + tuple(f"order_{h:02d}" for h in range(1, N_ORDERS + 1))


# ===========================================================================
# per-window features
# ===========================================================================

def _bin_amplitudes(X: np.ndarray, fs: float, n: int,
                    freqs: Sequence[float]) -> np.ndarray:
    """
    Complex amplitude at each requested frequency, by nearest FFT bin.

    A 1.0 s window at 10 kHz gives 1 Hz resolution, and every frequency we ask
    for is an exact multiple of 200 Hz, so "nearest bin" is exact here rather
    than an approximation.
    """
    idx = np.rint(np.asarray(freqs) * n / fs).astype(int)
    return X[..., idx]


def window_features(win: np.ndarray, fs: float,
                    f_elec: float = F_ELEC_HZ) -> np.ndarray:
    """
    Features for one window of shape (3, n_samples).

    Returns a vector in FEATURE_NAMES order.
    """
    if win.shape[0] != 3:
        raise ValueError(f"expected 3 phases, got {win.shape[0]}")
    n = win.shape[1]

    x = win.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)          # drop DC before anything else
    X = np.fft.rfft(x, axis=1)

    orders = np.arange(1, N_ORDERS + 1) * f_elec
    nyq = fs / 2.0
    if orders[-1] >= nyq:
        raise ValueError(f"order {N_ORDERS} at {orders[-1]} Hz exceeds Nyquist {nyq}")

    A = _bin_amplitudes(X, fs, n, orders)          # (3, N_ORDERS) complex

    # -- 1. negative-sequence ratio at f_e ---------------------------------
    ph = [complex(A[k, 0]) for k in range(3)]
    _, i1, i2 = sequence_components(ph)
    hi = max(i1, i2)
    neg_seq = (min(i1, i2) / hi) if hi > 0 else 0.0

    # -- 2. Park's-vector ellipse eccentricity -----------------------------
    # Clarke transform to the stationary alpha-beta frame. A healthy machine
    # traces a circle; stator asymmetry flattens it into an ellipse. The
    # eccentricity of that ellipse is scale-free, which is what we need.
    ia, ib, ic = x
    alpha = math.sqrt(2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
    beta = math.sqrt(2.0 / 3.0) * (math.sqrt(3.0) / 2.0) * (ib - ic)
    cov = np.cov(np.vstack([alpha, beta]))
    ev = np.linalg.eigvalsh(cov)
    lo, hig = float(ev[0]), float(ev[1])
    ecc = math.sqrt(max(0.0, 1.0 - lo / hig)) if hig > 0 else 0.0

    # -- 3. third-harmonic ratio -------------------------------------------
    mag = np.abs(A)                                 # (3, N_ORDERS)
    fund = mag[:, 0]
    third = mag[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        r3 = float(np.mean(np.where(fund > 0, third / fund, 0.0)))

    # -- 4. harmonic band energies, median-normalised ----------------------
    # Mean magnitude over the three phases at each order, divided by the median
    # across orders. Median rather than the fundamental so that one dominant
    # component cannot flatten everything else, and log1p to compress the
    # dynamic range -- the same treatment config.py:186 uses for the bearing
    # order spectra, and for the same reason.
    band = mag.mean(axis=0)
    # The divisor is the median, FLOORED at a fixed fraction of the peak.
    #
    # The floor is not cosmetic. On a spectrum whose energy sits in one or two
    # orders -- a clean synthetic signal, or a real recording with an unusually
    # quiet noise floor -- the median across 20 orders collapses towards the
    # floating-point noise, and band/median explodes to ~1e15. That ratio is then
    # dominated by rounding rather than by the signal, and it stops being scale
    # invariant: scaling the input by 1000 changes the rounding and moves the
    # feature. Measured before the fix, order_01 moved from 35.963 to 35.989
    # under a x1000 scaling, which is the feature reading the instrument gain.
    #
    # Flooring at 1e-6 of the peak keeps exact scale invariance -- median and
    # peak scale together, so the ratio does not -- and bounds the output near
    # log1p(1e6).
    peak = float(band.max())
    scale = max(float(np.median(band)), 1e-6 * peak)
    band = np.log1p(band / scale) if scale > 0 else np.zeros_like(band)

    return np.concatenate([[neg_seq, ecc, r3], band]).astype(np.float32)


def recording_features(rec: Recording, window_s: float = WINDOW_S,
                       hop_s: float = HOP_S) -> np.ndarray:
    """(n_windows, n_features) for one Recording."""
    w = windows(rec, PHASES, window_s, hop_s)
    fs = rec.fs[PHASES[0]]
    if w.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([window_features(w[i], fs) for i in range(w.shape[0])])


# ===========================================================================
# dataset assembly
# ===========================================================================

def build_dataset(recs: Sequence[Recording], drop_healthy: bool = True
                  ) -> Dict[str, np.ndarray]:
    """
    Stack per-window features with every grouping variable a protocol needs.

    `drop_healthy` defaults True and should stay True -- see the module
    docstring. The flag exists so the exclusion is visible in the code rather
    than achieved by silently never passing healthy recordings in.
    """
    X, y, group, session, batch, chassis, sev, ftype, recid, i0res = (
        [], [], [], [], [], [], [], [], [], [])

    for r in recs:
        if drop_healthy and r.label == "healthy":
            continue
        f = recording_features(r)
        if f.shape[0] == 0:
            continue
        X.append(f)
        n = f.shape[0]
        y += [r.label] * n
        group += [r.group] * n
        session += [r.condition["session"]] * n
        batch += [r.condition["batch"]] * n
        chassis += [r.condition["daq_chassis"]] * n
        sev += [r.condition["severity_pct"]] * n
        ftype += [r.condition["fault_type"]] * n
        recid += [r.provenance["current_file"]] * n
        i0res += [r.provenance["i0rel_residual"]] * n

    return dict(
        X=np.concatenate(X) if X else np.empty((0, len(FEATURE_NAMES)), np.float32),
        y=np.asarray(y), group=np.asarray(group), session=np.asarray(session),
        batch=np.asarray(batch), chassis=np.asarray(chassis),
        severity=np.asarray(sev, dtype=np.float64),
        fault_type=np.asarray(ftype), recording=np.asarray(recid),
        i0rel_residual=np.asarray(i0res, dtype=np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
    )
