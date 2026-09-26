"""
B-S1 supply branch (D4 Thomas, stage S1).

THIS IS A THRESHOLD RULE, NOT A LEARNED MODEL, AND THAT IS THE DELIVERABLE
--------------------------------------------------------------------------
D4 has one recording per (motor x scenario) and two motors.  Under
leave-one-motor-out that means **one training example per class**, so any learned
metric would be separating two 20 s captures rather than learning phase loss.
Liu et al. report exactly this on this dataset: macro-F1 0.9682 under a random
split, 0.5856 under a within-label block split.

So the branch ships a documented rule and reports the learned classifier only as
a leaky reference, for contrast (`workflow_v2.md` §13.8).

THE RULE
--------
A phase is LOST in a window when its 0.2 s RMS falls below 5 % of the median of
the other two phases.

HOW SENSITIVE IS THAT 5 %?  MEASURED, NOT ASSERTED.
An earlier draft of this docstring claimed "anything between 1 % and 30 % gives
the same answer".  That was wrong, and measuring it is what caught it.

  RECORDING-LEVEL VERDICT -- what the branch actually reports -- is 10/10 for
  every threshold from 0.03 to 0.40, a 13x range.  At 0.02 it loses FILE 2 and
  at 0.01 it loses FILE 2 and FILE 7.  So at the reporting level the threshold
  is a separator, with roughly a factor of two of margin below and eight above.

  WINDOW-LEVEL classification has a much narrower safe band: (0.048, 0.053).
  The lower bound is the worst lost-phase ratio seen during a real event
  (0.0476, in FILE 10); the upper bound is the tightest ratio in a window
  labelled normal (0.0526, in FILE 5).

Both bounds come from the SAME few windows -- the ones straddling the instant a
phase dies.  A 0.2 s window that spans the transition contains both states and
its RMS lands between them.  That is an artefact of the window length, not a
property of the fault, and it is why the branch reports a recording-level verdict
requiring a >= 0.5 s event rather than trusting individual windows.

Stated plainly: the shipped 0.05 is comfortable for what the branch reports and
tight for a single window. Do not quote the window-level number as if it had the
recording-level margin.

THE DE-ENERGISED CASE, WHICH THE RULE ALONE GETS WRONG
------------------------------------------------------
Before a motor starts, ALL THREE phases read ~0.012 A.  The ratio test is then
0.012 vs a median of 0.012 and fires on nothing -- correctly, because no phase is
"lost" relative to the others when the contactor is simply open.  A window is
therefore classified as `off` when every phase is below an absolute floor, and
`off` windows are excluded from scoring.  Without that, FILE 1's first 8 seconds
would be scored as `normal` running.

SEPARATING THE TWO FAULT CLASSES BY ROTATION, NOT BY CURRENT
------------------------------------------------------------
`phase_loss_running` and `single_phasing_start` look nearly identical in current:
one phase at ~0.012 A, the other two elevated.  What separates them is whether
the machine is turning.

    rotating   vib_x RMS 0.017 - 0.041   (FILE 1,2,3,4,6,7,8,9)
    stalled    vib_x RMS 0.0008 - 0.0012 (FILE 5, 10)

A factor of about 20, with nothing in between.  A motor that never started is
stalled; a motor that lost a phase while running keeps turning on the remaining
two.  Using current for this would be guessing; using vibration is measuring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..adapters.thomas_motor import CURRENTS, FS, HOP_S, WINDOW_S
from ..common.schema import Recording

LABELS = ("normal", "phase_loss_running", "single_phasing_start")

RULE = {
    # A phase is lost when its RMS is below this fraction of the median of the
    # other two. Recording-level verdicts are 10/10 across 0.03-0.40; the
    # window-level safe band is only (0.048, 0.053) because of transition
    # windows. See the module docstring -- the difference matters.
    "lost_fraction": 0.05,
    # Absolute floor below which the machine is de-energised rather than faulted.
    # Measured dead-phase level is ~0.012 A; running phases are 0.28-1.3 A.
    "off_amps": 0.05,
    # vib_x RMS separating a turning rotor from a stalled one. Measured gap is
    # 0.0012 (stalled) to 0.017 (rotating); 0.005 sits in the empty middle.
    "rotating_vib": 0.005,
    "window_s": WINDOW_S,
    "hop_s": HOP_S,
}


@dataclass(frozen=True)
class WindowVerdict:
    t: float
    state: str                  # "off" | "normal" | "phase_loss_running" | "single_phasing_start"
    lost_phases: Tuple[int, ...]
    rms: Tuple[float, ...]
    rotating: bool
    vib_rms: float


# ===========================================================================
# the rule
# ===========================================================================

def rms_track(x: np.ndarray, fs: float, window_s: float = None,
              hop_s: float = None) -> Tuple[np.ndarray, np.ndarray]:
    window_s = window_s or RULE["window_s"]
    hop_s = hop_s or RULE["hop_s"]
    w = int(round(window_s * fs))
    h = int(round(hop_s * fs))
    if x.size < w:
        return np.empty(0), np.empty(0)
    n = (x.size - w) // h + 1
    out = np.empty(n)
    for i in range(n):
        seg = x[i * h:i * h + w]
        out[i] = math.sqrt(float(np.mean(seg * seg)))
    t = (np.arange(n) * h + w / 2) / fs
    return t, out


def classify(rec: Recording, cfg: Dict = None) -> List[WindowVerdict]:
    """
    Per-window verdict from the rule. No model, no fitting, no thresholds
    learned from the data it is scoring.
    """
    cfg = dict(RULE, **(cfg or {}))
    fs = rec.fs[CURRENTS[0]]
    tracks, t = [], None
    for c in CURRENTS:
        t, r = rms_track(np.asarray(rec.signals[c], dtype=np.float64), fs,
                         cfg["window_s"], cfg["hop_s"])
        tracks.append(r)
    R = np.vstack(tracks)                              # (3, n)

    if "vib_x" in rec.signals:
        _, v = rms_track(np.asarray(rec.signals["vib_x"], dtype=np.float64),
                         rec.fs["vib_x"], cfg["window_s"], cfg["hop_s"])
        v = v[:R.shape[1]] if v.size >= R.shape[1] else np.pad(
            v, (0, R.shape[1] - v.size), mode="edge")
    else:
        v = np.full(R.shape[1], np.nan)

    out: List[WindowVerdict] = []
    for i in range(R.shape[1]):
        col = R[:, i]
        rotating = bool(v[i] > cfg["rotating_vib"]) if np.isfinite(v[i]) else True

        # De-energised: every phase under the absolute floor. Checked BEFORE the
        # ratio test, which is blind to this case because all three agree.
        if float(col.max()) < cfg["off_amps"]:
            out.append(WindowVerdict(float(t[i]), "off", (), tuple(col), rotating,
                                     float(v[i])))
            continue

        lost = tuple(k for k in range(3)
                     if col[k] < cfg["lost_fraction"] * float(np.median(np.delete(col, k))))
        if not lost:
            state = "normal"
        elif rotating:
            state = "phase_loss_running"
        else:
            state = "single_phasing_start"
        out.append(WindowVerdict(float(t[i]), state, lost, tuple(col), rotating,
                                 float(v[i])))
    return out


def events(verdicts: Sequence[WindowVerdict],
           min_duration_s: float = 0.5) -> List[Dict]:
    """Contiguous runs of a fault state, long enough not to be a single glitch."""
    out, cur = [], None
    for v in verdicts:
        fault = v.state in ("phase_loss_running", "single_phasing_start")
        if fault and cur is None:
            cur = {"state": v.state, "t0": v.t, "t1": v.t,
                   "phases": set(v.lost_phases)}
        elif fault and cur is not None and v.state == cur["state"]:
            cur["t1"] = v.t
            cur["phases"] |= set(v.lost_phases)
        elif cur is not None:
            if cur["t1"] - cur["t0"] >= min_duration_s:
                out.append(cur)
            cur = None if not fault else {
                "state": v.state, "t0": v.t, "t1": v.t, "phases": set(v.lost_phases)}
    if cur is not None and cur["t1"] - cur["t0"] >= min_duration_s:
        out.append(cur)
    for e in out:
        e["phases"] = sorted(e["phases"])
        e["duration_s"] = round(e["t1"] - e["t0"], 3)
    return out


def detection_latency(verdicts: Sequence[WindowVerdict],
                      cfg: Dict = None) -> Optional[Dict]:
    """
    How long after the physical event the rule first flags it.

    Ground truth is taken as the first window whose lost-phase current drops
    below the midpoint between its running level and its dead level -- an
    amplitude crossing, independent of the rule's own threshold. Latency is then
    the gap to the rule's first fault verdict.

    Reported per event because "we detect phase loss" means nothing without it.
    """
    cfg = dict(RULE, **(cfg or {}))
    ev = events(verdicts)
    if not ev:
        return None
    e = ev[0]
    ph = e["phases"][0] if e["phases"] else 0

    series = np.array([v.rms[ph] for v in verdicts])
    times = np.array([v.t for v in verdicts])
    live = np.array([v.state != "off" for v in verdicts])
    if not live.any():
        return None

    during = (times >= e["t0"]) & (times <= e["t1"])
    before = live & (times < e["t0"])
    if not before.any() or not during.any():
        # Single-phasing at start: the phase is dead from the moment the machine
        # energises, so there is no "before" to measure a latency against.
        return {"event": e, "latency_s": 0.0, "phase": ph,
                "note": "phase dead from energisation; no pre-event level exists"}

    hi = float(np.median(series[before]))
    lo = float(np.median(series[during]))
    cross = (hi + lo) / 2.0
    idx = np.flatnonzero(live & (series < cross) & (times <= e["t1"]))
    t_true = float(times[idx[0]]) if idx.size else e["t0"]
    return {"event": e, "phase": ph, "t_truth": t_true, "t_detected": e["t0"],
            "latency_s": round(e["t0"] - t_true, 4),
            "pre_event_rms": round(hi, 4), "during_event_rms": round(lo, 4)}


def summarise(rec: Recording, cfg: Dict = None) -> Dict:
    v = classify(rec, cfg)
    ev = events(v)
    counts: Dict[str, int] = {}
    for w in v:
        counts[w.state] = counts.get(w.state, 0) + 1
    return {
        "file": rec.provenance["source_file"],
        "motor": rec.condition["motor"],
        "scenario": rec.condition["scenario"],
        "true_label": rec.label,
        "n_windows": len(v),
        "window_states": counts,
        "events": ev,
        "latency": detection_latency(v, cfg),
        "verdict": recording_verdict(v),
    }


def recording_verdict(verdicts: Sequence[WindowVerdict]) -> str:
    """
    One label per recording: the most severe state seen for long enough.

    `off` never wins -- a machine that was switched off for part of a capture is
    not "off" as a diagnosis.
    """
    ev = events(verdicts)
    if any(e["state"] == "single_phasing_start" for e in ev):
        return "single_phasing_start"
    if any(e["state"] == "phase_loss_running" for e in ev):
        return "phase_loss_running"
    return "normal"
