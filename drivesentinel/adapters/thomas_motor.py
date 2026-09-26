"""
D4 -- Thomas et al. motor fault dataset (stage S1, supply input).

Source: https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219
Paper:  Sci Data 12:1468, 2025.
Licence: **CC BY-NC-ND -- never commit derived data.**

Measured facts behind every decision here are in docs/data_notes_d4.md.

WHAT THE FILES ARE
------------------
Ten MATLAB v5 files, each a bare [1000000, 9] double array -- no metadata
variables, no timestamps, no channel names -- plus a flat label vector.
50 kHz, 20.0 s per file, columns [vib_x, vib_y, vib_z, I1, I2, I3, V1, V2, V3].

Column identity is confirmed by physics rather than by documentation: column 2
has mean 1.0044, i.e. gravity, so it is the vertical accelerometer axis.

  FILE 1-5   healthy motor      normal / phase-removal / 0.4 Nm / 0.8 Nm / single-phasing
  FILE 6-10  faulty motor       the same five scenarios in the same order
             (SKF 6202-Z outer ring, drilled holes)

THE LABEL VECTOR IS NOT USED
----------------------------
`LABEL DATASET.mat` holds 19,982 int64 labels in 13 contiguous runs, with no file
index, no class names and no window metadata.  The paper describes a 1000-sample
window at step 500 over the ten files merged in order, and that reconstruction was
tested against the phase-current collapse boundaries measured here:

    6 of 14 label-run boundaries coincide with a FILE boundary at 1998
      windows per file (against 1 of 14 at 1999), so the per-file window count
      is probably right;
    0 of 4 measured phase-loss events coincide with ANY label boundary.

So the file structure reconstructs and the within-file event structure does not.
**The reconstruction is not validated and the label vector is not used.**  Labels
come from the current-collapse rule in `branches/supply.py`, which is primary
either way, and the disagreement is recorded in data_notes_d4.md rather than
papered over.

NO SESSION CONFOUND HERE
------------------------
Unlike D2, six independent signal-derived fingerprints (mains frequency to
0.001 Hz, DC offsets on five channels, broadband noise floor) fail to separate the
two motors, and the mains-frequency spread across all ten files is 0.081 Hz --
consistent with one recording session.  The three supply classes also appear on
BOTH motors as matched pairs, so they stay balanced across motor identity
whatever the acquisition turns out to have been.

WHAT IS STILL CONFOUNDED, AND ALWAYS WILL BE
--------------------------------------------
Bearing condition equals motor identity: one motor per condition, so nothing
distinguishes an outer-race defect from any other difference between two physical
machines.  `bearing_confound_note()` states it, and the branch reports it as a
documented negative result, never as a capability.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import config as C
from ..common.schema import Recording

FS = 50_000.0
N_SAMPLES = 1_000_000
N_CHANNELS = 9
MAT5_DATA_OFFSET = 184              # 128-byte header + 8-byte tag + matrix preamble
BYTES = N_SAMPLES * N_CHANNELS * 8 + MAT5_DATA_OFFSET

CHANNELS = ("vib_x", "vib_y", "vib_z", "I1", "I2", "I3", "V1", "V2", "V3")
CURRENTS = ("I1", "I2", "I3")
VOLTAGES = ("V1", "V2", "V3")
PHASE_OF = {"I1": 0, "I2": 1, "I3": 2}

WINDOW_S = 0.2                      # 10 mains cycles at 50 Hz
HOP_S = 0.1

# Confirmed against the paper's Data Records section (workflow_v2.md §13.2).
SCENARIOS = {
    1:  ("healthy", "normal_no_load",        "normal"),
    2:  ("healthy", "phase_removal_running", "phase_loss_running"),
    3:  ("healthy", "load_0.4Nm",            "normal"),
    4:  ("healthy", "load_0.8Nm",            "normal"),
    5:  ("healthy", "single_phasing_start",  "single_phasing_start"),
    6:  ("faulty",  "normal_no_load",        "normal"),
    7:  ("faulty",  "phase_removal_running", "phase_loss_running"),
    8:  ("faulty",  "load_0.4Nm",            "normal"),
    9:  ("faulty",  "load_0.8Nm",            "normal"),
    10: ("faulty",  "single_phasing_start",  "single_phasing_start"),
}

LABELS = ("normal", "phase_loss_running", "single_phasing_start")

# Reconstruction parameters, recorded for reproducibility. NOT used for labelling
# -- see "THE LABEL VECTOR IS NOT USED" above.
LABEL_RECONSTRUCTION = {
    "window_samples": 1000,
    "step_samples": 500,
    "windows_per_file": 1998,
    "validated": False,
    "file_boundary_hits": "6/14",
    "event_boundary_hits": "0/4",
    "note": ("File structure reconstructs at 1998 windows/file; within-file event "
             "structure does not match the measured phase-current collapse. "
             "Labels are derived from the collapse rule instead."),
}


def default_root() -> str:
    return os.environ.get(
        "DRIVESENTINEL_EXT_DATA_D4",
        os.path.join(C.PROJECT_ROOT, "data_ext", "thomas_motor"))


def read_channel(path: str, channel: str, decimate: int = 1) -> np.ndarray:
    """
    One column, read by seeking straight to it.

    The array is column-major, so a channel is a contiguous 8 MB block and can be
    read without touching the other eight. That is the difference between 8 MB
    and 72 MB per channel access, and it is why this does not use
    `scipy.io.loadmat`.
    """
    c = CHANNELS.index(channel)
    with open(path, "rb") as fh:
        fh.seek(MAT5_DATA_OFFSET + c * N_SAMPLES * 8)
        x = np.fromfile(fh, dtype="<f8", count=N_SAMPLES)
    if x.size != N_SAMPLES:
        raise ValueError(f"{os.path.basename(path)}: read {x.size} of {N_SAMPLES} "
                         f"samples for {channel}")
    return x[::decimate] if decimate > 1 else x


def index(root: str = None) -> List[Dict]:
    root = root or default_root()
    out = []
    for i in sorted(SCENARIOS):
        p = os.path.join(root, f"FILE {i}.mat")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} (D4 expects FILE 1..10.mat)")
        size = os.path.getsize(p)
        if size != BYTES:
            raise ValueError(f"FILE {i}.mat is {size} bytes, expected {BYTES}. "
                             f"The [1000000, 9] layout in data_notes_d4.md no "
                             f"longer holds -- re-inspect before loading.")
        motor, scenario, label = SCENARIOS[i]
        out.append(dict(file_no=i, path=p, motor=motor, scenario=scenario,
                        label=label,
                        group=f"motor_{'A_healthy' if motor == 'healthy' else 'B_faulty'}"))
    return out


def load(root: str = None, channels: Sequence[str] = None,
         decimate: int = 1, limit: int = None) -> List[Recording]:
    """
    One Recording per file. `group` is the MOTOR, which is what leave-one-motor-out
    holds out.

    Two groups only, which is below the pre-registered fusion floor of three --
    B-S1 is INDICATIVE whatever it scores, and `claims_audit.md` §1.1 said so
    before this branch existed.
    """
    root = root or default_root()
    channels = tuple(channels or (CURRENTS + VOLTAGES + ("vib_x",)))
    entries = index(root)
    if limit:
        entries = entries[:limit]

    recs = []
    for e in entries:
        sig = {c: read_channel(e["path"], c, decimate).astype(np.float32)
               for c in channels}
        fs = {c: FS / max(decimate, 1) for c in channels}
        recs.append(Recording(
            source="thomas_motor", stage="S1",
            signals=sig, fs=fs,
            label=e["label"], group=e["group"],
            condition={
                "file_no": e["file_no"],
                "motor": e["motor"],
                "scenario": e["scenario"],
                "bearing_condition": ("outer_race_drilled" if e["motor"] == "faulty"
                                      else "healthy"),
                "mains_hz": 50.0,
                "duration_s": N_SAMPLES / FS,
            },
            provenance={
                "source_file": os.path.basename(e["path"]),
                "channels": list(channels),
                "decimate": decimate,
                "labels_from": "phase-current collapse rule (branches/supply.py)",
                "label_vector_used": False,
                "label_reconstruction": dict(LABEL_RECONSTRUCTION),
                "scenario_map_source": "Sci Data 12:1468 Data Records",
                "licence": "CC BY-NC-ND -- do not commit derived data",
            },
        ))
    return recs


def bearing_confound_note() -> str:
    """
    The negative result, in one place so it is quoted identically everywhere.

    Measured: vibration is 1.6-2.3x higher on the faulty motor across every
    matched rotating scenario pair. That is a real difference, and it is useless
    as evidence of a bearing fault, because there is exactly one motor per
    bearing condition.
    """
    return (
        "D4 has ONE motor per bearing condition, so 'outer-race fault' and "
        "'motor identity' are the same variable. Nothing distinguishes a bearing "
        "defect from any other difference between two physical machines -- "
        "winding tolerances, mounting, alignment, age. Vibration is measurably "
        "1.6-2.3x higher on the faulty motor across every matched scenario pair, "
        "and that measurement cannot be attributed to the bearing. Reported as a "
        "documented negative result, never as a capability."
    )
