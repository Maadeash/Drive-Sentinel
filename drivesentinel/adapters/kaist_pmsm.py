"""
D2 -- KAIST PMSM stator fault dataset (stage S4, motor winding).

Source: https://data.mendeley.com/datasets/rgn5brrgrn/5
Paper:  https://pmc.ncbi.nlm.nih.gov/articles/PMC9957734/  (Data in Brief)

Measured facts behind every decision here are in docs/data_notes_d2.md.

WHAT THE FILES ARE
------------------
96 files = 48 recordings x 2 modalities.  3 motor ratings (1.0/1.5/3.0 kW) x 2
fault types (inter-turn, inter-coil) x 8 severities, one run per cell, no repeats.
Current is 3-phase at 100 kHz; vibration is single-channel at 25.6 kHz; both run
120-150 s.  f_e is 200.00 Hz in every recording (3000 rpm, 4 pole pairs).

FIVE THINGS THIS ADAPTER HAS TO FIX, AND WHY EACH ONE MATTERS
-------------------------------------------------------------
1. CHANNEL NAMES ARE NOT CONSTANT.  41 current files are on cDAQ1Mod2 with
   channels ai0/ai2/ai3; seven are on a different chassis AND module, cDAQ5Mod1,
   with ai0/ai1/ai3.  Hardcoding ai0/ai2/ai3 silently drops a phase on those
   seven.  We enumerate channels by unit_string instead, which is stable across
   both rigs.

2. SEVERITY AND FAULT-TYPE TOKENS ARE INCONSISTENT.  47 files write severity with
   an underscore for the decimal point ("0_68"); exactly one writes a real dot
   ("1.01").  Eight vibration files say "coil" where their current partners say
   "intercoil".  A parser that splits on "_" and takes a fixed field breaks on
   both.

3. PROBE POLARITY VARIES PER FILE.  Taken at face value, 33 of 48 files give a
   zero-sequence ratio near 1.0, which is impossible for a three-wire machine.
   Two of the three current probes are reversed in some sessions.  Uncorrected,
   the negative-sequence ratio -- the branch's headline physics feature --
   measures the wiring, not the fault.

4. A RESIDUAL SURVIVES THE SIGN FIX, AND IT CLUSTERS BY SESSION.  After sign
   correction |I0|/|I+| is still 0.006-0.079 where it should be ~0, and within
   1500 W intercoil it alternates between two tight clusters (~0.035 and ~0.067)
   as severity increases.  That is a session signature sitting directly on top of
   the feature we care about.

   The plan called for a per-channel gain calibration to remove it.  It was
   built, measured, and then DISABLED -- it zeroes |I0| by construction, hits its
   plausibility clamp on 20 of 45 files, and moves the negative-sequence ratio by
   up to 0.05 on a dataset whose entire fault-signal range is 0.01-0.11.  See
   calibrate_gains() for the numbers.  The residual is exposed as a measured
   confound instead of being corrected away.

5. THE HEALTHY RECORDINGS ARE FILED TWICE.  Within each rating the 0_00 intercoil
   and 0_00 interturn files are byte-identical.  Loading all 48 puts healthy at
   12.5 % of recordings instead of its true 6.7 %.

WHAT THIS ADAPTER DELIBERATELY DOES NOT FIX
-------------------------------------------
The 3000 W motor is instrument-confounded: all seven of its non-healthy intercoil
recordings are on cDAQ5Mod1 and all its interturn recordings are on cDAQ1Mod2, so
fault type is perfectly predictable from the DAQ chassis.  No preprocessing can
remove that -- the information genuinely is not there.  The adapter therefore
EXPOSES it, as `condition["batch"]` and `condition["daq_chassis"]`, so the branch
can measure it (V4) and the metric card can declare it.  See data_notes_d2.md §7.

UNITS: PROBE VOLTS, NOT AMPS
----------------------------
Every current channel carries unit_string 'A', but also
NI_SensorSensitivity = 1.0 with a zero intercept and DAC~Channel~Sensor 'Voltage'
-- an identity scale.  The probe's V/A factor is not in the file, and measured RMS
(0.42-2.57) is not physical for a 1-3 kW PMSM.  We label the unit `probe_V` so
that an absolute-amplitude feature cannot be written by accident, and
`assert_ratio_only()` exists for feature code to call on itself.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config as C
from ..common.schema import Recording
from ..dsp import decimate_signal

FS_CURRENT = 100_000.0
FS_VIBRATION = 25_600.0
DECIMATE_FACTOR = 10                  # 100 kHz -> 10 kHz, Nyquist 5 kHz vs order-20 = 4 kHz
FS_DECIMATED = FS_CURRENT / DECIMATE_FACTOR

# Measured in every one of the 48 current recordings, to 0.25 Hz.  3000 rpm with
# 4 pole pairs.  It is a constant here, which is exactly why the f/f_e axis adds
# no invariance on this dataset -- see data_notes_d2.md section 8.
F_ELEC_HZ = 200.0
POLE_PAIRS = 4

PHASE_NAMES = ("ia", "ib", "ic")
CURRENT_UNIT = "probe_V"              # NOT 'A'.  See module docstring.
VIBRATION_UNIT = "g"

LABELS = ("healthy", "inter_turn", "inter_coil")
_TYPE_TO_LABEL = {"interturn": "inter_turn", "intercoil": "inter_coil"}

_A = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))

_NAME = re.compile(
    r"^(?P<rating>\d+)W_(?P<sev>[\d._]+?)_(?P<modality>current|vibration)_"
    r"(?P<ftype>coil|intercoil|interturn)\.tdms$"
)


# ===========================================================================
# filename parsing
# ===========================================================================

def parse_filename(name: str) -> Dict:
    """
    `1000W_0_68_current_intercoil.tdms` -> rating 1000, severity 0.68, ...

    Handles both severity spellings and folds the stray `coil` token into
    `intercoil`.  Raises rather than guessing: a filename this adapter cannot
    parse is a dataset change we want to hear about.
    """
    m = _NAME.match(os.path.basename(name))
    if not m:
        raise ValueError(f"unparseable D2 filename: {os.path.basename(name)!r}")
    g = m.groupdict()

    # "0_68" and "1.01" both mean 0.68 / 1.01.  Only one file uses the dot form,
    # and it is the current-side partner of a file that uses the underscore form.
    sev_token = g["sev"]
    severity = float(sev_token.replace("_", "."))

    ftype = "intercoil" if g["ftype"] == "coil" else g["ftype"]

    return {
        "rating_w": int(g["rating"]),
        "severity_pct": severity,
        "severity_token": sev_token,
        "modality": g["modality"],
        "fault_type": ftype,
        "fault_type_token": g["ftype"],
    }


def label_for(fault_type: str, severity_pct: float) -> str:
    """Severity 0.00 is healthy (Mendeley format description; workflow §13.1)."""
    if severity_pct == 0.0:
        return "healthy"
    return _TYPE_TO_LABEL[fault_type]


# ===========================================================================
# sequence components, polarity and gain
# ===========================================================================

def goertzel(x: np.ndarray, fs: float, f: float) -> complex:
    """
    Single-frequency complex amplitude.

    Vectorised rather than the textbook recurrence: for one bin over 100k samples
    the recurrence is a Python loop and costs ~0.7 s, while the direct product
    against a complex exponential is a few milliseconds. Same answer.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    k = np.exp(-2j * math.pi * f * np.arange(n) / fs)
    return complex(2.0 * np.dot(x, k) / n)


def sequence_components(phasors: Sequence[complex]) -> Tuple[float, float, float]:
    """Fortescue. Returns (|I0|, |I1|, |I2|)."""
    ia, ib, ic = phasors
    i0 = (ia + ib + ic) / 3
    i1 = (ia + _A * ib + _A * _A * ic) / 3
    i2 = (ia + _A * _A * ib + _A * ic) / 3
    return abs(i0), abs(i1), abs(i2)


def negative_sequence_ratio(phasors: Sequence[complex]) -> float:
    """
    min(|I1|,|I2|) / max(|I1|,|I2|).

    min-over-max rather than |I2|/|I1| because the channel-to-phase ORDER is not
    known per file -- some rigs wire ABC and some ACB, which swaps which
    component is "positive".  The ratio of the smaller to the larger is the same
    physical quantity either way, and it does not need the order resolved.
    """
    _, i1, i2 = sequence_components(phasors)
    hi = max(i1, i2)
    return (min(i1, i2) / hi) if hi > 0 else float("nan")


def detect_polarity(phasors: Sequence[complex]) -> Tuple[Tuple[int, int, int], float, float]:
    """
    Find the sign vector minimising |I0|/max(|I1|,|I2|).

    Returns (signs, i0rel_before, i0rel_after).

    Why |I0| is the right objective: in a three-wire machine the three phase
    currents sum to zero by Kirchhoff, so the true zero-sequence component is
    zero regardless of the fault.  Any large |I0| is a measurement artefact.  It
    is therefore an objective that carries no information about the fault, which
    is what makes correcting against it safe -- we are not tuning on the label.

    The first channel is held at +1: a global sign flip changes nothing physical
    and would make the answer ambiguous.
    """
    def i0rel(sgn):
        i0, i1, i2 = sequence_components([sgn[i] * phasors[i] for i in range(3)])
        hi = max(i1, i2)
        return (i0 / hi) if hi > 0 else float("inf")

    before = i0rel((1, 1, 1))
    best_signs, best = (1, 1, 1), before
    for tail in itertools.product((1, -1), repeat=2):
        sgn = (1,) + tail
        v = i0rel(sgn)
        if v < best:
            best_signs, best = sgn, v
    return best_signs, before, best


def calibrate_gains(phasors: Sequence[complex],
                    max_trim: float = 0.15) -> Tuple[Tuple[float, float, float], float]:
    """
    Per-channel gains g (g0 = 1) that further minimise |I0|, after sign fixing.

    *** IMPLEMENTED BUT OFF BY DEFAULT. DO NOT ENABLE WITHOUT READING THIS. ***

    The plan called for this to mop up the 3-7 % residual |I0| that survives the
    sign fix.  It was built, measured, and then disabled, because the measurement
    said it does more harm than good.  Three findings, in increasing order of
    seriousness:

    1. IT IS NOT A DIAGNOSTIC.  Two free real parameters against a complex
       residual with two real degrees of freedom has an exact solution almost
       always, so |I0| after calibration is 0.0000 by construction on 25 of 45
       recordings.  A number that is zero because the algebra forces it to be
       zero measures nothing.

    2. THE GAIN MODEL IS WRONG FOR 20 OF 45 FILES.  Those hit the +/-15 % clamp,
       i.e. the fit wanted corrections larger than any credible sensor gain
       mismatch.  Whatever produces the residual in those recordings, it is not
       a per-channel scalar gain -- inter-channel timing skew is the more likely
       explanation, and a real gain cannot represent it.

    3. IT MOVES THE FEATURE WE ARE TRYING TO MEASURE, BY MORE THAN THE FAULT
       DOES.  Measured change in the negative-sequence ratio when enabled:

         1000W healthy        0.0236 -> 0.0306   (+0.0071)
         1000W intercoil 7.56 0.0258 -> 0.0318   (+0.0060)
         1500W healthy        0.0655 -> 0.0151   (-0.0504)
         1500W interturn 8.74 0.0096 -> 0.0341   (+0.0245)
         3000W interturn17.86 0.0904 -> 0.1193   (+0.0289)

       The whole fault-signal range across this dataset is about 0.01 to 0.11.
       A "correction" that can shift a value by 0.05 is not a correction on that
       scale; it is a second, larger source of variance sitting on top of the
       first.  Zeroing |I0| is only safe if I0 is pure artefact, and per (2) it
       is not.

    What we do instead: apply the sign fix, which is discrete, physically
    unambiguous and verified on all 48 files, and treat the residual |I0| as a
    MEASURED PER-FILE CONFOUND -- exposed in `provenance["i0rel_after_sign"]` and
    offered to the V4 batch control as a candidate predictor.  If the batch is
    predictable from the residual, that is a finding to report, not a defect to
    paper over.

    `max_trim` clamps to +/-15 % so that an implausible fit is visible as a
    clamp rather than silently applied.
    """
    a, b, c = phasors
    # Solve  min || a + g1*b + g2*c ||  ->  real 2x2 normal equations.
    M = np.array([[b.real, c.real], [b.imag, c.imag]], dtype=np.float64)
    rhs = -np.array([a.real, a.imag], dtype=np.float64)
    try:
        g = np.linalg.lstsq(M, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return (1.0, 1.0, 1.0), float("nan")

    g1, g2 = (float(np.clip(v, 1.0 - max_trim, 1.0 + max_trim)) for v in g)
    gains = (1.0, g1, g2)
    i0, i1, i2 = sequence_components([gains[i] * phasors[i] for i in range(3)])
    hi = max(i1, i2)
    return gains, (i0 / hi if hi > 0 else float("nan"))


def _phasors(signals: Sequence[np.ndarray], fs: float,
             n: int = 100_000) -> List[complex]:
    n = min(n, min(len(s) for s in signals))
    return [goertzel(np.asarray(s[:n], dtype=np.float64), fs, F_ELEC_HZ)
            for s in signals]


def assert_ratio_only(name: str) -> None:
    """
    Call from feature code that is about to use a current amplitude.

    D2 current is in probe volts on an identity scale, so absolute amplitude is
    meaningless and not comparable across files.  This exists so the constraint
    lives in code rather than only in a document.
    """
    raise AssertionError(
        f"{name}: D2 current amplitudes are probe volts on an identity scale "
        f"(NI_SensorSensitivity = 1.0), not amps. Absolute-amplitude features "
        f"are not comparable across files. Use a ratio. See adapters/kaist_pmsm.py."
    )


# ===========================================================================
# reading
# ===========================================================================

# The three acquisition days, verified from TDMS `Date Created` and corroborated
# by chassis, duration discipline and root-name style -- four independent signals
# agreeing on the same partition.  See docs/data_notes_d2.md section 7D.
#
# This is the group axis for V5 (leave-one-session-out).  It is NOT the residual
# clustering: the residual resolves five groups, and those five nest perfectly
# inside these three days.  The days are the fact; the residual reveals them.
SESSIONS = ("2022-01-25", "2022-03-08", "2022-08-11")


def _read_tdms(path: str, want_unit: str):
    """Channels carrying `want_unit`, in file order, plus their properties."""
    from nptdms import TdmsFile
    with TdmsFile.open(path) as tf:
        chans = [ch for grp in tf.groups() for ch in grp.channels()
                 if ch.properties.get("unit_string") == want_unit]
        if not chans:
            raise ValueError(f"{os.path.basename(path)}: no channels with unit {want_unit!r}")
        names = [ch.name for ch in chans]
        props = dict(chans[0].properties)
        data = [np.asarray(ch[:], dtype=np.float64) for ch in chans]
        root = tf.properties.get("name")
        created = tf.properties.get("Date Created") or tf.properties.get("datetime")
    return names, data, props, root, created


def _file_digest(path: str, nbytes: int = 8_000_000) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read(nbytes)).hexdigest()


def _root_style(root_name: Optional[str]) -> str:
    """
    Coarse acquisition-session fingerprint from the FlexLogger test name.

    Three styles exist and they track the session, not the condition:
      'Current_<R>'              the cDAQ5 rig
      'kW_<R>ohm_current'        capital W
      'kw_<R>ohm_current_<type>' lower-case w, fault type appended
    The names themselves are unreliable as CONDITION labels (one 3000 W file's
    root duplicates another's), but as a session marker they are consistent.
    """
    if not root_name:
        return "unknown"
    if root_name.startswith(("Current_", "Normal_", "Vibration_")):
        return "Current_<R>"
    if re.match(r"^\d+(\.\d+)?kW_", root_name):
        return "kW_<R>ohm_current"
    if re.match(r"^\d+(\.\d+)?kw_", root_name):
        return "kw_<R>ohm_current_<type>"
    return "other"


# ===========================================================================
# the loader
# ===========================================================================

def index(root: str) -> List[Dict]:
    """
    Parse every filename under `root`, pair current with vibration, and mark the
    duplicate healthy recordings -- without reading any sample data.

    Kept separate from `load` so a caller can see the grid, the dedup and the
    counts in milliseconds, which is what the branch script prints before it
    trains.
    """
    entries: Dict[Tuple, Dict] = {}
    for sub in ("current", "vibration"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} (expected D2 laid out as current/ and vibration/)")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".tdms"):
                continue
            meta = parse_filename(fn)
            key = (meta["rating_w"], meta["fault_type"], meta["severity_pct"])
            e = entries.setdefault(key, {
                "rating_w": meta["rating_w"],
                "fault_type": meta["fault_type"],
                "severity_pct": meta["severity_pct"],
                "label": label_for(meta["fault_type"], meta["severity_pct"]),
            })
            e[f"{meta['modality']}_path"] = os.path.join(d, fn)
            e[f"{meta['modality']}_file"] = fn

    recs = sorted(entries.values(),
                  key=lambda e: (e["rating_w"], e["fault_type"], e["severity_pct"]))

    # Mark duplicates: within a rating, the two severity-0 entries are the same
    # bytes.  Keep the intercoil-named copy; the choice is arbitrary because the
    # files are identical, but it must be deterministic.
    for rating in sorted({e["rating_w"] for e in recs}):
        zeros = [e for e in recs if e["rating_w"] == rating and e["severity_pct"] == 0.0]
        if len(zeros) != 2:
            raise ValueError(f"{rating}W: expected 2 severity-0 entries, found {len(zeros)}")
        a, b = sorted(zeros, key=lambda e: e["fault_type"])   # intercoil, interturn
        same = (os.path.getsize(a["current_path"]) == os.path.getsize(b["current_path"])
                and _file_digest(a["current_path"]) == _file_digest(b["current_path"]))
        a["is_duplicate"], a["duplicate_of"] = False, None
        b["is_duplicate"] = same
        b["duplicate_of"] = a["current_file"]
        b["duplicate_verified_identical"] = same
        if not same:
            # Do not silently keep both: if the bytes ever differ, that is a
            # dataset change and the healthy count is no longer 3.
            raise ValueError(
                f"{rating}W severity-0 files are NOT byte-identical "
                f"({a['current_file']} vs {b['current_file']}). The dedup rule in "
                f"data_notes_d2.md no longer holds -- re-inspect before loading."
            )
    for e in recs:
        e.setdefault("is_duplicate", False)
        e.setdefault("duplicate_of", None)
    return recs


def load(root: str = None, with_vibration: bool = False,
         decimate: bool = True, drop_duplicates: bool = True,
         gain_calibration: bool = False, limit: int = None) -> List[Recording]:
    """
    Load D2 as `Recording` objects, one per unique recording.

    group = motor rating ("1000W" / "1500W" / "3000W"), which is the physical
    unit leave-one-motor-out holds out.

    `gain_calibration` defaults to False. See calibrate_gains() for the measured
    reason -- it moves the negative-sequence ratio by up to 0.05 on a dataset
    whose whole fault-signal range is 0.01-0.11. The gains are still COMPUTED and
    recorded in provenance so the decision stays auditable; they are just not
    applied.

    Every correction the adapter applies is recorded in `provenance` -- the
    detected sign vector, the calibrated gains, |I0| before and after, and the
    channel names actually found -- because all of them are INFERRED from the
    signal rather than read from the file, and a reader of the results has to be
    able to see what was done.
    """
    root = root or os.environ.get("DRIVESENTINEL_EXT_DATA_D2",
                                  os.path.join(C.PROJECT_ROOT, "data_ext", "kaist_pmsm"))
    entries = index(root)
    if drop_duplicates:
        entries = [e for e in entries if not e["is_duplicate"]]
    if limit:
        entries = entries[:limit]

    out: List[Recording] = []
    for e in entries:
        names, data, props, root_name, created = _read_tdms(e["current_path"], "A")
        if len(data) != 3:
            raise ValueError(f"{e['current_file']}: found {len(data)} current channels, "
                             f"expected 3 ({names})")

        ph = _phasors(data, FS_CURRENT)
        signs, i0_before, i0_after_sign = detect_polarity(ph)
        ph_signed = [signs[i] * ph[i] for i in range(3)]
        gains, i0_after_gain = calibrate_gains(ph_signed)
        applied_gains = gains if gain_calibration else (1.0, 1.0, 1.0)

        corrected = [signs[i] * applied_gains[i] * data[i] for i in range(3)]
        if decimate:
            corrected = [decimate_signal(s, DECIMATE_FACTOR).astype(np.float32)
                         for s in corrected]
            fs_out = FS_DECIMATED
        else:
            corrected = [s.astype(np.float32) for s in corrected]
            fs_out = FS_CURRENT

        signals = dict(zip(PHASE_NAMES, corrected))
        fs = {n: fs_out for n in PHASE_NAMES}
        units = {n: CURRENT_UNIT for n in PHASE_NAMES}

        if with_vibration and e.get("vibration_path"):
            vnames, vdata, vprops, _, _ = _read_tdms(e["vibration_path"], "g")
            signals["vib"] = vdata[0].astype(np.float32)
            fs["vib"] = FS_VIBRATION
            units["vib"] = VIBRATION_UNIT

        session = str(created)[:10] if created is not None else "unknown"
        if session not in SESSIONS:
            raise ValueError(
                f"{e['current_file']}: acquisition date {session!r} is not one of "
                f"the three known sessions {SESSIONS}. The session grouping in "
                f"data_notes_d2.md section 7D no longer holds -- re-verify before loading."
            )
        chassis = names[0].split("/")[0] if "/" in names[0] else "unknown"
        style = _root_style(root_name)
        batch = f"{chassis}|{''.join('+' if s > 0 else '-' for s in signs)}|{style}"

        out.append(Recording(
            source="kaist_pmsm",
            stage="S4",
            signals=signals,
            fs=fs,
            label=e["label"],
            group=f"{e['rating_w']}W",
            condition={
                "rating_w": e["rating_w"],
                "fault_type": e["fault_type"],
                "severity_pct": e["severity_pct"],
                "f_elec_hz": F_ELEC_HZ,
                "pole_pairs": POLE_PAIRS,
                # Exposed, not hidden: these are what V4 predicts instead of the
                # fault class, and what the 3000 W confound is made of.
                "batch": batch,
                "session": session,
                "daq_chassis": chassis,
                "root_style": style,
            },
            provenance={
                "current_file": e["current_file"],
                "vibration_file": e.get("vibration_file"),
                "channels_found": names,
                "root_name": root_name,
                "date_created": str(created),
                "units_declared": props.get("unit_string"),
                "units_actual": CURRENT_UNIT,
                "units_note": (
                    "NI_SensorSensitivity=1.0 with zero intercept and sensor type "
                    "'Voltage': the 'A' label is an identity scale, so these are "
                    "probe volts. Ratio features only."
                ),
                "sensor_sensitivity": props.get("NI_SensorSensitivity"),
                "polarity_signs": list(signs),
                "gains_computed": [round(g, 6) for g in gains],
                "gains_applied": list(applied_gains),
                "gain_calibration_enabled": bool(gain_calibration),
                "gain_clamped": any(abs(g - 1.0) >= 0.1499 for g in gains[1:]),
                "i0rel_raw": round(i0_before, 5),
                "i0rel_after_sign": round(i0_after_sign, 5),
                "i0rel_if_gain_applied": round(i0_after_gain, 5),
                # The residual after the sign fix is a per-file confound, not a
                # defect to hide. V4 gets to try predicting the batch from it.
                "i0rel_residual": round(i0_after_sign, 5),
                "decimated": bool(decimate),
                "decimate_factor": DECIMATE_FACTOR if decimate else 1,
                "fs_hz": fs_out,
                "duplicate_of": e.get("duplicate_of"),
            },
        ))
    return out


def grid_report(entries: List[Dict]) -> str:
    """Human-readable grid, printed by the branch script before it trains."""
    lines = []
    uniq = [e for e in entries if not e.get("is_duplicate")]
    dup = [e for e in entries if e.get("is_duplicate")]
    lines.append(f"  recordings  : {len(uniq)} unique ({len(entries)} files, "
                 f"{len(dup)} duplicate healthy dropped)")
    by_label: Dict[str, int] = {}
    for e in uniq:
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
    lines.append(f"  by label    : {by_label}")
    for rating in sorted({e["rating_w"] for e in uniq}):
        g = [e for e in uniq if e["rating_w"] == rating]
        for ft in sorted({e["fault_type"] for e in g}):
            sev = sorted(e["severity_pct"] for e in g if e["fault_type"] == ft)
            lines.append(f"    {rating:5}W {ft:10} n={len(sev)}  severities {sev}")
    return "\n".join(lines)
