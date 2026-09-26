"""
The one data structure every multi-stage branch sees.

WHY THIS EXISTS
---------------
Four datasets, four sets of quirks: KAIST writes one channel per TDMS segment with
per-file probe polarity that has to be detected rather than assumed; Bacha ships
10 Hz Arduino ADC integers whose published unit conversion is wrong; Thomas ships
a flat 1e6 x 9 MATLAB array with a label vector that does not carry a file index.

Adapters are the only code allowed to know any of that.  Branches only ever see a
`Recording`.  The rule is worth stating because the alternative -- a branch that
special-cases "if source == kaist" -- is how the v4 pipeline grew its leakage.

WHAT `group` MEANS, AND WHY IT IS NOT OPTIONAL
----------------------------------------------
`group` is the physical unit that must never straddle a train/test split: a motor,
a bearing, a run.  It is a required field rather than a convenience because every
honest number in this project is a group-holdout number, and the one time this
project shipped a leaky result it was because a split was made over rows instead
of over specimens (see folds.py, "THE INNER SPLIT IS THE ONE PEOPLE GET WRONG").

Making it required means an adapter author has to answer "what generalises here?"
before any model is fitted, which is the point at which the question is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

# Drive stages, in power-flow order.  See workflow_v2.md section 1.
STAGES = ("S1", "S2", "S3", "S4", "S5")

STAGE_NAMES = {
    "S1": "supply input",
    "S2": "DC link",
    "S3": "inverter",
    "S4": "motor winding",
    "S5": "bearing / mechanical",
}

SOURCES = ("paderborn", "kaist_pmsm", "bacha_inverter", "thomas_motor", "sim_gem")


@dataclass(frozen=True)
class Recording:
    """
    One continuous capture from one physical unit under one condition.

    signals   name -> 1-D float32 array.  Channels may differ in length and in
              sample rate; `fs` carries the rate per channel rather than one rate
              per recording, because D2 pairs 100 kHz current with 25.6 kHz
              vibration in the same logical recording.
    fs        name -> Hz.  Every key in `signals` must appear here.
    label     branch-level class name.
    group     the physical unit used for splitting.  Never a file name unless the
              file genuinely is the unit.
    condition free-form: load, speed, severity, direction, polarity applied...
    provenance how this Recording was derived -- source paths, adapter decisions,
              and anything an adapter had to infer rather than read.  This is what
              a data-notes document is generated from.
    """

    source: str
    stage: str
    signals: Dict[str, np.ndarray]
    fs: Dict[str, float]
    label: str
    group: str
    condition: Dict = field(default_factory=dict)
    provenance: Dict = field(default_factory=dict)
    simulated: bool = False

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage {self.stage!r} not in {STAGES}")
        if self.source not in SOURCES:
            raise ValueError(f"source {self.source!r} not in {SOURCES}")
        if not self.group:
            raise ValueError("group is required -- see module docstring")
        if not self.signals:
            raise ValueError("a Recording with no signals is a bug, not an edge case")

        missing = set(self.signals) - set(self.fs)
        if missing:
            raise ValueError(f"no sample rate for channel(s) {sorted(missing)}")

        for name, arr in self.signals.items():
            a = np.asarray(arr)
            if a.ndim != 1:
                raise ValueError(f"channel {name!r} is {a.ndim}-D; signals are 1-D")
            if a.size == 0:
                raise ValueError(f"channel {name!r} is empty")
            if self.fs[name] <= 0:
                raise ValueError(f"channel {name!r} has fs={self.fs[name]}")

    # -- derived views -----------------------------------------------------

    def duration(self, channel: Optional[str] = None) -> float:
        """
        Seconds.  With no channel, the SHORTEST channel's duration.

        Shortest rather than longest so that a window index computed from this is
        valid for every channel.  An earlier draft justified this with a measured
        within-file length spread in D2; that measurement was wrong -- it came
        from a hand-written TDMS parser that miscounted chunked segments, and
        nptdms reports all 48 D2 current recordings as having three exactly
        equal-length channels.  The rule stays because it costs nothing and
        because adapters legitimately produce ragged channels (a decimation that
        divides unevenly, a trimmed transient), not because D2 needs it today.
        """
        names = [channel] if channel else list(self.signals)
        return min(len(self.signals[n]) / self.fs[n] for n in names)

    def channels(self, *names: str) -> Tuple[np.ndarray, ...]:
        """Fetch several channels at once, erroring clearly on a missing one."""
        missing = [n for n in names if n not in self.signals]
        if missing:
            raise KeyError(
                f"{self.source}/{self.group}: no channel(s) {missing}; "
                f"have {sorted(self.signals)}"
            )
        return tuple(self.signals[n] for n in names)

    def aligned(self, *names: str) -> Tuple[np.ndarray, ...]:
        """
        Several channels of the SAME rate, truncated to a common length.

        Refuses to mix rates, because silently resampling one of three phase
        currents is exactly the kind of thing that produces a plausible and wrong
        negative-sequence number.
        """
        arrs = self.channels(*names)
        rates = {self.fs[n] for n in names}
        if len(rates) != 1:
            raise ValueError(
                f"{self.source}/{self.group}: channels {names} have rates "
                f"{sorted(rates)}; resample explicitly before aligning"
            )
        n = min(len(a) for a in arrs)
        return tuple(a[:n] for a in arrs)

    def with_signals(self, signals: Mapping[str, np.ndarray],
                     fs: Optional[Mapping[str, float]] = None,
                     **overrides) -> "Recording":
        """
        Copy with different signals -- how an adapter applies decimation or a
        polarity correction without mutating a frozen dataclass.
        """
        base = dict(
            source=self.source, stage=self.stage,
            signals=dict(signals),
            fs=dict(fs) if fs is not None else dict(self.fs),
            label=self.label, group=self.group,
            condition=dict(self.condition),
            provenance=dict(self.provenance),
            simulated=self.simulated,
        )
        base.update(overrides)
        return Recording(**base)

    def summary(self) -> str:
        chans = ", ".join(
            f"{n}@{self.fs[n]:g}Hz x{len(self.signals[n])}"
            for n in sorted(self.signals)
        )
        tag = " SIMULATED" if self.simulated else ""
        return (f"{self.source}/{self.stage} group={self.group} "
                f"label={self.label} [{chans}]{tag}")


def windows(rec: Recording, channels, window_s: float, hop_s: float,
            fs: Optional[float] = None) -> np.ndarray:
    """
    Stack several same-rate channels into (n_windows, n_channels, n_samples).

    Windows never straddle recordings, because a Recording is the unit an adapter
    emits and the unit a group owns.  That is the property that makes
    `splits.assert_no_group_overlap` sufficient rather than merely necessary.
    """
    arrs = rec.aligned(*channels)
    rate = fs or rec.fs[channels[0]]
    w = int(round(window_s * rate))
    h = int(round(hop_s * rate))
    if w <= 0 or h <= 0:
        raise ValueError(f"window={w} hop={h} samples: both must be positive")

    n = len(arrs[0])
    if n < w:
        return np.empty((0, len(channels), w), dtype=np.float32)

    starts = range(0, n - w + 1, h)
    out = np.empty((len(list(starts)), len(channels), w), dtype=np.float32)
    for i, s in enumerate(range(0, n - w + 1, h)):
        for c, a in enumerate(arrs):
            out[i, c] = a[s:s + w]
    return out
