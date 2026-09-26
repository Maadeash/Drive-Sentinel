"""
tests/test_dsp.py -- the front end must be right before anything else matters.

The load-bearing test is test_planted_fault_lands_in_the_right_order_bin: it
synthesises a current signal carrying a known BPFO amplitude modulation at two
different shaft speeds and asserts the peak appears in the SAME order bin both
times.  That single assertion catches a wrong POLE_PAIRS, an off-by-one in the
order axis, a broken envelope, and the class of bug where the pipeline silently
keys on running speed instead of on the fault.
"""

from __future__ import annotations

import numpy as np
import pytest

from drivesentinel import config as C
from drivesentinel import dsp


# ===========================================================================
# kinematics
# ===========================================================================

def test_fault_orders_match_published_values():
    orders = dsp.fault_orders()
    for name, expected in C.FAULT_ORDERS_NOMINAL.items():
        assert abs(orders[name] - expected) < 1e-3, f"{name}: {orders[name]}"


def test_bpfo_plus_bpfi_equals_ball_count():
    """Kinematic identity: BPFO + BPFI = Z, exactly, for any geometry."""
    o = dsp.fault_orders()
    assert abs(o["bpfo"] + o["bpfi"] - C.BEARING_GEOMETRY["n_balls"]) < 1e-9


def test_bpfo_is_ball_count_times_cage():
    o = dsp.fault_orders()
    assert abs(o["bpfo"] - C.BEARING_GEOMETRY["n_balls"] * o["ftf"]) < 1e-9


# ===========================================================================
# speed estimation
# ===========================================================================

@pytest.mark.parametrize("rpm,expected_f_elec", [(1500.0, 100.0), (900.0, 60.0)])
def test_shaft_speed_recovered_from_current(rpm, expected_f_elec):
    fs = 4000.0
    t = np.arange(int(2.0 * fs)) / fs
    f_elec = C.POLE_PAIRS * rpm / 60.0
    current = np.sin(2 * np.pi * f_elec * t) + 0.02 * np.random.default_rng(0).normal(size=t.size)

    fr, f_est = dsp.estimate_shaft_speed(current, fs)
    assert abs(f_est - expected_f_elec) < 0.5
    assert abs(fr * 60.0 - rpm) < 5.0


def test_speed_estimate_is_subbin_accurate():
    """Parabolic interpolation must beat the raw 0.5 Hz bin quantisation."""
    fs, dur = 4000.0, 1.0
    t = np.arange(int(dur * fs)) / fs
    true_f = 97.3                      # deliberately off-grid for a 1 s window
    sig = np.sin(2 * np.pi * true_f * t)
    est = dsp.estimate_electrical_fundamental(sig, fs)
    assert abs(est - true_f) < 0.35, f"estimated {est}, true {true_f}"


# ===========================================================================
# the order axis -- the whole premise of the design
# ===========================================================================

def _synth_current_with_bpfo(rpm: float, fs: float, seconds: float,
                             mod_depth: float = 0.08, seed: int = 0) -> np.ndarray:
    """
    Stator current carrying a BPFO amplitude modulation.

    This is the physical picture: a spalled outer race produces a periodic load
    impulse at BPFO, which modulates the machine's torque and therefore the
    stator current.  In the spectrum it appears as sidebands at f_elec +/- BPFO.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    fr = rpm / 60.0
    f_elec = C.POLE_PAIRS * fr
    f_bpfo = dsp.fault_orders()["bpfo"] * fr

    modulation = 1.0 + mod_depth * np.cos(2 * np.pi * f_bpfo * t)
    carrier = np.sin(2 * np.pi * f_elec * t)
    return modulation * carrier + 0.01 * rng.normal(size=t.size)


@pytest.mark.parametrize("rpm", [1500.0, 900.0])
def test_planted_fault_lands_in_the_right_order_bin(rpm):
    fs = C.FS_FAST / C.DECIMATE_CURRENT
    sig = _synth_current_with_bpfo(rpm, fs, C.WINDOW_SECONDS)

    fr, f_elec = dsp.estimate_shaft_speed(sig, fs)
    spec = dsp.current_sideband_order_spectrum(sig, fs, fr, f_elec)

    expected_bin = dsp.order_to_bin(dsp.fault_orders()["bpfo"])
    peak_bin = int(np.argmax(spec[4:])) + 4          # skip residual DC bins

    assert abs(peak_bin - expected_bin) <= 2, (
        f"{rpm} rpm: BPFO peak at bin {peak_bin}, expected ~{expected_bin}"
    )


def test_order_axis_is_speed_invariant():
    """
    The claim the whole architecture rests on: the same fault produces the same
    picture at 900 and 1500 rpm.  If this fails, the network can key on speed.
    """
    fs = C.FS_FAST / C.DECIMATE_CURRENT
    peaks = []
    for rpm in (1500.0, 900.0):
        sig = _synth_current_with_bpfo(rpm, fs, C.WINDOW_SECONDS)
        fr, f_elec = dsp.estimate_shaft_speed(sig, fs)
        spec = dsp.current_sideband_order_spectrum(sig, fs, fr, f_elec)
        peaks.append(int(np.argmax(spec[4:])) + 4)

    assert abs(peaks[0] - peaks[1]) <= 2, (
        f"BPFO moved between speeds: bin {peaks[0]} vs {peaks[1]} -- "
        f"the order axis is not speed-invariant"
    )


def test_no_fault_gives_no_peak_at_bpfo():
    """A clean carrier must not produce a BPFO line -- guards against the
    envelope detector manufacturing structure out of the band-pass ringing."""
    fs = C.FS_FAST / C.DECIMATE_CURRENT
    rng = np.random.default_rng(1)
    t = np.arange(int(C.WINDOW_SECONDS * fs)) / fs
    sig = np.sin(2 * np.pi * 100.0 * t) + 0.01 * rng.normal(size=t.size)

    fr, f_elec = dsp.estimate_shaft_speed(sig, fs)
    spec = dsp.current_sideband_order_spectrum(sig, fs, fr, f_elec)

    bpfo_bin = dsp.order_to_bin(dsp.fault_orders()["bpfo"])
    at_bpfo = spec[bpfo_bin - 2: bpfo_bin + 3].max()
    floor = np.median(spec[4:])
    assert at_bpfo < 20 * (floor + 1e-15), "spurious BPFO line on a healthy signal"


# ===========================================================================
# conditioning
# ===========================================================================

def test_conditioning_is_gain_invariant():
    """Doubling sensor gain must not change the conditioned spectrum."""
    rng = np.random.default_rng(2)
    spec = np.abs(rng.normal(size=C.N_ORDER_BINS)) + 0.1
    spec[98] += 5.0
    a = dsp.condition_spectrum(spec)
    b = dsp.condition_spectrum(spec * 7.5)
    assert np.allclose(a, b, atol=1e-9), "conditioning is not gain-invariant"


def test_conditioning_handles_all_zeros():
    out = dsp.condition_spectrum(np.zeros(C.N_ORDER_BINS))
    assert np.isfinite(out).all() and np.allclose(out, 0.0)


def test_window_bounds_cover_a_4s_record():
    n = int(C.RECORD_SECONDS * C.FS_FAST)
    bounds = dsp.window_bounds(n, C.FS_FAST)
    assert len(bounds) == 7, f"expected 7 windows per 4 s record, got {len(bounds)}"
    assert bounds[0][0] == 0
    assert bounds[-1][1] <= n


def test_short_record_yields_no_windows():
    assert dsp.window_bounds(1000, C.FS_FAST) == []
