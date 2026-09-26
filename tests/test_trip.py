"""
Trip gating.

Everything here runs on synthetic signals with known ground truth, because no real
dataset in the roster has a speed ramp. That is the point of the module docstring
and it is asserted below (`test_gating_is_off_by_default`).
"""

import math

import numpy as np
import pytest

from drivesentinel import config as C
from drivesentinel import trip

FS = 2000.0


def s_curve(t_accel=1.5, t_cruise=2.5, t_decel=2.0, f_lo=20.0, f_hi=60.0, fs=FS):
    """Jerk-free trapezoidal profile: ramp up, hold, ramp down. Returns (x, fs, truth)."""
    T = t_accel + t_cruise + t_decel
    t = np.arange(int(fs * T)) / fs
    f = np.piecewise(
        t,
        [t < t_accel, (t >= t_accel) & (t < t_accel + t_cruise), t >= t_accel + t_cruise],
        [lambda u: f_lo + (f_hi - f_lo) * u / t_accel,
         lambda u: f_hi,
         lambda u: f_hi - (f_hi - f_lo) * (u - t_accel - t_cruise) / t_decel])
    x = np.cos(2 * np.pi * np.cumsum(f) / fs)
    return x, fs, dict(accel=(0.0, t_accel),
                       cruise=(t_accel, t_accel + t_cruise),
                       decel=(t_accel + t_cruise, T))


def constant(f0=50.0, dur=4.0, fs=FS):
    t = np.arange(int(fs * dur)) / fs
    return np.cos(2 * np.pi * f0 * t), fs


# ---------------------------------------------------------------------------
# frequency tracking
# ---------------------------------------------------------------------------

def test_tracks_a_constant_tone():
    x, fs = constant(50.0)
    _, fe = trip.estimate_f_elec(x, fs)
    assert np.allclose(fe, 50.0, atol=0.5)


@pytest.mark.parametrize("f0", [30.0, 50.0, 120.0, 200.0])
def test_tracks_constant_tones_across_the_search_band(f0):
    x, fs = constant(f0)
    _, fe = trip.estimate_f_elec(x, fs)
    assert abs(float(np.median(fe)) - f0) < 0.5


def test_tracks_a_linear_chirp_within_tolerance():
    """Known f_e(t); the ridge must follow it, not just its average."""
    fs, T = FS, 4.0
    t = np.arange(int(fs * T)) / fs
    f_true = 30.0 + (90.0 - 30.0) * t / T
    x = np.cos(2 * np.pi * np.cumsum(f_true) / fs)
    tt, fe = trip.estimate_f_elec(x, fs)
    expect = np.interp(tt, t, f_true)
    err = np.abs(fe - expect)
    inner = (tt > 0.3) & (tt < T - 0.3)          # edges are unresolvable, by definition
    assert float(np.median(err[inner])) < 2.0
    assert float(np.max(err[inner])) < 6.0


def test_hilbert_refinement_beats_the_raw_ridge_on_a_steady_tone():
    """
    The STFT bin is 4 Hz at the default window. The phase-slope fit should do
    far better than that on a signal that is not ramping.
    """
    x, fs = constant(50.3)
    assert abs(trip.hilbert_frequency(x[:512], fs, 50.0) - 50.3) < 0.5


def test_ridge_is_not_fooled_by_a_weaker_harmonic():
    fs, T = FS, 3.0
    t = np.arange(int(fs * T)) / fs
    x = np.cos(2 * np.pi * 50 * t) + 0.3 * np.cos(2 * np.pi * 150 * t)
    _, fe = trip.estimate_f_elec(x, fs)
    assert abs(float(np.median(fe)) - 50.0) < 1.0


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def test_s_curve_segments_into_three_phases_in_order():
    x, fs, truth = s_curve()
    tt, fe = trip.estimate_f_elec(x, fs)
    segs = trip.segment(tt, fe)
    assert [s.phase for s in segs] == ["accel", "cruise", "decel"]


def test_s_curve_boundaries_land_close_to_truth():
    x, fs, truth = s_curve()
    tt, fe = trip.estimate_f_elec(x, fs)
    segs = {s.phase: s for s in trip.segment(tt, fe)}
    assert abs(segs["cruise"].t0 - truth["cruise"][0]) < 0.25
    assert abs(segs["cruise"].t1 - truth["cruise"][1]) < 0.25


def test_cruise_segment_reports_the_right_frequency():
    x, fs, _ = s_curve(f_hi=60.0)
    tt, fe = trip.estimate_f_elec(x, fs)
    cruise = [s for s in trip.segment(tt, fe) if s.phase == "cruise"][0]
    assert abs(cruise.f_mean - 60.0) < 1.0


def test_ramp_signs_are_right_way_round():
    x, fs, _ = s_curve()
    tt, fe = trip.estimate_f_elec(x, fs)
    segs = {s.phase: s for s in trip.segment(tt, fe)}
    assert segs["accel"].df_dt > 0
    assert segs["decel"].df_dt < 0
    assert abs(segs["cruise"].df_dt) < C.TRIP_CONFIG["accel_threshold_hz_s"]


def test_a_constant_tone_is_all_cruise():
    """The real-data case: every dataset in the roster looks like this."""
    x, fs = constant(50.0, dur=5.0)
    tt, fe = trip.estimate_f_elec(x, fs)
    segs = trip.segment(tt, fe)
    assert {s.phase for s in segs} == {"cruise"}


def test_short_runs_are_absorbed():
    """One noisy frame must not split a cruise in two."""
    t = np.arange(0, 5.0, 0.05)
    fe = np.full_like(t, 50.0)
    fe[40] = 60.0                                  # a single spike
    segs = trip.segment(t, fe)
    assert [s.phase for s in segs] == ["cruise"]


def test_segment_handles_a_degenerate_input():
    assert trip.segment(np.array([0.0]), np.array([50.0])) == []


# ---------------------------------------------------------------------------
# envelope segmentation -- for mains-fed machines
# ---------------------------------------------------------------------------

def test_envelope_segmentation_finds_a_startup():
    """
    D4 is mains-fed: f_e is pinned at 50 Hz, so a start-up is invisible to a
    frequency tracker and lives entirely in the current envelope.
    """
    fs, T = FS, 6.0
    t = np.arange(int(fs * T)) / fs
    env = np.clip((t - 2.0) / 0.5, 0.0, 1.0)
    x = env * np.cos(2 * np.pi * 50 * t)
    segs = trip.segment_envelope(x, fs)
    assert any(s.phase == "cruise" and s.t0 > 2.0 for s in segs)


def test_envelope_segmentation_reports_no_frequency():
    """It knows nothing about frequency and must not pretend to."""
    fs = FS
    t = np.arange(int(fs * 4.0)) / fs
    x = np.clip((t - 1.0), 0, 1) * np.cos(2 * np.pi * 50 * t)
    for s in trip.segment_envelope(x, fs):
        assert math.isnan(s.f_mean)


# ---------------------------------------------------------------------------
# gating
# ---------------------------------------------------------------------------

def test_gating_is_off_by_default():
    """
    No real dataset in the roster has a speed ramp, so gating would mark every
    window cruise and change nothing. Enabling it by default would ship an
    untested code path and an unsupported claim.
    """
    assert C.TRIP_CONFIG["enabled"] is False


def test_disabled_gate_passes_everything():
    """Disabled means 'use everything', not 'use nothing'."""
    t = np.arange(0, 5.0, 0.1)
    assert trip.gate([], t, enabled=False).all()


def test_enabled_gate_selects_only_cruise():
    x, fs, truth = s_curve()
    tt, fe = trip.estimate_f_elec(x, fs)
    segs = trip.segment(tt, fe)
    m = trip.gate(segs, tt, enabled=True)
    assert m.any() and not m.all()
    assert all(truth["cruise"][0] - 0.3 < tv < truth["cruise"][1] + 0.3
               for tv in tt[m])


def test_cruise_mask_is_empty_when_there_is_no_cruise():
    t = np.arange(0, 2.0, 0.1)
    segs = [trip.Segment("accel", 0.0, 2.0, 50.0, 10.0)]
    assert not trip.cruise_mask(segs, t).any()


# ---------------------------------------------------------------------------
# order resolution -- the analytic artefact
# ---------------------------------------------------------------------------

def test_order_resolution_matches_the_closed_form():
    r = trip.order_resolution(900.0, window_s=1.0)
    assert r["f_shaft_hz"] == pytest.approx(15.0)
    assert r["order_resolution"] == pytest.approx(60.0 / (1.0 * 900.0))
    assert r["order_resolution"] == pytest.approx(0.0667, abs=1e-4)


def test_resolution_degrades_as_the_shaft_slows():
    fast = trip.order_resolution(900.0)["order_resolution"]
    slow = trip.order_resolution(20.0)["order_resolution"]
    assert slow > fast * 40


def test_bpfo_and_bpfi_stop_being_separable_at_sheave_speed():
    """
    BPFO 3.05, BPFI 4.95 -- 1.9 orders apart. At 20 rpm a 1 s window resolves
    3.0 orders, so the two lines merge. This is the honest answer to the low-speed
    question, and it is arithmetic.
    """
    gap = C.FAULT_ORDERS_NOMINAL["bpfi"] - C.FAULT_ORDERS_NOMINAL["bpfo"]
    assert trip.order_resolution(900.0)["order_resolution"] < gap
    assert trip.order_resolution(20.0)["order_resolution"] > gap


def test_longer_window_recovers_resolution():
    """The fix is 1/T, not a cleverer algorithm."""
    a = trip.order_resolution(20.0, window_s=1.0)["order_resolution"]
    b = trip.order_resolution(20.0, window_s=45.0)["order_resolution"]
    assert b == pytest.approx(a / 45.0, rel=1e-9)
    assert b < trip.order_resolution(900.0, window_s=1.0)["order_resolution"] + 1e-9


def test_resolution_report_labels_extrapolation():
    txt = trip.resolution_report()
    assert "EXTRAPOLATION" in txt
    assert "measured" in txt
    assert "gearless sheave" in txt


def test_resolution_table_covers_measured_and_extrapolated_speeds():
    rpms = [r["shaft_rpm"] for r in trip.resolution_table()]
    assert 900.0 in rpms and 1500.0 in rpms      # measured
    assert 20.0 in rpms                          # sheave range
