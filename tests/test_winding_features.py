"""
B-S4 winding features.

The central property is amplitude invariance: D2 current is in probe volts on an
identity scale, so any feature that responds to a uniform scaling of all three
phases is comparing instrument settings between files rather than windings.
Every feature is checked for it individually and as a block.
"""

import math

import numpy as np
import pytest

from drivesentinel.adapters import kaist_pmsm as K
from drivesentinel.branches import winding as W

FS = 10_000.0
FE = 200.0
N = 10_000                      # 1.0 s


def synth(neg_frac=0.0, third=0.0, amp=1.0, fs=FS, n=N, fe=FE, seed=0):
    """
    A physically valid three-wire set: positive sequence, optional negative
    sequence, optional third harmonic. Sums to zero at every instant.
    """
    t = np.arange(n) / fs
    a = 2 * math.pi / 3
    out = []
    for k in range(3):
        sig = np.cos(2 * math.pi * fe * t - k * a)
        if neg_frac:
            sig = sig + neg_frac * np.cos(2 * math.pi * fe * t + k * a)
        if third:
            sig = sig + third * np.cos(3 * (2 * math.pi * fe * t) - k * a)
        out.append(amp * sig)
    return np.vstack(out)


def feat(win):
    return dict(zip(W.FEATURE_NAMES, W.window_features(win, FS)))


# ---------------------------------------------------------------------------
# negative sequence
# ---------------------------------------------------------------------------

def test_balanced_signal_has_near_zero_negative_sequence():
    assert feat(synth())["neg_seq_ratio"] < 1e-6


def test_injected_negative_sequence_raises_the_ratio():
    base = feat(synth())["neg_seq_ratio"]
    hurt = feat(synth(neg_frac=0.10))["neg_seq_ratio"]
    assert hurt > base
    assert hurt == pytest.approx(0.10, abs=0.01)


def test_negative_sequence_is_monotonic_in_injected_unbalance():
    prev = -1.0
    for frac in (0.0, 0.02, 0.05, 0.10, 0.25):
        v = feat(synth(neg_frac=frac))["neg_seq_ratio"]
        assert v > prev
        prev = v


# ---------------------------------------------------------------------------
# Park's vector eccentricity
# ---------------------------------------------------------------------------

def test_balanced_signal_traces_a_circle():
    """Eccentricity ~0: a healthy machine's Park vector is circular."""
    assert feat(synth())["park_eccentricity"] < 0.05


def test_unbalance_flattens_the_park_ellipse():
    assert feat(synth(neg_frac=0.20))["park_eccentricity"] > \
        feat(synth())["park_eccentricity"] + 0.3


def test_park_eccentricity_is_monotonic_in_unbalance():
    prev = -1.0
    for frac in (0.0, 0.05, 0.15, 0.30):
        v = feat(synth(neg_frac=frac))["park_eccentricity"]
        assert v > prev
        prev = v


# ---------------------------------------------------------------------------
# third-harmonic ratio
# ---------------------------------------------------------------------------

def test_clean_fundamental_has_no_third_harmonic():
    assert feat(synth())["third_harmonic_ratio"] < 1e-6


def test_injected_third_harmonic_is_recovered():
    assert feat(synth(third=0.08))["third_harmonic_ratio"] == pytest.approx(0.08, abs=0.01)


# ---------------------------------------------------------------------------
# harmonic axis
# ---------------------------------------------------------------------------

def test_fundamental_lands_in_order_01():
    f = feat(synth())
    band = [f[f"order_{h:02d}"] for h in range(1, W.N_ORDERS + 1)]
    assert int(np.argmax(band)) == 0


def test_third_harmonic_lands_in_order_03():
    f = feat(synth(third=0.3))
    assert f["order_03"] > f["order_02"]
    assert f["order_03"] > f["order_04"]


def test_harmonic_axis_is_invariant_to_f_elec():
    """
    Order h must mean h*f_e whatever f_e is. f_e is constant on D2, so this is
    not exercised by the data -- it is asserted here because the branch API
    claims it and the simulator branch will rely on it.
    """
    a = W.window_features(synth(third=0.2, fe=200.0), FS, f_elec=200.0)
    b = W.window_features(synth(third=0.2, fe=220.0), FS, f_elec=220.0)
    ia = dict(zip(W.FEATURE_NAMES, a))
    ib = dict(zip(W.FEATURE_NAMES, b))
    for h in (1, 2, 3, 4):
        assert ia[f"order_{h:02d}"] == pytest.approx(ib[f"order_{h:02d}"], abs=0.05)


def test_order_20_stays_below_nyquist():
    assert W.N_ORDERS * FE < FS / 2


def test_band_normalisation_survives_a_near_empty_median():
    """
    Regression: a spectrum with energy in one order drives the median to the
    floating-point noise floor, band/median to ~1e15, and the feature stops being
    scale invariant. The peak-relative floor is what stops that.
    """
    a = W.window_features(synth(), FS)
    b = W.window_features(synth(amp=1000.0), FS)
    assert np.allclose(a, b, atol=1e-5)
    assert np.all(np.isfinite(a))


def test_exceeding_nyquist_raises():
    with pytest.raises(ValueError, match="exceeds Nyquist"):
        W.window_features(synth(), fs=2_000.0)


# ---------------------------------------------------------------------------
# amplitude invariance -- the property the whole feature set rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.1, 0.5, 2.0, 37.0])
def test_no_feature_responds_to_uniform_amplitude_scaling(scale):
    """
    D2 amplitudes are probe volts on an identity scale. A feature that moves when
    all three phases are scaled together is reading the instrument, not the motor.
    """
    a = W.window_features(synth(neg_frac=0.07, third=0.05, amp=1.0), FS)
    b = W.window_features(synth(neg_frac=0.07, third=0.05, amp=scale), FS)
    assert np.allclose(a, b, atol=1e-5), \
        f"features moved under x{scale}: max delta {np.abs(a - b).max()}"


def test_amplitude_invariance_holds_for_every_named_feature():
    a = dict(zip(W.FEATURE_NAMES, W.window_features(synth(neg_frac=0.07, third=0.05), FS)))
    b = dict(zip(W.FEATURE_NAMES, W.window_features(
        synth(neg_frac=0.07, third=0.05, amp=1000.0), FS)))
    for name in W.FEATURE_NAMES:
        assert a[name] == pytest.approx(b[name], abs=1e-5), name


def test_features_are_finite_on_a_silent_window():
    """A dead channel must not produce NaN and poison a whole fold."""
    f = W.window_features(np.zeros((3, N), dtype=np.float32), FS)
    assert np.all(np.isfinite(f))


# ---------------------------------------------------------------------------
# shape and assembly
# ---------------------------------------------------------------------------

def test_feature_vector_length_matches_the_names():
    assert len(W.window_features(synth(), FS)) == len(W.FEATURE_NAMES)
    assert len(W.FEATURE_NAMES) == 3 + W.N_ORDERS


def test_window_features_rejects_the_wrong_channel_count():
    with pytest.raises(ValueError, match="expected 3 phases"):
        W.window_features(np.zeros((2, N)), FS)


def test_labels_exclude_healthy():
    """
    Healthy is not measurable under any session-aware split on D2 -- all three
    healthy recordings are from 2022-01-25. The branch must not offer the class.
    """
    assert "healthy" not in W.LABELS
    assert set(W.LABELS) == {"inter_coil", "inter_turn"}


def test_build_dataset_drops_healthy_by_default():
    import inspect
    assert inspect.signature(W.build_dataset).parameters["drop_healthy"].default is True
