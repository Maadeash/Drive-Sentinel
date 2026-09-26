"""
B-S2/S3 inverter telemetry features.

The property that matters most here is CALIBRATION INDEPENDENCE: the NTC
Steinhart-Hart refit in data_notes_d3.md is three parameters on four points, and
its absolute accuracy below 30 C is an extrapolation. No feature in this branch
may depend on it. That is asserted directly rather than assumed.
"""

import os

import numpy as np
import pytest

from drivesentinel.adapters import bacha_inverter as B
from drivesentinel.branches import inverter_telemetry as IT
from drivesentinel.common.schema import Recording

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data_ext", "bacha_inverter")
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "NORMAL_OP.txt")),
    reason="D3 not present (data_ext/ is gitignored)")

N = 50          # one 5 s window at 10 Hz


def win(ia=0.2, ib=0.2, t1=500, t2=500, t3=500, n=N, drift=0.0):
    d = np.arange(n, dtype=float) * drift
    return {"Ia": np.full(n, ia), "Ib": np.full(n, ib),
            "T1_adc": np.full(n, float(t1)) - d,
            "T2_adc": np.full(n, float(t2)),
            "T3_adc": np.full(n, float(t3))}


def named(f, use_temperature=True):
    return dict(zip(IT.feature_names(use_temperature), f))


# ---------------------------------------------------------------------------
# calibration independence -- the property the branch rests on
# ---------------------------------------------------------------------------

def test_no_feature_name_implies_a_converted_temperature():
    """
    Temperature enters only as raw ADC and ADC differences. A feature called
    `T1_mean` in degrees would tie this branch to the Steinhart-Hart refit.
    """
    for n in IT.feature_names(True):
        if n.startswith("T") or n.startswith("dT"):
            assert "adc" in n.lower(), n


def test_temperature_channels_used_are_the_raw_adc_ones():
    assert IT.TEMP_ADC_CH == ("T1_adc", "T2_adc", "T3_adc")
    assert all(c.endswith("_adc") for c in IT.TEMP_ADC_CH)


def test_features_are_unchanged_by_a_monotone_recalibration():
    """
    The decisive test. Shift every NTC ADC reading by a constant -- which is what
    a different calibration does to first order -- and the DIFFERENCE features
    must not move. Absolute ADC statistics do move, and that is fine: they are
    raw sensor readings, not temperatures.
    """
    a = named(IT.window_features(win(t1=480, t2=500, t3=520)))
    b = named(IT.window_features(win(t1=480 + 37, t2=500 + 37, t3=520 + 37)))
    for k in ("dT12_adc", "dT13_adc", "dT23_adc", "T_spread_adc"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_temperature_rate_sign_survives_recalibration():
    """A falling NTC ADC means warming, under any monotone curve."""
    f = named(IT.window_features(win(drift=0.5)))
    assert f["T_min_adc_rate"] < 0


# ---------------------------------------------------------------------------
# dropped channels
# ---------------------------------------------------------------------------

def test_dead_channels_are_not_in_the_feature_set():
    names = " ".join(IT.feature_names(True))
    for dead in ("VDC", "IDC", "VD"):
        assert dead not in names


def test_no_power_or_dc_rate_features():
    """
    Vdc*Idc, dVdc/dt and dIdc/dt were in the original plan. They are products and
    derivatives of channels that are constant to within their quantisation noise.
    """
    names = " ".join(IT.feature_names(True)).lower()
    for banned in ("power", "vdc", "idc"):
        assert banned not in names


# ---------------------------------------------------------------------------
# current features
# ---------------------------------------------------------------------------

def test_balanced_currents_give_zero_imbalance():
    f = named(IT.window_features(win(ia=0.3, ib=0.3)))
    assert f["Ia_Ib_imbalance"] == pytest.approx(0.0, abs=1e-9)


def test_imbalance_has_the_right_sign_and_grows():
    lo = named(IT.window_features(win(ia=0.35, ib=0.30)))["Ia_Ib_imbalance"]
    hi = named(IT.window_features(win(ia=0.50, ib=0.30)))["Ia_Ib_imbalance"]
    assert 0 < lo < hi


def test_imbalance_is_finite_on_zero_current():
    f = named(IT.window_features(win(ia=0.0, ib=0.0)))
    assert np.isfinite(f["Ia_Ib_imbalance"])


def test_features_are_finite_everywhere():
    assert np.all(np.isfinite(IT.window_features(win())))
    assert np.all(np.isfinite(IT.window_features(win(ia=0, ib=0, t1=1, t2=1, t3=1))))


# ---------------------------------------------------------------------------
# the ablation must genuinely remove temperature
# ---------------------------------------------------------------------------

def test_electrical_only_feature_set_is_strictly_smaller():
    full = IT.feature_names(True)
    elec = IT.feature_names(False)
    assert set(elec) < set(full)
    assert len(elec) == 10 and len(full) == 28


def test_electrical_only_has_no_temperature_feature():
    for n in IT.feature_names(False):
        assert not n.startswith("T") and not n.startswith("dT")


def test_electrical_only_features_ignore_temperature_entirely():
    a = IT.window_features(win(t1=500), use_temperature=False)
    b = IT.window_features(win(t1=300), use_temperature=False)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------

def test_window_and_hop_match_the_spec():
    assert IT.WINDOW_S == 5.0 and IT.HOP_S == 1.0


def test_feature_vector_length_matches_the_names():
    assert len(IT.window_features(win())) == len(IT.feature_names(True))
    assert len(IT.window_features(win(), use_temperature=False)) == \
        len(IT.feature_names(False))


def test_missing_channel_raises_rather_than_guessing():
    r = Recording(source="bacha_inverter", stage="S3",
                  signals={"Ia": np.zeros(200)}, fs={"Ia": 10.0},
                  label="normal", group="F0", condition={"f_code": "F0"},
                  provenance={})
    with pytest.raises(KeyError, match="missing channel"):
        IT.recording_windows(r)


# ---------------------------------------------------------------------------
# labels and notes
# ---------------------------------------------------------------------------

def test_four_class_family():
    assert IT.LABELS == ("normal", "open_circuit", "short_circuit", "over_temp")


def test_confound_note_states_the_early_late_problem():
    n = IT.CONFOUND_NOTE
    assert "EARLY-IN-RUN from LATE-IN-RUN" in n
    assert "no group axis" in n
    assert "INDICATIVE" in n


def test_temperature_note_warns_the_four_class_score_is_a_thermometer():
    assert "thermometer" in IT.TEMPERATURE_NOTE


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

@needs_data
def test_dataset_builds_from_the_nine_conditions():
    D = IT.build_dataset(B.load(ROOT))
    assert D["X"].shape[1] == len(IT.feature_names(True))
    assert set(D["run"].tolist()) == {f"F{i}" for i in range(9)}
    assert set(D["family"].tolist()) == set(IT.LABELS)


@needs_data
def test_position_is_the_within_run_index():
    D = IT.build_dataset(B.load(ROOT))
    for run in set(D["run"].tolist()):
        pos = D["position"][D["run"] == run]
        assert pos.min() == 0
        assert np.array_equal(np.sort(pos), np.arange(pos.size))


@needs_data
def test_smallest_condition_still_yields_windows():
    """F4 has 341 samples -> (341-50)//10 + 1 = 30 windows."""
    D = IT.build_dataset(B.load(ROOT))
    assert int((D["run"] == "F4").sum()) == 30


@needs_data
def test_electrical_only_dataset_has_the_same_rows():
    recs = B.load(ROOT)
    a = IT.build_dataset(recs, use_temperature=True)
    b = IT.build_dataset(recs, use_temperature=False)
    assert a["X"].shape[0] == b["X"].shape[0]
    assert np.array_equal(a["family"], b["family"])
    assert b["X"].shape[1] < a["X"].shape[1]


@needs_data
def test_run_identification_control_is_recorded():
    """
    The control that converts the over_temp inference into a measurement. If it
    ever disappears from the results JSON, the 4-class number loses the context
    that makes it quotable.
    """
    import json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "multistage", "inverter_telemetry",
                     "inverter_telemetry_results.json")
    if not os.path.exists(p):
        pytest.skip("branch not run")
    with open(p) as fh:
        r = json.load(fh)
    rid = r["run_identification_control"]
    assert rid["n_runs"] == 9
    for k in ("with_temperature", "electrical_only"):
        assert rid[k]["accuracy"] > rid[k]["majority_baseline"], k
    # The point of the control: run identity is close to the condition score.
    assert rid["with_temperature"]["accuracy"] > 0.9
