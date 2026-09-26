"""
D3 Bacha adapter.

Split in two: unit tests on the conversions, which always run, and integration
tests against `data_ext/bacha_inverter/`, which skip when the dataset is absent
(it is gitignored, so CI without the data must still be green).

The integration tests assert the published label counts exactly. That is
deliberate -- a silently short file would change class balance without changing
anything a metric would show.
"""

import os

import numpy as np
import pytest

from drivesentinel.adapters import bacha_inverter as B

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data_ext", "bacha_inverter")
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "NORMAL_OP.txt")),
    reason="D3 not present (data_ext/ is gitignored)",
)


# ---------------------------------------------------------------------------
# current conversion
# ---------------------------------------------------------------------------

def test_current_midrail_is_zero_amps():
    """ACS712 sits at VREF/2 with no current through it."""
    assert B.adc_to_current_acs712(np.array([511.5]))[0] == pytest.approx(0.0, abs=1e-9)


def test_current_sensitivity_is_100mv_per_amp():
    lo = B.adc_to_current_acs712(np.array([511.5]))[0]
    hi = B.adc_to_current_acs712(np.array([511.5 + 1023 / 5 * 0.1]))[0]
    assert hi - lo == pytest.approx(1.0, rel=1e-6)


def test_current_reproduces_the_published_value():
    """The author's own formula is correct for the current channels; pin it."""
    assert B.adc_to_current_acs712(np.array([532.0]))[0] == pytest.approx(
        1.0019550342130978, rel=1e-9)


def test_current_is_signed():
    assert B.adc_to_current_acs712(np.array([388.0]))[0] < 0


# ---------------------------------------------------------------------------
# temperature conversion -- the bug this adapter exists to fix
# ---------------------------------------------------------------------------

def test_script_coefficients_reproduce_the_published_csv():
    """
    Proof that we understand what the shipped CSV did, before rejecting it:
    ADC 515 -> 10.577 C is exactly converted_dataset-2.csv's first T1 value.
    """
    got = B.adc_to_temperature_ntc(np.array([515.0]), coefficients=B.SH_SCRIPT)[0]
    assert got == pytest.approx(10.577241530953813, rel=1e-6)


def test_calibrated_fit_recovers_its_own_calibration_points():
    for adc, celsius in B.NTC_CALIBRATION_POINTS:
        got = B.adc_to_temperature_ntc(np.array([adc]))[0]
        assert got == pytest.approx(celsius, abs=0.5)


def test_calibrated_and_script_coefficients_disagree_materially():
    """
    The whole reason the default changed. If this ever stops being true,
    something has been edited by accident.
    """
    adc = np.array([462.3, 386.8, 300.1, 211.6])
    diff = B.adc_to_temperature_ntc(adc) - B.adc_to_temperature_ntc(
        adc, coefficients=B.SH_SCRIPT)
    assert diff.min() > 12.0
    assert diff.max() < 22.0


def test_temperature_is_monotonic_across_the_whole_data_range():
    """NTC: higher ADC must mean lower temperature, everywhere D3 actually sits."""
    adc = np.arange(275.0, 560.0, 1.0)
    t = B.adc_to_temperature_ntc(adc)
    assert np.all(np.diff(t) < 0)


def test_temperatures_over_the_data_range_are_physically_plausible():
    t = B.adc_to_temperature_ntc(np.array([281.0, 554.0]))
    assert 45.0 < t[0] < 70.0        # hottest sample seen
    assert 10.0 < t[1] < 30.0        # coldest sample seen


# ---------------------------------------------------------------------------
# bus voltage -- kept, unscaled, and NOT offset
# ---------------------------------------------------------------------------

def test_bus_voltage_has_no_midrail_offset():
    """The bug: subtracting 2.5 V from a unipolar divider gives VDC = -22 V."""
    assert B.adc_to_bus_voltage(np.array([507.0]))[0] > 0


def test_bus_voltage_is_pin_volts_by_default():
    assert B.adc_to_bus_voltage(np.array([1023.0]))[0] == pytest.approx(5.0)


def test_bus_voltage_applies_a_divider_ratio():
    assert B.adc_to_bus_voltage(np.array([1023.0]), divider_ratio=10.0)[0] == \
        pytest.approx(50.0)


# ---------------------------------------------------------------------------
# label map
# ---------------------------------------------------------------------------

def test_nine_conditions_totalling_the_published_count():
    assert len(B.FILE_TO_LABEL) == 9
    assert sum(n for _, _, n in B.FILE_TO_LABEL.values()) == 10_892


def test_f_codes_are_unique_and_complete():
    codes = sorted(c for c, _, _ in B.FILE_TO_LABEL.values())
    assert codes == [f"F{i}" for i in range(9)]


def test_families_match_the_published_grouping():
    fam = {c: f for c, f, _ in B.FILE_TO_LABEL.values()}
    assert fam["F0"] == "normal"
    assert fam["F1"] == fam["F2"] == "open_circuit"
    assert fam["F3"] == fam["F4"] == fam["F5"] == "short_circuit"
    assert fam["F6"] == fam["F7"] == fam["F8"] == "over_temp"


def test_dead_channels_are_not_in_the_used_set():
    assert set(B.USED_CHANNELS).isdisjoint(B.DROPPED_CHANNELS)
    assert set(B.DROPPED_CHANNELS) == {"VDC", "IDC", "VD"}


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

@needs_data
def test_loads_nine_recordings():
    assert len(B.load(ROOT)) == 9


@needs_data
def test_row_counts_match_the_published_labels_exactly():
    for r in B.load(ROOT):
        assert r.provenance["row_count_verified"], r.group
        assert r.condition["n_samples"] == r.provenance["expected_rows"]


@needs_data
def test_total_sample_count():
    assert sum(r.condition["n_samples"] for r in B.load(ROOT)) == 10_892


@needs_data
def test_sample_rate_is_ten_hertz():
    for r in B.load(ROOT):
        assert r.provenance["sample_interval_s"]["median"] == pytest.approx(0.1, abs=0.005)
        assert all(fs == 10.0 for fs in r.fs.values())


@needs_data
def test_groups_are_the_condition_codes():
    assert sorted(r.group for r in B.load(ROOT)) == [f"F{i}" for i in range(9)]


@needs_data
def test_dropped_channels_are_absent_from_signals():
    for r in B.load(ROOT):
        for dead in B.DROPPED_CHANNELS:
            assert dead not in r.signals


@needs_data
def test_raw_adc_is_retained_alongside_the_conversion():
    """So a data-notes document can show both without re-reading the files."""
    r = B.load(ROOT)[0]
    assert "Ia_adc" in r.signals and "T1_adc" in r.signals


@needs_data
def test_over_temp_conditions_heat_the_named_half_bridge():
    """
    The physical check that validates the whole conversion chain:
    F6 = HB1, F7 = HB1&2, F8 = HB3, per the source filenames.
    """
    by = {r.group: r for r in B.load(ROOT)}
    base = by["F0"]
    b1, b2, b3 = (float(base.signals[c].mean()) for c in ("T1", "T2", "T3"))

    f6 = by["F6"]
    assert float(f6.signals["T1"].mean()) > b1 + 15

    f7 = by["F7"]
    assert float(f7.signals["T1"].mean()) > b1 + 10
    assert float(f7.signals["T2"].mean()) > b2 + 5

    f8 = by["F8"]
    assert float(f8.signals["T3"].mean()) > b3 + 10


@needs_data
def test_over_temp_faults_reach_a_plausible_temperature():
    """Not 28 C. The shipped coefficients said 28 C and that was the tell."""
    by = {r.group: r for r in B.load(ROOT)}
    assert float(by["F6"].signals["T1"].max()) > 40.0
    assert float(by["F6"].signals["T1"].max()) < 80.0


@needs_data
def test_normal_condition_sits_near_ambient():
    f0 = {r.group: r for r in B.load(ROOT)}["F0"]
    for c in ("T1", "T2", "T3"):
        assert 15.0 < float(f0.signals[c].mean()) < 35.0


@needs_data
def test_short_circuit_perturbs_phase_current():
    """F3 (HB1 low-side SC) is the clearest electrical signature in the set."""
    by = {r.group: r for r in B.load(ROOT)}
    assert abs(float(by["F3"].signals["Ia"].mean())
               - float(by["F0"].signals["Ia"].mean())) > 1.0


@needs_data
def test_provenance_records_the_rejected_csv_and_the_reason():
    r = B.load(ROOT)[0]
    assert r.provenance["converted_csv_used"] is False
    assert "-22" in r.provenance["converted_csv_reason"]
    assert "REFITTED" in r.provenance["conversion"]["T1/T2/T3"]


@needs_data
def test_calibration_files_are_seen_and_excluded():
    present = B.calibration_files_present(ROOT)
    assert all(present.values())
    groups = {r.group for r in B.load(ROOT)}
    assert not any("SETPOINT" in g for g in groups)


@needs_data
def test_stages_split_electrical_from_thermal_conditions():
    by = {r.group: r for r in B.load(ROOT)}
    assert by["F1"].stage == "S3"        # inverter switch fault
    assert by["F3"].stage == "S3"
    assert by["F6"].stage == "S2"        # thermal, DC-link side


@needs_data
def test_a_short_file_would_be_caught(tmp_path):
    """The strict_counts guard must actually fire."""
    import shutil
    for f in B.FILE_TO_LABEL:
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    victim = tmp_path / "HB1_LOW_SIDE_SC.txt"
    lines = victim.read_text(encoding="utf-8", errors="replace").splitlines(True)
    victim.write_text("".join(lines[:-10]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 407"):
        B.load(str(tmp_path))
